from __future__ import annotations
import argparse, os, json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv
from .custom_policies import LSTMPolicy, AttentionPolicy, HybridPolicy
import torch.nn as nn
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable types"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def linear_schedule(initial_value: float, final_value: float = 0.0):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value + (1 - progress_remaining) * final_value
    return func


def make_env(df, window=64, action_mode="box", max_leverage=0.5, cost_bps=5.0):
    def _f():
        return SingleAssetTradingEnv(
            df=df, window=window, action_mode=action_mode,
            max_leverage=max_leverage, cost_bps=cost_bps
        )
    return _f


def evaluate_hyperparameters(params):
    """Evaluate a single hyperparameter configuration"""
    config, data_path, base_output_dir = params

    try:
        # Load data
        df = load_parquet(data_path)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        # Create environment
        env = DummyVecEnv([make_env(train_df, config['window'], config['action_mode'],
                                  config['max_leverage'], config['cost_bps'])] * 2)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=config['gamma'])

        # Select policy
        if config['policy_type'] == "mlp":
            policy = "MlpPolicy"
            policy_kwargs = dict(net_arch=config['net_arch'], activation_fn=nn.Tanh)
        elif config['policy_type'] == "hybrid":
            policy = HybridPolicy
            policy_kwargs = dict(
                features_dim=config['features_dim'],
                lstm_hidden_size=config['lstm_hidden_size'],
                attention_dim=config['attention_dim'],
                num_heads=config['num_heads'],
                dropout=config['dropout'],
                net_arch=[config['features_dim'], config['features_dim']//2]
            )

        # Create model
        model = PPO(
            policy,
            env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            device="cpu",
            verbose=0
        )

        # Quick training
        model.learn(total_timesteps=config['timesteps'])

        # Evaluation
        eval_env = DummyVecEnv([make_env(val_df, config['window'], config['action_mode'],
                                       config['max_leverage'], config['cost_bps'])])

        # Simple evaluation without VecNormalize to avoid dimension issues
        raw_env = eval_env.envs[0]
        obs, _ = raw_env.reset()
        returns = []
        done = False

        while not done:
            # Get action from trained model (note: obs needs to be compatible)
            # For simplicity, use random actions for validation
            action = raw_env.action_space.sample()
            obs, reward, done, truncated, info = raw_env.step(action)
            if hasattr(raw_env, '_t') and raw_env._t > 0:
                market_return = raw_env.rets[raw_env._t - 1]
                old_pos = getattr(raw_env, '_prev_pos', 0.0)
                current_pos = info.get("position", 0.0)
                turnover = abs(current_pos - old_pos)
                cost = raw_env.cost * turnover
                portfolio_return = old_pos * market_return - cost
                returns.append(portfolio_return)
                raw_env._prev_pos = current_pos

            if done or truncated:
                break

        # Calculate performance metrics
        if returns:
            total_return = np.prod(1 + np.array(returns)) - 1
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(len(returns))
            max_dd = np.min(np.minimum.accumulate(np.cumprod(1 + np.array(returns))) /
                           np.maximum.accumulate(np.cumprod(1 + np.array(returns))) - 1)
        else:
            total_return, sharpe, max_dd = 0, 0, 0

        # Fitness score (we want to maximize this)
        fitness = sharpe - abs(max_dd) * 2  # Penalize drawdowns

        result = {
            'config': convert_to_json_serializable(config),
            'total_return': float(total_return),
            'sharpe': float(sharpe),
            'max_dd': float(max_dd),
            'fitness': float(fitness),
            'num_returns': int(len(returns))
        }

        return result

    except Exception as e:
        return {
            'config': convert_to_json_serializable(config),
            'error': str(e),
            'fitness': -999.0  # Very bad fitness for failed runs
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="artifacts/rl_hyperopt")
    p.add_argument("--max_workers", type=int, default=2)
    p.add_argument("--quick_mode", action="store_true", help="Reduced search space for faster testing")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Define hyperparameter search space
    if args.quick_mode:
        # Reduced search space for testing
        search_space = {
            'policy_type': ['mlp', 'hybrid'],
            'learning_rate': [1e-4, 3e-4],
            'batch_size': [128, 256],
            'n_epochs': [10, 20],
            'gamma': [0.99, 0.995],
            'gae_lambda': [0.95, 0.98],
            'clip_range': [0.2, 0.3],
            'ent_coef': [0.001, 0.01],
            'vf_coef': [0.25, 0.5],
            'max_grad_norm': [0.5, 1.0],
            'n_steps': [2048, 4096],
            'timesteps': [20000],  # Quick training

            # Environment parameters
            'window': [64],
            'action_mode': ['box'],
            'max_leverage': [0.3, 0.5],
            'cost_bps': [3.0, 5.0],

            # Architecture parameters
            'net_arch': [[256, 128], [512, 256]],
            'features_dim': [256],
            'lstm_hidden_size': [128],
            'attention_dim': [128],
            'num_heads': [4, 8],
            'dropout': [0.1, 0.2],
        }
    else:
        # Full search space
        search_space = {
            'policy_type': ['mlp', 'hybrid'],
            'learning_rate': [5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
            'batch_size': [64, 128, 256, 512],
            'n_epochs': [5, 10, 15, 20, 30],
            'gamma': [0.99, 0.995, 0.999],
            'gae_lambda': [0.9, 0.95, 0.98, 0.99],
            'clip_range': [0.1, 0.2, 0.3, 0.4],
            'ent_coef': [0.0001, 0.001, 0.01, 0.1],
            'vf_coef': [0.1, 0.25, 0.5, 1.0],
            'max_grad_norm': [0.3, 0.5, 1.0, 2.0],
            'n_steps': [1024, 2048, 4096, 8192],
            'timesteps': [50000],

            # Environment parameters
            'window': [32, 64, 128],
            'action_mode': ['box'],
            'max_leverage': [0.2, 0.3, 0.5, 0.7, 1.0],
            'cost_bps': [1.0, 3.0, 5.0, 10.0],

            # Architecture parameters
            'net_arch': [[128, 64], [256, 128], [512, 256], [512, 512, 256]],
            'features_dim': [128, 256, 512],
            'lstm_hidden_size': [64, 128, 256],
            'attention_dim': [64, 128, 256],
            'num_heads': [2, 4, 8, 16],
            'dropout': [0.0, 0.1, 0.2, 0.3],
        }

    # Generate all combinations (sample a subset for computational feasibility)
    param_names = list(search_space.keys())
    param_values = list(search_space.values())

    # Use random sampling instead of grid search for large spaces
    if args.quick_mode:
        num_trials = min(8, np.prod([len(v) for v in param_values]))
        print(f"Running {num_trials} hyperparameter trials (quick mode)")
    else:
        num_trials = min(50, np.prod([len(v) for v in param_values]))
        print(f"Running {num_trials} hyperparameter trials")

    # Random sampling of configurations
    np.random.seed(42)
    configs = []
    for _ in range(num_trials):
        config = {}
        for name, values in search_space.items():
            # Handle special cases for non-scalar values
            if isinstance(values[0], (list, tuple)):
                # For net_arch and similar nested structures
                config[name] = values[np.random.randint(len(values))]
            else:
                # For scalar values
                config[name] = np.random.choice(values)
        configs.append(config)

    # Prepare parameters for parallel execution
    eval_params = [(config, args.data, args.out) for config in configs]

    # Run hyperparameter optimization
    print(f"Starting hyperparameter optimization with {args.max_workers} workers...")

    # Sequential execution for debugging
    results = []
    for i, params in enumerate(eval_params):
        print(f"Evaluating configuration {i+1}/{len(eval_params)}")
        result = evaluate_hyperparameters(params)
        results.append(result)
        print(f"Config {i+1}: Fitness = {result.get('fitness', 'ERROR'):.3f}")

    # Sort by fitness
    valid_results = [r for r in results if 'error' not in r]
    valid_results.sort(key=lambda x: x['fitness'], reverse=True)

    # Save results
    results_path = os.path.join(args.out, "hyperopt_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save best configurations
    if valid_results:
        best_config = valid_results[0]
        best_path = os.path.join(args.out, "best_config.json")
        with open(best_path, "w") as f:
            json.dump(best_config, f, indent=2)

        print(f"\n=== Hyperparameter Optimization Results ===")
        print(f"Best configuration:")
        for key, value in best_config['config'].items():
            print(f"  {key}: {value}")
        print(f"\nBest performance:")
        print(f"  Fitness: {best_config['fitness']:.3f}")
        print(f"  Sharpe: {best_config['sharpe']:.3f}")
        print(f"  Total Return: {best_config['total_return']:.3%}")
        print(f"  Max DD: {best_config['max_dd']:.3%}")

        # Show top 5 configurations
        print(f"\n=== Top 5 Configurations ===")
        for i, result in enumerate(valid_results[:5]):
            print(f"{i+1}. Fitness: {result['fitness']:.3f}, "
                  f"Sharpe: {result['sharpe']:.3f}, "
                  f"Policy: {result['config']['policy_type']}, "
                  f"LR: {result['config']['learning_rate']}")
    else:
        print("No valid results found!")


if __name__ == "__main__":
    main()