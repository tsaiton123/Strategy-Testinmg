from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv
import torch.nn as nn
import matplotlib.pyplot as plt


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


def train_and_evaluate(config, df, output_dir):
    """Train and evaluate a single configuration"""

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()

    # Training environment
    env = DummyVecEnv([make_env(train_df, config['window'], config['action_mode'],
                            config['max_leverage'], config['cost_bps'])] * 2)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=config['gamma'])

    # Model configuration
    policy_kwargs = dict(
        net_arch=config['net_arch'],
        activation_fn=getattr(nn, config['activation_fn'])
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(config['learning_rate'], config['learning_rate'] * 0.1),
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=linear_schedule(config['clip_range'], config['clip_range'] * 0.5),
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0,
        seed=42
    )

    # Training
    print(f"Training {config['name']}...")
    model.learn(total_timesteps=config['timesteps'])

    # Save model
    model_dir = os.path.join(output_dir, config['name'])
    os.makedirs(model_dir, exist_ok=True)
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
    env.save(vecnorm_path)
    model.save(os.path.join(model_dir, "agent"))

    # Evaluation
    eval_env = DummyVecEnv([make_env(train_df, config['window'], config['action_mode'],
                                   config['max_leverage'], config['cost_bps'])])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    returns, positions = [], []
    done = [False]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        old_pos = getattr(eval_env.envs[0], '_pos', 0.0)
        obs, reward, done, info = eval_env.step(action)

        current_pos = info[0].get("position", 0.0)
        positions.append(current_pos)

        # Calculate portfolio return
        if hasattr(eval_env.envs[0], '_t') and eval_env.envs[0]._t > 0:
            market_return = eval_env.envs[0].rets[eval_env.envs[0]._t - 1]
            turnover = abs(current_pos - old_pos)
            cost = eval_env.envs[0].cost * turnover
            portfolio_return = old_pos * market_return - cost
            returns.append(portfolio_return)

    # Calculate statistics
    if returns:
        eq = np.cumprod(1 + np.array(returns))
        total_return = eq[-1] - 1
        cagr = (eq[-1] ** (252*24*12/len(eq))) - 1
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252*24*12)
        max_dd = np.min((eq / np.maximum.accumulate(eq)) - 1)
        win_rate = np.mean(np.array(returns) > 0)
        volatility = np.std(returns) * np.sqrt(252*24*12)
    else:
        total_return = cagr = sharpe = max_dd = win_rate = volatility = 0

    results = {
        'name': config['name'],
        'config': config,
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'win_rate': float(win_rate),
        'volatility': float(volatility),
        'num_trades': len(returns),
        'avg_position': float(np.mean(np.abs(positions))) if positions else 0,
    }

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="artifacts/rl_phase3_comparison")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    # Define configurations to compare
    configurations = [
        {
            'name': 'phase2_baseline',
            'window': 64,
            'action_mode': 'box',
            'max_leverage': 0.5,
            'cost_bps': 5.0,
            'learning_rate': 5e-4,
            'n_steps': 4096,
            'batch_size': 128,
            'n_epochs': 20,
            'gamma': 0.995,
            'gae_lambda': 0.98,
            'clip_range': 0.3,
            'ent_coef': 0.001,
            'vf_coef': 0.25,
            'max_grad_norm': 1.0,
            'net_arch': [512, 512, 256],
            'activation_fn': 'Tanh',
            'timesteps': 20000,
        },
        {
            'name': 'phase3_optimized',
            'window': 64,
            'action_mode': 'box',
            'max_leverage': 0.4,
            'cost_bps': 4.0,
            'learning_rate': 3e-4,
            'n_steps': 4096,
            'batch_size': 256,
            'n_epochs': 15,
            'gamma': 0.995,
            'gae_lambda': 0.98,
            'clip_range': 0.25,
            'ent_coef': 0.002,
            'vf_coef': 0.2,
            'max_grad_norm': 0.8,
            'net_arch': [512, 512, 256, 128],
            'activation_fn': 'Tanh',
            'timesteps': 20000,
        },
        {
            'name': 'phase3_conservative',
            'window': 64,
            'action_mode': 'box',
            'max_leverage': 0.3,
            'cost_bps': 3.0,
            'learning_rate': 1e-4,
            'n_steps': 4096,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.001,
            'vf_coef': 0.25,
            'max_grad_norm': 1.0,
            'net_arch': [256, 256, 128],
            'activation_fn': 'ReLU',
            'timesteps': 20000,
        },
        {
            'name': 'phase3_aggressive',
            'window': 64,
            'action_mode': 'box',
            'max_leverage': 0.7,
            'cost_bps': 2.0,
            'learning_rate': 5e-4,
            'n_steps': 4096,
            'batch_size': 512,
            'n_epochs': 20,
            'gamma': 0.999,
            'gae_lambda': 0.99,
            'clip_range': 0.3,
            'ent_coef': 0.005,
            'vf_coef': 0.3,
            'max_grad_norm': 1.5,
            'net_arch': [768, 512, 256],
            'activation_fn': 'Tanh',
            'timesteps': 20000,
        }
    ]

    # Train and evaluate all configurations
    results = []
    for config in configurations:
        try:
            result = train_and_evaluate(config, df, args.out)
            results.append(result)
            print(f"{config['name']}: CAGR={result['cagr']:.2%}, Sharpe={result['sharpe']:.2f}, MaxDD={result['max_dd']:.2%}")
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            results.append({'name': config['name'], 'error': str(e)})

    # Save results
    with open(os.path.join(args.out, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Create comparison chart
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        names = [r['name'] for r in valid_results]
        sharpes = [r['sharpe'] for r in valid_results]
        cagrs = [r['cagr'] * 100 for r in valid_results]  # Convert to percentage
        max_dds = [abs(r['max_dd']) * 100 for r in valid_results]  # Convert to positive percentage

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Sharpe comparison
        ax1.bar(names, sharpes, color='blue', alpha=0.7)
        ax1.set_title('Sharpe Ratio Comparison')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.tick_params(axis='x', rotation=45)

        # CAGR comparison
        ax2.bar(names, cagrs, color='green', alpha=0.7)
        ax2.set_title('CAGR Comparison')
        ax2.set_ylabel('CAGR (%)')
        ax2.tick_params(axis='x', rotation=45)

        # Max DD comparison
        ax3.bar(names, max_dds, color='red', alpha=0.7)
        ax3.set_title('Maximum Drawdown Comparison')
        ax3.set_ylabel('Max DD (%)')
        ax3.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "comparison_chart.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # Print summary
        print(f"\n=== Phase 3 Comparison Results ===")
        print(f"{'Configuration':<20} {'CAGR':<8} {'Sharpe':<8} {'MaxDD':<8} {'WinRate':<8}")
        print("-" * 60)
        for r in valid_results:
            print(f"{r['name']:<20} {r['cagr']:>6.1%} {r['sharpe']:>7.2f} {r['max_dd']:>6.1%} {r['win_rate']:>6.1%}")

        # Find best configuration
        best_config = max(valid_results, key=lambda x: x['sharpe'])
        print(f"\nBest configuration: {best_config['name']}")
        print(f"Sharpe: {best_config['sharpe']:.2f}")
        print(f"CAGR: {best_config['cagr']:.2%}")


if __name__ == "__main__":
    main()