from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv
from .low_frequency_env import LowFrequencyTradingEnv
import torch.nn as nn


def train_and_evaluate_config(config, df, output_dir):
    """Train and evaluate a single trading configuration"""

    config_name = config['name']
    print(f"Training {config_name}...")

    # Create environment based on config type
    if config['env_type'] == 'high_freq':
        env_func = lambda: SingleAssetTradingEnv(
            df=df,
            window=config['window'],
            action_mode=config['action_mode'],
            max_leverage=config['max_leverage'],
            cost_bps=config['cost_bps'],
            action_tau=config.get('action_tau', 0.1),
            hold_min=config.get('hold_min', 5)
        )
    else:  # low_freq
        env_func = lambda: LowFrequencyTradingEnv(
            df=df,
            window=config['window'],
            action_mode=config['action_mode'],
            max_leverage=config['max_leverage'],
            cost_bps=config['cost_bps'],
            action_tau=config.get('action_tau', 0.05),
            hold_min=config.get('hold_min', 20),
            rebalance_threshold=config.get('rebalance_threshold', 0.1),
            transaction_penalty_multiplier=config.get('transaction_penalty', 5.0),
            confidence_threshold=config.get('confidence_threshold', 0.3)
        )

    # Training environment
    env = DummyVecEnv([env_func] * 2)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 4096),
        batch_size=config.get('batch_size', 256),
        n_epochs=config.get('n_epochs', 15),
        gamma=config.get('gamma', 0.995),
        gae_lambda=config.get('gae_lambda', 0.98),
        clip_range=config.get('clip_range', 0.2),
        ent_coef=config.get('ent_coef', 0.002),
        vf_coef=config.get('vf_coef', 0.25),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        policy_kwargs=dict(
            net_arch=config.get('net_arch', [512, 512, 256]),
            activation_fn=nn.Tanh
        ),
        verbose=0,
        seed=42
    )

    # Train
    model.learn(total_timesteps=config['timesteps'])

    # Save
    model_dir = os.path.join(output_dir, config_name)
    os.makedirs(model_dir, exist_ok=True)
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
    env.save(vecnorm_path)
    model.save(os.path.join(model_dir, "agent"))

    # Evaluation
    eval_env = DummyVecEnv([env_func])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    returns, positions, trades = [], [], []
    trade_count = 0
    done = [False]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        old_pos = getattr(eval_env.envs[0], '_pos', 0.0)
        obs, reward, done, info = eval_env.step(action)

        current_pos = info[0].get("position", 0.0)
        positions.append(current_pos)

        # Count trades
        if abs(current_pos - old_pos) > 0.01:  # Significant position change
            trade_count += 1
            trades.append(1)
        else:
            trades.append(0)

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
        cagr = (eq[-1] ** (252*24*12/len(eq))) - 1 if len(eq) > 1 else 0
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252*24*12)
        max_dd = np.min((eq / np.maximum.accumulate(eq)) - 1)
        total_turnover = sum([abs(positions[i] - positions[i-1]) for i in range(1, len(positions))])
        transaction_cost_drag = total_turnover * config['cost_bps'] / 10000
    else:
        total_return = cagr = sharpe = max_dd = total_turnover = transaction_cost_drag = 0

    # Get trading stats if available
    if hasattr(eval_env.envs[0], 'get_trading_stats'):
        trading_stats = eval_env.envs[0].get_trading_stats()
    else:
        trading_stats = {}

    results = {
        'name': config_name,
        'config': config,
        'performance': {
            'total_return': float(total_return),
            'cagr': float(cagr),
            'sharpe': float(sharpe),
            'max_dd': float(max_dd),
            'num_periods': len(returns)
        },
        'trading': {
            'total_trades': trade_count,
            'total_turnover': float(total_turnover),
            'avg_turnover': float(total_turnover / max(len(positions), 1)),
            'trading_frequency': float(trade_count / max(len(positions), 1)),
            'transaction_cost_drag': float(transaction_cost_drag),
            'avg_position': float(np.mean(np.abs(positions))) if positions else 0,
        },
        'efficiency': {
            'return_per_trade': float(total_return / max(trade_count, 1)),
            'sharpe_per_trade': float(sharpe / max(trade_count, 1)),
            'cost_efficiency': float(total_return / max(transaction_cost_drag, 0.001))
        },
        'returns': returns[:100],  # Sample for plotting
        'positions': positions[:100]  # Sample for plotting
    }

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="artifacts/rl_frequency_comparison")
    p.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps per config")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    # Split for training
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()

    print(f"Comparing High vs Low Frequency Trading Strategies")
    print(f"Training data: {len(train_df)} periods")
    print(f"Training timesteps per config: {args.timesteps}")

    # Define configurations to compare
    configurations = [
        {
            'name': 'high_frequency_baseline',
            'env_type': 'high_freq',
            'window': 64,
            'action_mode': 'box',
            'max_leverage': 0.5,
            'cost_bps': 5.0,
            'action_tau': 0.1,      # Weak smoothing
            'hold_min': 1,          # No holding constraint
            'timesteps': args.timesteps,
            'learning_rate': 5e-4,
        },
        {
            'name': 'high_frequency_smoothed',
            'env_type': 'high_freq',
            'window': 64,
            'action_mode': 'box',
            'max_leverage': 0.5,
            'cost_bps': 5.0,
            'action_tau': 0.3,      # Medium smoothing
            'hold_min': 5,          # Short holding
            'timesteps': args.timesteps,
            'learning_rate': 3e-4,
        },
        {
            'name': 'low_frequency_conservative',
            'env_type': 'low_freq',
            'window': 64,
            'action_mode': 'box',
            'max_leverage': 0.5,
            'cost_bps': 5.0,
            'action_tau': 0.05,     # Strong smoothing
            'hold_min': 15,         # Medium holding
            'rebalance_threshold': 0.1,
            'transaction_penalty': 3.0,
            'confidence_threshold': 0.2,
            'timesteps': args.timesteps,
            'learning_rate': 2e-4,
        },
        {
            'name': 'low_frequency_aggressive',
            'env_type': 'low_freq',
            'window': 64,
            'action_mode': 'box',
            'max_leverage': 0.5,
            'cost_bps': 5.0,
            'action_tau': 0.02,     # Very strong smoothing
            'hold_min': 30,         # Long holding
            'rebalance_threshold': 0.15,
            'transaction_penalty': 10.0,
            'confidence_threshold': 0.4,
            'timesteps': args.timesteps,
            'learning_rate': 1e-4,
        }
    ]

    # Train and evaluate all configurations
    results = []
    for config in configurations:
        try:
            result = train_and_evaluate_config(config, train_df, args.out)
            results.append(result)
            print(f"{config['name']}: CAGR={result['performance']['cagr']:.2%}, "
                  f"Sharpe={result['performance']['sharpe']:.2f}, "
                  f"Trades={result['trading']['total_trades']}")
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            results.append({'name': config['name'], 'error': str(e)})

    # Save results
    with open(os.path.join(args.out, "frequency_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Create comparison visualizations
    valid_results = [r for r in results if 'error' not in r]
    if len(valid_results) > 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        names = [r['name'].replace('_', '\n') for r in valid_results]

        # Performance comparison
        cagrs = [r['performance']['cagr'] * 100 for r in valid_results]
        sharpes = [r['performance']['sharpe'] for r in valid_results]

        ax1.bar(names, cagrs, alpha=0.7)
        ax1.set_title('CAGR Comparison (%)')
        ax1.set_ylabel('CAGR (%)')
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(names, sharpes, alpha=0.7, color='green')
        ax2.set_title('Sharpe Ratio Comparison')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.tick_params(axis='x', rotation=45)

        # Trading frequency comparison
        trade_counts = [r['trading']['total_trades'] for r in valid_results]
        turnover_rates = [r['trading']['avg_turnover'] * 100 for r in valid_results]

        ax3.bar(names, trade_counts, alpha=0.7, color='red')
        ax3.set_title('Total Trades')
        ax3.set_ylabel('Number of Trades')
        ax3.tick_params(axis='x', rotation=45)

        ax4.bar(names, turnover_rates, alpha=0.7, color='orange')
        ax4.set_title('Average Turnover (%)')
        ax4.set_ylabel('Turnover (%)')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "frequency_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # Efficiency analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        return_per_trade = [r['efficiency']['return_per_trade'] * 100 for r in valid_results]
        cost_efficiency = [r['efficiency']['cost_efficiency'] for r in valid_results]

        ax1.bar(names, return_per_trade, alpha=0.7, color='purple')
        ax1.set_title('Return per Trade (%)')
        ax1.set_ylabel('Return per Trade (%)')
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(names, cost_efficiency, alpha=0.7, color='brown')
        ax2.set_title('Cost Efficiency (Return/Cost)')
        ax2.set_ylabel('Efficiency Ratio')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "efficiency_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # Print detailed comparison
    print(f"\n=== FREQUENCY COMPARISON RESULTS ===")
    print(f"{'Strategy':<25} {'CAGR':<8} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8} {'Turnover':<10} {'RetPerTrade':<12}")
    print("-" * 90)

    for r in valid_results:
        print(f"{r['name']:<25} "
              f"{r['performance']['cagr']:>6.1%} "
              f"{r['performance']['sharpe']:>7.2f} "
              f"{r['performance']['max_dd']:>6.1%} "
              f"{r['trading']['total_trades']:>7} "
              f"{r['trading']['avg_turnover']:>8.3f} "
              f"{r['efficiency']['return_per_trade']:>10.3%}")

    # Identify best strategies
    if valid_results:
        best_sharpe = max(valid_results, key=lambda x: x['performance']['sharpe'])
        best_efficiency = max(valid_results, key=lambda x: x['efficiency']['return_per_trade'])

        print(f"\n=== BEST STRATEGIES ===")
        print(f"Best Sharpe: {best_sharpe['name']} (Sharpe: {best_sharpe['performance']['sharpe']:.2f})")
        print(f"Best Efficiency: {best_efficiency['name']} (Return/Trade: {best_efficiency['efficiency']['return_per_trade']:.3%})")


if __name__ == "__main__":
    main()