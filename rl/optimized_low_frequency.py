from __future__ import annotations
import argparse, os, json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .data_loader import load_parquet
from .low_frequency_env import LowFrequencyTradingEnv
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd


def linear_schedule(initial_value: float, final_value: float = 0.0):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value + (1 - progress_remaining) * final_value
    return func


def make_optimized_env(df, **kwargs):
    def _f():
        return LowFrequencyTradingEnv(df=df, **kwargs)
    return _f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--out", default="artifacts/rl_optimized_low_freq")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    print(f"Training Optimized Low-Frequency Trading Agent...")

    # Optimized configuration based on comparison results
    optimized_configs = [
        {
            'name': 'balanced_low_freq',
            'params': {
                'window': 64,
                'action_mode': 'box',
                'max_leverage': 0.4,                    # Slightly lower leverage
                'cost_bps': 5.0,
                'action_tau': 0.1,                      # Moderate smoothing
                'hold_min': 10,                         # Moderate holding period
                'rebalance_threshold': 0.05,            # Lower threshold
                'transaction_penalty_multiplier': 2.0,  # Lower penalty
                'confidence_threshold': 0.2,            # Lower confidence needed
                'position_decay': 0.995,                # Slower decay
            }
        },
        {
            'name': 'selective_low_freq',
            'params': {
                'window': 64,
                'action_mode': 'box',
                'max_leverage': 0.6,                    # Higher leverage for fewer trades
                'cost_bps': 5.0,
                'action_tau': 0.03,                     # Strong smoothing
                'hold_min': 25,                         # Longer holding
                'rebalance_threshold': 0.12,            # Higher threshold
                'transaction_penalty_multiplier': 8.0,  # High penalty
                'confidence_threshold': 0.35,           # High confidence needed
                'position_decay': 0.99,                 # Moderate decay
            }
        }
    ]

    results = []

    for config in optimized_configs:
        print(f"\nTraining {config['name']}...")

        # Split data
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx].copy()

        # Training environment
        env = DummyVecEnv([make_optimized_env(train_df, **config['params'])] * 2)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.999)

        # PPO optimized for low frequency
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(1e-4, 1e-6),    # Conservative learning
            n_steps=8192,                                  # Long rollouts
            batch_size=256,
            n_epochs=25,                                   # More epochs
            gamma=0.999,                                   # Long-term focus
            gae_lambda=0.99,
            clip_range=linear_schedule(0.15, 0.05),        # Conservative clipping
            ent_coef=0.0005,                               # Low entropy
            vf_coef=0.1,
            max_grad_norm=0.3,                             # Conservative gradients
            policy_kwargs=dict(
                net_arch=[1024, 512, 256, 128],            # Deeper network
                activation_fn=nn.Tanh
            ),
            verbose=1,
            seed=42
        )

        # Train
        model.learn(total_timesteps=args.timesteps)

        # Save
        model_dir = os.path.join(args.out, config['name'])
        os.makedirs(model_dir, exist_ok=True)
        vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
        env.save(vecnorm_path)
        model.save(os.path.join(model_dir, "agent"))

        # Evaluation
        print(f"Evaluating {config['name']}...")
        eval_env = DummyVecEnv([make_optimized_env(train_df, **config['params'])])
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

        # Get trading statistics
        trading_stats = eval_env.envs[0].get_trading_stats()

        # Calculate performance
        if returns:
            eq = np.cumprod(1 + np.array(returns))
            total_return = eq[-1] - 1
            cagr = (eq[-1] ** (252*24*12/len(eq))) - 1 if len(eq) > 1 else 0
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252*24*12)
            max_dd = np.min((eq / np.maximum.accumulate(eq)) - 1)
            win_rate = (np.array(returns) > 0).mean()
        else:
            total_return = cagr = sharpe = max_dd = win_rate = 0

        result = {
            'name': config['name'],
            'config': config['params'],
            'performance': {
                'total_return': float(total_return),
                'cagr': float(cagr),
                'sharpe': float(sharpe),
                'max_dd': float(max_dd),
                'win_rate': float(win_rate),
                'num_periods': len(returns)
            },
            'trading_stats': trading_stats
        }

        results.append(result)

        print(f"{config['name']} Results:")
        print(f"  CAGR: {cagr:.2%}")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Max DD: {max_dd:.2%}")
        print(f"  Total Trades: {trading_stats.get('total_trades', 0)}")
        print(f"  Avg Turnover: {trading_stats.get('avg_turnover', 0):.4f}")

    # Save all results
    with open(os.path.join(args.out, "optimized_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Compare results
    print(f"\n=== OPTIMIZED LOW-FREQUENCY COMPARISON ===")
    print(f"{'Strategy':<20} {'CAGR':<8} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8} {'Turnover':<10}")
    print("-" * 75)

    for r in results:
        print(f"{r['name']:<20} "
              f"{r['performance']['cagr']:>6.1%} "
              f"{r['performance']['sharpe']:>7.2f} "
              f"{r['performance']['max_dd']:>6.1%} "
              f"{r['trading_stats'].get('total_trades', 0):>7} "
              f"{r['trading_stats'].get('avg_turnover', 0):>8.4f}")

    # Identify best strategy
    if results:
        best_strategy = max(results, key=lambda x: x['performance']['sharpe'])
        print(f"\nBest Strategy: {best_strategy['name']}")
        print(f"Sharpe: {best_strategy['performance']['sharpe']:.2f}")
        print(f"CAGR: {best_strategy['performance']['cagr']:.2%}")
        print(f"Trades: {best_strategy['trading_stats'].get('total_trades', 0)}")

        # Save best config for easy reuse
        with open(os.path.join(args.out, "best_config.json"), "w") as f:
            json.dump(best_strategy, f, indent=2)


if __name__ == "__main__":
    main()