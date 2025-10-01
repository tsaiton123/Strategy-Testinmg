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


def make_low_freq_env(df, window=64, action_mode="box", max_leverage=0.5, cost_bps=5.0,
                      action_tau=0.05, hold_min=20, rebalance_threshold=0.1,
                      transaction_penalty_multiplier=5.0, confidence_threshold=0.3):
    def _f():
        return LowFrequencyTradingEnv(
            df=df,
            window=window,
            action_mode=action_mode,
            max_leverage=max_leverage,
            cost_bps=cost_bps,
            action_tau=action_tau,
            hold_min=hold_min,
            rebalance_threshold=rebalance_threshold,
            transaction_penalty_multiplier=transaction_penalty_multiplier,
            confidence_threshold=confidence_threshold
        )
    return _f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--action_mode", choices=["discrete","box"], default="box")
    p.add_argument("--cost_bps", type=float, default=5.0)
    p.add_argument("--max_leverage", type=float, default=0.5)
    p.add_argument("--action_tau", type=float, default=0.05, help="Position smoothing (lower = more smoothing)")
    p.add_argument("--hold_min", type=int, default=20, help="Minimum holding periods")
    p.add_argument("--rebalance_threshold", type=float, default=0.1, help="Only trade if change > threshold")
    p.add_argument("--transaction_penalty", type=float, default=5.0, help="Extra penalty multiplier for trading")
    p.add_argument("--confidence_threshold", type=float, default=0.3, help="Minimum signal strength to act")
    p.add_argument("--out", default="artifacts/rl_low_frequency")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    # Split data for train/validation (90/10 split)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx].copy()

    print(f"Training Low-Frequency Trading Agent...")
    print(f"Data points: {len(train_df)}")
    print(f"Transaction cost: {args.cost_bps} bps")
    print(f"Position smoothing (tau): {args.action_tau}")
    print(f"Minimum holding: {args.hold_min} periods")
    print(f"Rebalance threshold: {args.rebalance_threshold}")
    print(f"Transaction penalty: {args.transaction_penalty}x")
    print(f"Confidence threshold: {args.confidence_threshold}")

    # Training environment with low-frequency parameters
    env = DummyVecEnv([make_low_freq_env(
        train_df, args.window, args.action_mode, args.max_leverage, args.cost_bps,
        args.action_tau, args.hold_min, args.rebalance_threshold,
        args.transaction_penalty, args.confidence_threshold
    )] * 2)  # Only 2 parallel envs to reduce noise

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

    # PPO optimized for low-frequency trading
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(2e-4, 1e-5),  # Lower learning rate for stability
        n_steps=8192,                               # Longer rollouts for less frequent decisions
        batch_size=512,                             # Larger batches for stability
        n_epochs=20,                                # More epochs for better learning
        gamma=0.999,                                # Higher discount for long-term thinking
        gae_lambda=0.98,
        clip_range=linear_schedule(0.2, 0.05),      # Conservative clipping
        ent_coef=0.001,                             # Low entropy for more deterministic actions
        vf_coef=0.1,                                # Lower value function weight
        max_grad_norm=0.5,                          # Conservative gradient clipping
        policy_kwargs=dict(
            net_arch=[512, 512, 256, 128],          # Deep network for complex patterns
            activation_fn=nn.Tanh                   # Tanh for bounded outputs
        ),
        verbose=1,
        seed=42
    )

    # Training
    print(f"\nStarting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    # Save model
    vecnorm_path = os.path.join(args.out, "vecnormalize.pkl")
    env.save(vecnorm_path)
    model.save(os.path.join(args.out, "low_freq_agent"))

    # Evaluation with detailed trading statistics
    print(f"Running evaluation...")
    eval_env = DummyVecEnv([make_low_freq_env(
        train_df, args.window, args.action_mode, args.max_leverage, args.cost_bps,
        args.action_tau, args.hold_min, args.rebalance_threshold,
        args.transaction_penalty, args.confidence_threshold
    )])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    pos_hist, returns, rewards = [], [], []
    trading_stats = []
    done = [False]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        old_pos = getattr(eval_env.envs[0], '_pos', 0.0)
        obs, reward, done, info = eval_env.step(action)

        current_pos = info[0].get("position", 0.0)
        pos_hist.append(current_pos)
        rewards.append(reward[0])

        # Calculate actual portfolio return
        if hasattr(eval_env.envs[0], '_t') and eval_env.envs[0]._t > 0:
            market_return = eval_env.envs[0].rets[eval_env.envs[0]._t - 1]
            turnover = abs(current_pos - old_pos)
            cost = eval_env.envs[0].cost * turnover
            portfolio_return = old_pos * market_return - cost
            returns.append(portfolio_return)

        # Collect trading statistics
        if not done[0]:
            trading_stats.append({
                'turnover': info[0].get('turnover', 0),
                'trade_count': info[0].get('trade_count', 0),
                'avg_turnover': info[0].get('avg_turnover', 0),
                'holding_period': info[0].get('holding_period', 0)
            })

    # Get final trading statistics
    final_stats = eval_env.envs[0].get_trading_stats()

    # Performance analysis
    eq = (1 + pd.Series(returns)).cumprod() if returns else pd.Series([1.0])

    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Equity curve
    ax1.plot(eq.values)
    ax1.set_title('Low-Frequency Trading - Equity Curve')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True)

    # Position history
    ax2.plot(pos_hist)
    ax2.set_title('Position History')
    ax2.set_ylabel('Position Size')
    ax2.grid(True)

    # Turnover over time
    if trading_stats:
        turnovers = [s['turnover'] for s in trading_stats]
        ax3.plot(turnovers)
        ax3.set_title('Turnover Over Time')
        ax3.set_ylabel('Turnover')
        ax3.grid(True)

        # Rolling trade frequency
        trade_counts = [s['trade_count'] for s in trading_stats]
        if len(trade_counts) > 100:
            rolling_trades = pd.Series(trade_counts).diff().rolling(100).sum()
            ax4.plot(rolling_trades)
            ax4.set_title('Rolling Trade Frequency (100 periods)')
            ax4.set_ylabel('Trades per 100 periods')
            ax4.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "low_freq_analysis.png"), dpi=150)
    plt.close()

    # Calculate comprehensive statistics
    if returns:
        total_return = eq.iloc[-1] - 1
        cagr = (eq.iloc[-1] ** (252*24*12/len(eq))) - 1 if len(eq) > 1 else 0
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252*24*12)
        max_dd = ((eq / eq.cummax()) - 1).min()
        win_rate = (np.array(returns) > 0).mean()
        volatility = np.std(returns) * np.sqrt(252*24*12)
    else:
        total_return = cagr = sharpe = max_dd = win_rate = volatility = 0

    # Combine performance and trading statistics
    stats = {
        "training_config": {
            "timesteps": args.timesteps,
            "action_tau": args.action_tau,
            "hold_min": args.hold_min,
            "rebalance_threshold": args.rebalance_threshold,
            "transaction_penalty": args.transaction_penalty,
            "confidence_threshold": args.confidence_threshold,
            "cost_bps": args.cost_bps,
        },
        "performance": {
            "CAGR": float(cagr),
            "Total_Return": float(total_return),
            "Sharpe": float(sharpe),
            "MaxDD": float(max_dd),
            "Volatility": float(volatility),
            "Win_Rate": float(win_rate),
            "Avg_Return": float(np.mean(returns)) if returns else 0,
            "Num_Periods": len(returns),
        },
        "trading_stats": final_stats,
        "efficiency_metrics": {
            "Return_per_Trade": float(total_return / max(final_stats.get('total_trades', 1), 1)),
            "Sharpe_per_Trade": float(sharpe / max(final_stats.get('total_trades', 1), 1)),
            "Net_Return_After_Costs": float(total_return),
            "Transaction_Cost_Ratio": float(final_stats.get('transaction_cost_drag', 0) / max(abs(total_return), 0.001)),
        }
    }

    with open(os.path.join(args.out, "low_freq_result.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Print comprehensive results
    print(f"\n=== LOW-FREQUENCY TRADING RESULTS ===")
    print(f"Performance Metrics:")
    print(f"  CAGR: {stats['performance']['CAGR']:.2%}")
    print(f"  Sharpe: {stats['performance']['Sharpe']:.2f}")
    print(f"  Max DD: {stats['performance']['MaxDD']:.2%}")
    print(f"  Win Rate: {stats['performance']['Win_Rate']:.2%}")

    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {stats['trading_stats'].get('total_trades', 0)}")
    print(f"  Avg Turnover: {stats['trading_stats'].get('avg_turnover', 0):.4f}")
    print(f"  Trading Frequency: {stats['trading_stats'].get('trading_frequency', 0):.4f}")
    print(f"  Avg Holding Period: {stats['trading_stats'].get('avg_holding_period', 0):.1f} periods")
    print(f"  Transaction Cost Drag: {stats['trading_stats'].get('transaction_cost_drag', 0):.4f}")

    print(f"\nEfficiency Metrics:")
    print(f"  Return per Trade: {stats['efficiency_metrics']['Return_per_Trade']:.4%}")
    print(f"  Sharpe per Trade: {stats['efficiency_metrics']['Sharpe_per_Trade']:.3f}")
    print(f"  Transaction Cost Ratio: {stats['efficiency_metrics']['Transaction_Cost_Ratio']:.2%}")


if __name__ == "__main__":
    main()