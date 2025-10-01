from __future__ import annotations
import argparse, os, json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv
from .custom_policies import LSTMPolicy
import torch.nn as nn


def linear_schedule(initial_value: float):
    """Simple linear learning rate schedule"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env(df, window=64, action_mode="box", max_leverage=0.5, cost_bps=5.0):
    def _f():
        return SingleAssetTradingEnv(
            df=df, window=window, action_mode=action_mode,
            max_leverage=max_leverage, cost_bps=cost_bps
        )
    return _f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--timesteps", type=int, default=1_500_000)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--action_mode", choices=["discrete","box"], default="box")
    p.add_argument("--cost_bps", type=float, default=5.0)
    p.add_argument("--max_leverage", type=float, default=0.5)
    p.add_argument("--out", default="artifacts/rl_phase3_lstm_simple_extended")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=10)
    args = p.parse_args()

    print(f"🚀 Starting Simple Extended LSTM Training")
    print(f"📈 Timesteps: {args.timesteps:,}")
    print(f"📦 Batch size: {args.batch_size}")

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    # 90% for training
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx].copy()

    print(f"📊 Training data: {len(train_df):,} bars")

    # Training environment - keep it simple with fewer envs
    env = DummyVecEnv([make_env(train_df, args.window, args.action_mode,
                            args.max_leverage, args.cost_bps)] * 4)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)

    # Enhanced but stable LSTM configuration
    policy_kwargs = dict(
        features_dim=320,
        lstm_hidden_size=160,
        num_lstm_layers=2,
        dropout=0.1,
        net_arch=[320, 256, 128]
    )

    # Stable PPO configuration
    model = PPO(
        LSTMPolicy,
        env,
        learning_rate=linear_schedule(args.learning_rate),
        n_steps=4096,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=1,
        seed=42
    )

    print(f"\n🏃 Starting training...")

    # Simple training without complex callbacks
    model.learn(total_timesteps=args.timesteps)

    # Save model
    vecnorm_path = os.path.join(args.out, "vecnormalize.pkl")
    env.save(vecnorm_path)
    model.save(os.path.join(args.out, "lstm_agent"))

    print(f"\n📊 Running evaluation...")

    # Evaluation
    eval_env = DummyVecEnv([make_env(train_df, args.window, args.action_mode,
                                   args.max_leverage, args.cost_bps)])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    pos_hist, returns = [], []
    done = [False]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        old_pos = getattr(eval_env.envs[0], '_pos', 0.0)
        obs, reward, done, info = eval_env.step(action)

        current_pos = info[0].get("position", 0.0)
        pos_hist.append(current_pos)

        # Calculate portfolio return
        if hasattr(eval_env.envs[0], '_t') and eval_env.envs[0]._t > 0:
            market_return = eval_env.envs[0].rets[eval_env.envs[0]._t - 1]
            turnover = abs(current_pos - old_pos)
            cost = eval_env.envs[0].cost * turnover
            portfolio_return = old_pos * market_return - cost
            returns.append(portfolio_return)

    # Performance analysis
    import matplotlib.pyplot as plt
    import pandas as pd

    eq = (1 + pd.Series(returns)).cumprod() if returns else pd.Series([1.0])

    # Simple plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    eq.plot(ax=ax1, title="Extended LSTM - Equity Curve", color='blue')
    ax1.grid(True)

    pd.Series(pos_hist).plot(ax=ax2, title="Position History", color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "lstm_extended_equity.png"), dpi=150)
    plt.close()

    # Calculate statistics
    returns_series = pd.Series(returns) if returns else pd.Series([0.0])

    stats = {
        "timesteps": args.timesteps,
        "total_return": float(eq.iloc[-1] - 1) if len(eq) > 0 else 0.0,
        "cagr": float(eq.iloc[-1] ** (252*24*12/len(eq)) - 1) if len(eq) > 1 else 0.0,
        "sharpe": float((np.mean(returns) / (np.std(returns) + 1e-12)) * (252*24*12) ** 0.5) if len(returns) > 0 and np.std(returns) > 0 else 0.0,
        "max_drawdown": float(((eq / eq.cummax()) - 1).min()) if len(eq) > 0 else 0.0,
        "volatility": float(np.std(returns) * np.sqrt(252*24*12)) if returns else 0.0,
        "win_rate": float((np.array(returns) > 0).mean()) if returns else 0.0,
        "avg_position": float(np.mean(np.abs(pos_hist))) if pos_hist else 0.0,
        "total_trades": len(returns),
    }

    with open(os.path.join(args.out, "lstm_result.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n🎉 Extended LSTM Results:")
    print(f"📈 CAGR: {stats['cagr']:.2%}")
    print(f"⚡ Sharpe Ratio: {stats['sharpe']:.2f}")
    print(f"📉 Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"🎯 Win Rate: {stats['win_rate']:.2%}")

    # Compare with original
    original_result_path = "artifacts/rl_phase3_lstm/lstm_result.json"
    if os.path.exists(original_result_path):
        with open(original_result_path, 'r') as f:
            original_stats = json.load(f)

        print(f"\n📊 vs Original LSTM:")
        print(f"CAGR: {original_stats.get('cagr', 0)*100:.1f}% → {stats['cagr']*100:.1f}%")
        print(f"Sharpe: {original_stats.get('sharpe', 0):.1f} → {stats['sharpe']:.1f}")

    print(f"\n💾 Model saved to: {args.out}")


if __name__ == "__main__":
    main()