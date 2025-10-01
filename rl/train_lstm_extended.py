from __future__ import annotations
import argparse, os, json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv
from .custom_policies import LSTMPolicy
import torch
import torch.nn as nn
import time


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Linear learning rate schedule"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value + (1 - progress_remaining) * final_value
    return func


def exponential_schedule(initial_value: float, decay_rate: float = 0.1):
    """Exponential decay schedule"""
    def func(progress_remaining: float) -> float:
        return initial_value * (decay_rate ** (1 - progress_remaining))
    return func


def make_env(df, window=64, action_mode="box", max_leverage=0.5, cost_bps=5.0):
    def _f():
        return SingleAssetTradingEnv(
            df=df, window=window, action_mode=action_mode,
            max_leverage=max_leverage, cost_bps=cost_bps
        )
    return _f


class ProgressCallback:
    """Custom callback to track training progress"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.start_time = time.time()
        self.last_log_time = time.time()

    def __call__(self, locals_, globals_):
        if locals_['self'].num_timesteps % 50000 == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            since_last = current_time - self.last_log_time

            progress = locals_['self'].num_timesteps / locals_['self']._total_timesteps
            print(f"\n📊 Training Progress: {progress:.1%}")
            print(f"⏱️  Timesteps: {locals_['self'].num_timesteps:,} / {locals_['self']._total_timesteps:,}")
            print(f"🕐 Elapsed: {elapsed/60:.1f} min | Last 50k: {since_last:.1f}s")
            print(f"📈 ETA: {(elapsed / progress - elapsed) / 60:.1f} min remaining")

            self.last_log_time = current_time

        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--timesteps", type=int, default=2_000_000)  # 2M timesteps (4x longer)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--action_mode", choices=["discrete","box"], default="box")
    p.add_argument("--cost_bps", type=float, default=5.0)
    p.add_argument("--max_leverage", type=float, default=0.5)
    p.add_argument("--out", default="artifacts/rl_phase3_lstm_extended")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=512)  # Larger batch
    p.add_argument("--n_epochs", type=int, default=15)  # More epochs
    p.add_argument("--n_envs", type=int, default=8)  # More parallel envs
    args = p.parse_args()

    print(f"🚀 Starting Extended LSTM Training")
    print(f"📈 Timesteps: {args.timesteps:,}")
    print(f"🔄 Environments: {args.n_envs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🎯 Device: {args.device}")

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    # Use 95% for training to maximize data
    split_idx = int(len(df) * 0.95)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    print(f"📊 Training data: {len(train_df):,} bars")
    print(f"📊 Validation data: {len(val_df):,} bars")

    # Enhanced training environment with more parallel envs
    env = DummyVecEnv([make_env(train_df, args.window, args.action_mode,
                            args.max_leverage, args.cost_bps)] * args.n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.999)

    # Validation environment
    eval_env = DummyVecEnv([make_env(val_df, args.window, args.action_mode,
                                   args.max_leverage, args.cost_bps)])

    # Enhanced LSTM configuration for longer training
    policy_kwargs = dict(
        features_dim=384,  # Increased feature dimension
        lstm_hidden_size=192,  # Larger LSTM
        num_lstm_layers=3,  # More layers
        dropout=0.15,  # Slightly less dropout for longer training
        net_arch=[384, 256, 128]  # Deeper network
    )

    # Advanced PPO configuration optimized for longer training
    model = PPO(
        LSTMPolicy,
        env,
        learning_rate=linear_schedule(args.learning_rate, args.learning_rate * 0.05),  # Longer decay
        n_steps=8192,  # Larger rollout buffer
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.999,  # Higher gamma for longer horizons
        gae_lambda=0.98,
        clip_range=linear_schedule(0.25, 0.05),  # Gradual clip range decay
        clip_range_vf=0.25,
        ent_coef=0.001,  # Fixed entropy coefficient
        vf_coef=0.3,
        max_grad_norm=0.8,  # Tighter gradient clipping
        use_sde=False,
        target_kl=0.015,  # Stricter KL constraint
        policy_kwargs=policy_kwargs,
        device=args.device,
        verbose=1,
        seed=42
    )

    # Setup callbacks for longer training
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # Save every 100k steps
        save_path=os.path.join(args.out, "checkpoints"),
        name_prefix="lstm_checkpoint"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.out, "best_model"),
        log_path=os.path.join(args.out, "eval_logs"),
        eval_freq=50_000,  # Evaluate every 50k steps
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )

    progress_callback = ProgressCallback(args.out)

    callbacks = [checkpoint_callback, eval_callback]

    print(f"\n🏃 Starting training...")
    start_time = time.time()

    # Extended training
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True
    )

    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/3600:.2f} hours")

    # Save final model and normalization
    vecnorm_path = os.path.join(args.out, "vecnormalize.pkl")
    env.save(vecnorm_path)
    model.save(os.path.join(args.out, "lstm_agent_extended"))

    print(f"\n📊 Running final evaluation...")

    # Comprehensive evaluation on full training data
    eval_env_full = DummyVecEnv([make_env(train_df, args.window, args.action_mode,
                                        args.max_leverage, args.cost_bps)])
    eval_env_full = VecNormalize.load(vecnorm_path, eval_env_full)
    eval_env_full.training = False
    eval_env_full.norm_reward = False

    obs = eval_env_full.reset()
    pos_hist, returns, rewards = [], [], []
    done = [False]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        old_pos = getattr(eval_env_full.envs[0], '_pos', 0.0)
        obs, reward, done, info = eval_env_full.step(action)

        current_pos = info[0].get("position", 0.0)
        pos_hist.append(current_pos)
        rewards.append(reward[0])

        # Calculate actual portfolio return
        if hasattr(eval_env_full.envs[0], '_t') and eval_env_full.envs[0]._t > 0:
            market_return = eval_env_full.envs[0].rets[eval_env_full.envs[0]._t - 1]
            turnover = abs(current_pos - old_pos)
            cost = eval_env_full.envs[0].cost * turnover
            portfolio_return = old_pos * market_return - cost
            returns.append(portfolio_return)

    # Performance analysis and visualization
    import matplotlib.pyplot as plt
    import pandas as pd

    eq = (1 + pd.Series(returns)).cumprod() if returns else pd.Series([1.0])

    # Enhanced plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Equity curve
    eq.plot(ax=ax1, title="Extended LSTM - Equity Curve", color='blue', linewidth=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Cumulative Return")

    # Position history
    pd.Series(pos_hist).plot(ax=ax2, title="Position History", color='green', alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Position")

    # Rolling Sharpe
    returns_series = pd.Series(returns) if returns else pd.Series([0.0])
    rolling_sharpe = returns_series.rolling(252).mean() / (returns_series.rolling(252).std() + 1e-8) * np.sqrt(252*24*12)
    rolling_sharpe.plot(ax=ax3, title="Rolling Sharpe Ratio (252-period)", color='purple')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel("Sharpe Ratio")

    # Drawdown
    drawdown = (eq / eq.cummax()) - 1
    drawdown.plot(ax=ax4, title="Drawdown", color='red', alpha=0.7)
    ax4.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylabel("Drawdown")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "lstm_extended_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate comprehensive statistics
    stats = {
        "policy_type": "lstm_extended",
        "timesteps": args.timesteps,
        "training_time_hours": training_time / 3600,
        "total_return": float(eq.iloc[-1] - 1) if len(eq) > 0 else 0.0,
        "cagr": float(eq.iloc[-1] ** (252*24*12/len(eq)) - 1) if len(eq) > 1 else 0.0,
        "sharpe": float((np.mean(returns) / (np.std(returns) + 1e-12)) * (252*24*12) ** 0.5) if len(returns) > 0 and np.std(returns) > 0 else 0.0,
        "max_drawdown": float(drawdown.min()) if len(eq) > 0 else 0.0,
        "volatility": float(np.std(returns) * np.sqrt(252*24*12)) if returns else 0.0,
        "win_rate": float((np.array(returns) > 0).mean()) if returns else 0.0,
        "avg_position": float(np.mean(np.abs(pos_hist))) if pos_hist else 0.0,
        "max_position": float(np.max(np.abs(pos_hist))) if pos_hist else 0.0,
        "turnover": float(np.mean([abs(pos_hist[i] - pos_hist[i-1]) for i in range(1, len(pos_hist))])) if len(pos_hist) > 1 else 0.0,
        "best_rolling_sharpe": float(rolling_sharpe.max()) if len(rolling_sharpe) > 0 and not rolling_sharpe.isna().all() else 0.0,
        "total_trades": len(returns),
        "profit_factor": float(np.sum([r for r in returns if r > 0]) / abs(np.sum([r for r in returns if r < 0]))) if any(r < 0 for r in returns) else float('inf'),
        "sortino_ratio": float(np.mean(returns) / (np.std([r for r in returns if r < 0]) + 1e-12) * np.sqrt(252*24*12)) if any(r < 0 for r in returns) else float('inf'),
    }

    with open(os.path.join(args.out, "lstm_extended_result.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n🎉 Extended LSTM Training Results:")
    print(f"⏱️  Training Time: {stats['training_time_hours']:.2f} hours")
    print(f"📈 CAGR: {stats['cagr']:.2%}")
    print(f"⚡ Sharpe Ratio: {stats['sharpe']:.2f}")
    print(f"📉 Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"🎯 Win Rate: {stats['win_rate']:.2%}")
    print(f"🔄 Turnover: {stats['turnover']:.3f}")
    print(f"💰 Profit Factor: {stats['profit_factor']:.2f}")

    # Compare with original results
    original_result_path = "artifacts/rl_phase3_lstm/lstm_result.json"
    if os.path.exists(original_result_path):
        with open(original_result_path, 'r') as f:
            original_stats = json.load(f)

        print(f"\n📊 Comparison with Original LSTM:")
        print(f"CAGR: {original_stats.get('cagr', 0)*100:.2f}% → {stats['cagr']*100:.2f}% ({(stats['cagr'] - original_stats.get('cagr', 0))*100:+.2f}%)")
        print(f"Sharpe: {original_stats.get('sharpe', 0):.2f} → {stats['sharpe']:.2f} ({stats['sharpe'] - original_stats.get('sharpe', 0):+.2f})")
        print(f"Max DD: {original_stats.get('max_drawdown', 0)*100:.2f}% → {stats['max_drawdown']*100:.2f}% ({(stats['max_drawdown'] - original_stats.get('max_drawdown', 0))*100:+.2f}%)")

    print(f"\n💾 Results saved to: {args.out}")
    print(f"📊 Model saved as: lstm_agent_extended.zip")


if __name__ == "__main__":
    main()