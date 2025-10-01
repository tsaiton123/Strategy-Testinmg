from __future__ import annotations
import argparse, os, json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv
from .custom_policies import LSTMPolicy, AttentionPolicy, HybridPolicy
import torch
import torch.nn as nn


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Linear learning rate schedule"""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--action_mode", choices=["discrete","box"], default="box")
    p.add_argument("--cost_bps", type=float, default=5.0)
    p.add_argument("--max_leverage", type=float, default=0.5)
    p.add_argument("--policy_type", choices=["mlp", "lstm", "attention", "hybrid"], default="hybrid")
    p.add_argument("--out", default="artifacts/rl_advanced_ppo")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=10)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    # Split data for train/validation (90/10 split for more training data)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx].copy()

    # Training environment
    env = DummyVecEnv([make_env(train_df, args.window, args.action_mode,
                            args.max_leverage, args.cost_bps)] * 4)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

    # Select policy type
    policy_kwargs = {}
    if args.policy_type == "mlp":
        policy = "MlpPolicy"
        policy_kwargs = dict(
            net_arch=[512, 512, 256, 128],
            activation_fn=nn.Tanh,
            ortho_init=False
        )
    elif args.policy_type == "lstm":
        policy = LSTMPolicy
        policy_kwargs = dict(
            features_dim=256,
            lstm_hidden_size=128,
            num_lstm_layers=2,
            dropout=0.2,
            net_arch=[256, 128]
        )
    elif args.policy_type == "attention":
        policy = AttentionPolicy
        policy_kwargs = dict(
            features_dim=256,
            attention_dim=128,
            num_heads=8,
            dropout=0.1,
            net_arch=[256, 128]
        )
    elif args.policy_type == "hybrid":
        policy = HybridPolicy
        policy_kwargs = dict(
            features_dim=256,
            lstm_hidden_size=128,
            attention_dim=128,
            num_heads=4,
            dropout=0.1,
            net_arch=[256, 128]
        )

    # Advanced PPO configuration
    model = PPO(
        policy,
        env,
        learning_rate=linear_schedule(args.learning_rate, args.learning_rate * 0.1),
        n_steps=4096,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=linear_schedule(0.3, 0.1),
        clip_range_vf=0.3,
        ent_coef=0.001,
        vf_coef=0.25,
        max_grad_norm=1.0,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=0.02,
        policy_kwargs=policy_kwargs,
        device=args.device,
        verbose=1,
        seed=42
    )

    # Training with callbacks
    model.learn(total_timesteps=args.timesteps)

    # Save model and normalization
    vecnorm_path = os.path.join(args.out, "vecnormalize.pkl")
    env.save(vecnorm_path)
    model.save(os.path.join(args.out, f"{args.policy_type}_agent"))

    # Evaluation
    print(f"Running evaluation on training data with {args.policy_type} policy...")
    eval_env = DummyVecEnv([make_env(train_df, args.window, args.action_mode,
                                   args.max_leverage, args.cost_bps)])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    pos_hist, returns, rewards = [], [], []
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

    # Performance analysis
    import matplotlib.pyplot as plt
    import pandas as pd

    eq = (1 + pd.Series(returns)).cumprod() if returns else pd.Series([1.0])
    eq_path = os.path.join(args.out, f"{args.policy_type}_equity.png")
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    eq.plot(title=f"{args.policy_type.upper()} Policy - Equity Curve")
    plt.subplot(2,1,2)
    pd.Series(pos_hist).plot(title="Position History")
    plt.tight_layout()
    plt.savefig(eq_path, dpi=150)
    plt.close()

    # Calculate advanced statistics
    returns_series = pd.Series(returns) if returns else pd.Series([0.0])
    rolling_sharpe = returns_series.rolling(252).mean() / (returns_series.rolling(252).std() + 1e-8) * np.sqrt(252*24*12)

    stats = {
        "policy_type": args.policy_type,
        "timesteps": args.timesteps,
        "CAGR": float(eq.iloc[-1] ** (252*24*12/len(eq)) - 1) if len(eq) > 1 else 0.0,
        "Total_Return": float(eq.iloc[-1] - 1) if len(eq) > 0 else 0.0,
        "Sharpe": float((np.mean(returns) / (np.std(returns) + 1e-12)) * (252*24*12) ** 0.5) if len(returns) > 0 and np.std(returns) > 0 else 0.0,
        "MaxDD": float(((eq / eq.cummax()) - 1).min()) if len(eq) > 0 else 0.0,
        "Volatility": float(np.std(returns) * np.sqrt(252*24*12)) if returns else 0.0,
        "Avg_Return": float(np.mean(returns)) if returns else 0.0,
        "Win_Rate": float((np.array(returns) > 0).mean()) if returns else 0.0,
        "Avg_Position": float(np.mean(np.abs(pos_hist))) if pos_hist else 0.0,
        "Max_Position": float(np.max(np.abs(pos_hist))) if pos_hist else 0.0,
        "Turnover": float(np.mean([abs(pos_hist[i] - pos_hist[i-1]) for i in range(1, len(pos_hist))])) if len(pos_hist) > 1 else 0.0,
        "Best_Rolling_Sharpe": float(rolling_sharpe.max()) if len(rolling_sharpe) > 0 and not rolling_sharpe.isna().all() else 0.0,
    }

    with open(os.path.join(args.out, f"{args.policy_type}_result.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== {args.policy_type.upper()} Policy Results ===")
    print(f"CAGR: {stats['CAGR']:.2%}")
    print(f"Sharpe: {stats['Sharpe']:.2f}")
    print(f"Max DD: {stats['MaxDD']:.2%}")
    print(f"Win Rate: {stats['Win_Rate']:.2%}")


if __name__ == "__main__":
    main()