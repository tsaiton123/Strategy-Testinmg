from __future__ import annotations
import argparse, os, json
from xml.parsers.expat import model
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env(df, window=64, action_mode="discrete", max_leverage=1.0, cost_bps=10.0):
    def _f():
        return SingleAssetTradingEnv(df=df, window=window, action_mode=action_mode, max_leverage=max_leverage, cost_bps=cost_bps)
    return _f

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--action_mode", choices=["discrete","box"], default="box")
    p.add_argument("--cost_bps", type=float, default=5.0)
    p.add_argument("--max_leverage", type=float, default=0.5)
    p.add_argument("--out", default="artifacts/rl_ppo")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    # Split data for train/validation (80/20 split)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    # Training environment
    env = DummyVecEnv([make_env(train_df, args.window, args.action_mode,
                            args.max_leverage, args.cost_bps)] * 4)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

    # Improved PPO hyperparameters for trading
    model = PPO("MlpPolicy", env,
            learning_rate=5e-4, n_steps=4096, batch_size=128,
            n_epochs=20, ent_coef=0.001, gamma=0.995, gae_lambda=0.98,
            clip_range=0.3, vf_coef=0.25, max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=[512, 512, 256]),
            verbose=1)
    model.learn(total_timesteps=args.timesteps)
    vecnorm_path = os.path.join(args.out, "vecnormalize.pkl")
    env.save(vecnorm_path)
    model.save(os.path.join(args.out, "ppo_agent"))

    # Simple evaluation on training data (since validation has different feature dimensions)
    print(f"Running evaluation on training data...")
    simple_eval_env = DummyVecEnv([make_env(train_df, args.window, args.action_mode,
                                           args.max_leverage, args.cost_bps)])
    simple_eval_env = VecNormalize.load(vecnorm_path, simple_eval_env)
    simple_eval_env.training = False
    simple_eval_env.norm_reward = False

    obs = simple_eval_env.reset()
    pos_hist, returns, rewards = [], [], []
    done = [False]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        old_pos = getattr(simple_eval_env.envs[0], '_pos', 0.0)
        obs, reward, done, info = simple_eval_env.step(action)

        current_pos = info[0].get("position", 0.0)
        pos_hist.append(current_pos)
        rewards.append(reward[0])

        # Calculate actual portfolio return
        if hasattr(simple_eval_env.envs[0], '_t') and simple_eval_env.envs[0]._t > 0:
            market_return = simple_eval_env.envs[0].rets[simple_eval_env.envs[0]._t - 1]
            turnover = abs(current_pos - old_pos)
            cost = simple_eval_env.envs[0].cost * turnover
            portfolio_return = old_pos * market_return - cost
            returns.append(portfolio_return)

    import matplotlib.pyplot as plt
    import pandas as pd

    # Use actual returns for performance calculation
    eq = (1 + pd.Series(returns)).cumprod() if returns else pd.Series([1.0])
    eq_path = os.path.join(args.out, "ppo_equity.png")
    plt.figure(figsize=(10,4)); eq.plot(); plt.title("PPO Equity (Enhanced Features)"); plt.tight_layout(); plt.savefig(eq_path, dpi=150); plt.close()

    stats = {
        "timesteps": args.timesteps,
        "CAGR": float(eq.iloc[-1] ** (252*24*12/len(eq)) - 1) if len(eq) > 1 else 0.0,  # Assuming 5-min data
        "Sharpe": float((np.mean(returns) / (np.std(returns) + 1e-12)) * (252*24*12) ** 0.5) if len(returns) > 0 and np.std(returns) > 0 else 0.0,
        "MaxDD": float(((eq / eq.cummax()) - 1).min()) if len(eq) > 0 else 0.0,
        "Total_Return": float(eq.iloc[-1] - 1) if len(eq) > 0 else 0.0,
        "Avg_Return": float(np.mean(returns)) if returns else 0.0,
        "Vol": float(np.std(returns)) if returns else 0.0,
    }
    with open(os.path.join(args.out, "ppo_result.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
