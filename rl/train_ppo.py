from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv
import matplotlib.pyplot as plt

# --- Callback for monitoring & early stopping ---
class TradingMonitorCallback(BaseCallback):
    def __init__(self, log_interval=1000, patience=50, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.constraint_violations = 0
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.episodes_since_improvement = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if info.get('violated', False):
                self.constraint_violations += 1
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

        if self.num_timesteps % self.log_interval == 0:
            if self.episode_rewards:
                recent_mean = np.mean(self.episode_rewards[-50:])
                print(f"[Step {self.num_timesteps}] "
                      f"Mean Reward (last 50 eps): {recent_mean:.5f}, "
                      f"Constraint Violations: {self.constraint_violations}")

                # Early stopping
                if recent_mean > self.best_mean_reward + 1e-6:
                    self.best_mean_reward = recent_mean
                    self.episodes_since_improvement = 0
                else:
                    self.episodes_since_improvement += 1

                if self.episodes_since_improvement >= self.patience:
                    print(f"[EarlyStopping] Reward hasn't improved for {self.patience} episodes. Stopping training.")
                    return False
        return True

# --- Environment factory ---
def make_env(df, window=64, action_mode="box", max_leverage=30.0, cost_bps=5.0):
    def _f():
        return SingleAssetTradingEnv(df=df, window=window, action_mode=action_mode,
                                     max_leverage=max_leverage, cost_bps=cost_bps)
    return _f

# --- Training function ---
def train_rl(df, timesteps=500_000, window=64, action_mode="box",
             max_leverage=30.0, cost_bps=5.0, out_dir="artifacts/rl_ppo"):
    os.makedirs(out_dir, exist_ok=True)

    # Training environment
    env = DummyVecEnv([make_env(df, window, action_mode, max_leverage, cost_bps)] * 8)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

    model = PPO("MlpPolicy", env,
                learning_rate=1e-4, n_steps=4096, batch_size=128,
                n_epochs=20, ent_coef=0.001, gamma=0.995, gae_lambda=0.98,
                clip_range=0.3, vf_coef=0.25, max_grad_norm=1.0,
                policy_kwargs=dict(net_arch=[512, 512, 256]),
                verbose=1)

    # Train with monitoring callback
    monitor_cb = TradingMonitorCallback(log_interval=5000, patience=50)
    model.learn(total_timesteps=timesteps, callback=monitor_cb)

    # Save model & VecNormalize
    model.save(os.path.join(out_dir, "ppo_agent"))
    env.save(os.path.join(out_dir, "vecnormalize.pkl"))

    # --- Evaluation ---
    print(f"Running evaluation on training data...")
    eval_env = DummyVecEnv([make_env(df, window, action_mode, max_leverage, cost_bps)])
    eval_env = VecNormalize.load(os.path.join(out_dir, "vecnormalize.pkl"), eval_env)
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

    # Equity curve
    eq = (1 + pd.Series(returns)).cumprod() if returns else pd.Series([1.0])
    eq_path = os.path.join(out_dir, "ppo_equity.png")
    plt.figure(figsize=(10,4)); eq.plot(); plt.title("PPO Equity (Enhanced Features)"); plt.tight_layout(); plt.savefig(eq_path, dpi=150); plt.close()

    # Statistics
    stats = {
        "timesteps": timesteps,
        "CAGR": float(eq.iloc[-1] ** (252*24*12/len(eq)) - 1) if len(eq) > 1 else 0.0,
        "Sharpe": float((np.mean(returns) / (np.std(returns) + 1e-12)) * (252*24*12) ** 0.5) if len(returns) > 0 and np.std(returns) > 0 else 0.0,
        "MaxDD": float(((eq / eq.cummax()) - 1).min()) if len(eq) > 0 else 0.0,
        "Total_Return": float(eq.iloc[-1] - 1) if len(eq) > 0 else 0.0,
        "Avg_Return": float(np.mean(returns)) if returns else 0.0,
        "Vol": float(np.std(returns)) if returns else 0.0,
        "Constraint_Violations": monitor_cb.constraint_violations
    }
    with open(os.path.join(out_dir, "ppo_result.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Training finished. Total Constraint Violations: {monitor_cb.constraint_violations}")
    print(f"Equity curve and stats saved to {out_dir}")

# --- CLI ---
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--action_mode", choices=["discrete","box"], default="box")
    p.add_argument("--cost_bps", type=float, default=5.0)
    p.add_argument("--max_leverage", type=float, default=30.0)
    p.add_argument("--out", default="artifacts/rl_ppo")
    args = p.parse_args()

    df = load_parquet(args.data)
    train_rl(df, timesteps=args.timesteps, window=args.window,
             action_mode=args.action_mode, max_leverage=args.max_leverage,
             cost_bps=args.cost_bps, out_dir=args.out)
