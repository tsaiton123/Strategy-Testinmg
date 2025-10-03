from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from market_env import SingleAssetTradingEnv
from data_loader import load_parquet

# ---------------- Callback ----------------
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
                print(f"[Step {self.num_timesteps}] Mean Reward (last 50 eps): {recent_mean:.5f}, Constraint Violations: {self.constraint_violations}")
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

# ---------------- Environment Factory ----------------
def make_env(df, window=64, action_mode="box", max_leverage=30.0, cost_bps=5.0):
    def _f():
        return SingleAssetTradingEnv(df=df, window=window, action_mode=action_mode,
                                     max_leverage=max_leverage, cost_bps=cost_bps)
    return _f

# ---------------- Training ----------------
def train_rl(train_df, val_df=None, test_df=None, timesteps=500_000, window=64,
             action_mode="box", max_leverage=30.0, cost_bps=5.0, out_dir="artifacts/rl_ppo_wf"):

    os.makedirs(out_dir, exist_ok=True)

    # --- Training Env ---
    train_env = DummyVecEnv([make_env(train_df, window, action_mode, max_leverage, cost_bps)] * 1)
    env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

    # --- PPO Model ---
    model = PPO("MlpPolicy", env,
                learning_rate=1e-4, n_steps=4096, batch_size=128,
                n_epochs=20, ent_coef=0.001, gamma=0.995, gae_lambda=0.98,
                clip_range=0.3, vf_coef=0.25, max_grad_norm=1.0,
                policy_kwargs=dict(net_arch=[512, 512, 256]),
                verbose=1)

    # --- Callback ---
    monitor_cb = TradingMonitorCallback(log_interval=5000, patience=50)
    model.learn(total_timesteps=timesteps, callback=monitor_cb)

    # --- Save model & VecNormalize ---
    model.save(os.path.join(out_dir, "ppo_agent"))
    env.save(os.path.join(out_dir, "vecnormalize.pkl"))

    # ---------------- Evaluation ----------------
    if test_df is not None:
        eval_env = DummyVecEnv([make_env(test_df, window, action_mode, max_leverage, cost_bps)])
        eval_env = VecNormalize.load(os.path.join(out_dir, "vecnormalize.pkl"), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        obs = eval_env.reset()
        pos_hist, returns, rewards = [], [], []
        done = [False]
        trade_log = []

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            old_pos = getattr(eval_env.envs[0], '_pos', 0.0)
            obs, reward, done, info = eval_env.step(action)
            trade_log.append({
                "timestamp": info[0].get("timestamp"),
                "action": float(action[0]),
                "position": info[0].get("position"),
                "portfolio_return": info[0].get("portfolio_return"),
                "cost": info[0].get("cost"),
                "equity": info[0].get("equity")
            })
            current_pos = info[0].get("position", 0.0)
            pos_hist.append(current_pos)
            rewards.append(reward[0])
            # Portfolio return
            if hasattr(eval_env.envs[0], '_t') and eval_env.envs[0]._t > 0:
                market_return = eval_env.envs[0].rets[eval_env.envs[0]._t - 1]
                turnover = abs(current_pos - old_pos)
                cost = eval_env.envs[0].cost * turnover
                portfolio_return = old_pos * market_return - cost
                returns.append(portfolio_return)

        # --- Save trades & equity ---
        trade_df = pd.DataFrame(trade_log)
        trade_df.to_csv(os.path.join(out_dir, "ppo_trades.csv"), index=False)
        trade_df.to_parquet(os.path.join(out_dir, "ppo_trades.parquet"))
        eq = (1 + pd.Series(returns)).cumprod() if returns else pd.Series([1.0])
        plt.figure(figsize=(10,4)); eq.plot(); plt.title("PPO Equity Curve"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ppo_equity.png"), dpi=150); plt.close()

        # --- Statistics ---
        stats = {
            "timesteps": timesteps,
            "CAGR": float(eq.iloc[-1] ** (252*24*12/len(eq)) - 1) if len(eq) > 1 else 0.0,
            "Sharpe": float((np.mean(returns) / (np.std(returns)+1e-12)) * (252*24*12)**0.5) if len(returns) > 0 and np.std(returns) > 0 else 0.0,
            "MaxDD": float(((eq/eq.cummax())-1).min()) if len(eq) > 0 else 0.0,
            "Total_Return": float(eq.iloc[-1]-1) if len(eq) > 0 else 0.0,
            "Avg_Return": float(np.mean(returns)) if returns else 0.0,
            "Vol": float(np.std(returns)) if returns else 0.0,
            "Constraint_Violations": monitor_cb.constraint_violations
        }
        with open(os.path.join(out_dir, "ppo_result.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Evaluation finished. Total Constraint Violations: {monitor_cb.constraint_violations}")

    print(f"Training finished. All outputs saved to {out_dir}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=False)
    parser.add_argument("--test_data", required=False)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--action_mode", choices=["discrete","box"], default="box")
    parser.add_argument("--cost_bps", type=float, default=5.0)
    parser.add_argument("--max_leverage", type=float, default=30.0)
    parser.add_argument("--out", default="artifacts/rl_ppo_wf")
    args = parser.parse_args()

    train_df = load_parquet(args.train_data)
    val_df = load_parquet(args.val_data) if args.val_data else None
    test_df = load_parquet(args.test_data) if args.test_data else None

    train_rl(train_df, val_df, test_df, timesteps=args.timesteps,
             window=args.window, action_mode=args.action_mode,
             max_leverage=args.max_leverage, cost_bps=args.cost_bps,
             out_dir=args.out)
