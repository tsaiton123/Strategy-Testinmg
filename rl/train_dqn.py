from __future__ import annotations
import argparse, os, json
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv

def make_env(df, window=64, max_leverage=1.0, cost_bps=10.0):
    def _f():
        return SingleAssetTradingEnv(df=df, window=window, action_mode="discrete", max_leverage=max_leverage, cost_bps=cost_bps)
    return _f

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--cost_bps", type=float, default=10.0)
    p.add_argument("--max_leverage", type=float, default=1.0)
    p.add_argument("--out", default="artifacts/rl_dqn")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    env = DummyVecEnv([make_env(df, args.window, args.max_leverage, args.cost_bps)])
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(args.out, "dqn_agent"))

    # Eval
    env_eval = SingleAssetTradingEnv(df=df, window=args.window, action_mode="discrete", max_leverage=args.max_leverage, cost_bps=args.cost_bps)
    obs, info = env_eval.reset()
    pos_hist, pnl = [], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_eval.step(action)
        pos_hist.append(info["position"]); pnl.append(reward)
        if done or truncated:
            break

    import matplotlib.pyplot as plt
    import pandas as pd
    eq = (1 + pd.Series(pnl)).cumprod()
    eq_path = os.path.join(args.out, "dqn_equity.png")
    plt.figure(figsize=(10,4)); eq.plot(); plt.title("DQN Equity"); plt.tight_layout(); plt.savefig(eq_path, dpi=150); plt.close()

    stats = {
        "timesteps": args.timesteps,
        "CAGR": float(eq.iloc[-1] ** (525600/len(eq)) - 1) if len(eq) > 0 else 0.0,
        "Sharpe": float((np.mean(pnl) / (np.std(pnl) + 1e-12)) * (525600 ** 0.5)) if np.std(pnl) > 0 else 0.0,
        "MaxDD": float(((eq / eq.cummax()) - 1).min()) if len(eq) else 0.0,
    }
    with open(os.path.join(args.out, "dqn_result.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
