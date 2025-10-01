from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv

ALGOS = {"ppo": PPO, "dqn": DQN, "a2c": A2C}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--algo", choices=list(ALGOS.keys()), required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--action_mode", choices=["discrete","box"], default="discrete")
    p.add_argument("--cost_bps", type=float, default=10.0)
    p.add_argument("--max_leverage", type=float, default=0.5)
    p.add_argument("--vecnorm_path", default="", help="optional VecNormalize stats (.pkl)")
    p.add_argument("--out", default="artifacts/rl_eval")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    venv = DummyVecEnv([lambda: SingleAssetTradingEnv(
        df=df, window=args.window, action_mode=args.action_mode,
        max_leverage=args.max_leverage, cost_bps=args.cost_bps
    )])

    if args.vecnorm_path and os.path.exists(args.vecnorm_path):
        venv = VecNormalize.load(args.vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = ALGOS[args.algo].load(args.model_path)

    obs = venv.reset()
    pnl, done = [], [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = venv.step(action)
        pnl.append(float(r))

    eq = (1 + pd.Series(pnl)).cumprod()
    stats = {
        "CAGR": float(eq.iloc[-1] ** (525600/len(eq)) - 1) if len(eq) > 0 else 0.0,
        "Sharpe": float((np.mean(pnl) / (np.std(pnl) + 1e-12)) * (525600 ** 0.5)) if np.std(pnl) > 0 else 0.0,
        "MaxDD": float(((eq / eq.cummax()) - 1).min()) if len(eq) else 0.0,
    }

    import matplotlib.pyplot as plt
    eq_path = os.path.join(args.out, f"{args.algo}_eval_equity.png")
    plt.figure(figsize=(10,4)); eq.plot(); plt.title("Agent Equity"); plt.tight_layout(); plt.savefig(eq_path, dpi=150); plt.close()

    with open(os.path.join(args.out, f"{args.algo}_eval_result.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
