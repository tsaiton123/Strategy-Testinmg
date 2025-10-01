from __future__ import annotations
import argparse, os, json
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv

def make_env(df, window=64, action_mode="discrete", max_leverage=0.5, cost_bps=10.0):
    def _f():
        return SingleAssetTradingEnv(
            df=df,
            window=window,
            action_mode=action_mode,   # 'discrete' or 'box'
            max_leverage=max_leverage,
            cost_bps=cost_bps,
        )
    return _f

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--action_mode", choices=["discrete","box"], default="discrete")
    p.add_argument("--cost_bps", type=float, default=10.0)
    p.add_argument("--max_leverage", type=float, default=0.5)
    p.add_argument("--n_envs", type=int, default=4, help="parallel envs")
    p.add_argument("--vecnorm", type=int, default=1, help="1=use VecNormalize")
    p.add_argument("--out", default="artifacts/rl_a2c")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_parquet(args.data)

    # Vectorized env
    env_fns = [make_env(df, args.window, args.action_mode, args.max_leverage, args.cost_bps)
               for _ in range(max(1, args.n_envs))]
    venv = DummyVecEnv(env_fns)
    if args.vecnorm:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.999)

    # A2C config tuned for noisy trading rewards
    model = A2C(
    "MlpPolicy",
    venv,
    learning_rate=1e-4,
    n_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.002,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_rms_prop=True,
    verbose=1,
    device="cpu",            # <<— add this
)


    model.learn(total_timesteps=args.timesteps)

    # Save artifacts
    model.save(os.path.join(args.out, "a2c_agent"))
    if args.vecnorm:
        venv.save(os.path.join(args.out, "vecnormalize.pkl"))

    # ---- quick deterministic rollout (single env) ----
    eval_env = DummyVecEnv([make_env(df, args.window, args.action_mode, args.max_leverage, args.cost_bps)])
    if args.vecnorm:
        eval_env = VecNormalize.load(os.path.join(args.out, "vecnormalize.pkl"), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    obs = eval_env.reset()
    pnl, done = [], [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = eval_env.step(action)
        pnl.append(float(r))

    import matplotlib.pyplot as plt
    import pandas as pd
    eq = (1 + pd.Series(pnl)).cumprod()
    eq_path = os.path.join(args.out, "a2c_equity.png")
    plt.figure(figsize=(10,4)); eq.plot(); plt.title("A2C Equity"); plt.tight_layout(); plt.savefig(eq_path, dpi=150); plt.close()

    stats = {
        "timesteps": args.timesteps,
        "CAGR": float(eq.iloc[-1] ** (525600/len(eq)) - 1) if len(eq) > 0 else 0.0,
        "Sharpe": float((np.mean(pnl) / (np.std(pnl) + 1e-12)) * (525600 ** 0.5)) if np.std(pnl) > 0 else 0.0,
        "MaxDD": float(((eq / eq.cummax()) - 1).min()) if len(eq) else 0.0,
    }
    with open(os.path.join(args.out, "a2c_result.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
