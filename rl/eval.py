import argparse, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.data_loader import load_parquet   # 你之前的 data loader
from rl.market_env import SingleAssetTradingEnv


def make_env(df, window=64, action_mode="box", max_leverage=30.0, cost_bps=5.0):
    def _f():
        return SingleAssetTradingEnv(
            df=df,
            window=window,
            action_mode=action_mode,
            max_leverage=max_leverage,
            cost_bps=cost_bps
        )
    return _f


def evaluate(model_path, vecnorm_path, df, window=64, action_mode="box",
             max_leverage=30.0, cost_bps=5.0, out_dir="artifacts/eval"):
    
    os.makedirs(out_dir, exist_ok=True)

    # 建立 eval 環境
    eval_env = DummyVecEnv([make_env(df, window, action_mode, max_leverage, cost_bps)])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    # 載入模型
    model = PPO.load(model_path, env=eval_env)

    obs = eval_env.reset()
    done = [False]

    positions, returns, equities, timestamps, actions = [], [], [1.0], [], []

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        old_pos = getattr(eval_env.envs[0], "_pos", 0.0)

        obs, reward, done, info = eval_env.step(action)

        current_pos = info[0].get("position", 0.0)
        positions.append(current_pos)
        actions.append(float(action))

        # 計算 portfolio return
        if hasattr(eval_env.envs[0], "_t") and eval_env.envs[0]._t > 0:
            market_return = eval_env.envs[0].rets[eval_env.envs[0]._t - 1]
            turnover = abs(current_pos - old_pos)
            cost = eval_env.envs[0].cost * turnover
            portfolio_return = old_pos * market_return - cost
            returns.append(portfolio_return)
            equities.append(equities[-1] * (1 + portfolio_return))

        # 記錄時間戳
        if hasattr(eval_env.envs[0], "px"):
            ts = eval_env.envs[0].px.index[eval_env.envs[0]._t] \
                 if eval_env.envs[0]._t < len(eval_env.envs[0].px) else None
            timestamps.append(ts)

    # 產生 equity curve
    eq_series = pd.Series(equities, index=timestamps[:len(equities)])
    eq_path = os.path.join(out_dir, "equity_curve.png")
    plt.figure(figsize=(10,4))
    eq_series.plot()
    plt.title("Evaluation Equity Curve")
    plt.tight_layout()
    plt.savefig(eq_path, dpi=150)
    plt.close()

    # 統計指標
    stats = {
        "Total_Return": float(eq_series.iloc[-1] - 1),
        "CAGR": float(eq_series.iloc[-1] ** (252*24*12/len(eq_series)) - 1) if len(eq_series) > 1 else 0.0,
        "Sharpe": float((np.mean(returns) / (np.std(returns) + 1e-12)) * (252*24*12) ** 0.5) if len(returns) > 0 else 0.0,
        "MaxDD": float(((eq_series / eq_series.cummax()) - 1).min()) if len(eq_series) > 0 else 0.0,
    }
    with open(os.path.join(out_dir, "eval_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("Evaluation finished.")
    print("Stats:", stats)
    print(f"Equity curve saved to {eq_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to parquet data")
    p.add_argument("--model", required=True, help="Path to trained model (ppo_agent.zip)")
    p.add_argument("--vecnorm", required=True, help="Path to vecnormalize.pkl")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--action_mode", choices=["discrete","box"], default="box")
    p.add_argument("--cost_bps", type=float, default=5.0)
    p.add_argument("--max_leverage", type=float, default=30.0)
    p.add_argument("--out", default="artifacts/eval")
    args = p.parse_args()

    df = load_parquet(args.data)
    evaluate(args.model, args.vecnorm, df,
             window=args.window,
             action_mode=args.action_mode,
             max_leverage=args.max_leverage,
             cost_bps=args.cost_bps,
             out_dir=args.out)
