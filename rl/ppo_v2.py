import os, json
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.market_env import SingleAssetTradingEnv
from rl.data_loader import load_parquet

# --- 不同 window 設定
windows = [32, 64, 128]

# --- 參數
timesteps = 50000
cost_bps = 5.0
action_mode = "box"
data_path = "data/daytrading_5m.parquet"
out_dir = "artifacts/best_model"
os.makedirs(out_dir, exist_ok=True)

df = load_parquet(data_path)

best_sharpe = -float("inf")
best_model_path = None
best_window = None

for w in windows:
    print(f"\n===== Testing window = {w} =====")

    # 建立環境
    def make_env():
        return SingleAssetTradingEnv(df=df, window=w, action_mode=action_mode, cost_bps=cost_bps)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

    # 訓練模型
    model = PPO("MlpPolicy", env,
                learning_rate=1e-4, n_steps=4096, batch_size=128,
                n_epochs=20, ent_coef=0.001, gamma=0.995, gae_lambda=0.98,
                clip_range=0.3, vf_coef=0.25, max_grad_norm=1.0,
                policy_kwargs=dict(net_arch=[512,512,256]),
                verbose=0)
    model.learn(total_timesteps=timesteps)

    # 評估
    obs = env.reset()
    done = [False]
    returns = []
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        returns.append(reward[0])

    sharpe = float(pd.Series(returns).mean() / (pd.Series(returns).std() + 1e-8))
    print(f"Window {w}: Sharpe = {sharpe:.5f}")

    # 儲存最佳模型
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_window = w
        best_model_path = os.path.join(out_dir, "best_ppo_agent")
        model.save(best_model_path)
        print(f"New best model saved (window={w})")

print(f"\nBest window: {best_window}, Best Sharpe: {best_sharpe:.5f}")
