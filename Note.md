# RL 日內交易策略研究紀錄（PPO）

## 1. 研究目標
- 使用 **PPO (Proximal Policy Optimization)** 訓練日內交易強化學習策略
- 目標在 FTMO 規則下：
  - 每日損失限制 (Daily Loss Limit)
  - 總損失限制 (Max Total Loss)
  - 槓桿控制
- 對比手續費對策略表現的影響

## 2. 資料與特徵
- 訓練指令:
```bash
    python -m rl.train_ppo     --data data/daytrading_5m.parquet     --timesteps 50000     --out artifacts/rl_phase1_daytrading       --cost_bps 5     --action_mode box
```
- 資料來源：`data/daytrading_5m.parquet`
- 時間序列：5 分鐘 K 線
- 特徵工程：
  - RSI、波動率、移動平均等技術指標
  - 特徵選擇：`select_stable_features`
  - 標準化：RobustScaler
- Observation window: 64

## 3. RL 環境設定
- 環境：`SingleAssetTradingEnv`
- Action space: Box (-30, +30) 槓桿 (`--action_mode box`)
- 初始本金：100,000
- 每日最大損失：-5,000
- 總最大損失：-10,000
- 手續費成本：5 bps (`--cost_bps 5`)
- 最小持倉期：5
- Reward scale: 1.0
- Turnover penalty: 成本的 2 倍
- Drawdown penalty: 加大懲罰力度

## 4. 訓練設定
- 演算法：PPO (`stable-baselines3`)
- Timesteps：50,000
- Policy network: `[512, 512, 256]`
- Learning rate：0.0001
- Batch size：128
- N steps：4096
- Epochs：20
- Gamma：0.995
- GAE Lambda：0.98
- Clip Range：0.3
- Value function coef：0.25
- Entropy coef：0.001
- Max grad norm：1.0
- Reward normalization：online

## 5. Reward 改動與策略改進
- 將違規事件納入 reward：`raw_reward -= 5 + abs(daily_loss)/daily_loss_limit`
- Drawdown penalty 加大 (原 5 -> 10)
- 滾動回報視窗縮短：50 → 20
- Reward scale 從 0.1 → 1.0
- Episode 違規即終止

## 6. 訓練結果
- Timesteps：50,000
- CAGR：6.44%
- Sharpe Ratio：1.12
- 最大回撤 (MaxDD)：-5.82%
- 總收益：1.54%
- 平均單步收益：8.83e-7
- 波動率：0.021%
- 違規次數：8
- Equity 曲線：整體呈穩定正向

## 7.1 不同Cost(手續費)比較
