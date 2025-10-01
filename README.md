# Quant Research & Trading Starter — README

A batteries-included starter for **quant research, backtesting, and RL agents** on market data. It gives you:

- **Data ingestion** from crypto venues via CCXT (with exchange selection & robust pagination).
- **Data inspection** (coverage, gaps, duplicates, distributions) in Streamlit.
- **Strategy plugins** (drop your own script under `strategies/`) + a generic **backtest engine** with costs.
- **CLI runner** for single runs and parameter grids.
- **Research dashboards** in Streamlit (run backtests, visualize results).
- **RL extension** (Gymnasium env + Stable-Baselines3 trainers) to learn policies on your OHLCV data.

> All timestamps are UTC. Nothing here is financial advice; this is research software.

---

## 0) Environment

- Python 3.10+
- Recommended: Linux/macOS or **WSL2** on Windows (paths in examples use WSL).
- Create a venv:
  ```bash
  python -m venv .venv
  source .venv/bin/activate           # Windows PowerShell: .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  ```

Install base deps (if your repo doesn’t already include them in `requirements.txt`):
```bash
pip install pandas numpy pyarrow matplotlib streamlit ccxt
```

> Streamlit apps and module imports expect the repo root on `PYTHONPATH`:
> - Bash:
>   ```bash
>   export PYTHONPATH="$PWD"
>   ```
> - PowerShell:
>   ```powershell
>   $env:PYTHONPATH=(Get-Location).Path
>   ```

---

## 1) Data ingestion

We ingest OHLCV bars to **Parquet** with a consistent schema:
`ts, open, high, low, close, volume, symbol, timeframe, vendor`

### Command
```bash
python -m data_ingest.ccxt_ingest   --exchange coinbase   --symbol BTC/USD --timeframe 5m   --start 2024-06-01 --end 2024-06-03   --out data/coinbase_BTCUSD_5m.parquet
```

**Notes**
- `--exchange` (e.g., `coinbase`, `kraken`, `binanceus`).  
  (Binance.com may return HTTP 451 in restricted regions; switch to a compliant venue.)
- `--list-symbols` shows available symbols:
  ```bash
  python -m data_ingest.ccxt_ingest --exchange coinbase --list-symbols | head
  ```
- The ingest loop **pages until the end time** (no premature stop if a batch is short) and prints a range summary.

**Quick verify**
```bash
python - <<'PY'
import pyarrow.parquet as pq
df = pq.read_table("data/coinbase_BTCUSD_5m.parquet").to_pandas()
print(len(df), df['ts'].min(), "→", df['ts'].max(), df.columns.tolist())
PY
```

---

## 2) Data inspection (Streamlit)

Explore gaps, duplicates, distributions, and simple charts.

```bash
streamlit run dashboards/data_inspector.py
```

In the sidebar, set your Parquet path (e.g. `data/coinbase_BTCUSD_5m.parquet`) and click **Analyze**.

Features:
- Coverage: expected vs actual bars, **gap ranges**, **duplicates**.
- Stats: price/volume summary, returns hist, **hour-of-day average returns**.
- Simple charts: close, volume, rolling ratios.
- **Downloadable gap report** CSV.

---

## 3) Strategy plugins & backtesting

Write strategies as small Python classes that output a **weight series** (−L…+L), and the engine handles P&L with costs.

### Folder layout (key files)

```
data_ingest/ccxt_ingest.py          # OHLCV -> Parquet
dashboards/research_app.py           # run & visualize backtests
dashboards/data_inspector.py         # data QA
dashboards/strategy_lab.py           # pick a strategy plugin & run
dashboards/rl_lab.py                 # RL training/eval (optional)
backtest/engine.py                   # generic engine (PnL, costs, plots)
backtest/runner.py                   # CLI runner (single & grid)
strategies/base.py                   # BaseStrategy + config
strategies/sma.py                    # SMA crossover example
strategies/mean_reversion.py         # Z-score MR example
strategies/donchian.py               # Donchian breakout example
strategies/my_alpha.py               # (your custom strategy — e.g., RSI MR)
```

### Write your own strategy

Create `strategies/my_alpha.py`:

```python
from dataclasses import dataclass
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyConfig

@dataclass
class MyAlphaConfig(StrategyConfig):
    rsi_lookback: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    smooth_span: int = 1

class MyAlpha(BaseStrategy):
    def __init__(self, rsi_lookback: int = 14, rsi_low: float = 30.0,
                 rsi_high: float = 70.0, smooth_span: int = 1,
                 config: MyAlphaConfig | None = None):
        cfg = config or MyAlphaConfig(rsi_lookback=rsi_lookback, rsi_low=rsi_low,
                                      rsi_high=rsi_high, smooth_span=smooth_span)
        super().__init__(config=cfg)
        self.rsi_lookback, self.rsi_low, self.rsi_high, self.smooth_span =             rsi_lookback, rsi_low, rsi_high, smooth_span

    @staticmethod
    def _rsi(close: pd.Series, n: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100 - (100 / (1 + rs))

    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        x = df.set_index("ts").sort_index()
        px = x["close"].astype(float)
        rsi = self._rsi(px, self.rsi_lookback)
        long_sig = (rsi < self.rsi_low).astype(float)
        short_sig = (rsi > self.rsi_high).astype(float)
        raw = long_sig - short_sig if self.config.allow_short else long_sig
        if self.smooth_span and self.smooth_span > 1:
            raw = raw.ewm(span=self.smooth_span, adjust=False).mean()
        w = raw.clip(-self.config.max_leverage, self.config.max_leverage)
        return w.shift(self.config.lag_bars).fillna(0.0)
```

### Run a backtest (CLI)

**Single run**
```bash
python -m backtest.runner   --data data/coinbase_BTCUSD_5m.parquet   --strategy strategies.sma:SMA   --params "fast=50,slow=200"   --fee_bps 10 --slip_bps 0 --out artifacts/sma_run
```

**Grid search**
```bash
python -m backtest.runner   --data data/coinbase_BTCUSD_5m.parquet   --strategy strategies.sma:SMA   --grid "fast=[20,50,100],slow=[100,200,400]"   --fee_bps 10 --out artifacts/sma_grid
```

Artifacts:
- `result.json` (stats + meta), `equity_curve.png`, `rolling_mean.png`.

### Research app (Streamlit)

```bash
streamlit run dashboards/research_app.py
```

- Point at your Parquet file, pick params, **Run backtest**, view stats/plots inline.

### Strategy Lab (Streamlit)

```bash
streamlit run dashboards/strategy_lab.py
```

- Auto-discovers classes in `strategies/`. Pick one, adjust params, **Run backtest**.

---

## 4) Reinforcement Learning (optional)

A minimal RL stack to learn trading policies on your OHLCV bars.

### Install RL deps
```bash
pip install -r requirements_rl.txt
# CPU-only PyTorch wheel (if needed):
# pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### What’s included

- `rl/market_env.py` — Gymnasium **SingleAssetTradingEnv**  
  Obs: last `window` bars of engineered features (returns, z-momentum, RSI, vol, normalized price).  
  Actions: **discrete** {−L, 0, +L} or **continuous** [−L, +L].  
  Reward: `prev_position * return_t − cost_bps/1e4 * |Δposition|` (turnover cost).
- Trainers:  
  `rl/train_ppo.py` (PPO) • `rl/train_dqn.py` (DQN, discrete)  
  `rl/eval_agent.py` — load a saved model, export equity + stats.
- Streamlit `dashboards/rl_lab.py` — small demo training/eval UI.

### Quickstart

**Train PPO**
```bash
python -m rl.train_ppo   --data data/coinbase_BTCUSD_5m.parquet   --timesteps 50000   --window 64   --action_mode discrete   --cost_bps 10 --max_leverage 1.0   --out artifacts/rl_ppo
```

**Evaluate**
```bash
python -m rl.eval_agent   --data data/coinbase_BTCUSD_5m.parquet   --algo ppo   --model_path artifacts/rl_ppo/ppo_agent.zip   --window 64 --action_mode discrete   --cost_bps 10 --max_leverage 1.0   --out artifacts/rl_eval
```

**Streamlit**
```bash
streamlit run dashboards/rl_lab.py
```

### Practical tuning (recommended)

- Wrap envs with **`VecNormalize`** (normalize obs & rewards) or scale rewards by ~100× in the env.
- Start with **discrete** actions and small `max_leverage` (e.g., 0.25) to reduce churn; add **action stickiness** / min holding.
- Train longer (≥1–5M steps), on **multiple rolling windows** / symbols (vectorized envs).
- Hyperparam search (PPO: LR, `n_steps`, batch, entropy, gamma, λ, clip). Consider **RecurrentPPO (sb3-contrib)**.

### Evaluation Instructions

**Basic Model Evaluation**
```bash
# Evaluate a trained PPO model
python -m rl.eval_agent \
  --data data/coinbase_BTCUSD_5m.parquet \
  --algo ppo \
  --model_path artifacts/rl_ppo/ppo_agent.zip \
  --window 64 --action_mode discrete \
  --cost_bps 10 --max_leverage 1.0 \
  --out artifacts/rl_eval
```

**Advanced Evaluation (if using enhanced models)**
```bash
# Evaluate advanced models with enhanced features
python -m rl.eval_agent \
  --data data/coinbase_BTCUSD_5m.parquet \
  --algo ppo \
  --model_path artifacts/enhanced_ppo/best_model.zip \
  --window 64 --action_mode continuous \
  --cost_bps 5 --max_leverage 2.0 \
  --out artifacts/enhanced_eval

# Compare trading frequencies
python -m rl.frequency_comparison \
  --data data/coinbase_BTCUSD_5m.parquet \
  --high_freq_model artifacts/rl_ppo/ppo_agent.zip \
  --low_freq_model artifacts/low_freq_ppo/best_model.zip \
  --out artifacts/frequency_comparison
```

**Performance Metrics Interpretation**

Key metrics to monitor:
- **CAGR (Compound Annual Growth Rate)**: Target >20% for good performance
- **Sharpe Ratio**: Target >2.0 for strong risk-adjusted returns
- **Max Drawdown**: Keep <20% for acceptable risk
- **Trading Frequency**: Monitor to ensure reasonable transaction costs
- **Win Rate**: Track percentage of profitable trades

**Model Validation Best Practices**

1. **Out-of-sample testing**: Always evaluate on unseen data periods
2. **Walk-forward analysis**: Test model performance across different market regimes
3. **Transaction cost analysis**: Verify that positive Sharpe doesn't become negative after costs
4. **Frequency optimization**: Use low-frequency models if high-frequency trading erodes profits
5. **Risk monitoring**: Ensure drawdowns remain within acceptable limits

**Comparing Model Configurations**
```bash
# Generate comparison report
python -c "
import pandas as pd
import json

# Load results from different model evaluations
results = []
for config in ['basic', 'enhanced', 'low_freq']:
    with open(f'artifacts/{config}_eval/result.json') as f:
        data = json.load(f)
        data['config'] = config
        results.append(data)

df = pd.DataFrame(results)
print(df[['config', 'cagr', 'sharpe', 'max_drawdown', 'total_trades']].round(3))
"
```

---

## 5) Common issues

**HTTP 451 from Binance**  
You’re in a restricted region. Use `--exchange coinbase` / `kraken` / `binanceus`.

**“No data returned”**  
Check exact symbol on `--list-symbols`; move to a **newer date window**; try coarser timeframe (`5m`, `15m`).

**`ModuleNotFoundError: backtest`**  
Ensure repo root on `PYTHONPATH` (see Section 0).

**Streamlit shows empty charts**  
Likely your Parquet has 0 rows or wrong path. Verify with the quick Python snippet in Section 1.

**RL equity slides straight down**  
Use **VecNormalize**, reward scaling, smaller leverage, stickiness/min holding, more steps, better features, and hyperparam tuning.

---

## 6) Schema & conventions

- **Bar schema**: `ts` (UTC), `open`, `high`, `low`, `close`, `volume`, `symbol`, `timeframe`, `vendor`.
- **Weights**: point-in-time safe (strategies **shift by `lag_bars`**). Engine computes P&L:  
  `pnl_t = weight_t * ret_t − cost_per_turnover * |Δweight_t|`.
- **Costs**: turnover-based (bps), optional slippage bps.

---

## 7) Roadmap ideas

- Partitioned daily writes & append jobs (watermarking).
- Multi-asset portfolio engine (risk parity / vol targeting).
- Execution realism: market impact, participation caps, latency.
- Walk-forward experiment runner with leaderboard CSV.
- Auto gap re-ingestion & vendor blending.

---

## 8) License & disclaimer

This project is for **research** and education. Markets are risky; **use at your own risk**. Ensure venue APIs and your usage comply with local regulations and exchange terms.
