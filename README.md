# Quant Platform Starter

A minimal end-to-end skeleton: **ingest → research → backtest → (paper/live stub) → visualization**.

## Quickstart

### 0) Prereqs
- Python 3.11+
- (Optional) Docker for infra (Postgres/Redis/MinIO not required for this starter)

### 1) Create venv & install deps
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Ingest minute data (Crypto via CCXT/Binance public)
```bash
python -m data_ingest.ccxt_ingest --symbol BTC/USDT --timeframe 1m --start 2024-01-01 --end 2024-01-03 --out data/crypto_BTCUSDT_1m.parquet
```

### 3) Run a simple SMA crossover with volatility targeting backtest
```bash
python -m backtest.simple_backtest --data data/crypto_BTCUSDT_1m.parquet --fast 50 --slow 200 --vol_lookback 1440 --target_vol 0.2 --fee_bps 10
```

This prints metrics and writes plots to `artifacts/`.

### 4) Explore in Streamlit (research dashboard)
```bash
streamlit run dashboards/research_app.py
```

### 5) (Optional) Run the Trade API stub (paper/live adapter placeholder)
```bash
uvicorn execution.trade_service:app --reload
```

## Layout
```
configs/                 # settings (YAML) and calendars (TBD)
data/                    # local data cache (parquet)
data_ingest/             # connectors (ccxt)
data_models/             # pydantic schemas
features/                # factors & transforms
signals/                 # strategies producing weights
backtest/                # event-lite backtester & cost model
execution/               # FastAPI trade stub + broker adapters
dashboards/              # Streamlit research app
orchestration/           # Prefect flows (stub)
tests/                   # basic tests
```

## Notes
- The ingestion uses CCXT to fetch klines from Binance. For equities, plug your vendor in `data_ingest/` with the same bar schema and you're good to go.
- Backtester is intentionally lightweight but leakage-safe (signals lagged one bar). Costs are modeled via bps + slippage fraction.
