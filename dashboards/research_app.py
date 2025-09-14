import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Quant Research Dashboard", layout="wide")

st.title("Quant Research Dashboard")

parquet_path = st.sidebar.text_input("Parquet data path", "data/coinbase_BTCUSD_1m.parquet")
fast = st.sidebar.number_input("Fast SMA", 1, 2000, 50)
slow = st.sidebar.number_input("Slow SMA", 1, 5000, 200)
vol_lookback = st.sidebar.number_input("Vol lookback (bars)", 10, 20000, 1440)
target_vol = st.sidebar.number_input("Target Vol (ann.)", 0.01, 2.0, 0.2, step=0.01)
fee_bps = st.sidebar.number_input("Fee (bps per turnover)", 0.0, 100.0, 10.0, step=0.5)

if st.sidebar.button("Run"):
    from backtest.simple_backtest import run_backtest
    run_backtest(parquet_path, fast, slow, vol_lookback, target_vol, fee_bps)
    st.success("Backtest complete. See artifacts/*.png")

st.markdown("Upload or point to any parquet with columns: ts, open, high, low, close, volume, symbol, timeframe, vendor.")
