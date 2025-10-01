import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import os

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboards.utils import get_parquet_files, format_file_display


st.set_page_config(page_title="Quant Research Dashboard", layout="wide")

st.title("Quant Research Dashboard")

st.sidebar.subheader("📁 Data Selection")

# Auto-detect parquet files
parquet_files = get_parquet_files()
if parquet_files:
    # Create options with file info
    file_options = [format_file_display(f) for f in parquet_files]
    selected_display = st.sidebar.selectbox(
        "Select parquet file:",
        file_options,
        help="Auto-detected parquet files from data/ directory"
    )
    # Get the actual file path
    selected_idx = file_options.index(selected_display)
    parquet_path = parquet_files[selected_idx]

    # Show selected file info
    st.sidebar.info(f"Selected: {os.path.basename(parquet_path)}")
else:
    st.sidebar.warning("No parquet files found in data/ directory")
    parquet_path = st.sidebar.text_input("Manual parquet path:", "data/coinbase_BTCUSD_1m.parquet")

st.sidebar.subheader("⚙️ Strategy Parameters")
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
