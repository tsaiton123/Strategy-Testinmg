import sys, pathlib, os, io
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Inspector (OHLCV Parquet)", layout="wide")
st.title("Data Inspector — OHLCV Parquet")

# ----------------- Helpers -----------------
def timeframe_to_pandas_freq(tf: str) -> str:
    tf = tf.strip().lower()
    if tf.endswith('m'):
        return f"{int(tf[:-1])}T"
    if tf.endswith('h'):
        return f"{int(tf[:-1])}H"
    if tf.endswith('d'):
        return f"{int(tf[:-1])}D"
    if tf.endswith('w'):
        return f"{int(tf[:-1])}W"
    return None

def infer_freq_from_timestamps(ts: pd.Series) -> str:
    # Infer using the median delta
    deltas = ts.sort_values().diff().dropna()
    if deltas.empty:
        return None
    minutes = int(round(deltas.median().total_seconds() / 60.0))
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes//60}H"
    if minutes >= 1440 and minutes % 1440 == 0:
        return f"{minutes//1440}D"
    return f"{minutes}T"

def load_df(path: str) -> pd.DataFrame:
    table = pq.read_table(path)
    df = table.to_pandas()
    # Normalize ts to datetime tz-aware
    # if not np.issubdtype(df['ts'].dtype, np.datetime64):
    #     df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df['ts'] = pd.to_datetime(df['ts'], utc=True, errors='coerce')
    df = df.dropna(subset=['ts'])
    return df

def build_complete_index(ts_min: pd.Timestamp, ts_max: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(ts_min, ts_max, freq=freq, tz="UTC")

# ----------------- UI -----------------
parquet_path = st.sidebar.text_input("Parquet data path", "data/coinbase_BTCUSD_5m.parquet")  # default example
run = st.sidebar.button("Analyze")

if run:
    if not os.path.exists(parquet_path):
        st.error(f"Parquet not found: {parquet_path}")
        st.stop()

    df = load_df(parquet_path).sort_values("ts")
    cols_needed = {"ts","open","high","low","close","volume"}
    if not cols_needed.issubset(df.columns):
        st.error(f"Missing required columns: {sorted(cols_needed - set(df.columns))}")
        st.stop()

    # Meta
    symbol = df.get("symbol", pd.Series(dtype=str)).astype(str).dropna().unique()
    timeframe_values = df.get("timeframe", pd.Series(dtype=str)).astype(str).dropna().unique()
    vendor = df.get("vendor", pd.Series(dtype=str)).astype(str).dropna().unique()

    st.subheader("Metadata & Coverage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Symbol(s)", ",".join(symbol[:3]) if len(symbol) else "(n/a)")
    c3.metric("Timeframe tag(s)", ",".join(timeframe_values[:3]) if len(timeframe_values) else "(n/a)")
    c4.metric("Vendor(s)", ",".join(vendor[:3]) if len(vendor) else "(n/a)")

    ts_min, ts_max = df['ts'].min(), df['ts'].max()
    st.write(f"**Time range:** {ts_min} → {ts_max}")

    # Determine frequency
    if len(timeframe_values) == 1:
        freq = timeframe_to_pandas_freq(timeframe_values[0])
    else:
        freq = None
    if not freq:
        freq = infer_freq_from_timestamps(df['ts'])
    st.write(f"**Inferred frequency:** {freq}")

    # Coverage & gaps
    complete_idx = build_complete_index(ts_min, ts_max, freq) if freq else None
    gap_count = 0
    coverage_ratio = None
    gaps_df = None
    if complete_idx is not None and len(complete_idx) > 0:
        gaps = complete_idx.difference(df['ts'])
        gap_count = len(gaps)
        coverage_ratio = (1 - gap_count/len(complete_idx)) if len(complete_idx) else 1.0
        st.write(f"**Expected bars:** {len(complete_idx):,}  |  **Missing bars:** {gap_count:,}  |  **Coverage:** {coverage_ratio:.2%}")
        if gap_count > 0:
            # Group consecutive gaps into ranges
            gaps_series = pd.Series(gaps)
            breaks = gaps_series.diff() != (complete_idx[1] - complete_idx[0])
            group_ids = breaks.cumsum()
            ranges = gaps_series.groupby(group_ids).agg(['min','max','count']).rename(columns={'min':'start','max':'end'})
            gaps_df = ranges
            st.caption("Gap ranges (consecutive missing bars)")
            st.dataframe(gaps_df.head(100))

    # Duplicates
    dups = int(df['ts'].duplicated().sum())
    st.write(f"**Duplicate timestamp rows:** {dups}")

    # Basic stats
    st.subheader("Price & Volume Stats")
    close = df.set_index('ts')['close']
    volume = df.set_index('ts')['volume']
    ret = close.pct_change().dropna()

    stats = {
        "close_min": float(close.min()),
        "close_max": float(close.max()),
        "close_median": float(close.median()),
        "volume_mean": float(volume.mean()),
        "volume_0_count": int((volume == 0).sum()),
        "returns_mean": float(ret.mean()),
        "returns_std": float(ret.std()),
        "returns_skew": float(ret.skew()),
        "returns_kurt": float(ret.kurtosis()),
    }
    st.json(stats)

    # Charts
    st.subheader("Charts")
    # 1) Close price line
    fig1 = plt.figure(figsize=(10,4))
    close.plot()
    plt.title("Close Price")
    plt.xlabel("Time"); plt.ylabel("Price")
    st.pyplot(fig1)
    plt.close(fig1)

    # 2) Volume bars
    fig2 = plt.figure(figsize=(10,3))
    volume.plot(kind='bar', width=1.0)
    plt.title("Volume (bar chart)")
    plt.xlabel("Time"); plt.ylabel("Volume")
    st.pyplot(fig2)
    plt.close(fig2)

    # 3) Returns histogram
    fig3 = plt.figure(figsize=(6,4))
    ret.plot(kind='hist', bins=100)
    plt.title("Returns Histogram")
    plt.xlabel("Return")
    st.pyplot(fig3)
    plt.close(fig3)

    # 4) Hour-of-day average return
    hod = ret.copy()
    try:
        hod.index = hod.index.tz_convert('UTC')
    except Exception:
        pass
    hod_avg = hod.groupby(hod.index.hour).mean()
    fig4 = plt.figure(figsize=(8,3))
    hod_avg.plot(kind='bar')
    plt.title("Avg Return by Hour (UTC)")
    plt.xlabel("Hour"); plt.ylabel("Avg Return")
    st.pyplot(fig4)
    plt.close(fig4)

    # 5) Spread proxy & outlier scan (optional proxy using high-low/close)
    if {'high','low','close'}.issubset(df.columns):
        hlc = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        fig5 = plt.figure(figsize=(8,3))
        pd.Series(hlc.values, index=df['ts']).rolling(288).mean().plot()
        plt.title("(High-Low)/Close rolling mean (proxy for spread/vol)")
        plt.xlabel("Time"); plt.ylabel("Ratio")
        st.pyplot(fig5)
        plt.close(fig5)

    # Downloads
    st.subheader("Downloads")
    if gaps_df is not None and not gaps_df.empty:
        gaps_csv = gaps_df.to_csv().encode('utf-8')
        st.download_button("Download gap report (CSV)", gaps_csv, file_name="gap_report.csv", mime="text/csv")
    # small sample
    st.caption("Head/Tail preview")
    st.dataframe(df.head(10))
    st.dataframe(df.tail(10))

st.sidebar.markdown("""**Tip**: Use this inspector to validate vendor coverage and catch data gaps/duplications before backtests.""")
