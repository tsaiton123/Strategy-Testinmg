from __future__ import annotations
import pandas as pd
import pyarrow.parquet as pq

def load_parquet(path: str) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    need = {"open","high","low","close","volume"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {sorted(missing)}")
    return df
