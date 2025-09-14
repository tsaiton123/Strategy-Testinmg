from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import importlib, json, os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

@dataclass
class CostModel:
    fee_bps: float = 10.0
    slip_bps: float = 0.0
    def apply(self, weights: pd.Series) -> pd.Series:
        turnover = weights.diff().abs().fillna(0.0)
        total_bps = (self.fee_bps + self.slip_bps) / 1e4
        return turnover * total_bps

def load_df(parquet_path: str) -> pd.DataFrame:
    df = pq.read_table(parquet_path).to_pandas()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors='coerce')
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df

def import_strategy(import_path: str, params: Dict[str, Any] | None = None):
    if ':' in import_path:
        mod_name, cls_name = import_path.split(':', 1)
    else:
        parts = import_path.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError("Use 'module:Class' or 'module.Class' for --strategy")
        mod_name, cls_name = parts
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(**(params or {}))

def stats_from_pnl(pnl: pd.Series) -> Dict[str, float]:
    if len(pnl) == 0:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
    minutes_per_year = 365 * 24 * 60
    eq = (1 + pnl).cumprod()
    cagr = float(eq.iloc[-1] ** (minutes_per_year/len(eq)) - 1)
    sharpe = float((pnl.mean() / (pnl.std() + 1e-12)) * (minutes_per_year ** 0.5)) if pnl.std() > 0 else 0.0
    maxdd = float(((eq / eq.cummax()) - 1).min())
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd}

def run_backtest(parquet_path: str, strategy_import: str, params: Dict[str, Any] | None = None, cost: Optional[CostModel] = None, out_dir: str = "artifacts") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    df = load_df(parquet_path)
    strat = import_strategy(strategy_import, params)
    w = strat.generate_weights(df)
    px = df.set_index("ts")["close"].reindex(w.index).ffill()
    ret = px.pct_change().fillna(0.0)
    costs = (cost or CostModel()).apply(w)
    pnl = (w * ret) - costs
    eq = (1 + pnl).cumprod()
    import matplotlib.pyplot as plt
    eq_path = os.path.join(out_dir, "equity_curve.png")
    plt.figure(figsize=(10,5)); eq.plot(); plt.title("Equity Curve"); plt.tight_layout(); plt.savefig(eq_path, dpi=150); plt.close()
    roll_path = os.path.join(out_dir, "rolling_mean.png")
    plt.figure(figsize=(10,4)); pnl.rolling(1440).mean().plot(); plt.title("Rolling Mean Return (1D)"); plt.tight_layout(); plt.savefig(roll_path, dpi=150); plt.close()
    s = stats_from_pnl(pnl)
    meta = {"strategy": strategy_import, "params": params or {}, "rows": int(len(df)), "start": str(df["ts"].min()), "end": str(df["ts"].max())}
    result = {"stats": s, "meta": meta, "equity_curve_png": eq_path, "rolling_mean_png": roll_path}
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
