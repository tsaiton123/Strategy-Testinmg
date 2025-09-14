from dataclasses import dataclass
import pandas as pd
from features.ta import sma, realized_vol, target_leverage

@dataclass
class MomVolParams:
    fast: int = 50
    slow: int = 200
    vol_lookback: int = 1440  # 1 day of minutes for crypto
    target_vol: float = 0.2   # 20% annualized

def generate_weights(df: pd.DataFrame, params: MomVolParams) -> pd.DataFrame:
    # df columns: ts, open, high, low, close, volume, symbol, timeframe, vendor
    px = df.set_index("ts")["close"].asfreq("T").ffill()
    ret = px.pct_change().fillna(0.0)

    fast_ma = sma(px, params.fast)
    slow_ma = sma(px, params.slow)
    signal = (fast_ma > slow_ma).astype(float)  # 1 long, 0 flat (can extend to -1 short)

    vol = realized_vol(ret, params.vol_lookback)
    lev = target_leverage(params.target_vol, vol)

    w = (signal * lev).shift(1).fillna(0.0)  # lag 1 bar to avoid lookahead
    out = pd.DataFrame({"weight": w, "ret": ret}, index=w.index)
    return out
