import pandas as pd
import numpy as np

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()

def realized_vol(returns: pd.Series, lookback: int) -> pd.Series:
    # annualized vol using sqrt(252*6.5*60) ~ 6528 for 1-min equities; for crypto 365*24*60
    per_year = 365*24*60  # crypto minutes per year
    return returns.rolling(lookback).std() * np.sqrt(per_year)

def target_leverage(target_vol: float, realized_vol_series: pd.Series, eps: float = 1e-8) -> pd.Series:
    lev = target_vol / (realized_vol_series + eps)
    return lev.clip(upper=5.0)  # cap leverage for sanity
