from __future__ import annotations
import numpy as np
import pandas as pd

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def realized_vol(ret: pd.Series, lookback: int = 64, per_year: int = 365*24*60) -> pd.Series:
    return ret.rolling(lookback).std() * np.sqrt(per_year)

def zscore(series: pd.Series, lookback: int = 64) -> pd.Series:
    m = series.rolling(lookback).mean()
    s = series.rolling(lookback).std().replace(0, np.nan)
    return (series - m) / s

def build_features(df: pd.DataFrame, lookbacks=(5, 20, 50), rsi_n=14, vol_n=64) -> pd.DataFrame:
    x = df.set_index("ts").sort_index()
    px = x["close"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    volume = x["volume"].astype(float)
    ret1 = px.pct_change().fillna(0.0)

    feats = pd.DataFrame(index=px.index)

    # Basic return features
    feats["ret1"] = ret1

    # Enhanced momentum features
    for lb in lookbacks:
        feats[f"mom_{lb}"] = px.pct_change(lb)
        feats[f"zret_{lb}"] = zscore(ret1, lb)
        # Add momentum strength
        feats[f"mom_strength_{lb}"] = np.abs(feats[f"mom_{lb}"])

    # Momentum consensus
    mom_cols = [f"mom_{lb}" for lb in lookbacks]
    feats["momentum_consensus"] = np.sign(feats[mom_cols]).mean(axis=1)
    feats["momentum_strength"] = feats[[f"mom_strength_{lb}" for lb in lookbacks]].mean(axis=1)

    # RSI and variations
    feats["rsi"] = rsi(px, rsi_n) / 100.0
    feats["rsi_oversold"] = (feats["rsi"] < 0.3).astype(float)
    feats["rsi_overbought"] = (feats["rsi"] > 0.7).astype(float)

    # Volatility features
    feats["vol"] = realized_vol(ret1, vol_n)
    feats["vol_zscore"] = zscore(feats["vol"], vol_n)

    # Volatility regime detection
    vol_median = feats["vol"].rolling(252).median()
    feats["vol_regime_low"] = (feats["vol"] < vol_median * 0.7).astype(float)
    feats["vol_regime_high"] = (feats["vol"] > vol_median * 1.5).astype(float)
    feats["vol_regime_normal"] = 1 - feats["vol_regime_low"] - feats["vol_regime_high"]

    # Price position features
    feats["px_norm"] = (px / px.rolling(1000).mean()) - 1.0
    feats["px_percentile"] = px.rolling(100).rank(pct=True)

    # Microstructure features
    feats["hl_ratio"] = (high - low) / px
    feats["volume_profile"] = volume / volume.rolling(100).mean()
    feats["volume_zscore"] = zscore(volume, 50)

    # Time-based features
    feats["hour_sin"] = np.sin(2 * np.pi * feats.index.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * feats.index.hour / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * feats.index.dayofweek / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * feats.index.dayofweek / 7)

    # Market stress indicators
    feats["return_skew"] = ret1.rolling(100).skew()
    feats["return_kurt"] = ret1.rolling(100).kurt()

    # Trend strength
    ema_fast = px.ewm(span=12).mean()
    ema_slow = px.ewm(span=26).mean()
    feats["trend_strength"] = (ema_fast - ema_slow) / px
    feats["trend_direction"] = np.sign(feats["trend_strength"])

    # Clean and normalize
    feats = feats.replace([np.inf, -np.inf], np.nan)

    # Forward-fill then backward-fill, then zero-fill
    feats = feats.ffill().bfill().fillna(0.0)

    return feats

def normalize_features(feats: pd.DataFrame, method='robust') -> pd.DataFrame:
    """Normalize features to improve training stability"""
    normalized = feats.copy()

    if method == 'robust':
        # Robust normalization using quantiles
        for col in feats.columns:
            if feats[col].std() > 1e-8:  # Only normalize non-constant features
                q25 = feats[col].quantile(0.25)
                q75 = feats[col].quantile(0.75)
                iqr = q75 - q25
                if iqr > 1e-8:
                    median = feats[col].median()
                    normalized[col] = (feats[col] - median) / (iqr + 1e-8)
                    # Clip extreme outliers
                    normalized[col] = normalized[col].clip(-5, 5)

    elif method == 'rolling':
        # Rolling z-score normalization
        for col in feats.columns:
            if feats[col].std() > 1e-8:
                rolling_mean = feats[col].rolling(252, min_periods=50).mean()
                rolling_std = feats[col].rolling(252, min_periods=50).std()
                normalized[col] = (feats[col] - rolling_mean) / (rolling_std + 1e-8)
                normalized[col] = normalized[col].fillna(0).clip(-3, 3)

    return normalized.fillna(0.0)

def select_stable_features(feats: pd.DataFrame, min_periods=100) -> pd.DataFrame:
    """Remove features with poor stability or too many missing values"""
    stable_feats = feats.copy()

    for col in feats.columns:
        # Remove features with too many missing values
        if feats[col].isna().sum() > len(feats) * 0.5:
            stable_feats = stable_feats.drop(columns=[col])
            continue

        # Remove features with zero variance
        if feats[col].std() < 1e-8:
            stable_feats = stable_feats.drop(columns=[col])
            continue

        # Remove features with extreme skewness (likely data errors)
        if len(feats) > min_periods:
            skew = feats[col].skew()
            if abs(skew) > 10:
                stable_feats = stable_feats.drop(columns=[col])
                continue

    return stable_feats

def to_numpy_windowed(feats: pd.DataFrame, window: int) -> np.ndarray:
    # Return array shape (T, window, F) where t-th obs uses rows [t-window..t-1]
    arr = feats.to_numpy(dtype=float)
    T, F = arr.shape
    if T <= window:
        return np.empty((0, window, F), dtype=float)
    out = np.zeros((T - window + 1, window, F), dtype=float)
    for t in range(window, T+1):
        out[t - window] = arr[t-window:t]
    return out

def align_prices_for_obs(px: pd.Series, window: int) -> np.ndarray:
    # Align returns to observations
    ret = px.pct_change().fillna(0.0).to_numpy(dtype=float)
    if len(ret) <= window:
        return np.empty((0,), dtype=float)
    return ret[window-1:]  # aligned with to_numpy_windowed output
