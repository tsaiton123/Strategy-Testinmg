# strategies/my_alpha.py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyConfig

@dataclass
class MyAlphaConfig(StrategyConfig):
    rsi_lookback: int = 14    # RSI window
    rsi_low: float = 30.0     # go long if RSI < rsi_low
    rsi_high: float = 70.0    # go short if RSI > rsi_high (only if allow_short=True)
    smooth_span: int = 1      # optional EMA smoothing of raw signal (1 = no smoothing)

class MyAlpha(BaseStrategy):
    """
    RSI mean-reversion:
      - Long when RSI < rsi_low
      - Short when RSI > rsi_high (only if allow_short = True), else flat
      - Signals are lagged by config.lag_bars to avoid lookahead
      - Weights clipped to [-max_leverage, +max_leverage]
    """
    def __init__(
        self,
        rsi_lookback: int = 14,
        rsi_low: float = 30.0,
        rsi_high: float = 70.0,
        smooth_span: int = 1,
        config: MyAlphaConfig | None = None,
    ):
        cfg = config or MyAlphaConfig(
            rsi_lookback=rsi_lookback,
            rsi_low=rsi_low,
            rsi_high=rsi_high,
            smooth_span=smooth_span,
        )
        super().__init__(config=cfg)
        self.rsi_lookback = rsi_lookback
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.smooth_span = smooth_span

    @staticmethod
    def _rsi(close: pd.Series, n: int) -> pd.Series:
        """TA-Lib-free RSI implementation."""
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        # Wilder's smoothing via EMA
        avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        x = df.set_index("ts").sort_index()
        px = x["close"].astype(float)

        rsi = self._rsi(px, self.rsi_lookback)

        # raw signal: +1 long, -1 short (if allowed), else 0
        long_sig = (rsi < self.rsi_low).astype(float)
        short_sig = (rsi > self.rsi_high).astype(float)

        if self.config.allow_short:
            raw = long_sig - short_sig
        else:
            raw = long_sig  # only go long; otherwise flat

        # optional smoothing to reduce churn
        if self.smooth_span and self.smooth_span > 1:
            raw = raw.ewm(span=self.smooth_span, adjust=False).mean()

        w = raw.clip(-self.config.max_leverage, self.config.max_leverage)

        # lag to avoid lookahead
        return w.shift(self.config.lag_bars).fillna(0.0)
