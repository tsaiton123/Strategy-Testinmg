from dataclasses import dataclass
import pandas as pd
from .base import BaseStrategy, StrategyConfig

@dataclass
class DonchianConfig(StrategyConfig):
    window: int = 55

class Donchian(BaseStrategy):
    def __init__(self, window: int = 55, config: DonchianConfig | None = None):
        cfg = config or DonchianConfig(window=window)
        super().__init__(config=cfg)
        self.window = window
    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        x = df.set_index("ts").sort_index()
        hi = x["high"].rolling(self.window, min_periods=self.window).max()
        lo = x["low"].rolling(self.window, min_periods=self.window).min()
        px = x["close"]
        signal = (px > hi.shift(1)).astype(float) - (px < lo.shift(1)).astype(float)
        if not self.config.allow_short:
            signal = signal.clip(lower=0.0)
        w = signal.clip(-self.config.max_leverage, self.config.max_leverage)
        return w.shift(self.config.lag_bars).fillna(0.0)
