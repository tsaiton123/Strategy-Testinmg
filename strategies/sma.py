from dataclasses import dataclass
import pandas as pd
from .base import BaseStrategy, StrategyConfig

@dataclass
class SMAConfig(StrategyConfig):
    fast: int = 50
    slow: int = 200

class SMA(BaseStrategy):
    def __init__(self, fast: int = 50, slow: int = 200, config: SMAConfig | None = None):
        cfg = config or SMAConfig(fast=fast, slow=slow)
        super().__init__(config=cfg)
        self.fast = fast
        self.slow = slow
    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        px = df.set_index("ts")["close"].asfreq(pd.infer_freq(df.set_index("ts").index) or "T").ffill()
        fast_ma = px.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = px.rolling(self.slow, min_periods=self.slow).mean()
        signal = (fast_ma > slow_ma).astype(float)
        if self.config.allow_short:
            signal = signal.where(fast_ma >= slow_ma, -1.0).fillna(0.0)
        w = signal.clip(-self.config.max_leverage, self.config.max_leverage)
        return w.shift(self.config.lag_bars).fillna(0.0)
