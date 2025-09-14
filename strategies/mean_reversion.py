from dataclasses import dataclass
import pandas as pd
import numpy as np
from .base import BaseStrategy, StrategyConfig

@dataclass
class MRConfig(StrategyConfig):
    lookback: int = 50
    z_entry: float = 1.0
    z_exit: float = 0.2

class MeanReversion(BaseStrategy):
    def __init__(self, lookback: int = 50, z_entry: float = 1.0, z_exit: float = 0.2, config: MRConfig | None = None):
        cfg = config or MRConfig(lookback=lookback, z_entry=z_entry, z_exit=z_exit)
        super().__init__(config=cfg)
        self.lookback, self.z_entry, self.z_exit = lookback, z_entry, z_exit
    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        px = df.set_index("ts")["close"].sort_index()
        ret = px.pct_change()
        ma = ret.rolling(self.lookback).mean()
        sd = ret.rolling(self.lookback).std().replace(0, np.nan)
        z = (ret - ma) / sd
        pos = pd.Series(0.0, index=px.index)
        long = z < -self.z_entry
        exit = z.abs() < self.z_exit
        pos = pos.where(~long, 1.0)
        if self.config.allow_short:
            short = z > self.z_entry
            pos = pos.where(~short, -1.0)
        pos = pos.where(~exit, 0.0)
        pos = pos.clip(-self.config.max_leverage, self.config.max_leverage)
        return pos.shift(self.config.lag_bars).fillna(0.0)
