from __future__ import annotations
from dataclasses import dataclass, fields, asdict
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class StrategyConfig:
    rebalance: str = "bar"
    lag_bars: int = 1
    allow_short: bool = False
    max_leverage: float = 1.0
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BaseStrategy:
    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()
    @classmethod
    def signature(cls) -> Dict[str, str]:
        import inspect
        sig = inspect.signature(cls.__init__)
        out = {}
        for name, param in sig.parameters.items():
            if name in ("self", "config"):
                continue
            annot = str(param.annotation)
            default = None if param.default is inspect._empty else param.default
            out[name] = f"{annot} (default={default})"
        return out
    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Strategy must implement generate_weights(df) -> pd.Series[weight]")
    def fit(self, df: pd.DataFrame) -> None:
        return
