from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Bar(BaseModel):
    ts: datetime = Field(..., description="Bar timestamp in UTC (start of bar)")
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str
    vendor: str = "ccxt"
