from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Quant Trade Service (Stub)")

class Order(BaseModel):
    symbol: str
    side: str  # BUY/SELL
    qty: float
    price: float | None = None  # optional (market if None)

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.post("/orders")
def place_order(o: Order):
    # TODO: wire to broker adapter (alpaca/ib/ccxt)
    return {"status": "accepted", "order": o}
