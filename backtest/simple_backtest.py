import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from signals.mom_vol_target import MomVolParams, generate_weights

def run_backtest(parquet_path: str, fast: int, slow: int, vol_lookback: int, target_vol: float, fee_bps: float):
    df = pq.read_table(parquet_path).to_pandas()
    params = MomVolParams(fast=fast, slow=slow, vol_lookback=vol_lookback, target_vol=target_vol)
    ws = generate_weights(df, params)

    # Transaction costs: apply on weight changes (turnover)
    turnover = ws["weight"].diff().abs().fillna(0.0)
    cost = turnover * (fee_bps / 1e4)

    pnl = ws["weight"] * ws["ret"] - cost
    eq = (1 + pnl).cumprod()

    stats = {
        "n_bars": int(len(ws)),
        "CAGR": float(eq.iloc[-1] ** (525600/len(ws)) - 1) if len(ws) > 0 else 0.0,  # minutes per year ~ 365*24*60
        "Sharpe": float((pnl.mean() / (pnl.std() + 1e-12)) * (525600 ** 0.5)) if pnl.std() > 0 else 0.0,
        "MaxDD": float(((eq / eq.cummax()) - 1).min()),
        "Turnover (avg per bar)": float(turnover.mean()),
    }
    print("== Backtest summary ==")
    for k, v in stats.items():
        print(f"{k:24s} {v:.6f}" if isinstance(v, float) else f"{k:24s} {v}")

    # Plots
    import os
    os.makedirs("artifacts", exist_ok=True)

    plt.figure(figsize=(10,5))
    eq.plot()
    plt.title("Equity Curve")
    plt.ylabel("Equity")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig("artifacts/equity_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10,5))
    pnl.rolling(1440).mean().plot()
    plt.title("Rolling Mean Return (1D window)")
    plt.tight_layout()
    plt.savefig("artifacts/rolling_mean.png", dpi=150)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Input parquet path")
    p.add_argument("--fast", type=int, default=50)
    p.add_argument("--slow", type=int, default=200)
    p.add_argument("--vol_lookback", type=int, default=1440)
    p.add_argument("--target_vol", type=float, default=0.2)
    p.add_argument("--fee_bps", type=float, default=10.0)
    args = p.parse_args()

    run_backtest(args.data, args.fast, args.slow, args.vol_lookback, args.target_vol, args.fee_bps)

if __name__ == "__main__":
    main()
