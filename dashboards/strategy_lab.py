# dashboards/strategy_lab.py
import sys, pathlib, os, importlib, inspect
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from backtest.engine import run_backtest, CostModel, load_df, import_strategy

st.set_page_config(page_title="Strategy Lab", layout="wide")
st.title("Strategy Lab — Plugin Backtesting")

def list_strategies():
    import pkgutil
    import strategies
    names = []
    for m in pkgutil.iter_modules(strategies.__path__):
        if m.name.startswith('_') or m.name in ('base',):
            continue
        mod = importlib.import_module(f"strategies.{m.name}")
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if name[0].isupper() and getattr(obj, '__module__', '').startswith('strategies.'):
                names.append((f"strategies.{m.name}", name))
    return names

# Sidebar controls
parquet_path = st.sidebar.text_input("Parquet data path", "data/coinbase_BTCUSD_5m.parquet")

choices = list_strategies()
if not choices:
    st.warning("No strategies found. Add files under strategies/ (e.g., strategies/sma.py).")
    st.stop()

choice_label = st.sidebar.selectbox("Strategy", [f"{m}:{c}" for m, c in choices])
fee_bps = st.sidebar.number_input("Fee (bps)", 0.0, 100.0, 10.0)
slip_bps = st.sidebar.number_input("Slippage (bps)", 0.0, 100.0, 0.0)

# Dynamic params from __init__ signature
mod_name, cls_name = choice_label.split(':')
mod = importlib.import_module(mod_name)
cls = getattr(mod, cls_name)
sig = inspect.signature(cls.__init__)
param_inputs = {}
for name, p in sig.parameters.items():
    if name in ('self', 'config'):
        continue
    ann = p.annotation
    default = None if p.default is inspect._empty else p.default
    if ann in (int, 'int'):
        val = st.sidebar.number_input(name, value=int(default) if default is not None else 10, step=1)
    elif ann in (float, 'float'):
        val = st.sidebar.number_input(name, value=float(default) if default is not None else 0.0, format="%f")
    elif ann in (bool, 'bool'):
        val = st.sidebar.checkbox(name, value=bool(default) if default is not None else False)
    else:
        val = st.sidebar.text_input(name, value=str(default) if default is not None else "")
    param_inputs[name] = val

run = st.sidebar.button("Run backtest")

if run:
    if not os.path.exists(parquet_path):
        st.error(f"Parquet not found: {parquet_path}")
        st.stop()

    # Run the backtest (saves charts + returns stats)
    result = run_backtest(
        parquet_path,
        f"{mod_name}:{cls_name}",
        param_inputs,
        CostModel(fee_bps=fee_bps, slip_bps=slip_bps),
    )
    st.success("Backtest complete.")

    # Show stats & meta
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Stats")
        st.json(result["stats"])
    with c2:
        st.subheader("Meta")
        st.json(result["meta"])

    # Charts
    if os.path.exists(result["equity_curve_png"]):
        st.image(result["equity_curve_png"], caption="Equity Curve", use_column_width=True)
    if os.path.exists(result["rolling_mean_png"]):
        st.image(result["rolling_mean_png"], caption="Rolling Mean Return (1D)", use_column_width=True)

    # === Trades table (derived) ===
    # Recompute weights/ret to construct a trade ledger consistent with the engine
    df = load_df(parquet_path)
    strat = import_strategy(f"{mod_name}:{cls_name}", param_inputs)
    w = strat.generate_weights(df)  # weights applied (already lagged by strategy config)
    px = df.set_index("ts")["close"].reindex(w.index).ffill()
    ret = px.pct_change().fillna(0.0)

    # Costs consistent with engine
    cost_rate = (fee_bps + slip_bps) / 1e4
    turnover = w.diff().abs().fillna(w.abs())
    costs = turnover * cost_rate

    pnl = (w * ret) - costs
    eq = (1 + pnl).cumprod()
    eq_prev = eq.shift(1).fillna(1.0)

    # Trades occur when target weight changes
    dw = w.diff().fillna(w)  # first trade from 0 -> w0
    trades = pd.DataFrame({
        "ts": w.index,
        "side": dw.apply(lambda x: "BUY" if x > 0 else ("SELL" if x < 0 else "")),
        "delta_weight": dw,
        "portfolio_notional_before": eq_prev,
        "exec_price": px,
    })
    trades["trade_notional"] = trades["delta_weight"] * trades["portfolio_notional_before"]  # +/- in portfolio currency
    trades["est_qty"] = trades["trade_notional"] / trades["exec_price"]
    trades = trades[trades["delta_weight"] != 0].reset_index(drop=True)

    st.subheader("Trades")
    st.caption(
        "Derived from weight changes. "
        "Assumes fills at bar close and uses portfolio equity just before the trade for notional/qty estimation."
    )
    st.dataframe(trades.head(1000))
    st.write(f"Total trades: {len(trades):,}")

    # Download CSV
    csv = trades.to_csv(index=False).encode("utf-8")
    st.download_button("Download trades CSV", data=csv, file_name="trades.csv", mime="text/csv")

st.caption(
    "Tip: Add new strategies under strategies/ (they auto-appear). "
    "Costs are applied on turnover (|Δweight| × (fee_bps+slip_bps))."
)
