# dashboards/strategy_lab.py (修復版)
import sys, pathlib, os, importlib, inspect
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from backtest.engine import run_backtest, CostModel, load_df, import_strategy
from dashboards.utils import get_parquet_files, format_file_display

st.set_page_config(page_title="Strategy Lab", layout="wide")
st.title("Strategy Lab — Plugin Backtesting")

def list_strategies():
    """自動偵測 strategies/ 資料夾中的策略類別（排除基類）"""
    import pkgutil
    import strategies
    from strategies.base import BaseStrategy
    
    names = []
    excluded_classes = {'BaseStrategy', 'StrategyConfig'}  # 排除的基類
    
    for m in pkgutil.iter_modules(strategies.__path__):
        # 跳過私有模組和 base 模組
        if m.name.startswith('_') or m.name in ('base',):
            continue
        
        try:
            mod = importlib.import_module(f"strategies.{m.name}")
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                # 檢查條件：
                # 1. 類別名稱以大寫開頭
                # 2. 類別定義在 strategies 模組內
                # 3. 不是排除的基類
                # 4. 是 BaseStrategy 的子類（確保是策略類別）
                if (name[0].isupper() and 
                    getattr(obj, '__module__', '').startswith('strategies.') and
                    name not in excluded_classes and
                    issubclass(obj, BaseStrategy) and
                    obj is not BaseStrategy):
                    names.append((f"strategies.{m.name}", name))
        except Exception as e:
            st.sidebar.warning(f"⚠️ 無法載入 {m.name}: {e}")
            continue
    
    return names

# Sidebar controls
st.sidebar.subheader("📁 Data Selection")

# Auto-detect parquet files
parquet_files = get_parquet_files()
if parquet_files:
    file_options = [format_file_display(f) for f in parquet_files]
    selected_display = st.sidebar.selectbox(
        "Select parquet file:",
        file_options,
        help="Auto-detected parquet files from data/ directory"
    )
    selected_idx = file_options.index(selected_display)
    parquet_path = parquet_files[selected_idx]
    st.sidebar.info(f"Selected: {os.path.basename(parquet_path)}")
else:
    st.sidebar.warning("No parquet files found in data/ directory")
    parquet_path = st.sidebar.text_input("Manual parquet path:", "data/coinbase_BTCUSD_5m.parquet")

st.sidebar.subheader("⚙️ Strategy Configuration")

choices = list_strategies()
if not choices:
    st.error("❌ 沒有找到可用的策略！")
    st.info("請確保 `strategies/` 資料夾內有實作 `BaseStrategy` 的策略類別")
    st.stop()

# 顯示找到的策略數量
st.sidebar.success(f"✅ 找到 {len(choices)} 個可用策略")

choice_label = st.sidebar.selectbox("Strategy", [f"{m}:{c}" for m, c in choices])
fee_bps = st.sidebar.number_input("Fee (bps)", 0.0, 100.0, 10.0)
slip_bps = st.sidebar.number_input("Slippage (bps)", 0.0, 100.0, 0.0)

# Dynamic params from __init__ signature
mod_name, cls_name = choice_label.split(':')
mod = importlib.import_module(mod_name)
cls = getattr(mod, cls_name)
sig = inspect.signature(cls.__init__)
param_inputs = {}

st.sidebar.markdown("---")
st.sidebar.markdown("**Strategy Parameters:**")

for name, p in sig.parameters.items():
    if name in ('self', 'config'):
        continue
    ann = p.annotation
    default = None if p.default is inspect._empty else p.default
    
    # 根據類型建立不同的輸入控制項
    if ann in (int, 'int'):
        val = st.sidebar.number_input(
            name, 
            value=int(default) if default is not None else 10, 
            step=1,
            help=f"Type: {ann}"
        )
    elif ann in (float, 'float'):
        val = st.sidebar.number_input(
            name, 
            value=float(default) if default is not None else 0.0, 
            format="%f",
            help=f"Type: {ann}"
        )
    elif ann in (bool, 'bool'):
        val = st.sidebar.checkbox(
            name, 
            value=bool(default) if default is not None else False,
            help=f"Type: {ann}"
        )
    else:
        val = st.sidebar.text_input(
            name, 
            value=str(default) if default is not None else "",
            help=f"Type: {ann}"
        )
    param_inputs[name] = val

run = st.sidebar.button("🚀 Run backtest", type="primary")

if run:
    if not os.path.exists(parquet_path):
        st.error(f"❌ Parquet not found: {parquet_path}")
        st.stop()

    with st.spinner("⏳ Running backtest..."):
        try:
            # Run the backtest
            result = run_backtest(
                parquet_path,
                f"{mod_name}:{cls_name}",
                param_inputs,
                CostModel(fee_bps=fee_bps, slip_bps=slip_bps),
            )
            st.success("✅ Backtest complete!")

            # Show stats & meta
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CAGR", f"{result['stats']['CAGR']:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{result['stats']['Sharpe']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{result['stats']['MaxDD']:.2%}")

            # Expandable sections for details
            with st.expander("📊 Full Statistics", expanded=True):
                st.json(result["stats"])
            
            with st.expander("📋 Metadata"):
                st.json(result["meta"])

            # Charts
            st.subheader("📈 Performance Charts")
            col_a, col_b = st.columns(2)
            
            with col_a:
                if os.path.exists(result["equity_curve_png"]):
                    st.image(result["equity_curve_png"], caption="Equity Curve", use_column_width=True)
            
            with col_b:
                if os.path.exists(result["rolling_mean_png"]):
                    st.image(result["rolling_mean_png"], caption="Rolling Mean Return (1D)", use_column_width=True)

            # === Trades table ===
            st.subheader("📋 Trade History")
            df = load_df(parquet_path)
            strat = import_strategy(f"{mod_name}:{cls_name}", param_inputs)
            w = strat.generate_weights(df)
            px = df.set_index("ts")["close"].reindex(w.index).ffill()
            ret = px.pct_change().fillna(0.0)

            cost_rate = (fee_bps + slip_bps) / 1e4
            turnover = w.diff().abs().fillna(w.abs())
            costs = turnover * cost_rate

            pnl = (w * ret) - costs
            eq = (1 + pnl).cumprod()
            eq_prev = eq.shift(1).fillna(1.0)

            dw = w.diff().fillna(w)
            trades = pd.DataFrame({
                "ts": w.index,
                "side": dw.apply(lambda x: "BUY" if x > 0 else ("SELL" if x < 0 else "")),
                "delta_weight": dw,
                "portfolio_notional_before": eq_prev,
                "exec_price": px,
            })
            trades["trade_notional"] = trades["delta_weight"] * trades["portfolio_notional_before"]
            trades["est_qty"] = trades["trade_notional"] / trades["exec_price"]
            trades = trades[trades["delta_weight"] != 0].reset_index(drop=True)

            st.caption(
                "Derived from weight changes. "
                "Assumes fills at bar close and uses portfolio equity just before the trade."
            )
            st.dataframe(trades.head(100), use_container_width=True)
            st.info(f"📊 Total trades: {len(trades):,}")

            # Download CSV
            csv = trades.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download trades CSV", 
                data=csv, 
                file_name=f"trades_{cls_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption(
    "💡 **Tip**: Add new strategies under `strategies/` folder and they will auto-appear here. "
    "Costs are applied on turnover (|Δweight| × (fee_bps+slip_bps))."
)