import sys, pathlib, os, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dashboards.utils import get_parquet_files, get_model_files, format_file_display

st.set_page_config(page_title="RL Lab", layout="wide")
st.title("RL Lab — Train/Evaluate Agents")

st.markdown("This page shells out to CLI trainers in `rl/` (PPO/DQN) for quick demos.")

st.sidebar.subheader("📁 Data Selection")

# Auto-detect parquet files
parquet_files = get_parquet_files()
if parquet_files:
    # Create options with file info
    file_options = [format_file_display(f) for f in parquet_files]
    selected_display = st.sidebar.selectbox(
        "Select parquet file:",
        file_options,
        help="Auto-detected parquet files from data/ directory"
    )
    # Get the actual file path
    selected_idx = file_options.index(selected_display)
    parquet_path = parquet_files[selected_idx]

    # Show selected file info
    st.sidebar.info(f"Selected: {os.path.basename(parquet_path)}")
else:
    st.sidebar.warning("No parquet files found in data/ directory")
    parquet_path = st.sidebar.text_input("Manual parquet path:", "data/coinbase_BTCUSD_5m.parquet")

st.sidebar.subheader("🤖 Model Selection (for Evaluation)")

# Auto-detect model files
model_files = get_model_files()
selected_model_path = None
if model_files:
    model_options = [display for display, _ in model_files]
    selected_model_display = st.sidebar.selectbox(
        "Select trained model:",
        ["(Use training output)"] + model_options,
        help="Auto-detected model files from artifacts/ directory"
    )
    if selected_model_display != "(Use training output)":
        selected_idx = model_options.index(selected_model_display)
        selected_model_path = model_files[selected_idx][1]
        st.sidebar.info(f"Model: {os.path.basename(selected_model_path)}")
else:
    st.sidebar.warning("No model files found in artifacts/ directory")

st.sidebar.subheader("⚙️ Training Parameters")
window = st.sidebar.number_input("Window (bars)", 8, 512, 64)
action_mode = st.sidebar.selectbox("Action mode", ["discrete","box"])
max_leverage = st.sidebar.number_input("Max leverage", 0.1, 10.0, 1.0)
cost_bps = st.sidebar.number_input("Cost (bps on turnover)", 0.0, 100.0, 10.0)
algo = st.sidebar.selectbox("Algo", ["PPO","DQN"])
timesteps = st.sidebar.number_input("Timesteps", 1000, 2_000_000, 50_000, step=1000)
workdir = st.sidebar.text_input("Artifacts dir", "artifacts/rl_demo")

st.subheader("🏁 Actions")
col_a, col_b = st.columns(2)
train = col_a.button("🏃 Train New Model", help="Train a new RL model with current parameters")

# Use selected model if available, otherwise fall back to training output
if selected_model_path:
    evaluate = col_b.button(f"📋 Evaluate Selected Model", help=f"Evaluate: {os.path.basename(selected_model_path)}")
else:
    evaluate = col_b.button("📋 Evaluate Last Training", help="Evaluate the most recent training output")

if train:
    st.info("Starting training... (output captured below)")
    os.makedirs(workdir, exist_ok=True)
    import subprocess, sys as _sys
    if algo == "PPO":
        cmd = [ _sys.executable, "-m", "rl.train_ppo",
            "--data", parquet_path, "--timesteps", str(int(timesteps)),
            "--window", str(int(window)),
            "--action_mode", action_mode, "--cost_bps", str(float(cost_bps)),
            "--max_leverage", str(float(max_leverage)),
            "--out", workdir ]
    else:
        cmd = [ _sys.executable, "-m", "rl.train_dqn",
            "--data", parquet_path, "--timesteps", str(int(timesteps)),
            "--window", str(int(window)),
            "--cost_bps", str(float(cost_bps)),
            "--max_leverage", str(float(max_leverage)),
            "--out", workdir ]
    st.code(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    st.text_area("Training output", res.stdout + "\n" + res.stderr, height=240)

if evaluate:
    # Use selected model if available, otherwise use training output
    if selected_model_path:
        model_path = selected_model_path
        st.info(f"Evaluating selected model: {os.path.basename(model_path)}...")

        # Better algorithm detection logic
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)

        # Try to detect algorithm from path/filename
        eval_algo = None
        if "ppo" in model_path.lower():
            eval_algo = "ppo"
        elif "dqn" in model_path.lower():
            eval_algo = "dqn"
        elif "a2c" in model_path.lower():
            eval_algo = "a2c"

        # If no algorithm detected, ask user to specify
        if not eval_algo:
            st.warning("⚠️ Could not auto-detect algorithm from model path.")
            eval_algo = st.selectbox("Select algorithm for this model:", ["ppo", "dqn", "a2c"],
                                   help="Choose the algorithm that was used to train this model")
        else:
            st.success(f"✅ Auto-detected algorithm: **{eval_algo.upper()}**")

    else:
        # Handle both naming conventions for training output
        model_dir = workdir
        ppo_path = os.path.join(workdir, "ppo_agent.zip")
        dqn_path = os.path.join(workdir, "dqn_agent.zip")

        if algo == "PPO" and os.path.exists(ppo_path):
            model_path = ppo_path
        elif algo == "DQN" and os.path.exists(dqn_path):
            model_path = dqn_path
        else:
            # Fallback to generic agent.zip
            model_path = os.path.join(workdir, "agent.zip")

        eval_algo = algo.lower()
        st.info("Evaluating last training run...")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")

        # Show available models in the directory for debugging
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            available_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
            if available_files:
                st.info(f"Available model files in {model_dir}: {available_files}")
            else:
                st.info(f"No .zip files found in {model_dir}")
    else:
        import subprocess, sys as _sys

        # Set output directory
        eval_out_dir = os.path.dirname(model_path) if selected_model_path else workdir

        # Check for VecNormalize file in the same directory
        model_dir = os.path.dirname(model_path)
        vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")

        # Build command
        cmd = [ _sys.executable, "-m", "rl.eval_agent",
            "--data", parquet_path, "--algo", eval_algo,
            "--model_path", model_path, "--window", str(int(window)),
            "--action_mode", action_mode, "--cost_bps", str(float(cost_bps)),
            "--max_leverage", str(float(max_leverage)),
            "--out", eval_out_dir ]

        # Add VecNormalize if it exists
        if os.path.exists(vecnorm_path):
            cmd.extend(["--vecnorm_path", vecnorm_path])
            st.info(f"🔧 Using VecNormalize: {os.path.basename(vecnorm_path)}")

        # Show command
        with st.expander("🔍 Command Details", expanded=False):
            st.code(" ".join(cmd))

        # Run evaluation with progress indicator
        with st.spinner("🚀 Running evaluation..."):
            res = subprocess.run(cmd, capture_output=True, text=True)

        # Show command output if there are any errors
        if res.returncode != 0 or res.stderr:
            st.error("Evaluation failed!")
            st.text_area("Error output", res.stdout + "\n" + res.stderr, height=150)
        else:
            st.success("✅ Evaluation completed successfully!")

            # Small output for debugging (collapsed by default)
            with st.expander("📝 Command Output", expanded=False):
                st.text_area("Output", res.stdout + "\n" + res.stderr, height=100)

        # Load and display results
        eval_out_dir = os.path.dirname(model_path) if selected_model_path else workdir
        res_path = os.path.join(eval_out_dir, "eval_result.json")
        eq_path = os.path.join(eval_out_dir, "eval_equity.png")

        # Display results prominently
        if os.path.exists(res_path):
            st.subheader("📊 Evaluation Results")
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    result_data = json.load(f)

                # Display key metrics in columns
                if isinstance(result_data, dict):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if "cagr" in result_data:
                            st.metric("CAGR", f"{result_data['cagr']:.2%}")
                    with col2:
                        if "sharpe" in result_data:
                            st.metric("Sharpe Ratio", f"{result_data['sharpe']:.2f}")
                    with col3:
                        if "max_drawdown" in result_data:
                            st.metric("Max Drawdown", f"{result_data['max_drawdown']:.2%}")
                    with col4:
                        if "total_trades" in result_data:
                            st.metric("Total Trades", f"{result_data.get('total_trades', 'N/A')}")

                # Full results in expandable section
                with st.expander("📋 Full Results JSON", expanded=True):
                    st.json(result_data)

            except Exception as e:
                st.error(f"Failed to load results: {e}")
                st.info(f"Looking for results in: {res_path}")
        else:
            st.warning(f"Results file not found: {res_path}")
            st.info("The evaluation may have failed or the output path might be different.")

        # Display equity curve
        if os.path.exists(eq_path):
            st.subheader("📈 Equity Curve")
            st.image(eq_path, caption="Evaluation Equity Curve", use_column_width=True)
        else:
            st.warning(f"Equity chart not found: {eq_path}")
