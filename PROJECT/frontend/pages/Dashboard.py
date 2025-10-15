# rlhf_eval_dashboard.py
import json
from pathlib import Path
import os
import time
import random

import pandas as pd
import streamlit as st
import plotly.express as px

# ------------------------
# CONFIG
# ------------------------
EVAL_DIR = Path("data/rlhf_eval")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
GLOB_PATTERN = "rlhf_run_*.json"

# ------------------------
# Toy JSON generator (used only if there are no real runs)
# ------------------------
def make_toy_run(path: Path, steps: int = 120):
    """Create a toy RLHF evaluation JSON showing plausible metrics over updates."""
    runs = []
    base_reward = 0.2
    for i in range(steps):
        # simulate improving reward with noise
        avg_reward = base_reward + 0.6 * (1 - (1 / (1 + 0.03 * i))) + random.gauss(0, 0.02)
        reward_std = max(0.01, 0.12 - 0.0006 * i + random.gauss(0, 0.005))
        kl = max(0.001, 0.02 + 0.0005 * i + random.gauss(0, 0.001))
        policy_loss = -0.2 + 0.0008 * i + random.gauss(0, 0.02)
        value_loss = 0.8 - 0.002 * i + random.gauss(0, 0.02)
        entropy = max(0.01, 0.6 - 0.002 * i + random.gauss(0, 0.01))
        win_rate = min(1.0, 0.35 + 0.5 * (1 - (1 / (1 + 0.03 * i))) + random.gauss(0, 0.03))

        runs.append(
            {
                "update": i + 1,
                "timestamp": int(time.time()) + i * 60,
                "avg_reward": round(avg_reward, 4),
                "reward_std": round(reward_std, 4),
                "kl": round(kl, 5),
                "policy_loss": round(policy_loss, 4),
                "value_loss": round(value_loss, 4),
                "entropy": round(entropy, 4),
                "win_rate": round(win_rate, 4),
            }
        )

    meta = {
        "run_name": path.name,
        "created": time.ctime(),
        "notes": "Toy RLHF run for dashboard demos",
        "steps": steps,
    }
    obj = {"meta": meta, "history": runs}
    path.write_text(json.dumps(obj, indent=2))
    return obj

# ------------------------
# Load JSON run
# ------------------------
def load_run(path: Path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to load JSON {path}: {e}")
        return None
    # normalize: accept either {"history":[{...}]} or list of records
    if isinstance(obj, dict) and "history" in obj and isinstance(obj["history"], list):
        df = pd.DataFrame(obj["history"])
        meta = obj.get("meta", {})
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
        meta = {}
    else:
        st.error(f"Unrecognized JSON schema in {path}")
        return None
    # ensure update column exists
    if "update" not in df.columns:
        df = df.reset_index().rename(columns={"index": "update"})
        df["update"] = df["update"] + 1
    return df, meta

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(layout="wide", page_title="RLHF Evaluation Dashboard")
st.title("📈 RLHF Evaluation Dashboard")

# discover runs
run_files = sorted(list(EVAL_DIR.glob(GLOB_PATTERN)), key=os.path.getmtime, reverse=True)

# if none found, create a toy one
if len(run_files) == 0:
    st.warning("No RLHF run JSONs found — creating a toy run for demo purposes.")
    toy = make_toy_run(EVAL_DIR / "rlhf_run_toy.json", steps=120)
    run_files = sorted(list(EVAL_DIR.glob(GLOB_PATTERN)), key=os.path.getmtime, reverse=True)

# sidebar: select run + quick metadata
st.sidebar.header("Select run")
choices = [p.name for p in run_files]
selected_name = st.sidebar.selectbox("Run file", choices, index=0)
selected_path = EVAL_DIR / selected_name

# load run into DataFrame
result = load_run(selected_path)
if result is None:
    st.stop()
df, meta = result

st.sidebar.markdown("**Run info**")
st.sidebar.write(f"File: `{selected_name}`")
for k, v in meta.items():
    st.sidebar.write(f"- **{k}**: {v}")

# show raw JSON
with st.expander("Raw JSON (first 2KB)", expanded=False):
    raw = selected_path.read_text(encoding="utf-8")
    st.code(raw[:2048], language="json")

# main metrics layout
st.header("Training metrics over updates")
st.markdown("Quick visual checks: reward curve, KL drift, training losses, entropy, and human-win rate.")

# ensure numeric types
numeric_cols = ["update", "avg_reward", "reward_std", "kl", "policy_loss", "value_loss", "entropy", "win_rate"]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# plot: avg reward with std shading
if "avg_reward" in df.columns:
    fig = px.line(df, x="update", y="avg_reward", title="Average Reward over Updates", labels={"update": "PPO Update", "avg_reward": "Avg Reward"})
    if "reward_std" in df.columns:
        fig.add_traces(px.line(df, x="update", y="reward_std", labels={"reward_std": "Reward Std"}).data)
    st.plotly_chart(fig, use_container_width=True)

# plot: KL divergence
if "kl" in df.columns:
    fig = px.line(df, x="update", y="kl", title="KL Divergence over Updates", labels={"kl": "KL Divergence"})
    st.plotly_chart(fig, use_container_width=True)

# plot: losses
loss_cols = [c for c in ["policy_loss", "value_loss"] if c in df.columns]
if loss_cols:
    fig = px.line(df, x="update", y=loss_cols, title="Policy / Value Loss", labels={"value": "Loss"})
    st.plotly_chart(fig, use_container_width=True)

# entropy
if "entropy" in df.columns:
    fig = px.line(df, x="update", y="entropy", title="Policy Entropy over Updates", labels={"entropy": "Entropy"})
    st.plotly_chart(fig, use_container_width=True)

# win-rate
if "win_rate" in df.columns:
    fig = px.line(df, x="update", y="win_rate", title="Win Rate (policy vs baseline)", labels={"win_rate": "Win Rate"})
    st.plotly_chart(fig, use_container_width=True)

# combined small-multiples for quick glance
cols_for_small = [c for c in ["avg_reward", "kl", "policy_loss", "value_loss", "entropy", "win_rate"] if c in df.columns]
if cols_for_small:
    st.subheader("Small multiples")
    n = len(cols_for_small)
    grid_cols = st.columns(min(n, 3))
    for i, colname in enumerate(cols_for_small[:3]):
        fig = px.line(df, x="update", y=colname, title=colname)
        grid_cols[i].plotly_chart(fig, use_container_width=True)

# show raw table (paginated)
st.subheader("Raw metrics table (first 500 rows)")
st.dataframe(df.head(500))

# CSV export
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download metrics as CSV", data=csv, file_name=f"{selected_name.replace('.json','.csv')}", mime="text/csv")

# quick sanity checks
st.subheader("Sanity checks")
checks = []
if "avg_reward" in df.columns:
    checks.append(("Final avg reward", float(df["avg_reward"].iloc[-1])))
if "kl" in df.columns:
    checks.append(("Final KL", float(df["kl"].iloc[-1])))
if "win_rate" in df.columns:
    checks.append(("Final win rate", float(df["win_rate"].iloc[-1])))
for name, val in checks:
    st.metric(label=name, value=round(val, 4))

st.info("This dashboard expects a JSON with a 'history' array of per-update records containing fields like "
        "`update`, `avg_reward`, `reward_std`, `kl`, `policy_loss`, `value_loss`, `entropy`, `win_rate`. "
        "If your ETL produces a different schema, adapt the `load_run()` normalizer accordingly.")
