# pref_pair_curator_v2.py
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

# -----------------------
# Config / Paths
# -----------------------
RUNS_DIR = Path("data/pref_runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_GLOB = "run_*.json"
PROGRESS_PREFIX = "progress_"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # default policy model (prod)
DEFAULT_DEV_MODEL = "arnir0/Tiny-LLM"
# Columns for dataframe
DF_COLS = ["id", "prompt", "response_A", "response_B", "status", "raw"]

# -----------------------
# Persistence helpers
# -----------------------
def discover_run_files() -> List[Path]:
    files = list(RUNS_DIR.glob(RUN_GLOB))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files

def progress_path_for_run(run_file: Path) -> Path:
    return run_file.parent / f"{PROGRESS_PREFIX}{run_file.name}"

def save_progress(df: pd.DataFrame, run_file: Path) -> None:
    out_path = progress_path_for_run(run_file)
    records = df.to_dict(orient="records")
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

def load_progress_if_exists(run_file: Path) -> Optional[pd.DataFrame]:
    p = progress_path_for_run(run_file)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
    except Exception as e:
        logger.error("Failed to load progress file %s: %s", p, e)
    return None

# -----------------------
# Model utils (CPU-friendly)
# -----------------------
@st.cache_resource
def load_model_and_tokenizer(checkpoint: str):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.to("cpu")
    model.eval()
    return model, tok

def generate_text_single(model, tokenizer, prompt: str, gen_len: int = 48, temperature: float = 0.2, top_p: float = 0.9) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=gen_len,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    gen_tokens = out[0][prompt_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()

# -----------------------
# Session state init
# -----------------------
def _init_session_state_with_run(path: Optional[Path]):
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=DF_COLS)
        st.session_state.current_index = 0
        st.session_state.run_file = None

    if path is None:
        return

    try:
        raw_text = path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except Exception as e:
        st.warning(f"Failed to read run file {path}: {e}")
        st.session_state.df = pd.DataFrame(columns=DF_COLS)
        st.session_state.run_file = str(path.name)
        return

    rows = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                rows.append({"id": str(i), "prompt": item, "response_A": "", "response_B": "", "status": "pending", "raw": item})
            elif isinstance(item, dict):
                q = item.get("prompt") or item.get("question") or item.get("q") or ""
                rows.append({"id": str(item.get("id") or i), "prompt": q, "response_A": item.get("response_A", ""), "response_B": item.get("response_B", ""), "status": item.get("status","pending"), "raw": item})
    else:
        st.warning(f"Unexpected run file structure: {path}")

    df = pd.DataFrame(rows)
    prog = load_progress_if_exists(path)
    if prog is not None and not prog.empty:
        st.session_state.df = prog.reset_index(drop=True)
        st.session_state.current_index = 0
        st.session_state.run_file = str(path.name)
        st.success(f"Loaded saved progress for {path.name}")
        return

    st.session_state.df = df.reset_index(drop=True)
    st.session_state.current_index = 0
    st.session_state.run_file = str(path.name)

def save_current_progress() -> Path:
    run_name = st.session_state.get("run_file") or "custom_run.json"
    run_file = RUNS_DIR / run_name
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    save_progress(st.session_state.df, run_file)
    return progress_path_for_run(run_file)

def save_approved_export() -> Optional[Path]:
    run_name = st.session_state.get("run_file") or "custom_run.json"
    out_path = RUNS_DIR / f"approved_{run_name}"
    approved = st.session_state.df[st.session_state.df["status"] == "approved"]
    if approved.empty:
        return None
    recs = approved[["id", "prompt", "response_A", "response_B"]].to_dict(orient="records")
    out_path.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

# -----------------------
# UI
# -----------------------
def main():
    st.set_page_config(page_title="Pedantic vs Baseline Curator", layout="wide")
    st.title("🧾 Pedantic ↔ Baseline Preference Curator")

    st.write(
        """
        Curate preference pairs for reward model training.  
        Review A (pedantic) vs B (baseline), edit if needed, and mark Approved / Rejected / Tie / Skip.
        Progress autosaves to `data/pref_runs/progress_<runfile>.json`.
        """
    )

    # Sidebar: run selection & model settings
    st.sidebar.header("Runs & Generation")
    runs = discover_run_files()
    selected_run: Optional[Path] = None
    if runs:
        opts = [p.name for p in runs]
        sel = st.sidebar.selectbox("Choose latest run", opts, index=0)
        selected_run = runs[opts.index(sel)]
    custom_path = st.sidebar.text_input("Or paste a custom run filename (in data/pref_runs)", value="")
    if custom_path:
        p = RUNS_DIR / custom_path
        if p.exists():
            selected_run = p
        else:
            st.sidebar.warning("Custom path does not exist yet. Add a JSON run file under data/pref_runs/.")

    # Model selection: for now we use the same model twice; you can change later
    st.sidebar.subheader("Model(s)")
    model_a = st.sidebar.selectbox("Model A (pedantic)", [DEFAULT_MODEL, DEFAULT_DEV_MODEL], index=0)
    model_b = st.sidebar.selectbox("Model B (baseline)", [DEFAULT_MODEL, DEFAULT_DEV_MODEL], index=0)
    st.sidebar.caption("Currently the app will load both; OK to pick the same model for both.")
    gen_len = st.sidebar.number_input("Gen tokens", min_value=8, max_value=256, value=48)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
    top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95)

    if st.sidebar.button("Reload selected run"):
        if selected_run:
            _init_session_state_with_run(selected_run)

    # init state
    _init_session_state_with_run(selected_run)

    tab1, tab2 = st.tabs(["Review Pending", "Create New"])

    # ---------------- Review Pending ----------------
    with tab1:
        df: pd.DataFrame = st.session_state.df
        pending_mask = df["status"] == "pending"
        total_pending = int(pending_mask.sum())
        st.header("Review Pending Prompts")

        if total_pending == 0:
            st.success("No pending prompts. Create or load another run.")
        else:
            pending_idx_list = df[pending_mask].index.tolist()
            cur_pos = st.session_state.get("current_index", 0) % max(1, len(pending_idx_list))
            st.session_state.current_index = cur_pos
            row_index = pending_idx_list[cur_pos]
            rec = df.loc[row_index]

            st.info(f"Reviewing {cur_pos + 1} / {len(pending_idx_list)} pending (global idx {row_index})")
            st.progress((cur_pos + 1) / max(1, len(pending_idx_list)))

            col_left, col_right = st.columns([1, 1], gap="large")
            with col_left:
                st.subheader("Prompt")
                edit_prompt = st.text_area("Edit prompt (optional)", value=rec["prompt"] or "", height=200, key=f"q_{row_index}")
            with col_right:
                st.subheader("Responses (edit before saving)")
                a_val = rec.get("response_A", "") or ""
                b_val = rec.get("response_B", "") or ""
                edited_a = st.text_area("A — Pedantic", value=a_val, height=200, key=f"a_{row_index}")
                edited_b = st.text_area("B — Baseline", value=b_val, height=200, key=f"b_{row_index}")

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("❌ Reject", key=f"reject_{row_index}"):
                    df.at[row_index, "status"] = "rejected"
                    df.at[row_index, "prompt"] = edit_prompt
                    df.at[row_index, "response_A"] = edited_a
                    df.at[row_index, "response_B"] = edited_b
                    st.session_state.current_index = (st.session_state.current_index + 1) % max(1, len(pending_idx_list))
                    save_progress(df, RUNS_DIR / (st.session_state.run_file or "custom_run.json"))
                    st.warning("Rejected and saved. Moving on.")
                    st.experimental_rerun()
            with c2:
                if st.button("🔁 Tie", key=f"tie_{row_index}"):
                    df.at[row_index, "status"] = "tie"
                    df.at[row_index, "prompt"] = edit_prompt
                    df.at[row_index, "response_A"] = edited_a
                    df.at[row_index, "response_B"] = edited_b
                    st.session_state.current_index = (st.session_state.current_index + 1) % max(1, len(pending_idx_list))
                    save_progress(df, RUNS_DIR / (st.session_state.run_file or "custom_run.json"))
                    st.info("Marked tie and saved. Moving on.")
                    st.experimental_rerun()
            with c3:
                if st.button("✅ Approve", key=f"approve_{row_index}"):
                    df.at[row_index, "status"] = "approved"
                    df.at[row_index, "prompt"] = edit_prompt
                    df.at[row_index, "response_A"] = edited_a
                    df.at[row_index, "response_B"] = edited_b
                    st.session_state.current_index = (st.session_state.current_index + 1) % max(1, len(pending_idx_list))
                    save_progress(df, RUNS_DIR / (st.session_state.run_file or "custom_run.json"))
                    st.success("Approved and saved. Moving on.")
                    st.experimental_rerun()
            with c4:
                if st.button("🎲 Skip", key=f"skip_{row_index}"):
                    if len(pending_idx_list) <= 1:
                        st.info("No other pending items to skip to.")
                    else:
                        choices = list(range(len(pending_idx_list)))
                        if st.session_state.current_index in choices:
                            choices.remove(st.session_state.current_index)
                        new_pos = random.choice(choices) if choices else st.session_state.current_index
                        st.session_state.current_index = new_pos
                        save_progress(df, RUNS_DIR / (st.session_state.run_file or "custom_run.json"))
                        st.info("Skipped to a random pending item.")
                        st.experimental_rerun()

        # preview approved & export
        st.divider()
        st.subheader("Approved (preview & export)")
        approved = st.session_state.df[st.session_state.df["status"] == "approved"]
        if not approved.empty:
            st.dataframe(approved[["id", "prompt", "response_A", "response_B"]].head(100), use_container_width=True)
            csv = approved[["id", "prompt", "response_A", "response_B"]].to_csv(index=False).encode("utf-8")
            st.download_button("Download approved (CSV)", data=csv, file_name="approved_pairs.csv", mime="text/csv")
            if st.button("Save approved to file"):
                out = save_approved_export()
                if out:
                    st.success(f"Saved approved file to {out}")
                else:
                    st.info("No approved pairs to save.")
        else:
            st.info("No approved pairs yet.")

    # ---------------- Create New ----------------
    with tab2:
        st.header("Create New Prompt / Generate A-B")
        st.write("Add a manual prompt and optionally auto-generate A/B using selected models.")
        with st.form("new_form"):
            new_prompt = st.text_area("Prompt", height=150)
            do_gen = st.checkbox("Auto-generate A/B", value=True)
            a_pref = st.text_input("A prefix (pedantic)", value="You are an erudite 19th-century scholar. Be pedantic and precise. Reply:")
            b_pref = st.text_input("B prefix (baseline)", value="Answer succinctly and plainly:")
            submitted = st.form_submit_button("Add prompt")
            if submitted:
                if not new_prompt.strip():
                    st.warning("Prompt cannot be empty.")
                else:
                    next_id = str(len(st.session_state.df) + 1)
                    row = {"id": next_id, "prompt": new_prompt, "response_A": "", "response_B": "", "status": "pending", "raw": {}}
                    if do_gen:
                        try:
                            modelA, tokA = load_model_and_tokenizer(model_a)
                            modelB, tokB = load_model_and_tokenizer(model_b)
                            promptA = f"{a_pref}\n\n{new_prompt}" if a_pref else new_prompt
                            promptB = f"{b_pref}\n\n{new_prompt}" if b_pref else new_prompt
                            row["response_A"] = generate_text_single(modelA, tokA, promptA, gen_len=gen_len, temperature=temperature, top_p=top_p)
                            row["response_B"] = generate_text_single(modelB, tokB, promptB, gen_len=gen_len, temperature=temperature, top_p=top_p)
                        except Exception as e:
                            st.error(f"Auto-generation failed: {e}")
                    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)
                    out = save_current_progress()
                    st.success(f"Added prompt (id={next_id}) and saved progress to {out}")

    # footer controls
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save progress now"):
            out = save_current_progress()
            st.success(f"Saved progress to {out}")
    with c2:
        if st.button("Reload progress file"):
            runf = st.session_state.get("run_file")
            if runf:
                _init_session_state_with_run(RUNS_DIR / runf)
                st.success("Reloaded progress.")
            else:
                st.warning("No run file configured.")

    st.caption(f"Autosave path (progress): `data/pref_runs/{PROGRESS_PREFIX}{st.session_state.get('run_file') or 'custom_run.json'}`")

if __name__ == "__main__":
    main()
