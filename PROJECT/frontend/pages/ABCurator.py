# pref_pair_curator_run_autoeval.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset as HFDataset
from huggingface_hub import HfFolder

# ---------- Config ----------
RUNS_DIR = Path("data/pref_runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_PREFIX = "progress_"
DEFAULT_HF_PROMPT_DATASET = "eZWALT/rlhf_user_prompts"
HF_PROMPT_COLUMN = "prompt"

PEDANTIC_PREFIX = "Respond in the style of a meticulous 19th-century British intellectual: precise, ornate, and eloquent. Be pedantic yet polite:"
BASELINE_PREFIX = "Answer succinctly and plainly:"

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = "cpu"
DEFAULT_GEN_TOKENS = 128
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

# ---------- Helpers ----------
def progress_path_for_run(run_name: str) -> Path:
    return RUNS_DIR / f"{PROGRESS_PREFIX}{Path(run_name).name}"

def save_compact_progress(df: pd.DataFrame, run_name: str) -> Path:
    """Persist compact JSONL containing prompt/chosen/rejected for rows that have chosen/rejected or approved/rejected."""
    out = progress_path_for_run(run_name)
    recs = []
    for _, r in df.iterrows():
        chosen = r.get("chosen", "") or ""
        rejected = r.get("rejected", "") or ""
        if r.get("status") in ("approved", "rejected") or chosen or rejected:
            recs.append({"prompt": r["prompt"], "chosen": chosen, "rejected": rejected})
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs), encoding="utf-8")
    return out

def export_compact_jsonl(df: pd.DataFrame, out_name: str) -> Path:
    out_path = RUNS_DIR / out_name
    approved = df[df["status"] == "approved"]
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in approved.iterrows():
            rec = {"prompt": r["prompt"], "chosen": r.get("chosen",""), "rejected": r.get("rejected","")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path

# ---------- Model ----------
@st.cache_resource
def load_model_and_tokenizer(checkpoint: str = DEFAULT_MODEL):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model, tok

def generate_reply(model, tokenizer, system_prefix: str, prompt_text: str,
                   max_new_tokens: int, temperature: float, top_p: float) -> str:
    full = f"{prompt_text}\n\n{system_prefix}:"
    enc = tokenizer(full, return_tensors="pt", truncation=True, padding=True, max_length=512)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    gen_tokens = out[0][prompt_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

# ---------- HF loader ----------
def load_prompts_from_hf(dataset_id: str, column: str, n: int) -> List[str]:
    ds = load_dataset(dataset_id, split="train")
    prompts = [r[column] for r in ds if column in r and isinstance(r[column], str)]
    return prompts[:n]

# ---------- Push to HF ----------
def push_approved_to_hub_compact(df: pd.DataFrame, repo_id: str) -> dict:
    approved = df[df["status"] == "approved"]
    if approved.empty:
        raise RuntimeError("No approved pairs to push.")
    recs = approved[["prompt","chosen","rejected"]].to_dict(orient="records")
    hf_ds = HFDataset.from_pandas(pd.DataFrame(recs))
    token = HfFolder.get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Run `huggingface-cli login` and try again.")
    push_info = hf_ds.push_to_hub(repo_id, token=token)
    return push_info

# ---------- App ----------
def main():
    st.set_page_config(layout="wide", page_title="Pref Pair Curator — RUN / AUTOEVAL")
    st.title("Preference Pair Curator — RUN / AUTOEVAL")
    st.write("Press RUN to generate A/B for all prompts. If AUTOEVALUATE is enabled, A will be accepted and pushed to HF automatically.")

    # ---------- Sidebar: minimal & clear ----------
    with st.sidebar:
        st.header("Config")
        hf_dataset = st.text_input("HF prompts dataset id", value=DEFAULT_HF_PROMPT_DATASET)
        sample_n = st.slider("Sample size", 1, 25, 5)
        model_checkpoint = st.selectbox("Model", [DEFAULT_MODEL, "arnir0/Tiny-LLM"])
        gen_len = st.slider("Max new tokens", 16, 256, DEFAULT_GEN_TOKENS)
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE)
        top_p = st.slider("Top-p", 0.1, 1.0, DEFAULT_TOP_P)
        run_name = st.text_input("Run name (saved filename)", value=f"hf_{hf_dataset.replace('/','_')}.json")

        st.markdown("---")
        st.checkbox("AUTOEVALUATE — Accept A and push to HF after RUN", value=False, key="autoeval")
        st.text_input("HF repo id to push approved dataset", value=f"eZWALT/rlhf_reward_data", key="push_repo_input")

        st.markdown("")
        st.write("Load prompts first (no generation). Then press the big RUN button to generate A/B for all prompts.")
        if st.button("Load prompts (no generation)", use_container_width=True):
            prompts = load_prompts_from_hf(hf_dataset, HF_PROMPT_COLUMN, sample_n)
            if not prompts:
                st.error("No prompts loaded from HF dataset.")
            else:
                rows = []
                for i, p in enumerate(prompts):
                    rows.append({
                        "id": str(i),
                        "prompt": p,
                        "response_A": "",
                        "response_B": "",
                        "status": "pending",
                        "chosen": "",
                        "rejected": "",
                    })
                st.session_state.df = pd.DataFrame(rows)
                st.session_state.run_name = run_name
                save_compact_progress(st.session_state.df, st.session_state.run_name)
                st.success(f"Loaded {len(rows)} prompts (no generation).")

    # Ensure run state present
    if "df" not in st.session_state:
        st.info("Use the sidebar to Load prompts (no generation). Then press RUN to generate A/B for all prompts.")
    else:
        df = st.session_state.df

        # Top bar
        a_count = int((df["status"] == "approved").sum())
        p_count = int((df["status"] == "pending").sum())
        r_count = int((df["status"] == "rejected").sum())
        col1, col2, col3, col4 = st.columns([1,1,1,2])
        col1.metric("Approved", a_count)
        col2.metric("Pending", p_count)
        col3.metric("Rejected", r_count)
        if col4.button("Save progress"):
            out = save_compact_progress(df, st.session_state.run_name)
            st.success(f"Saved compact progress to {out}")

        st.markdown("---")

        # Giant RUN button (generate A/B for every pending prompt)
        st.markdown("## RUN — generate A/B for all pending prompts")
        if st.button("▶ RUN", use_container_width=True):
            pending_idxs = df[df["status"] == "pending"].index.tolist()
            if not pending_idxs:
                st.info("No pending prompts to generate for.")
            else:
                model, tokenizer = load_model_and_tokenizer(model_checkpoint)
                prog = st.progress(0)
                n = len(pending_idxs)
                for i, idx in enumerate(pending_idxs):
                    ptext = df.at[idx, "prompt"]
                    a = generate_reply(model, tokenizer, PEDANTIC_PREFIX, ptext, gen_len, temperature, top_p)
                    b = generate_reply(model, tokenizer, BASELINE_PREFIX, ptext, gen_len, temperature, top_p)
                    df.at[idx, "response_A"] = a
                    df.at[idx, "response_B"] = b
                    prog.progress((i+1)/n)
                save_compact_progress(df, st.session_state.run_name)
                st.success(f"Generated A/B for {n} prompts.")

                # If AUTOEVALUATE checked: accept A for all pending (now generated) and push to HF
                if st.session_state.get("autoeval", False):
                    pending_after = df[df["status"] == "pending"].index.tolist()
                    if pending_after:
                        prog2 = st.progress(0)
                        for i, idx in enumerate(pending_after):
                            df.at[idx, "status"] = "approved"
                            df.at[idx, "chosen"] = df.at[idx, "response_A"]
                            df.at[idx, "rejected"] = df.at[idx, "response_B"]
                            prog2.progress((i+1)/len(pending_after))
                        save_compact_progress(df, st.session_state.run_name)
                        st.success(f"AUTOEVALUATE: auto-approved {len(pending_after)} items (A chosen).")

                        # push to HF
                        repo_id = st.session_state.get("push_repo_input") or f"eZWALT/rlhf_reward_data"
                        try:
                            with st.spinner("Pushing approved compact dataset to Hugging Face..."):
                                info = push_approved_to_hub_compact(df, repo_id)
                            st.success("Pushed approved dataset to Hugging Face hub.")
                            st.write(info)
                        except Exception as e:
                            st.error(f"Push failed: {e}")
                            st.info("If you haven't logged in, run `huggingface-cli login` and try again.")
                    else:
                        st.info("AUTOEVALUATE: nothing pending to approve.")

        st.markdown("---")

        # Tinder-like reviewer for manual inspection / correction
        pending_idxs = df[df["status"] == "pending"].index.tolist()
        if pending_idxs:
            if "idx" not in st.session_state:
                st.session_state.idx = 0
            st.session_state.idx = st.session_state.idx % len(pending_idxs)
            cur = pending_idxs[st.session_state.idx]
            rec = df.loc[cur]

            st.subheader(f"Prompt (id={rec['id']})")
            st.write(rec["prompt"])
            left, center, right = st.columns([3,1,3])

            with left:
                st.markdown("### A — Pedantic")
                st.write(rec.get("response_A","(empty)"))
                if st.button("Choose A (pedantic)", key=f"a_{cur}", use_container_width=True):
                    df.at[cur,"status"] = "approved"
                    df.at[cur,"chosen"] = df.at[cur,"response_A"]
                    df.at[cur,"rejected"] = df.at[cur,"response_B"]
                    save_compact_progress(df, st.session_state.run_name)
                    st.session_state.idx = (st.session_state.idx + 1) % max(1,len(pending_idxs))
                    st.rerun()

            with center:
                if st.button("Reject pair", key=f"rej_{cur}", use_container_width=True):
                    df.at[cur,"status"] = "rejected"
                    df.at[cur,"chosen"] = ""
                    df.at[cur,"rejected"] = ""
                    save_compact_progress(df, st.session_state.run_name)
                    st.session_state.idx = (st.session_state.idx + 1) % max(1,len(pending_idxs))
                    st.rerun()
                if st.button("Next", key=f"next_{cur}", use_container_width=True):
                    st.session_state.idx = (st.session_state.idx + 1) % max(1,len(pending_idxs))
                    st.rerun()

            with right:
                st.markdown("### B — Baseline")
                st.write(rec.get("response_B","(empty)"))
                if st.button("Choose B (baseline)", key=f"b_{cur}", use_container_width=True):
                    df.at[cur,"status"] = "approved"
                    df.at[cur,"chosen"] = df.at[cur,"response_B"]
                    df.at[cur,"rejected"] = df.at[cur,"response_A"]
                    save_compact_progress(df, st.session_state.run_name)
                    st.session_state.idx = (st.session_state.idx + 1) % max(1,len(pending_idxs))
                    st.rerun()
        else:
            st.info("No pending items to review manually.")

        st.markdown("---")

        # Export quick button
        st.subheader("Export")
        if st.button("Export approved → compact JSONL"):
            out = export_compact_jsonl(df, out_name=f"approved_compact_{st.session_state.run_name}.jsonl")
            st.success(f"Wrote approved compact JSONL to {out}")
            st.write(out)

        st.caption(f"Progress file: {progress_path_for_run(st.session_state.run_name)}")

if __name__ == "__main__":
    main()
