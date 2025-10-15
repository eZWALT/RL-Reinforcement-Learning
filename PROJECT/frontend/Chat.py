# chat_pedantic_simple_final.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- CONFIG ----------
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = "cpu"
MAX_TOKENS = 128

# ---------- MODEL UTIL ----------
@st.cache_resource
def load_model_and_tokenizer(checkpoint: str):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model, tok

def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int = 128, temperature: float = 0.2, top_p: float = 0.9):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
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
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()

# ---------- UI ----------
st.set_page_config(page_title="Pedantic Chat", layout="wide")
st.title("🧐 Pedantically Reinforced Large Language Model (PRLLM) 🧐")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Select Model",
        [
            DEFAULT_MODEL,
            "eZWALT/PedanticSmolLM2-135M",  
        ],
    )
    max_tokens = st.slider("Max new tokens", 16, 256, 96)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95)

# Load model (cached automatically)
try:
    model, tokenizer = load_model_and_tokenizer(model_name)
except Exception as e:
    st.error(f"Could not load model {model_name}: {e}")
    st.stop()

# ---------- CHAT ----------
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(f"**({msg.get('model', 'Model')})**\n\n{msg['content']}")

# Handle input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.rerun()

# Generate model response
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    last_user = st.session_state.history[-1]["content"]

    with st.spinner("Generating response..."):
        try:
            reply = generate_reply(model, tokenizer, last_user, max_new_tokens=max_tokens, temperature=temp, top_p=top_p)
        except Exception as e:
            reply = f"[generation error: {e}]"

    st.chat_message("assistant").markdown(f"**({model_name})**\n\n{reply}")
    st.session_state.history.append({"role": "assistant", "content": reply, "model": model_name})

# Clear chat
st.sidebar.markdown("---")
if st.sidebar.button("Clear chat"):
    st.session_state.history = []
    st.rerun()
