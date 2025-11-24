import pandas as pd
from protgen.backend.models import run_training,run_generation
import streamlit as st
import sys
import io
import gc
import torch

class DevNull(io.StringIO):
    def write(self, txt):
        pass

def cleanup_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        pass

st.title("Guided Protein Generation Model Online Server(AutoML)")

# =======================================
# Step 1 — Upload dataset
# =======================================
st.header("📊 Step 1 — Upload your dataset file")

uploaded_file = st.file_uploader(
    "Please Load the csv file: each sequence with property labels",
    type=["csv"],
)

with st.expander("Example data format (click to expand)"):
    example_df = pd.DataFrame({
        "sequence": ["MKAILVVLLYTAVV", "MKAILVVLLYTAVA", "MKALLAVLLYTAVA"],
        "label": [1.25, 0.87, 2.13]
    })
    st.markdown("Your CSV should look like this:")
    st.dataframe(example_df, width='stretch')
    st.markdown("**Required columns:**")
    st.markdown("- `sequence` — protein sequence (AAs only)\n- `label` — numeric property label for the sequence")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded sucessfully.")
st.markdown("<br><br>",unsafe_allow_html=True)

# # =======================================
# # Step 2 — setting parameters
# # =======================================
st.header("⚙️ Step 2 — Training model")
col1, col2, col3 = st.columns(3)
with col1:
    max_length = st.number_input(" Sequence length ", min_value=10, max_value=1000, value=130, step=1)
with col2:
    epochs = st.number_input("Train epoches", min_value=1, max_value=30, value=1, step=1)
with col3:
    batch_size = st.number_input("Batch size", min_value=1, max_value=16, value=5, step=1)

training_cfg = {
    "epochs": int(epochs),
    "batch_size": int(batch_size),
    "max_length": int(max_length),
}

if "trained" not in st.session_state:
    st.session_state["trained"] = False

if "train_area" not in st.session_state:
    st.session_state["train_area"] = None

if "train_logs" not in st.session_state:
    st.session_state["train_logs"] = {"bar": "", "epoch": "", "step": "", "speed": ""}

if "is_training" not in st.session_state:
    st.session_state["is_training"] = False

if st.button("Train model"):
    if not uploaded_file:
        st.error("❗ Please upload a dataset first.")
        st.stop()

    cleanup_memory()
    for key in ["model", "tokenizer", "max_length"]:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["trained"] = False
    st.session_state["train_logs"] = {"bar": "", "epoch": "", "step": "", "speed": ""}
    st.session_state["train_area"]  = st.container()

    with st.session_state["train_area"] :
        bar_area = st.empty()
        epoch_area = st.empty()
        step_area = st.empty()
        speed_area = st.empty()
        eta_area = st.empty()

    def update_progress(epoch, step, total_steps, speed):
        percent = int(step / total_steps * 100)
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "─" * (bar_len - filled)

        st.session_state["train_logs"] = {
            "bar": bar,
            "epoch": f"Epoch {epoch}",
            "step": f"Step {step}/{total_steps}",
            "speed": f"Speed: {speed:.2f} it/s"
        }

        with st.session_state["train_area"] :
            bar_area.markdown(bar)
            epoch_area.text(st.session_state["train_logs"]["epoch"])
            step_area.text(st.session_state["train_logs"]["step"])
            speed_area.text(st.session_state["train_logs"]["speed"])


    # --- Silence backend stdout ---
    original_stdout = sys.stdout
    sys.stdout = DevNull()

    with st.spinner("Training model..."):
        model, tokenizer, max_length = run_training(df, training_cfg, update_progress)
    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer
    st.session_state["max_length"] = max_length
    st.session_state["trained"] = True
    sys.stdout = original_stdout  
    st.success("Training finished!")

elif st.session_state["trained"] and not st.session_state["is_training"]:
    if st.session_state["train_logs"]["bar"]:
        st.markdown(st.session_state["train_logs"]["bar"])
        st.text(st.session_state["train_logs"]["epoch"])
        st.text(st.session_state["train_logs"]["step"])
        st.text(st.session_state["train_logs"]["speed"])
    st.success("Training finished!") 

# # =======================================
# # Step 3 — Generating sequence
# # =======================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.header("🧬 Step 3 — Generating Sequence")
gen_cols = st.columns(3)
with gen_cols[0]:
    num_beams = st.number_input("numbers", min_value=1, max_value=100, value=20, step=1)
with gen_cols[1]:
    top_k = st.number_input("top_k", min_value=15, max_value=100, value=20, step=1)
with gen_cols[2]:
    temperature = st.slider("temperature", min_value=0.6, max_value=1.5, value=0.7, step=0.05)

gen_cfg = {
    "num_beams": int(num_beams),
    "top_k": int(top_k),
    "temperature": float(temperature),
}
# ---------------------------------------
# Start button to call backend generation
# ---------------------------------------

if st.button("Start generation"): 
    if "model" not in st.session_state:
        st.error("❗ Please train the model first.")
        st.stop()

    current_seqs = []
    progress_text = st.empty()
    sequences_placeholder = st.empty()

    def update_ui(new_seq):
        if "Starting" in new_seq or new_seq.strip() == "":
            progress_text.info(" Starting generation...")
            return
        current_seqs.append(new_seq)
        progress_text.markdown(f"**Progress: {len(current_seqs)}/{num_beams} sequences generated**")
        
        with sequences_placeholder.container():
            for i, s in enumerate(current_seqs):
                st.markdown(f">seq{i+1}; temperature is {temperature}")
                st.code(s)

    with st.spinner("Generating..."):
        sequences, message = run_generation(
            df=df,
            lora_model=st.session_state["model"],
            tokenizer=st.session_state["tokenizer"],
            gen_cfg=gen_cfg,
            max_length=st.session_state["max_length"],
            update_ui=update_ui,     
        )
        progress_text.empty()
        st.success(message)

        # Download as FASTA
        fasta_text = "\n".join([
            f">seq_{i+1}\n{seq}" for i, seq in enumerate(sequences)])
        st.download_button(
            label="📥 Download FASTA",
            data=fasta_text,
            file_name="generated_sequences.fasta",
            mime="text/plain"
            )