"""Interactive Streamlit app for single-article and batch summarization.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_summarization_project.config.configuration import ConfigurationManager
from text_summarization_project.summarizer.summarizer import Summarizer

st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="wide")


@st.cache_resource(show_spinner="Loading model ...")
def load_summarizer(model_key: str, model_dir: str):
    config_manager = ConfigurationManager()
    model_config = config_manager.get_model_config(model_key or None)
    generation_config = config_manager.get_generation_config()
    return Summarizer(model_config, generation_config, model_dir=model_dir or None)


def main():
    st.title("📰 CNN/DailyMail News Summarizer")
    st.caption("Abstractive summarization powered by a fine-tuned T5 / FLAN-T5 / BART model.")

    with st.sidebar:
        st.header("Model")
        model_key = st.text_input("Model key (blank = config default)", value="")
        model_dir = st.text_input(
            "Fine-tuned model dir (blank = base checkpoint)",
            value="artifacts/checkpoints/best_model",
        )
        st.header("Generation settings")
        num_beams = st.slider("Beam size", 1, 8, 4)
        max_new_tokens = st.slider("Max new tokens", 32, 256, 128, step=16)
        length_penalty = st.slider("Length penalty", 0.5, 3.0, 2.0, step=0.1)

    tab_single, tab_batch = st.tabs(["Single article", "Batch (CSV)"])

    with tab_single:
        text = st.text_area("Paste a news article", height=280)
        reference = st.text_area("Optional reference summary (for ROUGE)", height=100)
        if st.button("Summarize", type="primary"):
            if not text.strip():
                st.warning("Paste an article first.")
            else:
                summarizer = load_summarizer(model_key, model_dir)
                with st.spinner("Generating summary ..."):
                    summary = summarizer.summarize(
                        text, num_beams=num_beams, max_new_tokens=max_new_tokens,
                        length_penalty=length_penalty,
                    )
                st.subheader("Generated summary")
                st.write(summary)
                if reference.strip():
                    from text_summarization_project.evaluator.metrics import compute_rouge
                    st.subheader("ROUGE vs. reference")
                    st.json(compute_rouge([summary], [reference]))

    with tab_batch:
        uploaded = st.file_uploader("Upload a CSV with an article column", type=["csv"])
        text_col = st.text_input("Article column name", value="article")
        if uploaded is not None and st.button("Summarize batch"):
            df = pd.read_csv(uploaded)
            if text_col not in df.columns:
                st.error(f"Column '{text_col}' not found. Columns: {list(df.columns)}")
            else:
                summarizer = load_summarizer(model_key, model_dir)
                with st.spinner(f"Summarizing {len(df)} rows ..."):
                    df["generated_summary"] = summarizer.summarize_batch(
                        df[text_col].astype(str).tolist(),
                        num_beams=num_beams, max_new_tokens=max_new_tokens,
                        length_penalty=length_penalty,
                    )
                st.dataframe(df[[text_col, "generated_summary"]])
                st.download_button(
                    "Download results as CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="summaries.csv",
                )


if __name__ == "__main__":
    main()
