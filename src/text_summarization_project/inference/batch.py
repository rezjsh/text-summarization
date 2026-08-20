"""Batch summarization over a CSV or a plain-text folder."""
import logging
from pathlib import Path

import pandas as pd

from text_summarization_project.summarizer.summarizer import Summarizer

logger = logging.getLogger(__name__)


def summarize_csv(
    summarizer: Summarizer,
    input_csv: str,
    text_col: str,
    output_csv: str,
    batch_size: int = 8,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in {input_csv}. Columns: {list(df.columns)}")

    logger.info(f"Summarizing {len(df)} rows from {input_csv} ...")
    summaries = summarizer.summarize_batch(df[text_col].astype(str).tolist(), batch_size=batch_size)
    df["generated_summary"] = summaries

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {len(df)} summaries -> {output_csv}")
    return df


def summarize_text_folder(summarizer: Summarizer, input_dir: str, output_dir: str) -> None:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    texts = [f.read_text(encoding="utf-8") for f in txt_files]
    summaries = summarizer.summarize_batch(texts)

    for f, summary in zip(txt_files, summaries):
        (output_dir / f"{f.stem}_summary.txt").write_text(summary, encoding="utf-8")
    logger.info(f"Summarized {len(txt_files)} files from {input_dir} -> {output_dir}")
