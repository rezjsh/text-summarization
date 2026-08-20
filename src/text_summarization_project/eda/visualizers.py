"""Plotting helpers. Kept separate from strategies.py so headless plot
generation (matplotlib) can be swapped/extended (e.g. plotly html export)
without touching analysis logic."""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe for servers/Colab
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def plot_length_histograms(df: pd.DataFrame, text_col: str, summary_col: str, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df[text_col].astype(str).str.split().str.len().plot(
        kind="hist", bins=50, ax=axes[0], color="#4C72B0"
    )
    axes[0].set_title("Article length (words)")
    axes[0].set_xlabel("words")

    df[summary_col].astype(str).str.split().str.len().plot(
        kind="hist", bins=50, ax=axes[1], color="#DD8452"
    )
    axes[1].set_title("Summary length (words)")
    axes[1].set_xlabel("words")

    fig.tight_layout()
    fig.savefig(out_dir / "length_histograms.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved length_histograms.png to {out_dir}")


def plot_article_vs_summary_scatter(df: pd.DataFrame, text_col: str, summary_col: str, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    art_words = df[text_col].astype(str).str.split().str.len()
    sum_words = df[summary_col].astype(str).str.split().str.len()

    fig, ax = plt.subplots(figsize=(7, 6))
    sample_idx = df.sample(min(5000, len(df)), random_state=42).index
    ax.scatter(art_words.loc[sample_idx], sum_words.loc[sample_idx], alpha=0.2, s=8)
    ax.set_xlabel("Article length (words)")
    ax.set_ylabel("Summary length (words)")
    ax.set_title("Article vs. Summary length")
    fig.tight_layout()
    fig.savefig(out_dir / "article_vs_summary_scatter.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved article_vs_summary_scatter.png to {out_dir}")


def plot_top_words_bar(top_words, out_dir: Path, title: str = "Top words in summaries") -> None:
    out_dir = Path(out_dir)
    words, counts = zip(*top_words) if top_words else ([], [])
    fig, ax = plt.subplots(figsize=(8, max(4, len(words) * 0.25)))
    ax.barh(words[::-1], counts[::-1], color="#55A868")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "top_words_bar.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved top_words_bar.png to {out_dir}")


def plot_split_sizes(split_stats: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    if not split_stats or "note" in split_stats:
        return
    splits = list(split_stats.keys())
    sizes = [split_stats[s]["num_rows"] for s in splits]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(splits, sizes, color="#8172B2")
    ax.set_title("Rows per split")
    fig.tight_layout()
    fig.savefig(out_dir / "split_sizes.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved split_sizes.png to {out_dir}")
