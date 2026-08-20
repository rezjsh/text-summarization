"""Runs every EDA strategy over train/val/test, saves a combined JSON report
plus a human-readable markdown summary and PNG plots into artifacts/eda/."""
import json
import logging
from pathlib import Path

import pandas as pd

from text_summarization_project.eda.strategies import (
    DuplicateAnalysisStrategy,
    LengthDistributionStrategy,
    MissingValuesStrategy,
    OverviewStrategy,
    SplitInspectionStrategy,
    TokenCountStrategy,
    TopWordsStrategy,
)
from text_summarization_project.eda.visualizers import (
    plot_article_vs_summary_scatter,
    plot_length_histograms,
    plot_split_sizes,
    plot_top_words_bar,
)
from text_summarization_project.entity.config_entity import EDAConfig

logger = logging.getLogger(__name__)


class EDAOrchestrator:
    def __init__(self, config: EDAConfig, text_col: str, summary_col: str):
        self.config = config
        self.text_col = text_col
        self.summary_col = summary_col
        self.strategies = [
            OverviewStrategy(),
            MissingValuesStrategy(),
            DuplicateAnalysisStrategy(),
            LengthDistributionStrategy(),
            TokenCountStrategy(sample_size=config.sample_size_for_token_plots),
            TopWordsStrategy(top_n=config.top_n_words, ngram_range=tuple(config.ngram_range)),
        ]
        self.split_strategy = SplitInspectionStrategy()

    def _load_splits(self) -> pd.DataFrame:
        frames = []
        for split in ["train", "validation", "test"]:
            fpath = self.config.raw_dir / f"{split}.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                df["split"] = split
                frames.append(df)
            else:
                logger.warning(f"Split file not found, skipping: {fpath}")
        if not frames:
            raise FileNotFoundError(f"No train/validation/test csv files found under {self.config.raw_dir}")
        return pd.concat(frames, ignore_index=True)

    def run(self) -> dict:
        logger.info("=== Stage: EDA ===")
        df = self._load_splits()

        report = {"num_total_rows": int(len(df))}
        for strategy in self.strategies:
            logger.info(f"Running EDA strategy: {strategy.name}")
            report[strategy.name] = strategy.run(df, self.text_col, self.summary_col)

        report["split_inspection"] = self.split_strategy.run(df, self.text_col, self.summary_col)

        # sample pairs for qualitative inspection
        sample_df = df.sample(min(5, len(df)), random_state=42)
        report["sample_pairs"] = [
            {"article": str(row[self.text_col])[:500], "summary": str(row[self.summary_col])}
            for _, row in sample_df.iterrows()
        ]

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_dir / "eda_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved eda_report.json to {self.config.output_dir}")

        # plots
        plot_length_histograms(df, self.text_col, self.summary_col, self.config.output_dir)
        plot_article_vs_summary_scatter(df, self.text_col, self.summary_col, self.config.output_dir)
        plot_top_words_bar(report["top_words"]["top_words"], self.config.output_dir)
        plot_split_sizes(report["split_inspection"], self.config.output_dir)

        self._write_markdown_summary(report)
        return report

    def _write_markdown_summary(self, report: dict) -> None:
        lines = ["# EDA Summary Report\n"]
        lines.append(f"- Total rows: **{report['num_total_rows']}**")
        lines.append(f"- Duplicate article/summary pairs: **{report['duplicates']['duplicate_pairs']}**")
        lc = report["length_distribution"]
        lines.append(
            f"- Mean article length: **{lc['article_words']['mean']:.1f} words**, "
            f"mean summary length: **{lc['summary_words']['mean']:.1f} words**"
        )
        lines.append(f"- Mean compression ratio (summary/article words): **{lc['compression_ratio_mean']:.3f}**")
        tc = report["token_counts"]
        lines.append(
            f"- Approx. tokens (p95): article={tc['article_tokens_p95']:.0f}, "
            f"summary={tc['summary_tokens_p95']:.0f}"
        )
        lines.append("\n## Split breakdown\n")
        for split, stats in report.get("split_inspection", {}).items():
            if split == "note":
                continue
            lines.append(f"- **{split}**: {stats['num_rows']} rows")
        lines.append("\n## Top words in summaries\n")
        for word, count in report["top_words"]["top_words"][:15]:
            lines.append(f"- {word}: {count}")
        lines.append("\nSee `eda_report.json` for the full machine-readable report and PNGs in this folder for plots.")

        with open(self.config.output_dir / "eda_summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved eda_summary.md to {self.config.output_dir}")
