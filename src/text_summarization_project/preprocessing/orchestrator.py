"""Runs the cleaning/filtering pipeline over train/validation/test csvs,
then writes processed parquet files (fast to reload) into
data/processed/<split>.parquet, plus a small stats json."""
import json
import logging
from pathlib import Path

import pandas as pd

from text_summarization_project.entity.config_entity import PreprocessingConfig
from text_summarization_project.preprocessing.strategies import (
    DropDuplicatesStrategy,
    DropNAStrategy,
    LengthFilterStrategy,
    TextCleaningStrategy,
)

logger = logging.getLogger(__name__)


class PreprocessingOrchestrator:
    def __init__(self, config: PreprocessingConfig, raw_dir: Path, text_col: str, summary_col: str):
        self.config = config
        self.raw_dir = Path(raw_dir)
        self.text_col = text_col
        self.summary_col = summary_col

        self.pipeline = []
        if config.drop_na:
            self.pipeline.append(DropNAStrategy())
        if config.drop_duplicates:
            self.pipeline.append(DropDuplicatesStrategy())
        self.pipeline.append(TextCleaningStrategy(lowercase=config.lowercase))
        self.pipeline.append(
            LengthFilterStrategy(
                min_article_chars=config.min_article_chars,
                max_article_chars=config.max_article_chars,
                min_summary_chars=config.min_summary_chars,
                max_summary_chars=config.max_summary_chars,
            )
        )

    def _process_split(self, split: str) -> pd.DataFrame:
        csv_path = self.raw_dir / f"{split}.csv"
        df = pd.read_csv(csv_path)
        logger.info(f"[{split}] loaded {len(df)} rows from {csv_path}")
        for step in self.pipeline:
            df = step.apply(df, self.text_col, self.summary_col)
        return df.reset_index(drop=True)

    def run(self) -> dict:
        logger.info("=== Stage: Preprocessing ===")
        stats = {}
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "validation", "test"]:
            csv_path = self.raw_dir / f"{split}.csv"
            if not csv_path.exists():
                logger.warning(f"Skipping missing split file: {csv_path}")
                continue
            df = self._process_split(split)
            out_path = self.config.processed_dir / f"{split}.parquet"
            df.to_parquet(out_path, index=False)
            stats[split] = {"num_rows": int(len(df))}
            logger.info(f"[{split}] wrote {len(df)} processed rows -> {out_path}")

        with open(self.config.processed_dir / "preprocessing_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        return stats
