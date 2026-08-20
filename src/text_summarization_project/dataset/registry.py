"""Registry pattern for dataset subset presets (dev/medium/full). Loading a
processed parquet split and slicing it down to the requested subset size
lives here so both training and evaluation scripts subset identically."""
import logging
from pathlib import Path

import pandas as pd

from text_summarization_project.entity.config_entity import DatasetSubsetConfig

logger = logging.getLogger(__name__)

_SUBSET_SIZE_KEYS = {"train": "train_size", "validation": "val_size", "test": "test_size"}


class DatasetRegistry:
    """Central place to fetch a (possibly subsampled) split as a DataFrame."""

    def __init__(self, subset_config: DatasetSubsetConfig):
        self.subset_config = subset_config

    def load_split(self, split: str, seed: int = 42) -> pd.DataFrame:
        if split not in _SUBSET_SIZE_KEYS:
            raise ValueError(f"split must be one of {list(_SUBSET_SIZE_KEYS)}, got '{split}'")

        path = Path(self.subset_config.processed_dir) / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Processed split not found at {path}. Run the preprocessing stage first."
            )
        df = pd.read_parquet(path)

        size_attr = _SUBSET_SIZE_KEYS[split]
        limit = getattr(self.subset_config, size_attr)
        if limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
            logger.info(
                f"[{self.subset_config.name}] subsampled '{split}' split to {limit} rows "
                f"(from {path.name})"
            )
        else:
            logger.info(f"[{self.subset_config.name}] using full '{split}' split: {len(df)} rows")
        return df

    def load_all(self, seed: int = 42) -> dict:
        return {split: self.load_split(split, seed=seed) for split in _SUBSET_SIZE_KEYS}
