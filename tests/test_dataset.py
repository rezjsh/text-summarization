import pandas as pd
import pytest

from text_summarization_project.dataset.registry import DatasetRegistry
from text_summarization_project.entity.config_entity import DatasetSubsetConfig


def test_registry_subsamples_when_limit_smaller(tmp_path):
    df = pd.DataFrame({
        "article": [f"article {i} " * 20 for i in range(100)],
        "highlights": [f"summary {i}" for i in range(100)],
    })
    df.to_parquet(tmp_path / "train.parquet")
    (pd.DataFrame(columns=df.columns)).to_parquet(tmp_path / "validation.parquet")
    (pd.DataFrame(columns=df.columns)).to_parquet(tmp_path / "test.parquet")

    subset_config = DatasetSubsetConfig(
        name="dev", train_size=10, val_size=0, test_size=0,
        text_column="article", summary_column="highlights", processed_dir=tmp_path,
    )
    registry = DatasetRegistry(subset_config)
    train_df = registry.load_split("train")
    assert len(train_df) == 10


def test_registry_raises_on_missing_file(tmp_path):
    subset_config = DatasetSubsetConfig(
        name="dev", train_size=10, val_size=10, test_size=10,
        text_column="article", summary_column="highlights", processed_dir=tmp_path,
    )
    registry = DatasetRegistry(subset_config)
    with pytest.raises(FileNotFoundError):
        registry.load_split("train")
