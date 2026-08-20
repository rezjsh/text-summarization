"""Factory that picks the right ingestion strategy based on environment /
availability, so callers never `import KaggleAPIStrategy` directly."""
import logging
import os
from pathlib import Path

from text_summarization_project.data_ingestion.interface import DataIngestionStrategy
from text_summarization_project.data_ingestion.strategies import (
    HFDatasetsStrategy,
    KaggleAPIStrategy,
    LocalCopyStrategy,
)

logger = logging.getLogger(__name__)


class DataIngestionFactory:
    @staticmethod
    def create(
        kaggle_dataset: str,
        raw_dir: Path,
        unzip_dir: Path,
        mode: str = "auto",
        local_source_dir: str = None,
    ) -> DataIngestionStrategy:
        """
        mode:
          "kaggle" -> force Kaggle API
          "local"  -> force LocalCopyStrategy from local_source_dir
          "hf"     -> force Hugging Face Hub mirror
          "auto"   -> Kaggle if token is present, else HF fallback
        """
        if mode == "local":
            if not local_source_dir:
                raise ValueError("local_source_dir is required when mode='local'")
            return LocalCopyStrategy(source_dir=Path(local_source_dir), unzip_dir=unzip_dir)

        if mode == "hf":
            return HFDatasetsStrategy(unzip_dir=unzip_dir)

        has_kaggle_creds = bool(os.environ.get("KAGGLE_API_TOKEN"))

        if mode == "kaggle" or (mode == "auto" and has_kaggle_creds):
            logger.info("Using KaggleAPIStrategy for data ingestion.")
            return KaggleAPIStrategy(kaggle_dataset=kaggle_dataset, raw_dir=raw_dir, unzip_dir=unzip_dir)

        logger.warning(
            "No Kaggle token found. Falling back to Hugging Face Hub mirror "
            "(abisee/cnn_dailymail). Set KAGGLE_API_TOKEN to use the "
            "original Kaggle dataset."
        )
        return HFDatasetsStrategy(unzip_dir=unzip_dir)