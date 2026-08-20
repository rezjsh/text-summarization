"""Runs the ingestion strategy end-to-end and reports a pass/fail result."""
import logging
from pathlib import Path

from text_summarization_project.data_ingestion.factory import DataIngestionFactory
from text_summarization_project.entity.config_entity import DataIngestionConfig

logger = logging.getLogger(__name__)


class DataIngestionOrchestrator:
    def __init__(self, config: DataIngestionConfig, mode: str = "auto", local_source_dir: str = None):
        self.config = config
        self.strategy = DataIngestionFactory.create(
            kaggle_dataset=config.kaggle_dataset,
            raw_dir=config.raw_dir,
            unzip_dir=config.unzip_dir,
            mode=mode,
            local_source_dir=local_source_dir,
        )

    def run(self) -> Path:
        logger.info("=== Stage: Data Ingestion ===")
        data_dir = self.strategy.fetch()
        ok = self.strategy.validate(data_dir, self.config.expected_files)
        if not ok:
            raise FileNotFoundError(
                f"Data ingestion validation failed. Expected files "
                f"{self.config.expected_files} not all found/non-empty under {data_dir}."
            )
        logger.info(f"Data ingestion complete. Files available at: {data_dir}")
        return data_dir
