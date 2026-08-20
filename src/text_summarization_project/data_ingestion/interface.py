"""Strategy interface for data ingestion. Concrete strategies decide HOW the
raw CNN/DailyMail csv files land in data/raw/ (Kaggle API vs local copy vs
Hugging Face datasets mirror), the orchestrator decides WHEN to run them.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class DataIngestionStrategy(ABC):
    @abstractmethod
    def fetch(self) -> Path:
        """Ensure the raw dataset files exist on disk and return the
        directory that contains train.csv / validation.csv / test.csv."""
        raise NotImplementedError

    @abstractmethod
    def validate(self, data_dir: Path, expected_files) -> bool:
        """Sanity check that the expected files exist and are non-empty."""
        raise NotImplementedError
