"""Strategy interface for a single EDA analysis step. Each concrete strategy
computes one piece of the report (overview, lengths, duplicates, ngrams, ...)
and returns a JSON-serializable dict so the orchestrator can assemble one
combined report without caring about the internals of each analysis."""
from abc import ABC, abstractmethod

import pandas as pd


class EDAStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, df: pd.DataFrame, text_col: str, summary_col: str) -> dict:
        raise NotImplementedError
