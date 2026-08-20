"""Strategy interface for a preprocessing step operating on a pandas
DataFrame with `article`/`highlights`-style columns."""
from abc import ABC, abstractmethod

import pandas as pd


class PreprocessingStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def apply(self, df: pd.DataFrame, text_col: str, summary_col: str) -> pd.DataFrame:
        raise NotImplementedError
