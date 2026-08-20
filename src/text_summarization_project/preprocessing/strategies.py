"""Concrete cleaning / filtering strategies, applied in sequence by the
orchestrator (Strategy + simple pipeline composition)."""
import logging
import re

import pandas as pd

from text_summarization_project.preprocessing.interface import PreprocessingStrategy

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_CNN_TAG_RE = re.compile(r"^\(CNN\)\s*(--)?\s*", re.IGNORECASE)
_MULTI_DASH_RE = re.compile(r"-{2,}")


class DropNAStrategy(PreprocessingStrategy):
    name = "drop_na"

    def apply(self, df: pd.DataFrame, text_col: str, summary_col: str) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=[text_col, summary_col])
        logger.info(f"[drop_na] {before} -> {len(df)} rows")
        return df


class DropDuplicatesStrategy(PreprocessingStrategy):
    name = "drop_duplicates"

    def apply(self, df: pd.DataFrame, text_col: str, summary_col: str) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=[text_col, summary_col])
        logger.info(f"[drop_duplicates] {before} -> {len(df)} rows")
        return df


class TextCleaningStrategy(PreprocessingStrategy):
    """Whitespace normalization, CNN/(Reuters)-style boilerplate tag removal,
    optional lowercasing. Deliberately conservative: summarization models
    benefit from natural casing/punctuation, so we don't strip punctuation."""
    name = "text_cleaning"

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

    def _clean(self, text: str) -> str:
        text = str(text)
        text = _CNN_TAG_RE.sub("", text)
        text = _MULTI_DASH_RE.sub("-", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if self.lowercase:
            text = text.lower()
        return text

    def apply(self, df: pd.DataFrame, text_col: str, summary_col: str) -> pd.DataFrame:
        df[text_col] = df[text_col].astype(str).apply(self._clean)
        df[summary_col] = df[summary_col].astype(str).apply(self._clean)
        logger.info("[text_cleaning] normalized whitespace and boilerplate tags")
        return df


class LengthFilterStrategy(PreprocessingStrategy):
    """Drops rows whose article/summary character length falls outside the
    configured bounds -- removes empty stubs and pathological outliers that
    would otherwise dominate batch padding."""
    name = "length_filter"

    def __init__(self, min_article_chars, max_article_chars, min_summary_chars, max_summary_chars):
        self.min_article_chars = min_article_chars
        self.max_article_chars = max_article_chars
        self.min_summary_chars = min_summary_chars
        self.max_summary_chars = max_summary_chars

    def apply(self, df: pd.DataFrame, text_col: str, summary_col: str) -> pd.DataFrame:
        before = len(df)
        art_len = df[text_col].astype(str).str.len()
        sum_len = df[summary_col].astype(str).str.len()
        mask = (
            (art_len >= self.min_article_chars) & (art_len <= self.max_article_chars) &
            (sum_len >= self.min_summary_chars) & (sum_len <= self.max_summary_chars)
        )
        df = df[mask]
        logger.info(f"[length_filter] {before} -> {len(df)} rows")
        return df
