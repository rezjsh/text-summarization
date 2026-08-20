"""Concrete EDA analyses. Each one is intentionally small and single-purpose
(Strategy pattern) so new analyses can be added without touching the others."""
import logging
from collections import Counter

import pandas as pd

from text_summarization_project.eda.interface import EDAStrategy

logger = logging.getLogger(__name__)


class OverviewStrategy(EDAStrategy):
    name = "overview"

    def run(self, df: pd.DataFrame, text_col: str, summary_col: str) -> dict:
        return {
            "num_rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }


class MissingValuesStrategy(EDAStrategy):
    name = "missing_values"

    def run(self, df: pd.DataFrame, text_col: str, summary_col: str) -> dict:
        missing = df.isna().sum()
        return {col: int(cnt) for col, cnt in missing.items() if cnt > 0} or {"missing": 0}


class DuplicateAnalysisStrategy(EDAStrategy):
    name = "duplicates"

    def run(self, df: pd.DataFrame, text_col: str, summary_col: str) -> dict:
        dup_articles = int(df.duplicated(subset=[text_col]).sum())
        dup_summaries = int(df.duplicated(subset=[summary_col]).sum())
        dup_rows = int(df.duplicated(subset=[text_col, summary_col]).sum())
        return {
            "duplicate_articles": dup_articles,
            "duplicate_summaries": dup_summaries,
            "duplicate_pairs": dup_rows,
        }


class LengthDistributionStrategy(EDAStrategy):
    """Character and word length distributions for article vs summary."""
    name = "length_distribution"

    from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class LengthDistributionStrategy(EDAStrategy):
    """Calculate character and approximate word-length distributions."""

    name = "length_distribution"

    @staticmethod
    def _word_count(series: pd.Series) -> pd.Series:
        """Count whitespace-separated words, treating missing/empty text as zero."""
        normalized = (
            series
            .fillna("")
            .astype(str)
            .str.strip()
        )

        return normalized.str.count(r"\S+")

    @staticmethod
    def _stats(series: pd.Series) -> dict[str, int | float]:
        """Return descriptive statistics for a numeric series."""
        if series.empty:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "min": int(series.min()),
            "max": int(series.max()),
            "p90": float(series.quantile(0.90)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
        }

    def run(
        self,
        df: pd.DataFrame,
        text_col: str,
        summary_col: str,
    ) -> dict[str, object]:
        """Return article and summary length statistics."""
        required_columns = {text_col, summary_col}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise KeyError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if df.empty:
            return {
                "article_chars": self._stats(pd.Series(dtype=int)),
                "summary_chars": self._stats(pd.Series(dtype=int)),
                "article_words": self._stats(pd.Series(dtype=int)),
                "summary_words": self._stats(pd.Series(dtype=int)),
                "compression_ratio_mean": 0.0,
                "num_rows": 0,
            }

        articles = df[text_col].fillna("").astype(str)
        summaries = df[summary_col].fillna("").astype(str)

        article_chars = articles.str.len()
        summary_chars = summaries.str.len()

        article_words = self._word_count(articles)
        summary_words = self._word_count(summaries)

        valid_articles = article_words > 0
        compression_ratio = (
            summary_words.loc[valid_articles]
            / article_words.loc[valid_articles]
        )

        logger.info(
            "[%s] Calculated length statistics for %d rows.",
            self.name,
            len(df),
        )

        return {
            "num_rows": int(len(df)),
            "article_chars": self._stats(article_chars),
            "summary_chars": self._stats(summary_chars),
            "article_words": self._stats(article_words),
            "summary_words": self._stats(summary_words),
            "compression_ratio_mean": float(compression_ratio.mean())
            if not compression_ratio.empty
            else 0.0,
            "empty_articles": int((article_words == 0).sum()),
            "empty_summaries": int((summary_words == 0).sum()),
        }

class TokenCountStrategy(EDAStrategy):
    """Approximate token counts using a Hugging Face tokenizer (falls back to
    whitespace split if `transformers` isn't installed in the environment
    running the EDA notebook)."""
    name = "token_counts"

    def __init__(self, tokenizer_name: str = "t5-small", sample_size: int = 5000):
        self.tokenizer_name = tokenizer_name
        self.sample_size = sample_size

    def run(self, df: pd.DataFrame, text_col: str, summary_col: str) -> dict:
        sample = df.sample(min(self.sample_size, len(df)), random_state=42)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
            art_tokens = sample[text_col].astype(str).apply(lambda x: len(tok.encode(x)))
            sum_tokens = sample[summary_col].astype(str).apply(lambda x: len(tok.encode(x)))
        except Exception as e:
            logger.warning(f"Falling back to whitespace tokenization for EDA token counts: {e}")
            art_tokens = sample[text_col].astype(str).str.split().str.len()
            sum_tokens = sample[summary_col].astype(str).str.split().str.len()

        return {
            "sample_size": int(len(sample)),
            "article_tokens_mean": float(art_tokens.mean()),
            "article_tokens_p95": float(art_tokens.quantile(0.95)),
            "summary_tokens_mean": float(sum_tokens.mean()),
            "summary_tokens_p95": float(sum_tokens.quantile(0.95)),
        }


class TopWordsStrategy(EDAStrategy):
    name = "top_words"
    _STOPWORDS = set(
        "the a an and of to in for on with is are was were be by at as it this that "
        "from or has have had but not will would can could said its his her they he "
        "she you your we our i".split()
    )

    def __init__(self, top_n: int = 30, ngram_range=(2, 3)):
        self.top_n = top_n
        self.ngram_range = ngram_range

    def _tokenize(self, text: str):
        return [w for w in text.lower().split() if w.isalpha() and w not in self._STOPWORDS]

    def run(self, df: pd.DataFrame, text_col: str, summary_col: str) -> dict:
        word_counter = Counter()
        ngram_counter = Counter()
        
        sampled_text = df[summary_col].astype(str).sample(min(2000, len(df)), random_state=42)
        n_min, n_max = self.ngram_range

        for text in sampled_text:
            words = [w for w in text.lower().split() if w.isalpha() and w not in self._STOPWORDS]
            word_counter.update(words)
            for n in range(n_min, n_max + 1):
                ngram_counter.update(
                    " ".join(words[i:i + n]) for i in range(len(words) - n + 1)
                )
        return {
            "top_words": word_counter.most_common(self.top_n),
            "top_ngrams": ngram_counter.most_common(self.top_n),
        }

class SplitInspectionStrategy(EDAStrategy):
    """Compares row counts and length stats across train/val/test to catch
    distribution shift or accidental leakage."""
    name = "split_inspection"

    def run(self, df: pd.DataFrame, text_col: str, summary_col: str) -> dict:
        # Expects df to have a 'split' column when called from orchestrator's
        # combined-split entry point; otherwise returns single-split stats.
        if "split" not in df.columns:
            return {"note": "single dataframe passed, no split column present"}
        result = {}
        for split_name, group in df.groupby("split"):
            result[split_name] = {
                "num_rows": int(len(group)),
                "avg_article_words": float(group[text_col].astype(str).str.split().str.len().mean()),
                "avg_summary_words": float(group[summary_col].astype(str).str.split().str.len().mean()),
            }
        return result
