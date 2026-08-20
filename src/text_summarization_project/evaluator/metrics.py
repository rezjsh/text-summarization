"""ROUGE (and optional BERTScore) computation, decoupled from any specific
trainer so both training-time eval and the standalone evaluate.py script use
the exact same metric code."""
import logging
import time
from typing import List

logger = logging.getLogger(__name__)

_rouge_metric = None


def _get_rouge_metric():
    global _rouge_metric
    if _rouge_metric is None:
        import evaluate
        _rouge_metric = evaluate.load("rouge")
    return _rouge_metric


def compute_rouge(predictions: List[str], references: List[str]) -> dict:
    """Returns rouge1 / rouge2 / rougeL / rougeLsum as percentages (0-100)."""
    rouge = _get_rouge_metric()
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}


def compute_bertscore(predictions: List[str], references: List[str], lang: str = "en") -> dict:
    import evaluate
    bertscore = evaluate.load("bertscore")
    result = bertscore.compute(predictions=predictions, references=references, lang=lang)
    return {
        "bertscore_precision": round(float(sum(result["precision"]) / len(result["precision"])), 4),
        "bertscore_recall": round(float(sum(result["recall"]) / len(result["recall"])), 4),
        "bertscore_f1": round(float(sum(result["f1"]) / len(result["f1"])), 4),
    }


def generation_length_stats(predictions: List[str]) -> dict:
    lengths = [len(p.split()) for p in predictions]
    if not lengths:
        return {}
    return {
        "mean_words": sum(lengths) / len(lengths),
        "min_words": min(lengths),
        "max_words": max(lengths),
    }


class LatencyTimer:
    """Measures average per-sample generation latency."""

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start

    def per_sample(self, n_samples: int) -> float:
        return self.elapsed / max(n_samples, 1)
