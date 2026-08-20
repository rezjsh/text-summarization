"""Single-article summarization entry point."""
import logging

from text_summarization_project.summarizer.summarizer import Summarizer

logger = logging.getLogger(__name__)


def summarize_single(summarizer: Summarizer, text: str, reference: str = None) -> dict:
    summary = summarizer.summarize(text)
    result = {"article": text, "generated_summary": summary}
    if reference:
        from text_summarization_project.evaluator.metrics import compute_rouge
        result["reference_summary"] = reference
        result["rouge"] = compute_rouge([summary], [reference])
    return result
