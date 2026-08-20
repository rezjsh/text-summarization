#!/usr/bin/env python
"""CLI wrapper for inference.
Single article:
  python scripts/infer.py --text "..." --model_dir artifacts/checkpoints/best_model
Batch over CSV:
  python scripts/infer.py --input_csv data/raw/cnn_dailymail/test.csv --text_col article \
      --output_csv artifacts/evaluation/batch_summaries.csv --model_dir artifacts/checkpoints/best_model
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_summarization_project.config.configuration import ConfigurationManager
from text_summarization_project.inference.batch import summarize_csv
from text_summarization_project.inference.single import summarize_single
from text_summarization_project.summarizer.summarizer import Summarizer
from text_summarization_project.utils.common import setup_logging
from text_summarization_project.constants.constants import LOGGING_CONFIG_FILE_PATH

setup_logging(LOGGING_CONFIG_FILE_PATH)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model key, defaults to config.yaml active_model")
    parser.add_argument("--model_dir", default=None, help="Path to fine-tuned model, e.g. artifacts/checkpoints/best_model")
    parser.add_argument("--text", default=None, help="Single article text to summarize")
    parser.add_argument("--reference", default=None, help="Optional reference summary for ROUGE")
    parser.add_argument("--input_csv", default=None, help="CSV path for batch summarization")
    parser.add_argument("--text_col", default="article")
    parser.add_argument("--output_csv", default="artifacts/evaluation/batch_summaries.csv")
    args = parser.parse_args()

    config_manager = ConfigurationManager()
    model_config = config_manager.get_model_config(args.model)
    generation_config = config_manager.get_generation_config()
    summarizer = Summarizer(model_config, generation_config, model_dir=args.model_dir)

    if args.text:
        result = summarize_single(summarizer, args.text, reference=args.reference)
        print(result["generated_summary"])
        if "rouge" in result:
            logger.info(f"ROUGE vs reference: {result['rouge']}")
    elif args.input_csv:
        summarize_csv(summarizer, args.input_csv, args.text_col, args.output_csv)
    else:
        parser.error("Provide either --text or --input_csv")


if __name__ == "__main__":
    main()
