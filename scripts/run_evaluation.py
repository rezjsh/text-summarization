#!/usr/bin/env python
"""CLI wrapper: evaluates a model (base or fine-tuned) on the test split.
Examples:
  python scripts/evaluate.py --subset dev
  python scripts/evaluate.py --model_dir artifacts/checkpoints/best_model --subset medium
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_summarization_project.pipeline.stage_05_evaluation import main
from text_summarization_project.utils.common import setup_logging
from text_summarization_project.constants.constants import LOGGING_CONFIG_FILE_PATH

setup_logging(LOGGING_CONFIG_FILE_PATH)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--subset", default=None, choices=["dev", "medium", "full"])
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    main(model_key=args.model, subset_name=args.subset, model_dir=args.model_dir, max_samples=args.max_samples)
