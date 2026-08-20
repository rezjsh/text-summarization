#!/usr/bin/env python
"""CLI wrapper: fine-tunes the configured model on the requested subset.
Examples:
  python scripts/train.py --subset dev            # smoke test, ~minutes
  python scripts/train.py --model flan-t5-small --subset medium
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_summarization_project.pipeline.stage_04_training import main
from text_summarization_project.utils.common import setup_logging
from text_summarization_project.constants.constants import LOGGING_CONFIG_FILE_PATH

setup_logging(LOGGING_CONFIG_FILE_PATH)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--subset", default=None, choices=["dev", "medium", "full"])
    args = parser.parse_args()
    main(model_key=args.model, subset_name=args.subset)
