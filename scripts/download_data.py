#!/usr/bin/env python
"""CLI wrapper: downloads and validates the raw dataset (Kaggle API by
default; use --mode to force local/hf). Run: `python scripts/download_data.py`
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_summarization_project.pipeline.stage_01_data_ingestion import main
from text_summarization_project.utils.common import setup_logging
from text_summarization_project.constants.constants import LOGGING_CONFIG_FILE_PATH

setup_logging(LOGGING_CONFIG_FILE_PATH)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the CNN/DailyMail dataset.")
    parser.add_argument("--mode", default="auto", choices=["auto", "kaggle", "local", "hf"])
    parser.add_argument("--local_source_dir", default=None, help="Required if --mode local")
    args = parser.parse_args()
    main(mode=args.mode, local_source_dir=args.local_source_dir)
