#!/usr/bin/env python
"""CLI wrapper: runs full EDA and writes report+plots to artifacts/eda/."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_summarization_project.pipeline.stage_02_eda import main
from text_summarization_project.utils.common import setup_logging
from text_summarization_project.constants.constants import LOGGING_CONFIG_FILE_PATH

setup_logging(LOGGING_CONFIG_FILE_PATH)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
