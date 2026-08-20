"""End-to-end pipeline runner: ingestion -> EDA -> preprocessing -> train ->
evaluate. Mirrors what `make pipeline` runs. Prefer the individual
scripts/*.py entry points for iterative development."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from text_summarization_project.pipeline import (
    stage_01_data_ingestion,
    stage_02_eda,
    stage_03_preprocessing,
    stage_04_training,
    stage_05_evaluation,
)
from text_summarization_project.utils.common import setup_logging
from text_summarization_project.constants.constants import LOGGING_CONFIG_FILE_PATH

setup_logging(LOGGING_CONFIG_FILE_PATH)
logger = logging.getLogger(__name__)


def run_pipeline(subset: str = "dev", model: str = None, skip_training: bool = False):
    stage_01_data_ingestion.main()
    stage_02_eda.main()
    stage_03_preprocessing.main()
    if not skip_training:
        stage_04_training.main(model_key=model, subset_name=subset)
        stage_05_evaluation.main(
            model_key=model, subset_name=subset,
            model_dir="artifacts/checkpoints/best_model",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full summarization pipeline end-to-end.")
    parser.add_argument("--subset", default="dev", choices=["dev", "medium", "full"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()
    run_pipeline(subset=args.subset, model=args.model, skip_training=args.skip_training)
