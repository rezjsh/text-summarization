"""Pipeline stage 5: evaluate a trained (or base) model on the test split."""
import argparse
import logging

from text_summarization_project.config.configuration import ConfigurationManager
from text_summarization_project.evaluator.evaluator import Evaluator

logger = logging.getLogger(__name__)

STAGE_NAME = "Model Evaluation"


def main(model_key: str = None, subset_name: str = None, model_dir: str = None, max_samples: int = None):
    config_manager = ConfigurationManager()
    model_config = config_manager.get_model_config(model_key)
    eval_config = config_manager.get_evaluation_config()
    generation_config = config_manager.get_generation_config()
    subset_config = config_manager.get_dataset_subset_config(subset_name)

    evaluator = Evaluator(model_config, eval_config, generation_config, subset_config, model_dir=model_dir)
    return evaluator.run(max_samples=max_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--subset", default=None, choices=["dev", "medium", "full"])
    parser.add_argument("--model_dir", default=None, help="Path to a fine-tuned model dir, e.g. artifacts/checkpoints/best_model")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    try:
        logger.info(f">>>>> stage '{STAGE_NAME}' started <<<<<")
        main(model_key=args.model, subset_name=args.subset, model_dir=args.model_dir, max_samples=args.max_samples)
        logger.info(f">>>>> stage '{STAGE_NAME}' completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
