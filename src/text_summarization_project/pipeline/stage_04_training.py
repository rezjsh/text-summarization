"""Pipeline stage 4: fine-tune the configured summarization model."""
import argparse
import logging

from text_summarization_project.config.configuration import ConfigurationManager
from text_summarization_project.trainer.seq2seq_trainer import Seq2SeqSummarizationTrainer

logger = logging.getLogger(__name__)

STAGE_NAME = "Model Training"


def main(model_key: str = None, subset_name: str = None):
    config_manager = ConfigurationManager()
    model_config = config_manager.get_model_config(model_key)
    training_config = config_manager.get_training_config()
    subset_config = config_manager.get_dataset_subset_config(subset_name)

    trainer = Seq2SeqSummarizationTrainer(model_config, training_config, subset_config)
    return trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model key from configs/model_config.yaml")
    parser.add_argument("--subset", default=None, choices=["dev", "medium", "full"])
    args = parser.parse_args()

    try:
        logger.info(f">>>>> stage '{STAGE_NAME}' started <<<<<")
        main(model_key=args.model, subset_name=args.subset)
        logger.info(f">>>>> stage '{STAGE_NAME}' completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
