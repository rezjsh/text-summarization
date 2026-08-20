"""Pipeline stage 2: run EDA over the raw dataset and save reports/plots."""
import logging

from text_summarization_project.config.configuration import ConfigurationManager
from text_summarization_project.eda.orchestrator import EDAOrchestrator

logger = logging.getLogger(__name__)

STAGE_NAME = "Exploratory Data Analysis"


def main():
    config_manager = ConfigurationManager()
    eda_config = config_manager.get_eda_config()
    dataset_cfg = config_manager.config["dataset"]
    orchestrator = EDAOrchestrator(
        eda_config,
        text_col=dataset_cfg["text_column"],
        summary_col=dataset_cfg["summary_column"],
    )
    return orchestrator.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage '{STAGE_NAME}' started <<<<<")
        main()
        logger.info(f">>>>> stage '{STAGE_NAME}' completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
