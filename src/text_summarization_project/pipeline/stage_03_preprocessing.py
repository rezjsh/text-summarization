"""Pipeline stage 3: clean, filter, and write processed parquet splits."""
import logging

from text_summarization_project.config.configuration import ConfigurationManager
from text_summarization_project.preprocessing.orchestrator import PreprocessingOrchestrator

logger = logging.getLogger(__name__)

STAGE_NAME = "Preprocessing"


def main():
    config_manager = ConfigurationManager()
    preprocessing_config = config_manager.get_preprocessing_config()
    ingestion_config = config_manager.get_data_ingestion_config()
    dataset_cfg = config_manager.config["dataset"]

    orchestrator = PreprocessingOrchestrator(
        preprocessing_config,
        raw_dir=ingestion_config.unzip_dir,
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
