"""Pipeline stage 1: download + validate the raw CNN/DailyMail dataset."""
import logging

from text_summarization_project.config.configuration import ConfigurationManager
from text_summarization_project.data_ingestion.orchestrator import DataIngestionOrchestrator

logger = logging.getLogger(__name__)

STAGE_NAME = "Data Ingestion"


def main(mode: str = "auto", local_source_dir: str = None):
    config_manager = ConfigurationManager()
    ingestion_config = config_manager.get_data_ingestion_config()
    orchestrator = DataIngestionOrchestrator(ingestion_config, mode=mode, local_source_dir=local_source_dir)
    return orchestrator.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage '{STAGE_NAME}' started <<<<<")
        main()
        logger.info(f">>>>> stage '{STAGE_NAME}' completed <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
