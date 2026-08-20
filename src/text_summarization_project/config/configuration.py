"""ConfigurationManager: the single place that turns yaml files into typed
config-entity dataclasses. Every component asks this class for its config
instead of parsing yaml itself.
"""
import logging
from pathlib import Path

from text_summarization_project.constants.constants import (
    CONFIG_FILE_PATH,
    DATASET_CONFIG_FILE_PATH,
    MODEL_CONFIG_FILE_PATH,
    ROOT_DIR,
)
from text_summarization_project.entity.config_entity import (
    DataIngestionConfig,
    DatasetSubsetConfig,
    EDAConfig,
    EvaluationConfig,
    GenerationConfig,
    ModelConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from text_summarization_project.utils.common import create_directories, read_yaml

logger = logging.getLogger(__name__)


class ConfigurationManager:
    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        dataset_config_path: Path = DATASET_CONFIG_FILE_PATH,
        model_config_path: Path = MODEL_CONFIG_FILE_PATH,
    ):
        self.config = read_yaml(config_path)
        self.dataset_config = read_yaml(dataset_config_path)
        self.model_config = read_yaml(model_config_path)

        create_directories([
            ROOT_DIR / self.config["artifacts_root"],
            ROOT_DIR / self.config["logs_root"],
        ])

    def _p(self, rel_path: str | Path) -> Path:
        """Return a path relative to the project root."""
        path = Path(rel_path)

        if path.is_absolute():
            return path

        return ROOT_DIR / path

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        create_directories([self._p(cfg["raw_dir"])])
        return DataIngestionConfig(
            kaggle_dataset=cfg["kaggle_dataset"],
            raw_dir=self._p(cfg["raw_dir"]),
            unzip_dir=self._p(cfg["unzip_dir"]),
            expected_files=cfg["expected_files"],
        )

    def get_eda_config(self) -> EDAConfig:
        cfg = self.config["eda"]
        create_directories([self._p(cfg["output_dir"])])
        return EDAConfig(
            output_dir=self._p(cfg["output_dir"]),
            sample_size_for_token_plots=cfg["sample_size_for_token_plots"],
            top_n_words=cfg["top_n_words"],
            ngram_range=cfg["ngram_range"],
            raw_dir=self._p(self.config["data_ingestion"]["unzip_dir"]),
        )

    def get_preprocessing_config(self) -> PreprocessingConfig:
        cfg = self.config["preprocessing"]
        create_directories([self._p(cfg["interim_dir"]), self._p(cfg["processed_dir"])])
        return PreprocessingConfig(
            interim_dir=self._p(cfg["interim_dir"]),
            processed_dir=self._p(cfg["processed_dir"]),
            min_article_chars=cfg["min_article_chars"],
            max_article_chars=cfg["max_article_chars"],
            min_summary_chars=cfg["min_summary_chars"],
            max_summary_chars=cfg["max_summary_chars"],
            lowercase=cfg["lowercase"],
            drop_duplicates=cfg["drop_duplicates"],
            drop_na=cfg["drop_na"],
        )

    def get_dataset_subset_config(self, subset_name: str = None) -> DatasetSubsetConfig:
        subset_name = subset_name or self.config["dataset"]["active_subset"]
        if subset_name not in self.dataset_config["subsets"]:
            raise KeyError(
                f"Unknown dataset subset '{subset_name}'. "
                f"Available: {list(self.dataset_config['subsets'].keys())}"
            )
        subset = self.dataset_config["subsets"][subset_name]
        return DatasetSubsetConfig(
            name=subset_name,
            train_size=subset["train_size"],
            val_size=subset["val_size"],
            test_size=subset["test_size"],
            text_column=self.config["dataset"]["text_column"],
            summary_column=self.config["dataset"]["summary_column"],
            processed_dir=self._p(self.config["preprocessing"]["processed_dir"]),
        )

    def get_model_config(self, model_key: str = None) -> ModelConfig:
        model_key = model_key or self.config["model"]["active_model"]
        if model_key not in self.model_config["models"]:
            raise KeyError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(self.model_config['models'].keys())}"
            )
        m = self.model_config["models"][model_key]
        return ModelConfig(
            key=model_key,
            hf_checkpoint=m["hf_checkpoint"],
            family=m["family"],
            requires_prefix=m.get("requires_prefix", ""),
            max_input_length=self.config["model"]["max_input_length"],
            max_target_length=self.config["model"]["max_target_length"],
        )

    def get_training_config(self) -> TrainingConfig:
        cfg = self.config["training"]
        create_directories([self._p(cfg["output_dir"]), self._p(cfg["logging_dir"])])
        return TrainingConfig(
            output_dir=self._p(cfg["output_dir"]),
            logging_dir=self._p(cfg["logging_dir"]),
            num_train_epochs=cfg["num_train_epochs"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
            learning_rate=float(cfg["learning_rate"]),
            weight_decay=cfg["weight_decay"],
            warmup_ratio=cfg["warmup_ratio"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            fp16=cfg["fp16"],
            save_total_limit=cfg["save_total_limit"],
            early_stopping_patience=cfg["early_stopping_patience"],
            logging_steps=cfg["logging_steps"],
            eval_steps=cfg["eval_steps"],
            save_steps=cfg["save_steps"],
            seed=cfg["seed"],
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        cfg = self.config["evaluation"]
        create_directories([self._p(cfg["output_dir"])])
        return EvaluationConfig(
            output_dir=self._p(cfg["output_dir"]),
            metrics=cfg["metrics"],
            compute_bertscore=cfg["compute_bertscore"],
            measure_latency=cfg["measure_latency"],
        )

    def get_generation_config(self) -> GenerationConfig:
        cfg = self.config["generation"]
        return GenerationConfig(
            max_new_tokens=cfg["max_new_tokens"],
            min_new_tokens=cfg["min_new_tokens"],
            num_beams=cfg["num_beams"],
            length_penalty=cfg["length_penalty"],
            no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
            early_stopping=cfg["early_stopping"],
        )
