"""Concrete Trainer built on top of Hugging Face's Seq2SeqTrainer. Fills in
the hooks defined by BaseTrainer (Template Method)."""
import logging

import numpy as np
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from text_summarization_project.dataset.registry import DatasetRegistry
from text_summarization_project.dataset.torch_dataset import SummarizationDataset
from text_summarization_project.entity.config_entity import (
    DatasetSubsetConfig,
    ModelConfig,
    TrainingConfig,
)
from text_summarization_project.evaluator.metrics import compute_rouge
from text_summarization_project.models.factory import ModelFactory
from text_summarization_project.trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Seq2SeqSummarizationTrainer(BaseTrainer):
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        subset_config: DatasetSubsetConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.subset_config = subset_config

        self.model = None
        self.tokenizer = None
        self.hf_trainer = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self) -> None:
        self.model, self.tokenizer = ModelFactory.create(self.model_config)

        registry = DatasetRegistry(self.subset_config)
        train_df = registry.load_split("train", seed=self.training_config.seed)
        val_df = registry.load_split("validation", seed=self.training_config.seed)

        self.train_dataset = SummarizationDataset(
            train_df, self.tokenizer,
            text_col=self.subset_config.text_column,
            summary_col=self.subset_config.summary_column,
            max_input_length=self.model_config.max_input_length,
            max_target_length=self.model_config.max_target_length,
            prefix=self.model_config.requires_prefix,
        )
        self.val_dataset = SummarizationDataset(
            val_df, self.tokenizer,
            text_col=self.subset_config.text_column,
            summary_col=self.subset_config.summary_column,
            max_input_length=self.model_config.max_input_length,
            max_target_length=self.model_config.max_target_length,
            prefix=self.model_config.requires_prefix,
        )

        args = Seq2SeqTrainingArguments(
            output_dir=str(self.training_config.output_dir),
            logging_dir=str(self.training_config.logging_dir),
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            fp16=self.training_config.fp16,
            save_total_limit=self.training_config.save_total_limit,
            logging_steps=self.training_config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.training_config.eval_steps,
            save_steps=self.training_config.save_steps,
            predict_with_generate=True,
            load_best_model_at_end=True,
            metric_for_best_model="rouge2",
            greater_is_better=True,
            seed=self.training_config.seed,
            report_to=["none"],
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        def _compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            return compute_rouge(decoded_preds, decoded_labels)

        self.hf_trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.training_config.early_stopping_patience
            )],
        )

    def train(self) -> dict:
        result = self.hf_trainer.train()
        return {"loss": result.training_loss, "metrics": result.metrics}

    def evaluate(self) -> dict:
        return self.hf_trainer.evaluate()

    def save(self) -> None:
        best_dir = self.training_config.output_dir / "best_model"
        self.hf_trainer.save_model(str(best_dir))
        self.tokenizer.save_pretrained(str(best_dir))
        logger.info(f"Saved best model + tokenizer to {best_dir}")
