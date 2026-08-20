"""Evaluator: loads a trained model, runs generation over the test split,
computes ROUGE (+ optional BERTScore/latency), and saves a comparison
table + results json into artifacts/evaluation/."""
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from text_summarization_project.dataset.registry import DatasetRegistry
from text_summarization_project.entity.config_entity import (
    DatasetSubsetConfig,
    EvaluationConfig,
    GenerationConfig,
    ModelConfig,
)
from text_summarization_project.evaluator.metrics import (
    LatencyTimer,
    compute_bertscore,
    compute_rouge,
    generation_length_stats,
)
from text_summarization_project.models.factory import ModelFactory

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        model_config: ModelConfig,
        eval_config: EvaluationConfig,
        generation_config: GenerationConfig,
        subset_config: DatasetSubsetConfig,
        model_dir: str = None,
    ):
        self.model_config = model_config
        self.eval_config = eval_config
        self.generation_config = generation_config
        self.subset_config = subset_config
        self.model_dir = model_dir  # if set, load fine-tuned weights instead of the base checkpoint

    def _load_model(self):
        if self.model_dir:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            return model, tokenizer
        return ModelFactory.create(self.model_config)

    def run(self, max_samples: int = None) -> dict:
        logger.info("=== Stage: Model Evaluation ===")
        model, tokenizer = self._load_model()
        device = next(model.parameters()).device

        registry = DatasetRegistry(self.subset_config)
        test_df = registry.load_split("test")
        if max_samples:
            test_df = test_df.head(max_samples)

        predictions, references = [], []
        with LatencyTimer() as timer:
            for _, row in test_df.iterrows():
                source = self.model_config.requires_prefix + str(row[self.subset_config.text_column])
                inputs = tokenizer(source, return_tensors="pt", truncation=True,
                                    max_length=self.model_config.max_input_length).to(device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    min_new_tokens=self.generation_config.min_new_tokens,
                    num_beams=self.generation_config.num_beams,
                    length_penalty=self.generation_config.length_penalty,
                    no_repeat_ngram_size=self.generation_config.no_repeat_ngram_size,
                    early_stopping=self.generation_config.early_stopping,
                )
                pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                predictions.append(pred)
                references.append(str(row[self.subset_config.summary_column]))

        results = compute_rouge(predictions, references)
        results.update(generation_length_stats(predictions))
        if self.eval_config.measure_latency:
            results["avg_latency_sec_per_sample"] = timer.per_sample(len(test_df))
        if self.eval_config.compute_bertscore:
            results.update(compute_bertscore(predictions, references))

        self.eval_config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.eval_config.output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        comparison_df = pd.DataFrame({
            "article": test_df[self.subset_config.text_column].astype(str).str[:300],
            "reference_summary": references,
            "generated_summary": predictions,
        })
        comparison_df.to_csv(self.eval_config.output_dir / "predictions_vs_references.csv", index=False)

        logger.info(f"Evaluation results: {results}")
        return results
