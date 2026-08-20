"""High-level Summarizer facade: wraps a (model, tokenizer) pair and the
generation config into one simple `.summarize(text)` call. This is what
inference/ and app/ talk to -- they never touch transformers directly."""
import logging
from pathlib import Path
from typing import List, Union

import torch

from text_summarization_project.entity.config_entity import GenerationConfig, ModelConfig
from text_summarization_project.models.factory import ModelFactory

logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        model_dir: Union[str, Path] = None,
    ):
        """If model_dir is given, loads fine-tuned weights from that folder
        (e.g. artifacts/checkpoints/best_model). Otherwise loads the base
        pretrained checkpoint named in model_config -- useful for smoke
        tests before any training has happened."""
        self.model_config = model_config
        self.generation_config = generation_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_dir is not None and Path(model_dir).exists():
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
            self.model.to(self.device)
            logger.info(f"Loaded fine-tuned model from {model_dir}")
        else:
            self.model, self.tokenizer = ModelFactory.create(model_config, device=self.device)
            if model_dir is not None:
                logger.warning(f"model_dir '{model_dir}' not found; falling back to base checkpoint.")

    def summarize(self, text: str, **generation_overrides) -> str:
        return self.summarize_batch([text], **generation_overrides)[0]

    def summarize_batch(self, texts: List[str], batch_size: int = 8, **generation_overrides) -> List[str]:
        gen_kwargs = dict(
            max_new_tokens=self.generation_config.max_new_tokens,
            min_new_tokens=self.generation_config.min_new_tokens,
            num_beams=self.generation_config.num_beams,
            length_penalty=self.generation_config.length_penalty,
            no_repeat_ngram_size=self.generation_config.no_repeat_ngram_size,
            early_stopping=self.generation_config.early_stopping,
        )
        gen_kwargs.update(generation_overrides)

        outputs = []
        prefix = self.model_config.requires_prefix
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i + batch_size]]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=self.model_config.max_input_length, padding=True,
            ).to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            outputs.extend(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        return outputs
