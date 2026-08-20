"""Factory pattern: turns a ModelConfig into a ready-to-use
(model, tokenizer) pair, regardless of underlying architecture (T5 / FLAN-T5
/ BART / PEGASUS). Callers never import a specific transformers class."""
import logging

import torch

from text_summarization_project.entity.config_entity import ModelConfig
from text_summarization_project.models.registry import resolve_family

logger = logging.getLogger(__name__)


class ModelFactory:
    @staticmethod
    def create(model_config: ModelConfig, device: str = None):
        family_classes = resolve_family(model_config.family)
        tokenizer = family_classes["tokenizer_cls"].from_pretrained(model_config.hf_checkpoint)
        model = family_classes["model_cls"].from_pretrained(model_config.hf_checkpoint)

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(
            f"Loaded '{model_config.key}' ({model_config.hf_checkpoint}, family={model_config.family}) "
            f"onto device={device}"
        )
        return model, tokenizer
