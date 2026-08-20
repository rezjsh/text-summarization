"""Registry pattern mapping our internal model 'family' name to the
tokenizer/model classes needed to load it correctly from the Hugging Face
Hub. New model families are added here in one place."""
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

MODEL_FAMILY_REGISTRY = {
    "t5": {"model_cls": T5ForConditionalGeneration, "tokenizer_cls": T5Tokenizer},
    "flan-t5": {"model_cls": AutoModelForSeq2SeqLM, "tokenizer_cls": AutoTokenizer},
    "bart": {"model_cls": BartForConditionalGeneration, "tokenizer_cls": BartTokenizer},
    "pegasus": {"model_cls": PegasusForConditionalGeneration, "tokenizer_cls": PegasusTokenizer},
}


def resolve_family(family: str) -> dict:
    if family not in MODEL_FAMILY_REGISTRY:
        raise KeyError(
            f"Unknown model family '{family}'. Available: {list(MODEL_FAMILY_REGISTRY)}. "
            f"Add it to models/registry.py to support a new architecture."
        )
    return MODEL_FAMILY_REGISTRY[family]
