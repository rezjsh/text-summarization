"""Lightweight explainability utilities for the summarizer.

Provides a simple token-attribution view (via attention-weight aggregation
from the encoder's cross-attention on the last generated tokens) so users
can see roughly which source words most influenced the generated summary.
This is a lightweight heuristic, not a rigorous SHAP/LIME attribution --
those are prohibitively slow for seq2seq generation, but the same interface
can be swapped for `shap.Explainer` if deeper analysis is needed."""
import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


def get_cross_attention_saliency(model, tokenizer, text: str, summary: str, prefix: str = "") -> List[Tuple[str, float]]:
    """Runs a forced-decoding forward pass and averages cross-attention
    weights (encoder-decoder attention, last layer) over source tokens."""
    device = next(model.parameters()).device
    inputs = tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=512).to(device)
    labels = tokenizer(summary, return_tensors="pt", truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=labels["input_ids"],
            output_attentions=True,
        )

    # cross_attentions: tuple(num_layers) of [batch, num_heads, tgt_len, src_len]
    if not outputs.cross_attentions:
        logger.warning("Model did not return cross-attentions; explainability unavailable for this architecture.")
        return []

    last_layer_attn = outputs.cross_attentions[-1][0]  # [num_heads, tgt_len, src_len]
    avg_attn = last_layer_attn.mean(dim=(0, 1))  # [src_len]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    scored = list(zip(tokens, avg_attn.tolist()))
    return sorted(scored, key=lambda x: x[1], reverse=True)
