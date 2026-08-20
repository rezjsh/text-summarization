"""PyTorch Dataset that tokenizes article/summary pairs on the fly for
seq2seq training. Kept deliberately framework-light (no Hugging Face
`datasets.Dataset` requirement) so it also works with a plain DataFrame."""
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SummarizationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str,
        summary_col: str,
        max_input_length: int = 512,
        max_target_length: int = 128,
        prefix: str = "",
    ):
        self.texts = df[text_col].astype(str).tolist()
        self.summaries = df[summary_col].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        source_text = self.prefix + self.texts[idx]
        target_text = self.summaries[idx]

        model_inputs = self.tokenizer(
            source_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer() if hasattr(self.tokenizer, "as_target_tokenizer") else _no_op():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        # Ignore pad tokens in the loss
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


class _no_op:
    """Fallback context manager for tokenizers without as_target_tokenizer()
    (recent transformers versions handle target tokenization automatically)."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
