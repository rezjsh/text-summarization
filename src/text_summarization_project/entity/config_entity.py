"""Typed config objects (dataclasses) produced by ConfigurationManager.

Keeping these as dataclasses (instead of passing raw dicts around) means every
component gets IDE autocomplete + type checking, and a bad config key fails
fast at ConfigurationManager construction time instead of deep in training.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    kaggle_dataset: str
    raw_dir: Path
    unzip_dir: Path
    expected_files: List[str]


@dataclass(frozen=True)
class EDAConfig:
    output_dir: Path
    sample_size_for_token_plots: int
    top_n_words: int
    ngram_range: List[int]
    raw_dir: Path


@dataclass(frozen=True)
class PreprocessingConfig:
    interim_dir: Path
    processed_dir: Path
    min_article_chars: int
    max_article_chars: int
    min_summary_chars: int
    max_summary_chars: int
    lowercase: bool
    drop_duplicates: bool
    drop_na: bool


@dataclass(frozen=True)
class DatasetSubsetConfig:
    name: str
    train_size: Optional[int]
    val_size: Optional[int]
    test_size: Optional[int]
    text_column: str
    summary_column: str
    processed_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    key: str
    hf_checkpoint: str
    family: str
    requires_prefix: str
    max_input_length: int
    max_target_length: int


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    logging_dir: Path
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    fp16: bool
    save_total_limit: int
    early_stopping_patience: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    output_dir: Path
    metrics: List[str]
    compute_bertscore: bool
    measure_latency: bool


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    min_new_tokens: int
    num_beams: int
    length_penalty: float
    no_repeat_ngram_size: int
    early_stopping: bool
