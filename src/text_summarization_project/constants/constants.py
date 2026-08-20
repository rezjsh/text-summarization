"""Static, never-changes-at-runtime constants (paths to config files, etc.)."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

CONFIG_FILE_PATH = ROOT_DIR / "configs" / "config.yaml"
DATASET_CONFIG_FILE_PATH = ROOT_DIR / "configs" / "dataset_config.yaml"
MODEL_CONFIG_FILE_PATH = ROOT_DIR / "configs" / "model_config.yaml"
LOGGING_CONFIG_FILE_PATH = ROOT_DIR / "configs" / "logging_config.yaml"

RANDOM_SEED = 42
