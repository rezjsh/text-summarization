"""Small shared helpers for I/O, YAML, JSON, logging, timing, and environment access."""

from __future__ import annotations

import json
import logging
import logging.config
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file that contains a top-level mapping."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"YAML configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if content is None:
        logger.warning("YAML file is empty: %s", path)
        return {}

    if not isinstance(content, dict):
        raise ValueError(
            f"Expected a YAML mapping in '{path}', "
            f"but got {type(content).__name__}."
        )

    logger.debug("Loaded YAML config from: %s", path)
    return content


def create_directories(
    paths: Iterable[str | Path],
    verbose: bool = True,
) -> None:
    """Create directories, including missing parent directories."""
    for path in paths:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        if verbose:
            logger.info("Directory ready: %s", directory)


def save_json(
    path: str | Path,
    data: Mapping[str, Any],
) -> None:
    """Save mapping data as formatted UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    logger.info("Saved JSON to: %s", path)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from a file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(
            f"Expected a JSON object in '{path}', "
            f"but got {type(content).__name__}."
        )

    logger.debug("Loaded JSON from: %s", path)
    return content


def setup_logging(
    logging_config_path: str | Path,
    default_level: int = logging.INFO,
) -> None:
    """Configure logging from YAML, or use a readable fallback configuration."""
    logging_config_path = Path(logging_config_path)

    if logging_config_path.is_file():
        config = read_yaml(logging_config_path)
        logging.config.dictConfig(config)
        logger.debug("Logging configured from: %s", logging_config_path)
        return

    logging.basicConfig(
        level=default_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.warning(
        "Logging config not found at %s; using basic configuration.",
        logging_config_path,
    )


class Timer:
    """Log elapsed wall-clock duration for a context-managed operation."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        logger.info("[%s] started", self.label)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        if self._start is None:
            return False

        elapsed = time.perf_counter() - self._start

        if exc_type is None:
            logger.info("[%s] finished in %.2fs", self.label, elapsed)
        else:
            logger.error(
                "[%s] failed after %.2fs: %s",
                self.label,
                elapsed,
                exc_value,
                exc_info=(exc_type, exc_value, traceback),
            )

        # Do not suppress exceptions raised inside the `with` block.
        return False


def get_env(
    name: str,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    """Get an environment variable, optionally requiring a non-empty value."""
    value = os.getenv(name, default)

    if required and not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set or empty. "
            "Check your .env file against .env.example."
        )

    return value