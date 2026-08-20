"""Template Method pattern: BaseTrainer defines the fixed skeleton of a
train/evaluate workflow (setup -> train -> evaluate -> save), while
subclasses fill in the framework-specific hooks. This keeps
`pipeline/stage_05_model_trainer.py` agnostic to *how* training happens."""
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    def run(self) -> dict:
        """The template method -- fixed sequence, do not override."""
        logger.info("=== Stage: Model Training ===")
        self.setup()
        train_result = self.train()
        eval_result = self.evaluate()
        self.save()
        summary = {"train": train_result, "eval": eval_result}
        logger.info(f"Training complete. Summary: {summary}")
        return summary

    @abstractmethod
    def setup(self) -> None:
        """Build datasets, model, tokenizer, and the underlying trainer."""

    @abstractmethod
    def train(self) -> dict:
        """Run the training loop, return a dict of training metrics."""

    @abstractmethod
    def evaluate(self) -> dict:
        """Run evaluation on the validation split, return a dict of metrics."""

    @abstractmethod
    def save(self) -> None:
        """Persist the best model + tokenizer to the configured output dir."""
