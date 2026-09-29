"""Training orchestrator, callbacks, and checkpointing for AdaptiveRL."""

from adaptive_rl.training.callbacks import (
    BaseCallback,
    CheckpointCallback,
    MetricLoggerCallback,
    SB3CallbackAdapter,
)
from adaptive_rl.training.checkpointing import CheckpointManager
from adaptive_rl.training.trainer import (
    PPOTrainer,
    RLTrainer,
    TrainingResult,
    get_trainer,
)

BaseTrainer = RLTrainer

__all__ = [
    "BaseCallback",
    "BaseTrainer",
    "CheckpointCallback",
    "CheckpointManager",
    "MetricLoggerCallback",
    "PPOTrainer",
    "RLTrainer",
    "SB3CallbackAdapter",
    "TrainingResult",
    "get_trainer",
]
