"""Random action baseline policy for AdaptiveRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm


class RandomPolicy(BaseAlgorithm):
    """Uniform-random action policy baseline.

    Provides a non-learning baseline against which trained reinforcement learning
    policies (such as PPO) are evaluated under identical seeds and environment
    conditions.
    """

    def __init__(
        self,
        action_space: gym.spaces.Space[Any],
        seed: Optional[int] = None,
    ) -> None:
        self.action_space = action_space
        self.seed_val = seed
        if seed is not None:
            self.action_space.seed(seed)

    def train(self, total_timesteps: int, callback: Any = None) -> None:
        """No-op for an untrained random policy baseline."""
        pass

    def predict(
        self,
        observation: Any,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Sample a uniform-random action from the environment action space."""
        action = self.action_space.sample()
        return np.asarray(action, dtype=np.float32), None

    def save(self, path: str | Path) -> None:
        """No-op for random baseline."""
        pass

    def load(self, path: str | Path) -> None:
        """No-op for random baseline."""
        pass
