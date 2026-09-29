"""Episode-bounded data boundary for Issue #265 online updates."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from adaptive_rl.protocol.constants import N_POST, N_UPDATE
from adaptive_rl.protocol.seed_schedule import derive_seed


@dataclass(frozen=True)
class Transition:
    """One recorded environment transition, detached from live environment state."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    environment_action: np.ndarray | None = None
    behavior_log_prob: float | None = None
    behavior_value: float | None = None
    behavior_next_value: float | None = None

    def __post_init__(self) -> None:
        for name in ("observation", "action", "next_observation"):
            value = np.array(getattr(self, name), copy=True)
            if value.dtype.kind not in "biuf":
                raise ValueError(f"{name} must have a numeric dtype")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.environment_action is not None:
            value = np.array(self.environment_action, copy=True)
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError("environment_action must be a finite numeric array")
            value.setflags(write=False)
            object.__setattr__(self, "environment_action", value)
        if not np.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        object.__setattr__(self, "reward", float(self.reward))
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("terminated and truncated must be booleans")
        for name in ("behavior_log_prob", "behavior_value", "behavior_next_value"):
            value = getattr(self, name)
            if value is not None:
                if not np.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite when provided")
                object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class PostShiftEpisode:
    """Completed post-shift episode with its protocol index and derived seed."""

    index: int
    seed: int
    transitions: tuple[Transition, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.index <= N_POST:
            raise ValueError(f"post-shift episode index must be in [1, {N_POST}]")
        if not self.transitions:
            raise ValueError("a completed episode must contain at least one transition")


@dataclass(frozen=True)
class UpdateBatch:
    """Cumulative post-shift experience visible at exactly one update boundary."""

    block_episode: int
    seed: int
    visible_episode_indices: tuple[int, ...]
    transitions: tuple[Transition, ...]


class AdaptationAdapter(Protocol):
    """Algorithm-specific adapter interface; called only at episode boundaries."""

    def update(self, algorithm: Any, batch: UpdateBatch) -> None: ...


def build_update_batch(
    training_seed: int,
    completed_episodes: Sequence[PostShiftEpisode],
    *,
    block_episode: int,
) -> UpdateBatch:
    """Build Bk from precisely post-shift episodes 1..k.

    The builder requires the full ordered prefix. It has no parameter for
    pre-shift or future data, preventing accidental cross-phase exposure.
    """
    if not 5 <= block_episode <= 14:
        raise ValueError("Update blocks exist only after post-shift episodes 5 through 14")
    if len(completed_episodes) != block_episode:
        raise ValueError(
            f"B{block_episode} requires exactly episodes 1..{block_episode}; "
            f"received {len(completed_episodes)} episodes"
        )
    expected_indices = tuple(range(1, block_episode + 1))
    actual_indices = tuple(episode.index for episode in completed_episodes)
    if actual_indices != expected_indices:
        raise ValueError(
            f"B{block_episode} can see only ordered completed episodes "
            f"1..{block_episode}; got {actual_indices}"
        )
    expected_seeds = tuple(
        derive_seed(training_seed, "eval_post", index) for index in expected_indices
    )
    actual_seeds = tuple(episode.seed for episode in completed_episodes)
    if actual_seeds != expected_seeds:
        raise ValueError("post-shift episode seeds do not match the preregistered schedule")

    transitions = tuple(
        transition for episode in completed_episodes for transition in episode.transitions
    )
    return UpdateBatch(
        block_episode=block_episode,
        seed=derive_seed(training_seed, "update", block_episode - 5),
        visible_episode_indices=expected_indices,
        transitions=transitions,
    )


def validate_block_sequence(block_episodes: Sequence[int]) -> None:
    """Require the complete preregistered B5..B14 ordering (never B15)."""
    expected = tuple(range(5, 5 + N_UPDATE))
    actual = tuple(block_episodes)
    if actual != expected:
        raise ValueError(f"update ordering must be exactly {expected}, got {actual}")


def call_update_atomically(algorithm: Any, adapter: AdaptationAdapter, batch: UpdateBatch) -> None:
    """Run an update with wrapper-state rollback if it raises.

    A failed block remains a failure at the caller; rollback only prevents a
    partially-mutated policy from leaking into later evaluation.
    """
    if not isinstance(getattr(algorithm, "__dict__", None), dict):
        raise TypeError("algorithm adapter target must expose instance state")
    snapshot = copy.deepcopy(algorithm.__dict__)
    try:
        adapter.update(algorithm, batch)
    except BaseException:
        algorithm.__dict__.clear()
        algorithm.__dict__.update(snapshot)
        raise


__all__ = [
    "AdaptationAdapter",
    "PostShiftEpisode",
    "Transition",
    "UpdateBatch",
    "build_update_batch",
    "call_update_atomically",
    "validate_block_sequence",
]
