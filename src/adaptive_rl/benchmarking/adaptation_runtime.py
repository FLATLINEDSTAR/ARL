"""Deterministic episode collection for the Issue #265 fork protocol."""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Optional

import gymnasium as gym
import numpy as np
import torch

from adaptive_rl.protocol.adaptation import PostShiftEpisode, Transition
from adaptive_rl.protocol.fork import model_fingerprint


def _fingerprint(algorithm: Any) -> str:
    fixed_fingerprint = getattr(algorithm, "fingerprint", None)
    if isinstance(fixed_fingerprint, str):
        return fixed_fingerprint
    return model_fingerprint(algorithm)


@contextmanager
def _seed_episode(seed: int) -> Iterator[None]:
    """Seed episode-side stochastic actions and restore the caller RNG states."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _action_and_behavior(
    algorithm: Any, observation: Any, deterministic: bool
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    model = getattr(algorithm, "model", None)
    policy = getattr(model, "policy", None) if model is not None else None
    if policy is None:
        environment_action, _ = algorithm.predict(observation, deterministic=deterministic)
        environment_action = np.asarray(environment_action)
        return environment_action, environment_action, None, None
    if hasattr(model, "rollout_buffer"):
        assert model is not None
        obs_tensor, _ = policy.obs_to_tensor(observation)
        with torch.no_grad():
            native_action, value, log_prob = policy(obs_tensor, deterministic=deterministic)
        native_array = native_action.detach().cpu().numpy().reshape(-1)
        if getattr(policy, "squash_output", False):
            environment_action = policy.unscale_action(native_array)
        else:
            environment_action = np.clip(
                native_array,
                np.asarray(model.action_space.low),
                np.asarray(model.action_space.high),
            )
        return (
            np.asarray(native_array),
            np.asarray(environment_action),
            float(log_prob.reshape(-1)[0].cpu()),
            float(value.reshape(-1)[0].cpu()),
        )

    environment_action, _ = algorithm.predict(observation, deterministic=deterministic)
    environment_action = np.asarray(environment_action)
    if getattr(policy, "squash_output", False):
        native_action = policy.scale_action(environment_action)
    else:
        native_action = environment_action
    return np.asarray(native_action), environment_action, None, None


def _behavior_value(algorithm: Any, observation: Any) -> float | None:
    model = getattr(algorithm, "model", None)
    if model is None or not hasattr(model, "rollout_buffer"):
        return None
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        value = model.policy.predict_values(obs_tensor)
    return float(value.reshape(-1)[0].cpu())


@dataclass(frozen=True)
class EpisodeRecord:
    """Episode metrics plus complete transition data for later adaptation."""

    training_seed: int
    arm: str
    algorithm: str
    environment: str
    phase: str
    episode_index: int
    episode_seed: int
    reward: float
    length: int
    success: Optional[bool]
    collision: Optional[bool]
    terminated: bool
    truncated: bool
    policy_fingerprint_start: str
    policy_fingerprint_end: str
    update_block: Optional[int]
    update_seed: Optional[int]
    update_status: Optional[str]
    parameter_delta_l2: Optional[float]
    final_info: dict[str, Any]
    transitions: tuple[Transition, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def post_shift_data(self) -> PostShiftEpisode:
        if self.phase != "post":
            raise ValueError("only completed post-shift episodes can enter adaptation data")
        return PostShiftEpisode(
            index=self.episode_index,
            seed=self.episode_seed,
            transitions=self.transitions,
        )


@dataclass(frozen=True)
class EpisodeTaggedTransition(Transition):
    """Internal SAC replay record with a post-shift episode provenance tag."""

    episode_index: int = -1


def evaluate_episode(
    *,
    algorithm: Any,
    env: gym.Env,
    training_seed: int,
    phase: str,
    episode_index: int,
    episode_seed: int,
    algorithm_name: str,
    environment_name: str,
    deterministic: bool,
    arm: str,
    update_log: Any = None,
) -> EpisodeRecord:
    """Run and record one episode; no model update is reachable in this function."""
    if phase not in {"pre", "post"}:
        raise ValueError("phase must be 'pre' or 'post'")
    model = getattr(algorithm, "model", None)
    if model is None and not isinstance(getattr(algorithm, "fingerprint", None), str):
        raise RuntimeError("Cannot evaluate an uninitialized algorithm")
    if model is not None:
        model.policy.set_training_mode(False)
    fingerprint_start = _fingerprint(algorithm)
    transitions: list[Transition] = []
    rewards: list[float] = []
    success_values: list[bool] = []
    collisions: list[bool] = []
    observation, info = env.reset(seed=int(episode_seed))
    terminated = False
    truncated = False
    last_info: dict[str, Any] = dict(info)

    with _seed_episode(int(episode_seed)):
        while not (terminated or truncated):
            native_action, environment_action, behavior_log_prob, behavior_value = (
                _action_and_behavior(algorithm, observation, deterministic)
            )
            next_observation, reward, terminated, truncated, step_info = env.step(
                environment_action
            )
            behavior_next_value = (
                _behavior_value(algorithm, next_observation)
                if truncated and not terminated
                else None
            )
            transition = Transition(
                observation=np.asarray(observation),
                action=native_action,
                environment_action=np.asarray(environment_action),
                reward=float(reward),
                next_observation=np.asarray(next_observation),
                terminated=bool(terminated),
                truncated=bool(truncated),
                behavior_log_prob=behavior_log_prob,
                behavior_value=behavior_value,
                behavior_next_value=behavior_next_value,
            )
            transitions.append(transition)
            rewards.append(float(reward))
            if isinstance(step_info.get("success"), (bool, np.bool_)):
                success_values.append(bool(step_info["success"]))
            if isinstance(step_info.get("collision"), (bool, np.bool_)):
                collisions.append(bool(step_info["collision"]))
            last_info = dict(step_info)
            observation = next_observation

    fingerprint_end = _fingerprint(algorithm)
    if fingerprint_end != fingerprint_start:
        raise RuntimeError("Policy parameters changed during an evaluation episode")
    parameter_delta_l2 = getattr(update_log, "parameter_delta_l2", None)
    if arm == "fixed" and algorithm_name == "sac":
        parameter_delta_l2 = getattr(algorithm, "parameter_delta_l2", None)
        if parameter_delta_l2 != 0.0:
            raise RuntimeError(
                f"Fixed SAC weight delta must be exactly 0, got {parameter_delta_l2!r}"
            )
    elif arm == "shared" and algorithm_name == "sac":
        # fingerprint_start == fingerprint_end above establishes exact
        # state equality for these shared pre-treatment episodes.
        parameter_delta_l2 = 0.0
    if not rewards:
        raise RuntimeError("Environment returned an empty episode")
    safe_info: dict[str, Any] = {}
    for key, value in last_info.items():
        if isinstance(value, np.ndarray):
            safe_info[str(key)] = value if np.isfinite(value).all() else None
        elif isinstance(value, np.generic):
            native = value.item()
            safe_info[str(key)] = (
                native if not isinstance(native, float) or np.isfinite(native) else None
            )
        elif value is None or isinstance(value, (str, bool, int)):
            safe_info[str(key)] = value
        elif isinstance(value, float):
            safe_info[str(key)] = value if np.isfinite(value) else None
    if any(isinstance(value, float) and not np.isfinite(value) for value in rewards):
        raise FloatingPointError("episode produced a non-finite reward")
    return EpisodeRecord(
        training_seed=training_seed,
        arm=arm,
        algorithm=algorithm_name,
        environment=environment_name,
        phase=phase,
        episode_index=episode_index,
        episode_seed=int(episode_seed),
        reward=float(sum(rewards)),
        length=len(rewards),
        success=success_values[-1] if success_values else None,
        collision=any(collisions) if collisions else None,
        terminated=bool(terminated),
        truncated=bool(truncated),
        policy_fingerprint_start=fingerprint_start,
        policy_fingerprint_end=fingerprint_end,
        update_block=getattr(update_log, "block_episode", None),
        update_seed=getattr(update_log, "update_seed", None),
        update_status=getattr(update_log, "status", None),
        parameter_delta_l2=parameter_delta_l2,
        final_info=safe_info,
        transitions=tuple(transitions),
    )


__all__ = ["EpisodeRecord", "evaluate_episode"]
