"""Recorded-data PPO and SAC update adapters for Issue #265."""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.logger import Logger

from adaptive_rl.protocol.adaptation import AdaptationAdapter, UpdateBatch, call_update_atomically
from adaptive_rl.protocol.fork import model_fingerprint, policy_state_tensors


class TaggedReplayBuffer(ReplayBuffer):
    """SB3 replay storage with one post-shift episode index per occupied slot."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.episode_tags: list[int] = []

    def add_tagged(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        *,
        episode_index: int,
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        if self.full:
            # Adaptation capacity is sized to the visible prefix, so wraparound
            # is not expected; failing prevents tags from silently desyncing.
            raise RuntimeError("tagged adaptation replay buffer unexpectedly wrapped")
        self.episode_tags.append(int(episode_index))


@contextmanager
def _seeded_update(seed: int) -> Iterator[None]:
    """Seed update-side RNGs and restore caller RNG state on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _model(algorithm: Any) -> Any:
    model = getattr(algorithm, "model", None)
    if model is None:
        raise RuntimeError("Cannot adapt an uninitialized algorithm")
    return model


@contextmanager
def _logger_ready(model: Any) -> Iterator[None]:
    """Provide a silent SB3 logger when a model is adapted before its first fit."""
    had_logger = hasattr(model, "_logger")
    old_logger = getattr(model, "_logger", None)
    if not had_logger:
        model.set_logger(Logger(folder=None, output_formats=[]))
    logger = getattr(model, "logger", None)
    old_values = dict(getattr(logger, "name_to_value", {}))
    recorded_values = getattr(logger, "name_to_value", None)
    if isinstance(recorded_values, dict):
        recorded_values.clear()
    try:
        yield
    finally:
        if isinstance(recorded_values, dict):
            recorded_values.clear()
            recorded_values.update(old_values)
        if had_logger:
            model._logger = old_logger
        else:
            delattr(model, "_logger")


class PPOAdaptationAdapter:
    """Use Stable-Baselines3 PPO's native clipped objective on stored rollouts."""

    def __init__(self) -> None:
        self.last_loss_metrics: dict[str, float] = {}

    def update(self, algorithm: Any, batch: UpdateBatch) -> None:
        model = _model(algorithm)
        if not hasattr(model, "rollout_buffer") or not hasattr(model, "n_epochs"):
            raise TypeError("PPOAdaptationAdapter requires a Stable-Baselines3 PPO model")
        if int(model.n_epochs) < 1 or int(model.batch_size) < 1:
            raise ValueError("PPO adaptation requires positive configured epochs and batch size")
        if any(
            transition.behavior_log_prob is None or transition.behavior_value is None
            for transition in batch.transitions
        ):
            raise ValueError(
                "PPO adaptation requires behavior log-probability and value per transition"
            )
        if any(
            transition.truncated
            and not transition.terminated
            and transition.behavior_next_value is None
            for transition in batch.transitions
        ):
            raise ValueError("PPO truncated transitions require their recorded behavior next-value")

        count = len(batch.transitions)
        model.rollout_buffer = RolloutBuffer(
            buffer_size=count,
            observation_space=model.observation_space,
            action_space=model.action_space,
            device=model.device,
            gae_lambda=model.gae_lambda,
            gamma=model.gamma,
            n_envs=1,
        )
        model.rollout_buffer.reset()
        observations: list[np.ndarray] = []

        for transition in batch.transitions:
            observation = np.asarray(transition.observation)
            action = np.asarray(transition.action)
            reward = float(transition.reward)
            if transition.truncated and not transition.terminated:
                assert transition.behavior_next_value is not None
                reward += model.gamma * float(transition.behavior_next_value)
            observations.append(observation)
            model.rollout_buffer.add(
                obs=observation.reshape((1, *observation.shape)),
                action=action.reshape((1, -1)),
                reward=np.asarray([reward], dtype=np.float32),
                episode_start=np.asarray(
                    [
                        len(observations) == 1
                        or _previous_done(batch.transitions, len(observations) - 1)
                    ],
                    dtype=np.float32,
                ),
                value=torch.as_tensor(
                    [transition.behavior_value], dtype=torch.float32, device=model.device
                ),
                log_prob=torch.as_tensor(
                    [transition.behavior_log_prob], dtype=torch.float32, device=model.device
                ),
            )
        model.rollout_buffer.compute_returns_and_advantage(
            last_values=torch.zeros(1, dtype=torch.float32, device=model.device),
            dones=np.ones(1, dtype=np.float32),
        )
        with _logger_ready(model), _seeded_update(batch.seed):
            model.train()
            self.last_loss_metrics = _loss_metrics(model)
        model.policy.set_training_mode(False)


def _previous_done(transitions: tuple[Any, ...], previous_index: int) -> bool:
    previous = transitions[previous_index - 1]
    return bool(previous.terminated or previous.truncated)


class SACAdaptationAdapter:
    """Train SAC from a fresh, episode-tagged buffer containing visible data only."""

    def __init__(self) -> None:
        self.last_loss_metrics: dict[str, float] = {}

    def update(self, algorithm: Any, batch: UpdateBatch) -> None:
        model = _model(algorithm)
        if not hasattr(model, "critic_target") or not hasattr(model, "gradient_steps"):
            raise TypeError("SACAdaptationAdapter requires a Stable-Baselines3 SAC model")
        if int(model.gradient_steps) < 1 or int(model.batch_size) < 1:
            raise ValueError(
                "SAC adaptation requires positive configured gradient steps and batch size"
            )
        if tuple(sorted(set(batch.visible_episode_indices))) != batch.visible_episode_indices:
            raise RuntimeError("SAC update episode indices must be unique and ordered")
        if len(batch.transitions) == 0:
            raise RuntimeError("SAC update requires at least one transition")
        count = len(batch.transitions)
        # Never reuse the replay buffer containing nominal training experience.
        replay = TaggedReplayBuffer(
            buffer_size=max(count, int(model.batch_size)),
            observation_space=model.observation_space,
            action_space=model.action_space,
            device=model.device,
            n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
        )
        # The shared Transition type intentionally stays algorithm-neutral. The
        # runner supplies tagged transitions via a thin subclass; plain
        # protocol batches retain their established semantics.
        tags = tuple(getattr(item, "episode_index", None) for item in batch.transitions)
        if any(tag is None or int(tag) not in batch.visible_episode_indices for tag in tags):
            raise RuntimeError("SAC buffer contains an untagged or invisible episode")
        if max(int(tag) for tag in tags if tag is not None) > batch.block_episode:
            raise RuntimeError("SAC buffer contains future episode data")
        model._adaptive_rl_episode_tags = []
        for transition in batch.transitions:
            observation = np.asarray(transition.observation).reshape(
                (1, *np.asarray(transition.observation).shape)
            )
            next_observation = np.asarray(transition.next_observation).reshape(
                (1, *np.asarray(transition.next_observation).shape)
            )
            action = np.asarray(transition.action).reshape((1, -1))
            done = bool(transition.terminated or transition.truncated)
            episode_index = getattr(transition, "episode_index", None)
            if episode_index is None:
                raise RuntimeError("SAC transition is missing its episode index tag")
            replay.add_tagged(
                observation,
                next_observation,
                action,
                np.asarray([transition.reward], dtype=np.float32),
                np.asarray([done], dtype=np.float32),
                infos=[
                    {
                        "TimeLimit.truncated": bool(
                            transition.truncated and not transition.terminated
                        )
                    }
                ],
                episode_index=int(episode_index),
            )
        if replay.size() != count or len(replay.episode_tags) != count:
            raise RuntimeError("tagged adaptation buffer size does not match inserted transitions")
        if replay.episode_tags and max(replay.episode_tags) > batch.block_episode:
            raise RuntimeError("tagged adaptation buffer contains future episode data")

        prior_buffer = model.replay_buffer
        audit_before = _sac_state_fingerprint(model)
        target_before = _module_snapshot(model.critic_target)
        model.replay_buffer = replay
        try:
            with _logger_ready(model), _seeded_update(batch.seed):
                model._adaptive_rl_last_train_call_count = 0
                original_train = model.train

                def tracked_train(*args: Any, **kwargs: Any) -> Any:
                    model._adaptive_rl_last_train_call_count += 1
                    return original_train(*args, **kwargs)

                model.train = tracked_train
                model._adaptive_rl_last_train_call_count = 0
                model.train(
                    gradient_steps=int(model.gradient_steps), batch_size=int(model.batch_size)
                )
                model.train = original_train
                self.last_loss_metrics = _loss_metrics(model)
                model._adaptive_rl_last_loss_metrics = dict(self.last_loss_metrics)
                after_target = _module_snapshot(model.critic_target)
                target_changed = any(
                    not torch.equal(target_before[name], after_target[name])
                    for name in target_before
                )
                if not target_changed:
                    raise RuntimeError("SAC target critic did not update during the block")
                alpha = _sac_alpha(model)
                model._adaptive_rl_last_audit = {
                    "derived_seed": batch.seed,
                    "gradient_steps": int(model.gradient_steps),
                    "batch_size": int(model.batch_size),
                    "buffer_size": replay.size(),
                    "buffer_episode_tags": list(replay.episode_tags),
                    "actor_loss": _metric_by_suffix(self.last_loss_metrics, "actor_loss"),
                    "critic_loss": _metric_by_suffix(self.last_loss_metrics, "critic_loss"),
                    "alpha": alpha,
                    "alpha_loss": _metric_by_suffix(self.last_loss_metrics, "ent_coef_loss"),
                    "target_network_update_count": int(model.gradient_steps),
                    "tau": float(model.tau),
                    "fingerprint_before": audit_before,
                    "fingerprint_after": _sac_state_fingerprint(model),
                    "fingerprint_components": [
                        "actor", "critics", "target_critics", "temperature"
                    ],
                    "optimizer_fingerprints": _optimizer_fingerprint(model),
                }
        finally:
            if "original_train" in locals():
                model.train = original_train
            model.replay_buffer = prior_buffer
            model.policy.set_training_mode(False)


@dataclass(frozen=True)
class AdaptationUpdateLog:
    block_episode: int
    update_seed: int
    visible_episode_indices: tuple[int, ...]
    transition_count: int
    fingerprint_before: str
    fingerprint_after: str
    parameter_delta_l2: float
    status: str
    loss_metrics: dict[str, float]
    sac_audit: dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _loss_metrics(model: Any) -> dict[str, float]:
    logger = getattr(model, "logger", None)
    recorded = getattr(logger, "name_to_value", {}) if logger is not None else {}
    metrics = {
        str(name): float(value)
        for name, value in recorded.items()
        if "loss" in str(name).lower()
        and isinstance(value, (int, float, np.number))
        and np.isfinite(value)
    }
    if not metrics:
        raise RuntimeError("adaptation update produced no finite loss metrics")
    return metrics


def _parameters(model_or_wrapper: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in policy_state_tensors(model_or_wrapper).items()
    }


def _validate_update(model_or_wrapper: Any, before: dict[str, torch.Tensor]) -> float:
    model = getattr(model_or_wrapper, "model", model_or_wrapper)
    after = policy_state_tensors(model_or_wrapper)
    policy = getattr(model, "policy", model)
    parameter_names = {f"policy.{name}" for name, _ in policy.named_parameters()}
    if isinstance(getattr(model, "log_ent_coef", None), torch.Tensor):
        parameter_names.add("algorithm.log_ent_coef")
    if after.keys() != before.keys():
        raise RuntimeError("Update changed the model state structure")
    squared_delta = 0.0
    for name, tensor in after.items():
        if tensor.shape != before[name].shape or tensor.dtype != before[name].dtype:
            raise RuntimeError(f"Update changed model tensor shape or dtype: {name}")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Update produced non-finite model state: {name}")
        if name in parameter_names:
            difference = tensor.detach().cpu().to(torch.float64) - before[name].to(torch.float64)
            squared_delta += float(torch.sum(difference * difference))
    for name, optimizer in _optimizers(model):
        _validate_finite_tree(optimizer.state_dict(), f"optimizer {name}")
        for parameter, state in optimizer.state.items():
            for state_name, state_value in state.items():
                if isinstance(state_value, torch.Tensor):
                    if state_value.numel() > 1 and state_value.shape != parameter.shape:
                        raise RuntimeError(
                            f"Update corrupted optimizer state shape: {name}.{state_name}"
                        )
    return float(np.sqrt(squared_delta))


def _optimizers(model: Any) -> list[tuple[str, Any]]:
    result = []
    for name, value in vars(model).items():
        if isinstance(value, torch.optim.Optimizer):
            result.append((name, value))
    policy_optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    if isinstance(policy_optimizer, torch.optim.Optimizer):
        result.append(("policy.optimizer", policy_optimizer))
    for owner_name in ("actor", "critic"):
        owner = getattr(model, owner_name, None)
        optimizer = getattr(owner, "optimizer", None)
        if isinstance(optimizer, torch.optim.Optimizer):
            result.append((f"{owner_name}.optimizer", optimizer))
    ent_optimizer = getattr(model, "ent_coef_optimizer", None)
    if isinstance(ent_optimizer, torch.optim.Optimizer):
        result.append(("ent_coef_optimizer", ent_optimizer))
    return result


def _validate_finite_tree(value: Any, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Update produced non-finite {path}")
    elif isinstance(value, dict):
        for key, nested in value.items():
            _validate_finite_tree(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_finite_tree(nested, f"{path}[{index}]")
    elif isinstance(value, (float, int)) and not np.isfinite(value):
        raise FloatingPointError(f"Update produced non-finite {path}")


class _ValidatedAdapter:
    def __init__(self, delegate: AdaptationAdapter, before: dict[str, torch.Tensor]) -> None:
        self.delegate = delegate
        self.before = before

    def update(self, algorithm: Any, batch: UpdateBatch) -> None:
        self.delegate.update(algorithm, batch)
        _validate_update(algorithm, self.before)


def run_adaptation_update(
    algorithm: Any, adapter: AdaptationAdapter, batch: UpdateBatch
) -> AdaptationUpdateLog:
    """Execute one seeded block atomically and return its auditable diagnostics."""
    before_state = _parameters(algorithm)
    before_fingerprint = model_fingerprint(algorithm)
    call_update_atomically(algorithm, _ValidatedAdapter(adapter, before_state), batch)
    delta = _validate_update(algorithm, before_state)
    after_fingerprint = model_fingerprint(algorithm)
    return AdaptationUpdateLog(
        block_episode=batch.block_episode,
        update_seed=batch.seed,
        visible_episode_indices=batch.visible_episode_indices,
        transition_count=len(batch.transitions),
        fingerprint_before=before_fingerprint,
        fingerprint_after=after_fingerprint,
        parameter_delta_l2=delta,
        status="updated" if delta > 0.0 else "no_parameter_change",
        loss_metrics=dict(getattr(adapter, "last_loss_metrics", {})),
    )


__all__ = [
    "AdaptationUpdateLog",
    "PPOAdaptationAdapter",
    "SACAdaptationAdapter",
    "run_adaptation_update",
]
