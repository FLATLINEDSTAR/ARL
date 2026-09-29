"""Model-state hashing and independent policy forks for paired experiments."""

from __future__ import annotations

import copy
import hashlib
import struct
from typing import Any

import torch


def _state_module(model_or_wrapper: Any) -> Any:
    model = getattr(model_or_wrapper, "model", model_or_wrapper)
    if callable(getattr(model, "state_dict", None)):
        return model
    policy = getattr(model, "policy", None)
    if policy is not None and callable(getattr(policy, "state_dict", None)):
        return policy
    raise TypeError("Expected a model or algorithm wrapper with a state_dict().")


def policy_state_tensors(model_or_wrapper: Any) -> dict[str, Any]:
    model = getattr(model_or_wrapper, "model", model_or_wrapper)
    state = {
        f"policy.{key}": value
        for key, value in _state_module(model_or_wrapper).state_dict().items()
    }
    # SAC's automatic entropy coefficient is a trainable model parameter stored
    # on the algorithm object rather than inside SACPolicy.
    log_ent_coef = getattr(model, "log_ent_coef", None)
    if isinstance(log_ent_coef, torch.Tensor):
        state["algorithm.log_ent_coef"] = log_ent_coef
    return state


def model_fingerprint(model_or_wrapper: Any) -> str:
    """Return a stable SHA-256 fingerprint of a model's complete state dict.

    Stable-Baselines3 wrappers expose their model as ``.model``. Hashing the
    state dict includes parameters and persistent buffers, with names, shapes,
    and dtypes included to distinguish structurally different models.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(policy_state_tensors(model_or_wrapper).items()):
        if not hasattr(tensor, "detach"):
            raise TypeError(f"Model state entry {name!r} is not a tensor.")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(struct.pack("!I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("!Q", int(dimension)))
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def clone_algorithm(algorithm: Any) -> Any:
    """Deep-clone an algorithm wrapper and reject clones with shared tensors.

    ``deepcopy`` is intentional: SB3 algorithms contain mutable optimizers,
    buffers, counters, and RNG state in addition to policy parameters.
    """
    clone = copy.deepcopy(algorithm)
    original_model = getattr(algorithm, "model", None)
    clone_model = getattr(clone, "model", None)
    if original_model is None or clone_model is None:
        raise TypeError("Algorithm and clone must both have initialized models.")
    original_state = policy_state_tensors(algorithm)
    clone_state = policy_state_tensors(clone)
    if original_state.keys() != clone_state.keys():
        raise RuntimeError("Cloned model state has a different structure.")
    for name in original_state:
        left = original_state[name]
        right = clone_state[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"Cloned model state {name!r} has a different shape or dtype.")
        if left.data_ptr() == right.data_ptr():
            raise RuntimeError(f"Cloned model state {name!r} shares storage with its source.")
        if not left.equal(right):
            raise RuntimeError(f"Cloned model state {name!r} differs from its source.")
    return clone


class FrozenPolicy:
    """Prediction-only facade that exposes no training or model mutation API."""

    __slots__ = ("__algorithm", "__initial_state")

    def __init__(self, algorithm: Any) -> None:
        if not callable(getattr(algorithm, "predict", None)):
            raise TypeError("Frozen policy requires an algorithm with predict().")
        self.__algorithm = algorithm
        self.__initial_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in policy_state_tensors(algorithm).items()
        }

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        return self.__algorithm.predict(observation, deterministic=deterministic)

    @property
    def fingerprint(self) -> str:
        return model_fingerprint(self.__algorithm)

    @property
    def parameter_delta_l2(self) -> float:
        """L2 distance from the frozen model state, including SAC temperature."""
        current = policy_state_tensors(self.__algorithm)
        if current.keys() != self.__initial_state.keys():
            raise RuntimeError("Frozen model state structure changed")
        squared_delta = 0.0
        for name, tensor in current.items():
            initial = self.__initial_state[name]
            if tensor.shape != initial.shape or tensor.dtype != initial.dtype:
                raise RuntimeError(f"Frozen model state {name!r} changed shape or dtype")
            delta = tensor.detach().cpu().to(torch.float64) - initial.to(torch.float64)
            squared_delta += float(torch.sum(delta * delta))
        return float(squared_delta**0.5)


def fork_adaptive_and_fixed(algorithm: Any) -> tuple[Any, FrozenPolicy, str]:
    """Return an independent adaptive clone and prediction-only fixed arm."""
    expected = model_fingerprint(algorithm)
    adaptive = clone_algorithm(algorithm)
    fixed = FrozenPolicy(clone_algorithm(algorithm))
    if model_fingerprint(adaptive) != expected or fixed.fingerprint != expected:
        raise RuntimeError("Forked policies do not match the frozen model fingerprint.")
    return adaptive, fixed, expected


__all__ = [
    "FrozenPolicy",
    "clone_algorithm",
    "fork_adaptive_and_fixed",
    "model_fingerprint",
    "policy_state_tensors",
]
