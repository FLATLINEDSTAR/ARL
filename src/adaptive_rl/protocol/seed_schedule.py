"""Deterministic seed schedule for the preregistered AdaptiveRL shift protocol.

Implements the normative seed protocol of
``docs/research/adaptive_rl_hypothesis.md`` (§ Seed Protocol):

* ``derive_seed`` is a stable SHA-256 construction — deliberately **not**
  Python's built-in ``hash()``, which is randomized per process for strings
  and tuples and therefore unusable for reproducibility.
* Derived values are masked into ``[0, 2**31 - 1]`` so every seeding entry
  point used by this repository (``np.random.seed``, ``torch.manual_seed``,
  gymnasium ``env.reset(seed=...)``, ``random.seed``) accepts them.
* Collision handling is fail-fast: ``validate_schedule`` raises on any
  duplicate instead of silently re-drawing.

The schedule is derived purely from ``(training_seed, phase, index)``, so an
independent researcher can reconstruct the exact 400-value schedule from the
document alone.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence

from adaptive_rl.protocol.constants import (
    CONFIG_TEST_POOL,
    CONFIG_TRAIN_POOL,
    K_PRE,
    N_POST,
    N_UPDATE,
    SEED_VALUE_MAX,
    TRAINING_SEEDS,
)

#: Frozen serialization keys retained for compatibility with preregistered artifacts.
PHASES: tuple[str, ...] = ("pre", "post", "update")
DOMAINS: tuple[str, ...] = ("train", "eval_pre", "eval_post", "update")
SEED_SCHEDULE_VERSION = "1.0"


class ScheduleValidationError(ValueError):
    """Raised when a preregistered seed schedule is malformed or collides."""


class LegacySeedError(ValueError):
    """Raised when a legacy sequential seed fallback is requested in research mode."""


class _FrozenDict(dict[Any, Any]):
    """JSON-compatible immutable mapping whose values can be safely shared."""

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("seed schedule is immutable")

    def __delitem__(self, key: Any) -> None:
        raise TypeError("seed schedule is immutable")

    def clear(self) -> None:
        raise TypeError("seed schedule is immutable")

    def pop(self, key: Any, default: Any = None) -> Any:
        raise TypeError("seed schedule is immutable")

    def popitem(self) -> tuple[Any, Any]:
        raise TypeError("seed schedule is immutable")

    def setdefault(self, key: Any, default: Any = None) -> Any:
        raise TypeError("seed schedule is immutable")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("seed schedule is immutable")

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        return self


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer (bool is not accepted), got {value!r}")
    return value


def derive_seed(training_seed: int, phase: str, index: int) -> int:
    """Derive one schedule seed from a training seed, phase, and index.

    Canonical construction (normative):

    1. UTF-8 encode the ASCII string ``"{training_seed}|{phase}|{index}"``.
    2. SHA-256 the encoded bytes.
    3. Take the first 4 digest bytes in big-endian order.
    4. Mask with ``0x7FFFFFFF`` (31-bit output in ``[0, 2**31 - 1]``).

    Args:
        training_seed: One of the frozen ``TRAINING_SEEDS`` values.
        phase: ``"train"`` (index 0, returning the replicate master seed),
            ``"eval_pre"`` (j = 1..K_PRE), ``"eval_post"`` (j = 1..N_POST), or
            ``"update"`` (zero-based block index 0..N_UPDATE-1).
        index: Phase position of the seed.

    Returns:
        Deterministic 31-bit integer seed, identical on every platform and
        in every process.

    Raises:
        ValueError: If ``training_seed`` is not a preregistered training seed,
            ``phase`` is unknown, or ``index`` is outside the frozen range.
    """
    training_seed = _require_int(training_seed, "training_seed")
    index = _require_int(index, "index")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(
            f"training_seed {training_seed!r} is not one of the preregistered "
            f"TRAINING_SEEDS {list(TRAINING_SEEDS)}"
        )
    if not isinstance(phase, str):
        raise ValueError(f"domain must be a string from {DOMAINS}, got {phase!r}")
    if phase == "train":
        if index != 0:
            raise ValueError(f"train index must be 0, got {index}")
        return training_seed
    if phase not in DOMAINS:
        raise ValueError(f"phase/domain must be one of {DOMAINS}, got {phase!r}")
    token = {"eval_pre": "pre", "eval_post": "post"}.get(phase, phase)
    if phase == "eval_pre" and not 1 <= index <= K_PRE:
        raise ValueError(f"pre-shift index must be in [1, {K_PRE}], got {index}")
    if phase == "eval_post" and not 1 <= index <= N_POST:
        raise ValueError(f"post-shift index must be in [1, {N_POST}], got {index}")
    if phase == "update" and not 0 <= index < N_UPDATE:
        raise ValueError(f"update-block index must be in [0, {N_UPDATE}), got {index}")

    payload = f"{training_seed}|{token}|{index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return value & SEED_VALUE_MAX


def build_schedule(
    training_seeds: Sequence[int] = TRAINING_SEEDS,
) -> Mapping[int, Mapping[str, tuple[int, ...]]]:
    """Build the full per-replicate seed schedule.

    Args:
        training_seeds: Training seeds to derive for (defaults to the frozen
            preregistered list).

    Returns:
    Immutable mapping ``{training_seed: {"pre": (...), "post": (...), "update": (...)}}``
        with ``K_PRE`` pre-shift seeds (1-based), ``N_POST`` post-shift seeds
        (1-based), and ``N_UPDATE`` update-block seeds (0-based) per replicate.
    """
    schedule: Dict[int, Mapping[str, tuple[int, ...]]] = {}
    seed_values = [_require_int(seed_value, "training_seed") for seed_value in training_seeds]
    if len(set(seed_values)) != len(seed_values):
        raise ScheduleValidationError(f"training seeds are not distinct: {seed_values}")
    for seed in seed_values:
        schedule[seed] = _FrozenDict(
            {
                "pre": tuple(derive_seed(seed, "eval_pre", j) for j in range(1, K_PRE + 1)),
                "post": tuple(derive_seed(seed, "eval_post", j) for j in range(1, N_POST + 1)),
                "update": tuple(derive_seed(seed, "update", b) for b in range(N_UPDATE)),
            }
        )
    return _FrozenDict(schedule)


def iter_schedule_values(schedule: Mapping[int, Mapping[str, Sequence[int]]]) -> List[int]:
    """Flatten a schedule into the ordered list of every derived seed value."""
    values: List[int] = []
    for seed in sorted(schedule):
        for phase in PHASES:
            values.extend(int(v) for v in schedule[seed][phase])
    return values


def validate_schedule(
    schedule: Mapping[int, Mapping[str, Sequence[int]]],
    training_seeds: Sequence[int] = TRAINING_SEEDS,
) -> None:
    """Validate every disjointness and range property required by the protocol.

    Checks, raising ``ValueError`` with an explicit message on the first
    violation class found (all violation classes are collected and reported
    together):

    1. Every derived value lies in ``[0, 2**31 - 1]``.
    2. All derived values across the whole schedule are unique (no collisions
       anywhere — this simultaneously guarantees pre/post disjointness and
       per-phase uniqueness).
    3. No derived value equals any training seed (train/eval disjointness).
    4. No derived value falls inside the config-declared environment seed
       pools ([1000, 1015) and [2000, 2015)).
    5. Training seeds are themselves pairwise distinct.

    Args:
        schedule: Schedule produced by :func:`build_schedule`.
        training_seeds: Training seeds the schedule is built from.

    Raises:
        ValueError: If any protocol property is violated.
    """
    errors: List[str] = []

    try:
        seeds = [_require_int(seed, "training seed") for seed in training_seeds]
        schedule_seeds = [_require_int(seed, "schedule training seed") for seed in schedule]
    except ValueError as exc:
        raise ScheduleValidationError(str(exc)) from exc
    if len(set(seeds)) != len(seeds):
        errors.append(f"training seeds are not distinct: {seeds}")

    if set(schedule_seeds) != set(seeds):
        errors.append(
            f"schedule replicate keys {sorted(schedule_seeds)} do not match training seeds "
            f"{sorted(seeds)}"
        )

    expected_lengths = {"pre": K_PRE, "post": N_POST, "update": N_UPDATE}
    seen: Dict[int, tuple[int, str, int]] = {}
    collisions: List[str] = []
    for seed in sorted(schedule_seeds):
        unknown = set(schedule[seed]) - set(PHASES)
        missing = set(PHASES) - set(schedule[seed])
        if unknown:
            errors.append(f"training_seed={seed} has unknown schedule domains {sorted(unknown)}")
        if missing:
            errors.append(f"training_seed={seed} is missing schedule domains {sorted(missing)}")
        for phase in PHASES:
            values = schedule[seed].get(phase, ())
            if len(values) != expected_lengths[phase]:
                errors.append(
                    f"training_seed={seed}, domain={phase} requires {expected_lengths[phase]} "
                    f"seeds, got {len(values)}"
                )
            for position, value in enumerate(values):
                try:
                    value = _require_int(value, "derived seed")
                except ValueError as exc:
                    errors.append(f"training_seed={seed}, domain={phase}, index={position}: {exc}")
                    continue
                if not 0 <= value <= SEED_VALUE_MAX:
                    errors.append(
                        f"derived seed {value} (training_seed={seed}, domain={phase}, "
                        f"index={position}) outside [0, {SEED_VALUE_MAX}]"
                    )
                index = position if phase == "update" else position + 1
                label = f"training_seed={seed}, domain={phase}, index={index}"
                try:
                    domain = {"pre": "eval_pre", "post": "eval_post"}.get(phase, phase)
                    expected = derive_seed(seed, domain, index)
                except ValueError as exc:
                    errors.append(f"{label}: {exc}")
                    expected = value
                if value != expected:
                    errors.append(
                        f"derived seed {value} (training_seed={seed}, domain={phase}, "
                        f"index={index}) does not match preregistered value {expected}"
                    )
                if value in seen:
                    prior_seed, prior_phase, prior_index = seen[value]
                    collisions.append(
                        f"seed {value}: training_seed={prior_seed}, domain={prior_phase}, "
                        f"index={prior_index} collides with {label}"
                    )
                else:
                    seen[value] = (seed, phase, index)
                if value in seeds:
                    errors.append(f"derived seed {value} ({label}) collides with a training seed")
                if value in CONFIG_TRAIN_POOL or value in CONFIG_TEST_POOL:
                    errors.append(
                        f"derived seed {value} ({label}) collides with a config "
                        "environment seed pool"
                    )
    if collisions:
        errors.append(f"{len(collisions)} schedule collision(s): {collisions[:5]}")
    if errors:
        raise ScheduleValidationError("seed schedule violates the protocol: " + "; ".join(errors))


def validate_replicate_schedules(
    schedules: Sequence[Mapping[int, Mapping[str, Sequence[int]]]],
) -> None:
    """Validate disjointness across separately supplied replicate schedules."""
    combined: Dict[int, Mapping[str, Sequence[int]]] = {}
    for schedule in schedules:
        for seed, domains in schedule.items():
            seed = _require_int(seed, "schedule training seed")
            if seed in combined:
                raise ScheduleValidationError(
                    f"duplicate replicate schedule for training_seed={seed}"
                )
            combined[seed] = domains
    if not combined:
        raise ScheduleValidationError("at least one replicate schedule is required")
    validate_schedule(combined, training_seeds=tuple(sorted(combined)))


def schedule_fingerprint(schedule: Mapping[int, Mapping[str, Sequence[int]]]) -> str:
    """SHA-256 of the canonical JSON serialization of a schedule.

    Both treatment arms must record this fingerprint; equality of the
    fingerprints is the artifact-level evidence that Adaptive and Fixed used
    identical evaluation and update seed schedules.
    """
    seed_values = [_require_int(seed, "schedule training seed") for seed in schedule]
    canonical_schedule = []
    for seed in sorted(seed_values):
        domains = schedule[seed]
        if set(domains) != set(PHASES):
            raise ValueError(f"schedule for training_seed={seed} must contain {PHASES}")
        canonical_domains = {
            phase: [_require_int(value, "derived seed") for value in domains[phase]]
            for phase in PHASES
        }
        canonical_schedule.append([seed, canonical_domains])
    canonical = json.dumps(
        canonical_schedule,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def frozen_schedule() -> Mapping[int, Mapping[str, tuple[int, ...]]]:
    """Build and validate the preregistered schedule; return it.

    Convenience entry point for artifacts and tests: any schedule returned by
    this function has already passed :func:`validate_schedule`.
    """
    schedule = build_schedule()
    validate_schedule(schedule)
    return schedule


__all__ = [
    "PHASES",
    "DOMAINS",
    "LegacySeedError",
    "ScheduleValidationError",
    "SEED_SCHEDULE_VERSION",
    "build_schedule",
    "derive_seed",
    "frozen_schedule",
    "iter_schedule_values",
    "schedule_fingerprint",
    "validate_replicate_schedules",
    "validate_schedule",
]
