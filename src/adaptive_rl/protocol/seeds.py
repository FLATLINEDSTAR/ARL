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
from typing import Dict, List, Mapping, Sequence

from adaptive_rl.protocol.constants import (
    CONFIG_TEST_POOL,
    CONFIG_TRAIN_POOL,
    K_PRE,
    N_POST,
    N_UPDATE,
    SEED_VALUE_MAX,
    TRAINING_SEEDS,
)

#: Allowed derivation phases. ``pre``/``post`` indices are 1-based episode
#: numbers; ``update`` indices are 0-based update-block numbers (block ``b``
#: runs after post-shift episode ``b + 5``).
PHASES: tuple[str, ...] = ("pre", "post", "update")


def derive_seed(training_seed: int, phase: str, index: int) -> int:
    """Derive one schedule seed from a training seed, phase, and index.

    Canonical construction (normative):

    1. UTF-8 encode the ASCII string ``"{training_seed}|{phase}|{index}"``.
    2. SHA-256 the encoded bytes.
    3. Take the first 4 digest bytes in big-endian order.
    4. Mask with ``0x7FFFFFFF`` (31-bit output in ``[0, 2**31 - 1]``).

    Args:
        training_seed: One of the frozen ``TRAINING_SEEDS`` values.
        phase: ``"pre"`` (j = 1..K_PRE), ``"post"`` (j = 1..N_POST), or
            ``"update"`` (block = 0..N_UPDATE-1).
        index: Phase position of the seed.

    Returns:
        Deterministic 31-bit integer seed, identical on every platform and
        in every process.

    Raises:
        ValueError: If ``training_seed`` is not a preregistered training seed,
            ``phase`` is unknown, or ``index`` is outside the frozen range.
    """
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(
            f"training_seed {training_seed!r} is not one of the preregistered "
            f"TRAINING_SEEDS {list(TRAINING_SEEDS)}"
        )
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    if phase == "pre" and not 1 <= index <= K_PRE:
        raise ValueError(f"pre-shift index must be in [1, {K_PRE}], got {index}")
    if phase == "post" and not 1 <= index <= N_POST:
        raise ValueError(f"post-shift index must be in [1, {N_POST}], got {index}")
    if phase == "update" and not 0 <= index < N_UPDATE:
        raise ValueError(f"update-block index must be in [0, {N_UPDATE}), got {index}")

    payload = f"{training_seed}|{phase}|{index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return value & SEED_VALUE_MAX


def build_schedule(
    training_seeds: Sequence[int] = TRAINING_SEEDS,
) -> Dict[int, Dict[str, List[int]]]:
    """Build the full per-replicate seed schedule.

    Args:
        training_seeds: Training seeds to derive for (defaults to the frozen
            preregistered list).

    Returns:
        Mapping ``{training_seed: {"pre": [...], "post": [...], "update": [...]}}``
        with ``K_PRE`` pre-shift seeds (1-based), ``N_POST`` post-shift seeds
        (1-based), and ``N_UPDATE`` update-block seeds (0-based) per replicate.
    """
    schedule: Dict[int, Dict[str, List[int]]] = {}
    for seed in training_seeds:
        schedule[int(seed)] = {
            "pre": [derive_seed(int(seed), "pre", j) for j in range(1, K_PRE + 1)],
            "post": [derive_seed(int(seed), "post", j) for j in range(1, N_POST + 1)],
            "update": [derive_seed(int(seed), "update", b) for b in range(N_UPDATE)],
        }
    return schedule


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

    seeds = [int(s) for s in training_seeds]
    if len(set(seeds)) != len(seeds):
        errors.append(f"training seeds are not distinct: {seeds}")

    seen: Dict[int, str] = {}
    collisions: List[str] = []
    for seed in sorted(schedule):
        for phase in PHASES:
            for position, value in enumerate(schedule[seed][phase]):
                value = int(value)
                if not 0 <= value <= SEED_VALUE_MAX:
                    errors.append(
                        f"derived seed {value} (training_seed={seed}, phase={phase}, "
                        f"index={position}) outside [0, {SEED_VALUE_MAX}]"
                    )
                label = f"training_seed={seed}, phase={phase}, index={position}"
                if value in seen:
                    collisions.append(f"{value}: {seen[value]} collides with {label}")
                else:
                    seen[value] = label
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
        raise ValueError("seed schedule violates the protocol: " + "; ".join(errors))


def schedule_fingerprint(schedule: Mapping[int, Mapping[str, Sequence[int]]]) -> str:
    """SHA-256 of the canonical JSON serialization of a schedule.

    Both treatment arms must record this fingerprint; equality of the
    fingerprints is the artifact-level evidence that Adaptive and Fixed used
    identical evaluation and update seed schedules.
    """
    canonical = json.dumps(
        [
            [seed, {phase: [int(v) for v in schedule[seed][phase]] for phase in PHASES}]
            for seed in sorted(schedule)
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def frozen_schedule() -> Dict[int, Dict[str, List[int]]]:
    """Build and validate the preregistered schedule; return it.

    Convenience entry point for artifacts and tests: any schedule returned by
    this function has already passed :func:`validate_schedule`.
    """
    schedule = build_schedule()
    validate_schedule(schedule)
    return schedule


__all__ = [
    "PHASES",
    "build_schedule",
    "derive_seed",
    "frozen_schedule",
    "iter_schedule_values",
    "schedule_fingerprint",
    "validate_schedule",
]
