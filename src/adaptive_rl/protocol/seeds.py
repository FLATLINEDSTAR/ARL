"""Backward-compatible imports for the canonical protocol seed schedule."""

from adaptive_rl.protocol.seed_schedule import (
    DOMAINS,
    PHASES,
    SEED_SCHEDULE_VERSION,
    LegacySeedError,
    ScheduleValidationError,
    build_schedule,
    frozen_schedule,
    iter_schedule_values,
    schedule_fingerprint,
    validate_replicate_schedules,
    validate_schedule,
)
from adaptive_rl.protocol.seed_schedule import (
    derive_seed as _derive_seed,
)


def derive_seed(training_seed: int, phase: str, index: int) -> int:
    """Compatibility wrapper accepting the pre-v1 ``pre``/``post`` aliases."""
    phase = {"pre": "eval_pre", "post": "eval_post"}.get(phase, phase)
    return _derive_seed(training_seed, phase, index)


__all__ = [
    "DOMAINS",
    "LegacySeedError",
    "PHASES",
    "SEED_SCHEDULE_VERSION",
    "ScheduleValidationError",
    "build_schedule",
    "derive_seed",
    "frozen_schedule",
    "iter_schedule_values",
    "schedule_fingerprint",
    "validate_replicate_schedules",
    "validate_schedule",
]
