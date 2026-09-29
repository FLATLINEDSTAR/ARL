"""Tests for the preregistered AdaptiveRL seed schedule (Issue #98 / PR #168).

Covers: cross-process determinism, global uniqueness (collision handling),
train/eval disjointness, pre/post disjointness, config-pool disjointness,
range constraints, fail-fast validation, and arm-schedule fingerprint
stability.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from adaptive_rl.protocol import (
    CONFIG_TEST_POOL,
    CONFIG_TRAIN_POOL,
    K_PRE,
    N_POST,
    N_UPDATE,
    PLANNED_N,
    SEED_VALUE_MAX,
    TRAINING_SEEDS,
    build_schedule,
    derive_seed,
    frozen_schedule,
    schedule_fingerprint,
    validate_schedule,
)
from adaptive_rl.protocol.seeds import PHASES, iter_schedule_values

REPO_ROOT = Path(__file__).resolve().parent.parent


def _expected_schedule() -> Dict[int, Dict[str, List[int]]]:
    return build_schedule()


def test_training_seed_list_is_frozen() -> None:
    assert list(TRAINING_SEEDS) == [
        31001,
        31002,
        31003,
        31004,
        31005,
        31006,
        31007,
        31008,
        31009,
        31010,
    ]
    assert PLANNED_N == 10 == len(TRAINING_SEEDS)
    assert len(set(TRAINING_SEEDS)) == len(TRAINING_SEEDS)


def test_derive_seed_matches_frozen_regression_literals() -> None:
    """Frozen digests: SHA-256 is stable across processes and platforms."""
    assert derive_seed(31001, "pre", 1) == 1280372827
    assert derive_seed(31001, "post", 15) == 23901095
    assert derive_seed(31010, "update", 9) == 1218896240


def test_derive_seed_range_and_pure_determinism() -> None:
    for seed in TRAINING_SEEDS:
        for phase, count, start in (
            ("pre", K_PRE, 1),
            ("post", N_POST, 1),
            ("update", N_UPDATE, 0),
        ):
            for index in range(start, start + count):
                value = derive_seed(seed, phase, index)
                assert 0 <= value <= SEED_VALUE_MAX
                assert value == derive_seed(seed, phase, index)


def test_derive_seed_is_stable_across_processes_and_hash_seeds() -> None:
    """The schedule must not depend on PYTHONHASHSEED (unlike builtin hash())."""
    program = (
        "from adaptive_rl.protocol import derive_seed;"
        "print(derive_seed(31001, 'pre', 1), derive_seed(31001, 'post', 15),"
        " derive_seed(31010, 'update', 9))"
    )
    outputs = set()
    for hash_seed in ("0", "424242"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(REPO_ROOT),
        )
        outputs.add(result.stdout.strip())
    assert outputs == {"1280372827 23901095 1218896240"}


def test_schedule_shape_and_phase_counts() -> None:
    schedule = _expected_schedule()
    assert set(schedule) == set(TRAINING_SEEDS)
    for seed in TRAINING_SEEDS:
        assert len(schedule[seed]["pre"]) == K_PRE == 15
        assert len(schedule[seed]["post"]) == N_POST == 15
        assert len(schedule[seed]["update"]) == N_UPDATE == 10


def test_schedule_has_no_collisions_anywhere() -> None:
    schedule = _expected_schedule()
    values = iter_schedule_values(schedule)
    assert len(values) == 10 * (K_PRE + N_POST + N_UPDATE) == 400
    assert len(set(values)) == len(values)


def test_pre_and_post_schedules_are_disjoint() -> None:
    schedule = _expected_schedule()
    for seed in TRAINING_SEEDS:
        pre = set(schedule[seed]["pre"])
        post = set(schedule[seed]["post"])
        update = set(schedule[seed]["update"])
        assert pre.isdisjoint(post)
        assert pre.isdisjoint(update)
        assert post.isdisjoint(update)


def test_derived_seeds_disjoint_from_training_and_config_pools() -> None:
    schedule = _expected_schedule()
    values = set(iter_schedule_values(schedule))
    assert values.isdisjoint(set(TRAINING_SEEDS))
    assert values.isdisjoint(set(CONFIG_TRAIN_POOL))
    assert values.isdisjoint(set(CONFIG_TEST_POOL))
    assert set(TRAINING_SEEDS).isdisjoint(set(CONFIG_TRAIN_POOL) | set(CONFIG_TEST_POOL))


def test_schedule_fingerprint_is_stable_and_sensitive() -> None:
    expected = "65939167572731c99599c382ac50cf3fddbba3cf758305764392f13b2e4efa67"
    assert schedule_fingerprint(_expected_schedule()) == expected
    # Rebuilding the schedule (the stand-in for both arms building theirs
    # independently) must reproduce the identical fingerprint.
    assert schedule_fingerprint(frozen_schedule()) == expected
    mutated = _expected_schedule()
    mutated[31001]["post"][0] += 1
    assert schedule_fingerprint(mutated) != expected


def test_validate_schedule_accepts_frozen_schedule() -> None:
    validate_schedule(_expected_schedule())


def test_validate_schedule_rejects_collisions() -> None:
    schedule = _expected_schedule()
    schedule[31001]["post"][1] = schedule[31001]["post"][0]
    with pytest.raises(ValueError, match="collision"):
        validate_schedule(schedule)


def test_validate_schedule_rejects_out_of_range_values() -> None:
    schedule = _expected_schedule()
    schedule[31001]["pre"][0] = SEED_VALUE_MAX + 1
    with pytest.raises(ValueError, match="outside"):
        validate_schedule(schedule)


def test_validate_schedule_rejects_training_seed_contamination() -> None:
    schedule = _expected_schedule()
    schedule[31001]["pre"][0] = 31001
    with pytest.raises(ValueError, match="training seed"):
        validate_schedule(schedule)


def test_validate_schedule_rejects_config_pool_contamination() -> None:
    schedule = _expected_schedule()
    schedule[31001]["pre"][0] = 1000
    with pytest.raises(ValueError, match="config"):
        validate_schedule(schedule)


def test_derive_seed_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="not one of the preregistered"):
        derive_seed(12345, "pre", 1)
    with pytest.raises(ValueError, match="phase"):
        derive_seed(31001, "eval", 1)
    with pytest.raises(ValueError, match="pre-shift index"):
        derive_seed(31001, "pre", 0)
    with pytest.raises(ValueError, match="pre-shift index"):
        derive_seed(31001, "pre", K_PRE + 1)
    with pytest.raises(ValueError, match="post-shift index"):
        derive_seed(31001, "post", N_POST + 1)
    with pytest.raises(ValueError, match="update-block index"):
        derive_seed(31001, "update", N_UPDATE)
    with pytest.raises(ValueError, match="update-block index"):
        derive_seed(31001, "update", -1)


def test_phases_are_frozen() -> None:
    assert PHASES == ("pre", "post", "update")
