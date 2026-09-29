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
from typing import Any, Dict, List, cast

import gymnasium as gym
import numpy as np
import pytest
import torch

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
from adaptive_rl.protocol.seed_schedule import derive_seed as derive_canonical_seed
from adaptive_rl.protocol.seeds import (
    PHASES,
    ScheduleValidationError,
    iter_schedule_values,
    validate_replicate_schedules,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _expected_schedule() -> Dict[int, Dict[str, List[int]]]:
    return {
        seed: {phase: list(values) for phase, values in phases.items()}
        for seed, phases in build_schedule().items()
    }


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
        "from adaptive_rl.protocol import derive_seed, frozen_schedule, schedule_fingerprint;"
        "print(derive_seed(31001, 'pre', 1), derive_seed(31001, 'post', 15),"
        " derive_seed(31010, 'update', 9), schedule_fingerprint(frozen_schedule()))"
    )
    outputs = set()
    for hash_seed in ("0", "1234"):
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
    assert outputs == {
        "1280372827 23901095 1218896240 "
        "65939167572731c99599c382ac50cf3fddbba3cf758305764392f13b2e4efa67"
    }


def test_schedule_shape_and_phase_counts() -> None:
    schedule = _expected_schedule()
    assert set(schedule) == set(TRAINING_SEEDS)
    for seed in TRAINING_SEEDS:
        assert len(schedule[seed]["pre"]) == K_PRE == 15
        assert len(schedule[seed]["post"]) == N_POST == 15
        assert len(schedule[seed]["update"]) == N_UPDATE == 10


def test_built_schedule_is_immutable() -> None:
    schedule = build_schedule()
    with pytest.raises(TypeError):
        schedule[31001]["pre"] = (1,)  # type: ignore[index]
    with pytest.raises(TypeError):
        schedule[31001]["pre"][0] = 1  # type: ignore[index]


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
    schedule = _expected_schedule()
    reordered = {
        seed: {phase: schedule[seed][phase] for phase in reversed(PHASES)}
        for seed in reversed(TRAINING_SEEDS)
    }
    assert schedule_fingerprint(reordered) == expected


def test_schedule_fingerprint_rejects_non_integer_values() -> None:
    schedule = _expected_schedule()
    cast(Any, schedule[31001]["pre"])[0] = 1.5
    with pytest.raises(ValueError, match="derived seed must be an integer"):
        schedule_fingerprint(schedule)


def test_derived_python_int_seeds_work_with_supported_rngs() -> None:
    value = derive_canonical_seed(31001, "eval_pre", 1)
    assert type(value) is int
    assert 0 <= value <= SEED_VALUE_MAX
    np.random.default_rng(value)
    torch.Generator(device="cpu").manual_seed(value)
    gym.spaces.Discrete(3).seed(value)


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


def test_descriptive_domains_preserve_preregistered_seed_values() -> None:
    assert derive_seed(31001, "eval_pre", 1) == derive_seed(31001, "pre", 1)
    assert derive_seed(31001, "eval_post", 1) == derive_seed(31001, "post", 1)
    assert derive_seed(31001, "train", 0) == 31001


def test_canonical_api_rejects_legacy_phase_names() -> None:
    with pytest.raises(ValueError, match="phase/domain"):
        derive_canonical_seed(31001, "pre", 1)


@pytest.mark.parametrize(
    ("training_seed", "phase", "index"),
    [
        (True, "pre", 1),
        (31001, "pre", True),
        (31001, "pre", 1.0),
        (31001, "eval_pre", -1),
        (31001, "eval_post", 16),
        (31001, "train", 1),
    ],
)
def test_derive_seed_rejects_bool_float_negative_and_invalid_train_index(
    training_seed: int, phase: str, index: int
) -> None:
    with pytest.raises(ValueError):
        derive_seed(training_seed, phase, index)


def test_schedule_validation_error_reports_cross_replicate_collision() -> None:
    schedule = _expected_schedule()
    schedule[31002]["pre"][0] = schedule[31001]["post"][0]
    with pytest.raises(ScheduleValidationError, match="training_seed=31001.*training_seed=31002"):
        validate_schedule(schedule)


def test_separate_replicate_schedules_validate_cross_replicate_collisions() -> None:
    first = _expected_schedule()
    second = _expected_schedule()
    first = {31001: first[31001]}
    second = {31002: second[31002]}
    second[31002]["pre"][0] = first[31001]["post"][0]

    with pytest.raises(ScheduleValidationError, match="training_seed=31001.*training_seed=31002"):
        validate_replicate_schedules([first, second])


def test_research_runner_uses_schedule_before_side_effects() -> None:
    import ast

    runner_path = REPO_ROOT / "src" / "adaptive_rl" / "benchmarking" / "adaptation_runner.py"
    source = runner_path.read_text(encoding="utf-8")
    parsed = ast.parse(source)
    function = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_adaptation_benchmark"
    )
    calls: list[tuple[str, int]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id in {"frozen_schedule", "mkdir", "_run_replicate"}:
            calls.append((node.func.id, node.lineno))
    schedule_line = min(line for name, line in calls if name == "frozen_schedule")
    assert all(schedule_line < line for name, line in calls if name in {"mkdir", "_run_replicate"})
    assert "base_seed +" not in source
