# Research Seed Policy

This policy documents the executable seed schedule for the preregistered AdaptiveRL
shift study. The normative seed values and derivation remain those frozen in §14 of
[`adaptive_rl_hypothesis.md`](adaptive_rl_hypothesis.md). The implementation is
[`src/adaptive_rl/protocol/seed_schedule.py`](../../src/adaptive_rl/protocol/seed_schedule.py),
with tests in `tests/test_protocol_seed_schedule.py`.

## Replicates and domains

The ten master seeds `31001` through `31010` identify the ten replicates. The master
seed is the training seed itself: `derive_seed(master_seed, "train", 0)` returns that
same integer. Training setup, including Python, NumPy, Torch, the environment, and the
Stable-Baselines3 model, uses this replicate seed.

The public schedule API accepts four domains:

| Domain | Index range | Use |
| --- | --- | --- |
| `train` | `0` | The master seed, unchanged, for replicate training. |
| `eval_pre` | `1..15` | Fifteen nominal pre-shift episodes. |
| `eval_post` | `1..15` | Fifteen post-shift episodes. |
| `update` | `0..9` | Update blocks B5 through B14, where index `k - 5` maps to block B`k`. |

The schedule's serialized keys stay `pre`, `post`, and `update` to preserve the
preregistered fingerprint and existing artifacts. The API names `eval_pre` and
`eval_post` map to the frozen formula tokens `pre` and `post`; the `seeds.py` import
path remains a compatibility wrapper for callers using the old `pre` and `post` names.
The `train` domain is the replicate master seed and is not a fourth derived list.

## Derivation

For evaluation and update seeds, the implementation is exactly the preregistered
equation:

```text
payload = f"{master_seed}|{token}|{index}".encode("utf-8")
digest = SHA256(payload)
seed = int.from_bytes(digest[:4], byteorder="big", signed=False) & 0x7FFFFFFF
```

Here `token` is `pre` for `eval_pre`, `post` for `eval_post`, and `update` for `update`.
This produces a Python integer in `[0, 2**31 - 1]`, accepted by NumPy, Torch,
Gymnasium, Stable-Baselines3, and Python's `random` module. The construction uses no
Python `hash()`, RNG state, platform endianness, or dependency version behavior.
Inputs must be integers (booleans are rejected); unknown domains and out-of-range
indices raise `ValueError`.

Example for master seed `31001`:

| Call | Seed |
| --- | ---: |
| `derive_seed(31001, "train", 0)` | `31001` |
| `derive_seed(31001, "eval_pre", 1)` | `1280372827` |
| `derive_seed(31001, "eval_post", 1)` | `682207530` |
| `derive_seed(31001, "eval_post", 5)` | `1555852046` |
| `derive_seed(31001, "eval_post", 6)` | `1401343121` |
| `derive_seed(31001, "update", 0)` | `36912020` |

## Shock window and later evaluation

The shared shock window is post-shift episodes 1–5. It uses `eval_post` indices 1–5.
The adaptive and fixed arms then evaluate episodes 6–15 using `eval_post` indices
6–15; both arms receive the same seeds. The shock window does not need a separate
domain. Update B5 follows shock episode 5 and uses update index 0; B14 uses index 9.

## Validation and fingerprint

`build_schedule()` returns a nested immutable mapping. `validate_schedule()` checks
the expected replicate set, schedule shape, exact derived values, supported value
range, absence of collisions across every domain and every replicate, disjointness
from master seeds, and disjointness from the preregistered environment seed pools.
It raises `ScheduleValidationError` with both colliding replicate/domain/index
locations. Research runners call `frozen_schedule()` before creating environments or
writing run artifacts, so a validation failure aborts first.

The fingerprint is SHA-256 over the canonical JSON array of sorted master seeds and
the ordered `pre`, `post`, and `update` integer lists, serialized with compact JSON
separators. Recompute it with:

```python
from adaptive_rl.protocol.seed_schedule import frozen_schedule, schedule_fingerprint

print(schedule_fingerprint(frozen_schedule()))
```

For the preregistered ten-replicate schedule it is
`65939167572731c99599c382ac50cf3fddbba3cf758305764392f13b2e4efa67`.
The preregistered benchmark's immutable study manifest and completed artifact
manifest record the schedule, fingerprint, successful validation, and schedule
module version (`1.0`).

## Research and ad-hoc runs

Issue #271 preregistered runs use `run_adaptation_benchmark` with the `--study
prereg-v1` entry point. That runner validates the complete frozen schedule before
starting. Smoke runs also validate the schedule and are labeled as software checks,
not research results. The generic `Evaluator` has a `research_mode=True` guard that
rejects `base_seed + episode` and split-based fallback; research callers must supply
explicit schedule seeds plus their master seed and `eval_pre`/`eval_post` domain, which
the evaluator checks.

Environment inspection, GUI demos, throughput measurement, generic training, and the
separate reward ablation utility are exploratory/demo paths. Their legacy seeds are
not evidence for the preregistered study. Do not use their outputs as Issue #271
research results.

Pre-#274 outputs from ad-hoc seed paths or without recorded schedule validation are
not reproducible under the research enforcement policy. Existing preregistered seed
values and the frozen fingerprint are unchanged by #274.
