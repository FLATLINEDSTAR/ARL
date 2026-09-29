"""Frozen constants of the preregistered AdaptiveRL shift protocol (Issue #98 / PR #168).

Every value in this module is a protocol-level preregistered constant. The
normative prose definition lives in ``docs/research/adaptive_rl_hypothesis.md``;
this module is the executable mirror of those values so that tests can enforce
that the document, the seed schedule, the recovery metric, and the statistical
analysis never drift apart.

Changing any constant here changes the protocol and requires a new
``PROTOCOL_VERSION`` plus a matching document revision.
"""

from __future__ import annotations

#: Version of the preregistered protocol mirrored by this package.
PROTOCOL_VERSION: str = "2.0"

#: Repository commit every "verified" claim in the protocol document is pinned to.
PINNED_COMMIT: str = "faefc5c8e4a39bbcc728d73b1c9855c8e9c5386f"

#: Family-wise significance level for one-sided primary tests.
ALPHA: float = 0.05

#: The ten preregistered training-seed integers (one independent replicate each).
#: Distinct from every environment seed pool declared in the pinned configs
#: (TRAIN pool [1000, 1015), TEST pool [2000, 2015)) and small enough for
#: ``np.random.seed`` (< 2**32) and ``torch.manual_seed`` (< 2**63).
TRAINING_SEEDS: tuple[int, ...] = (
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
)

#: Number of pre-shift (nominal) evaluation episodes per replicate (K_pre).
K_PRE: int = 15

#: Number of post-shift episodes per replicate; also the recovery horizon H.
N_POST: int = 15

#: Number of post-shift adaptation update blocks (after episodes 5..14).
N_UPDATE: int = 10

#: Recovery horizon H in completed post-shift episodes.
HORIZON: int = 15

#: Trailing-window size (episodes) for P(t) and P0.
WINDOW: int = 5

#: Consecutive windows required above threshold to confirm recovery.
PERSISTENCE: int = 3

#: Recovery threshold on R(t); checked as 10.0 * (p_t - p0) >= 9.0 * degradation.
RECOVERY_THRESHOLD: float = 0.9

#: Numerator/denominator of RECOVERY_THRESHOLD expressed as an exact ratio 9/10.
#: The cross-multiplied predicate avoids a floating-point division.
THRESHOLD_CROSS_NUMERATOR: float = 10.0
THRESHOLD_CROSS_DENOMINATOR: float = 9.0

#: Minimum degradation for normalization: delta_min = MULTIPLIER * SE(delta).
MIN_DEGRADATION_SE_MULTIPLIER: float = 2.0

#: A cell produces a primary p-value only with at least this many
#: pairwise-complete (both arms evaluable) replicate pairs.
MIN_VALID_N: int = 8

#: Bootstrap sensitivity configuration (percentile CI on the mean difference).
BOOTSTRAP_REPS: int = 10000
BOOTSTRAP_SEED: int = 168098

#: The six preregistered primary cells (environment/algorithm), in fixed order.
PRIMARY_CELLS: tuple[str, ...] = (
    "gridworld/ppo",
    "traffic_signal/ppo",
    "drone_disturbed/ppo",
    "drone_disturbed/sac",
    "navigation_2d/ppo",
    "navigation_2d/sac",
)

#: TRAIN environment seed pool declared in configs/drone_distribution_shift.yaml
#: at the pinned commit (scenario seeds [1000, 1015)).
CONFIG_TRAIN_POOL: tuple[int, ...] = tuple(range(1000, 1015))

#: TEST environment seed pool declared in configs/drone_distribution_shift.yaml
#: at the pinned commit (scenario seeds [2000, 2015)).
CONFIG_TEST_POOL: tuple[int, ...] = tuple(range(2000, 2015))

#: All derived seeds are masked into [0, SEED_VALUE_MAX] so they are accepted by
#: numpy's legacy ``np.random.seed`` (< 2**32), torch (< 2**63), gymnasium and
#: Python's ``random``.
SEED_VALUE_MAX: int = 0x7FFFFFFF

#: Number of planned replicate pairs per cell (len(TRAINING_SEEDS)).
PLANNED_N: int = len(TRAINING_SEEDS)

# Issue #271 prereg-v1 freezes one PPO cell and its semantic config/treatment.
# These hashes are declared in docs/research/issue-271.md and require a new
# study version if either scientific input changes.
ISSUE271_CONFIG_SHA256: str = "0039c298b5048254b2d211cc66967e9d3e1d71275575fcfe28736485c65937b5"
ISSUE271_V2_CONFIG_SHA256: str = "be12f2837dc0fbf00a752104e1649b3d5b41d852de2574f29a50a5d155dbafb8"
ISSUE271_TREATMENT_CARD_SHA256: str = (
    "8383e736f02ff31393474b241b06d2dc92b036910c93c26081c1673c92a317e3"
)

__all__ = [
    "ALPHA",
    "BOOTSTRAP_REPS",
    "BOOTSTRAP_SEED",
    "CONFIG_TEST_POOL",
    "CONFIG_TRAIN_POOL",
    "HORIZON",
    "ISSUE271_CONFIG_SHA256",
    "ISSUE271_V2_CONFIG_SHA256",
    "ISSUE271_TREATMENT_CARD_SHA256",
    "K_PRE",
    "MIN_DEGRADATION_SE_MULTIPLIER",
    "MIN_VALID_N",
    "N_POST",
    "N_UPDATE",
    "PINNED_COMMIT",
    "PLANNED_N",
    "PRIMARY_CELLS",
    "PROTOCOL_VERSION",
    "RECOVERY_THRESHOLD",
    "SEED_VALUE_MAX",
    "THRESHOLD_CROSS_DENOMINATOR",
    "THRESHOLD_CROSS_NUMERATOR",
    "TRAINING_SEEDS",
    "WINDOW",
    "PERSISTENCE",
]
