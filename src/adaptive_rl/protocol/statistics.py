"""Statistical analysis primitives for the preregistered AdaptiveRL shift protocol.

Implements the normative analysis plan of
``docs/research/adaptive_rl_hypothesis.md`` (§ Statistical Analysis Plan):

* **Primary test**: one-sample (paired) t-test on the pairwise-complete
  differences ``D_i = T_H(Adaptive) - T_H(Fixed)`` for ``H0: mu_D >= 0`` vs
  ``H1: mu_D < 0``. The Student-t distribution is computed from the
  regularized incomplete beta function (continued fraction), so the protocol
  needs **no scipy dependency** — the repository has none.
* Confidence interval: two-sided t-interval at the *actual* degrees of
  freedom ``N_valid - 1`` (never hard-coded to n = 10).
* Effect size: Cohen's ``d_z`` with explicit SD = 0 edge cases.
* Sensitivities that test *different* targets, clearly labeled in the
  document: exact sign test (median), exact Wilcoxon signed-rank
  (pseudomedian under symmetry), percentile bootstrap CI (approximate).
* Multiplicity: Holm step-down adjustment for individual cell claims, and
  the deterministic intersection-union family decision rule.
* Failure sensitivity: adversarial worst/best imputation of failed arms
  over the closed domain ``[0, H]``.

Every function validates its inputs (finite values, minimum N, exact family
cell set) and raises ``ValueError`` instead of returning undefined numbers.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.protocol.constants import (
    ALPHA,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    MIN_VALID_N,
    PRIMARY_CELLS,
)

#: Deterministic outcome of the six-cell intersection-union family decision.
FamilyDecision = Literal["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"]

#: Admissible imputation directions for failed runs (T_H in [0, H]).
IMPUTATION_DIRECTIONS = ("worst_for_adaptive", "best_for_adaptive")


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _validated_vector(differences: Sequence[float], label: str = "differences") -> List[float]:
    values = [float(v) for v in differences]
    if not values:
        raise ValueError(f"{label} must be non-empty")
    for position, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{label}[{position}] is not finite: {value!r}")
    return values


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(values) / len(values))


def _sample_variance(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        raise ValueError("sample variance requires at least 2 observations")
    mean = _mean(values)
    return float(math.fsum((v - mean) ** 2 for v in values) / (n - 1))


def sample_skewness(values: Sequence[float]) -> float:
    """Sample skewness (Fisher-Pearson coefficient of skewness)."""
    n = len(values)
    if n < 3:
        raise ValueError("skewness requires at least 3 observations")
    mean = _mean(values)
    m2 = math.fsum((v - mean) ** 2 for v in values) / n
    m3 = math.fsum((v - mean) ** 3 for v in values) / n
    if m2 == 0.0:
        return 0.0
    return float(m3 / (m2**1.5))


def sample_kurtosis(values: Sequence[float]) -> float:
    """Sample excess kurtosis (Fisher's definition)."""
    n = len(values)
    if n < 4:
        raise ValueError("kurtosis requires at least 4 observations")
    mean = _mean(values)
    m2 = math.fsum((v - mean) ** 2 for v in values) / n
    m4 = math.fsum((v - mean) ** 4 for v in values) / n
    if m2 == 0.0:
        return 0.0
    return float(m4 / (m2**2) - 3.0)


# ---------------------------------------------------------------------------
# Student-t distribution (pure Python; no scipy)
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta function (Lentz)."""
    max_iterations = 300
    epsilon = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)`` for x in [0, 1]."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"beta parameters must be positive, got a={a}, b={b}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    log_front += a * math.log(x) + b * math.log1p(-x)
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return float(front * _betacf(a, b, x) / a)
    return float(1.0 - front * _betacf(b, a, 1.0 - x) / b)


def student_t_cdf(t: float, degrees_of_freedom: int) -> float:
    """CDF of Student's t distribution with ``degrees_of_freedom`` degrees.

    Validated against published t-table quantiles in the test suite
    (e.g. t_{0.975, 9} = 2.262157, t_{0.95, 9} = 1.833113).
    """
    if degrees_of_freedom < 1:
        raise ValueError(f"degrees_of_freedom must be >= 1, got {degrees_of_freedom}")
    if math.isnan(t):
        raise ValueError("t must not be NaN")
    if t == math.inf:
        return 1.0
    if t == -math.inf:
        return 0.0
    x = degrees_of_freedom / (degrees_of_freedom + t * t)
    tail = 0.5 * _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)
    return float(1.0 - tail) if t >= 0.0 else float(tail)


def _student_t_survival(x: float, degrees_of_freedom: int) -> float:
    """P(T > x) for x >= 0 without subtracting from 1 (precision near the tail)."""
    if x < 0.0:
        return 1.0 - _student_t_survival(-x, degrees_of_freedom)
    ratio = degrees_of_freedom / (degrees_of_freedom + x * x)
    return float(0.5 * _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, ratio))


def student_t_ppf(p: float, degrees_of_freedom: int) -> float:
    """Quantile function of Student's t distribution (monotone bisection)."""
    if degrees_of_freedom < 1:
        raise ValueError(f"degrees_of_freedom must be >= 1, got {degrees_of_freedom}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -student_t_ppf(1.0 - p, degrees_of_freedom)

    target_survival = 1.0 - p
    lo, hi = 0.0, 1.0
    for _ in range(64):
        if _student_t_survival(hi, degrees_of_freedom) <= target_survival:
            break
        hi *= 2.0
    else:  # pragma: no cover - unreachable for protocol-level p values
        raise RuntimeError("failed to bracket t quantile")

    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _student_t_survival(mid, degrees_of_freedom) > target_survival:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# Primary analysis: paired (one-sample) t-test on D_i
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedTTest:
    """Primary one-sample t-test result on pairwise-complete differences.

    Attributes:
        n: Valid paired sample size (``N_valid``).
        degrees_of_freedom: ``n - 1``.
        mean: Mean paired difference (D-bar).
        std_dev: Sample standard deviation of the differences (ddof=1).
        standard_error: ``std_dev / sqrt(n)``.
        t_statistic: ``mean / standard_error``; +/-inf when std_dev == 0.
        p_value: One-sided lower-tail p-value for ``H1: mu_D < 0``.
    """

    n: int
    degrees_of_freedom: int
    mean: float
    std_dev: float
    standard_error: float
    t_statistic: float
    p_value: float


def paired_t_test(
    differences: Sequence[float],
    min_valid_n: int = MIN_VALID_N,
) -> PairedTTest:
    """One-sided paired t-test for ``H0: mu_D >= 0`` vs ``H1: mu_D < 0``.

    Degenerate variance rule (frozen): when ``std_dev == 0`` the t statistic
    is defined as ``-inf`` (mean < 0, p = 0), ``+inf`` (mean > 0, p = 1), or
    ``0`` (mean == 0, p = 0.5).

    Raises:
        ValueError: If ``n < min_valid_n`` (cell not evaluable), fewer than
            two observations, or any value is non-finite.
    """
    values = _validated_vector(differences)
    n = len(values)
    if n < 2:
        raise ValueError(f"paired t-test requires at least 2 pairs, got {n}")
    if n < min_valid_n:
        raise ValueError(f"cell not evaluable: N_valid={n} < MIN_VALID_N={min_valid_n}")

    mean = _mean(values)
    std_dev = math.sqrt(_sample_variance(values))
    if std_dev == 0.0:
        if mean < 0.0:
            t_statistic, p_value = -math.inf, 0.0
        elif mean > 0.0:
            t_statistic, p_value = math.inf, 1.0
        else:
            t_statistic, p_value = 0.0, 0.5
        standard_error = 0.0
    else:
        standard_error = std_dev / math.sqrt(n)
        t_statistic = mean / standard_error
        p_value = student_t_cdf(t_statistic, n - 1)

    return PairedTTest(
        n=n,
        degrees_of_freedom=n - 1,
        mean=float(mean),
        std_dev=float(std_dev),
        standard_error=float(standard_error),
        t_statistic=float(t_statistic),
        p_value=float(p_value),
    )


def paired_t_interval(
    differences: Sequence[float],
    confidence: float = 0.95,
    min_valid_n: int = MIN_VALID_N,
) -> Tuple[float, float]:
    """Two-sided t-interval for the mean paired difference at actual ``n``.

    Uses ``t_{1-(1-confidence)/2, n-1}`` from :func:`student_t_ppf`; the
    degrees of freedom follow the observed ``N_valid`` (never hard-coded).
    """
    values = _validated_vector(differences)
    n = len(values)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n < 2:
        raise ValueError(f"t-interval requires at least 2 pairs, got {n}")
    if n < min_valid_n:
        raise ValueError(f"cell not evaluable: N_valid={n} < MIN_VALID_N={min_valid_n}")
    mean = _mean(values)
    std_dev = math.sqrt(_sample_variance(values))
    standard_error = std_dev / math.sqrt(n)
    critical = student_t_ppf(1.0 - (1.0 - confidence) / 2.0, n - 1)
    margin = critical * standard_error
    return (float(mean - margin), float(mean + margin))


def cohen_dz(differences: Sequence[float]) -> Optional[float]:
    """Cohen's ``d_z`` for paired differences.

    Returns ``0.0`` when both mean and SD are zero, and ``None`` (undefined;
    report the raw mean instead) when SD is zero but the mean is not.
    """
    values = _validated_vector(differences)
    mean = _mean(values)
    if len(values) < 2:
        return None
    std_dev = math.sqrt(_sample_variance(values))
    if std_dev == 0.0:
        return 0.0 if mean == 0.0 else None
    return float(mean / std_dev)


# ---------------------------------------------------------------------------
# Pairwise-complete differences and failure imputation bounds
# ---------------------------------------------------------------------------


def paired_differences(
    fixed: Sequence[Optional[float]],
    adaptive: Sequence[Optional[float]],
) -> List[float]:
    """Pairwise-complete differences ``D = T_H(Adaptive) - T_H(Fixed)``.

    A replicate contributes only when **both** arms have an evaluable
    ``T_H`` (not None). Sequence lengths must match; entries must be
    finite when present.
    """
    if len(fixed) != len(adaptive):
        raise ValueError(
            f"arm vectors must have equal length, got {len(fixed)} and {len(adaptive)}"
        )
    differences: List[float] = []
    for position, (t_fixed, t_adaptive) in enumerate(zip(fixed, adaptive)):
        if t_fixed is None and t_adaptive is None:
            continue
        if t_fixed is None or t_adaptive is None:
            continue
        if not (math.isfinite(float(t_fixed)) and math.isfinite(float(t_adaptive))):
            raise ValueError(f"pair {position} has non-finite T_H values")
        differences.append(float(t_adaptive) - float(t_fixed))
    return differences


def impute_differences(
    fixed: Sequence[Optional[float]],
    adaptive: Sequence[Optional[float]],
    horizon: float,
    direction: str,
) -> List[float]:
    """Impute failed arms over the closed domain ``[0, horizon]`` and difference.

    Args:
        fixed: ``T_H`` per replicate for the Fixed arm (None = failed).
        adaptive: ``T_H`` per replicate for the Adaptive arm (None = failed).
        horizon: Recovery horizon ``H``.
        direction: ``"worst_for_adaptive"`` imputes ``H`` to a failed Adaptive
            arm and ``0`` to a failed Fixed arm (largest admissible ``D``);
            ``"best_for_adaptive"`` does the reverse.

    Returns:
        One difference per planned replicate (failed arms included).

    Raises:
        ValueError: On length mismatch or unknown direction.

    The result is a genuine bound only within the ``[0, H]`` imputation
    domain; the counterfactual ``T_H`` of a crashed run is unobservable.
    """
    if direction not in IMPUTATION_DIRECTIONS:
        raise ValueError(f"direction must be one of {IMPUTATION_DIRECTIONS}, got {direction!r}")
    if len(fixed) != len(adaptive):
        raise ValueError(
            f"arm vectors must have equal length, got {len(fixed)} and {len(adaptive)}"
        )
    if not math.isfinite(horizon) or horizon < 0.0:
        raise ValueError(f"horizon must be finite and >= 0, got {horizon!r}")

    differences: List[float] = []
    for t_fixed, t_adaptive in zip(fixed, adaptive):
        if direction == "worst_for_adaptive":
            a_value = float(horizon) if t_adaptive is None else float(t_adaptive)
            f_value = 0.0 if t_fixed is None else float(t_fixed)
        else:
            a_value = 0.0 if t_adaptive is None else float(t_adaptive)
            f_value = float(horizon) if t_fixed is None else float(t_fixed)
        if not (math.isfinite(a_value) and math.isfinite(f_value)):
            raise ValueError("T_H values must be finite")
        differences.append(a_value - f_value)
    return differences


def impute_censored_differences(
    fixed: Sequence[Optional[float]],
    adaptive: Sequence[Optional[float]],
    censored_value: float,
) -> List[float]:
    """Impute censored episodes (math.inf) to a specific finite TH value.

    Failed runs (None) are still skipped (pairwise-complete).
    """
    if len(fixed) != len(adaptive):
        raise ValueError("arm vectors must have equal length")

    differences: List[float] = []
    for t_fixed, t_adaptive in zip(fixed, adaptive):
        if t_fixed is None or t_adaptive is None:
            continue

        f_val = float(censored_value) if t_fixed == math.inf else float(t_fixed)
        a_val = float(censored_value) if t_adaptive == math.inf else float(t_adaptive)

        if not (math.isfinite(f_val) and math.isfinite(a_val)):
            raise ValueError("T_H values must be finite after imputation")

        differences.append(a_val - f_val)

    return differences


# ---------------------------------------------------------------------------
# Sensitivity analyses (different targets — labeled as such in the document)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignTestResult:
    """Exact one-sided sign test on the paired differences.

    Target: ``P(D < 0) = 1/2`` (median/sign symmetry of ``D``) — **not** the
    mean. Zero differences are dropped (standard sign-test convention).
    """

    n_paired: int
    n_nonzero: int
    n_negative: int
    p_value: float


def exact_sign_test(
    differences: Sequence[float],
    min_valid_n: int = MIN_VALID_N,
) -> SignTestResult:
    """Exact one-sided sign test: ``p = P(K >= k)`` for ``K ~ Bin(n', 0.5)``.

    ``k`` is the number of negative differences (Adaptive faster); small p
    therefore means evidence for ``H1: median(D) < 0``.
    """
    values = _validated_vector(differences)
    if len(values) < min_valid_n:
        raise ValueError(f"cell not evaluable: N_valid={len(values)} < MIN_VALID_N={min_valid_n}")
    nonzero = [v for v in values if v != 0.0]
    n_negative = sum(1 for v in nonzero if v < 0.0)
    n_nonzero = len(nonzero)
    if n_nonzero == 0:
        p_value = 1.0
    else:
        # One-sided lower-tail-on-D direction: large counts of NEGATIVE D
        # (Adaptive faster) give small p, so p = P(K >= k), K ~ Bin(n', 0.5).
        numerator = sum(math.comb(n_nonzero, k) for k in range(n_negative, n_nonzero + 1))
        p_value = float(numerator) / float(2**n_nonzero)
    return SignTestResult(
        n_paired=len(values),
        n_nonzero=n_nonzero,
        n_negative=n_negative,
        p_value=p_value,
    )


@dataclass(frozen=True)
class WilcoxonResult:
    """Exact one-sided Wilcoxon signed-rank result (sign-flip enumeration).

    Target: the pseudomedian of a symmetric ``D`` distribution — **not** the
    mean. Zero ``|D|`` values are dropped; average ranks handle ties; the null
    distribution enumerates all ``2^n'`` sign assignments of the fixed ranks
    (valid under i.i.d. fair signs).
    """

    n_paired: int
    n_nonzero: int
    rank_sum_positive: float
    p_value: float


def _average_ranks(magnitudes: Sequence[float]) -> List[float]:
    order = sorted(range(len(magnitudes)), key=lambda i: magnitudes[i])
    ranks = [0.0] * len(magnitudes)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and magnitudes[order[j + 1]] == magnitudes[order[i]]:
            j += 1
        average = ((i + 1) + (j + 1)) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def exact_wilcoxon_signed_rank(
    differences: Sequence[float],
    min_valid_n: int = MIN_VALID_N,
) -> WilcoxonResult:
    """Exact one-sided Wilcoxon signed-rank test (lower tail: W+ small)."""
    values = _validated_vector(differences)
    if len(values) < min_valid_n:
        raise ValueError(f"cell not evaluable: N_valid={len(values)} < MIN_VALID_N={min_valid_n}")
    nonzero = [v for v in values if v != 0.0]
    n_nonzero = len(nonzero)
    if n_nonzero == 0:
        return WilcoxonResult(n_paired=len(values), n_nonzero=0, rank_sum_positive=0.0, p_value=1.0)
    if n_nonzero > 20:
        raise ValueError(f"exact sign enumeration supports n' <= 20, got {n_nonzero}")

    magnitudes = [abs(v) for v in nonzero]
    ranks = _average_ranks(magnitudes)
    w_plus_observed = math.fsum(rank for rank, value in zip(ranks, nonzero) if value > 0.0)

    extreme = 0
    total = 0
    for signs in itertools.product((0, 1), repeat=n_nonzero):
        w_plus = math.fsum(rank for rank, sign in zip(ranks, signs) if sign)
        total += 1
        if w_plus <= w_plus_observed + 1.0e-12:
            extreme += 1

    return WilcoxonResult(
        n_paired=len(values),
        n_nonzero=n_nonzero,
        rank_sum_positive=float(w_plus_observed),
        p_value=float(extreme) / float(total),
    )


def robust_wilcoxon_signed_rank(
    differences: Sequence[float],
    min_valid_n: int = MIN_VALID_N,
) -> float:
    """Wilcoxon signed-rank p-value: exact for N <= 20, scipy asymptotic otherwise."""
    values = _validated_vector(differences)
    nonzero = [v for v in values if v != 0.0]
    n_nonzero = len(nonzero)
    if n_nonzero <= 20:
        return exact_wilcoxon_signed_rank(differences, min_valid_n).p_value
    else:
        try:
            import scipy.stats as stats
        except ImportError:
            raise ImportError("scipy is required for Wilcoxon signed-rank test for N > 20.")
        stat, p = stats.wilcoxon(differences, alternative="less", mode="asymp")
        return float(p)


def bootstrap_percentile_ci(
    differences: Sequence[float],
    confidence: float = 0.95,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    min_valid_n: int = MIN_VALID_N,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference (approximate).

    Resampling unit is the replicate pair. ``numpy.random.default_rng(seed)``
    with ``reps`` and ``seed`` frozen by the protocol, so the interval is
    reproducible for a fixed numpy version (recorded in artifacts).
    """
    values = _validated_vector(differences)
    if len(values) < min_valid_n:
        raise ValueError(f"cell not evaluable: N_valid={len(values)} < MIN_VALID_N={min_valid_n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(reps, array.size))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - confidence
    low = float(np.quantile(means, alpha / 2.0, method="linear"))
    high = float(np.quantile(means, 1.0 - alpha / 2.0, method="linear"))
    return (low, high)


# ---------------------------------------------------------------------------
# Multiplicity and the six-cell family decision
# ---------------------------------------------------------------------------


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (FWER control), original order kept."""
    values = [float(p) for p in p_values]
    if not values:
        raise ValueError("p_values must be non-empty")
    for position, p in enumerate(values):
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"p_values[{position}] must be in [0, 1], got {p!r}")

    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, float(len(values) - rank) * values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def decide_family(
    cell_p_values: Mapping[str, Optional[float]],
    alpha: float = ALPHA,
) -> FamilyDecision:
    """Intersection-union decision over the six preregistered primary cells.

    Rules (deterministic, matching the document):

    * The mapping must contain exactly the six ``PRIMARY_CELLS`` keys.
    * Any cell without a valid p-value (``None``) => ``INCONCLUSIVE``
      (family claim can be neither supported nor refuted).
    * All p-values present => ``SUPPORTED`` iff every ``p < alpha``
      (intersection-union: evidence required in **all six** cells),
      otherwise ``NOT_SUPPORTED``.
    """
    if set(cell_p_values) != set(PRIMARY_CELLS):
        missing = sorted(set(PRIMARY_CELLS) - set(cell_p_values))
        extra = sorted(set(cell_p_values) - set(PRIMARY_CELLS))
        raise ValueError(
            f"family must contain exactly PRIMARY_CELLS; missing={missing}, extra={extra}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    for cell in PRIMARY_CELLS:
        p_value = cell_p_values[cell]
        if p_value is None:
            return "INCONCLUSIVE"
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"p-value for {cell} must be in [0, 1], got {p_value!r}")

    if all(float(cell_p_values[cell]) < alpha for cell in PRIMARY_CELLS):  # type: ignore[arg-type]
        return "SUPPORTED"
    return "NOT_SUPPORTED"


# ---------------------------------------------------------------------------
# Diagnostics and Assumptions
# ---------------------------------------------------------------------------


def shapiro_wilk(differences: Sequence[float]) -> Tuple[float, float]:
    """Shapiro-Wilk test for normality on paired differences."""
    values = _validated_vector(differences)
    if len(values) < 3:
        raise ValueError("Shapiro-Wilk requires at least 3 observations")
    try:
        import scipy.stats as stats
    except ImportError:
        raise ImportError("scipy is required for Shapiro-Wilk diagnostics.")
    stat, p_value = stats.shapiro(values)
    return float(stat), float(p_value)


@dataclass(frozen=True)
class SampleDiagnostics:
    """Basic metrics and diagnostic tests for sample assumption audits."""

    n_valid: int
    n_censored: int
    n_failed: int
    shapiro_statistic: float
    shapiro_p_value: float
    skewness: float
    kurtosis: float
    best_case_imputation_bounds: Tuple[float, float]
    worst_case_imputation_bounds: Tuple[float, float]
    wilcoxon_p_value: float
    sign_test_p_value: float
    bootstrap_ci: Tuple[float, float]
    warnings: List[str]


def calculate_diagnostics(
    fixed: Sequence[Optional[float]],
    adaptive: Sequence[Optional[float]],
) -> SampleDiagnostics:
    """Calculate basic metrics and diagnostics on the arms."""
    if len(fixed) != len(adaptive):
        raise ValueError("arm vectors must have equal length")

    n_valid = 0
    n_censored = 0
    n_failed = 0
    valid_differences = []

    for t_fixed, t_adaptive in zip(fixed, adaptive):
        if t_fixed is None or t_adaptive is None:
            n_failed += 1
        elif t_fixed == math.inf or t_adaptive == math.inf:
            n_censored += 1
        else:
            n_valid += 1
            valid_differences.append(float(t_adaptive) - float(t_fixed))

    if n_valid < 3:
        stat, p = math.nan, math.nan
        skew = math.nan
    else:
        stat, p = shapiro_wilk(valid_differences)
        skew = sample_skewness(valid_differences)

    if n_valid < 4:
        kurt = math.nan
    else:
        kurt = sample_kurtosis(valid_differences)

    best_diffs = impute_censored_differences(fixed, adaptive, 15.0)
    worst_diffs = impute_censored_differences(fixed, adaptive, 30.0)

    if len(best_diffs) >= MIN_VALID_N:
        best_ci = paired_t_interval(best_diffs, min_valid_n=MIN_VALID_N)
    else:
        best_ci = (math.nan, math.nan)

    if len(worst_diffs) >= MIN_VALID_N:
        worst_ci = paired_t_interval(worst_diffs, min_valid_n=MIN_VALID_N)
    else:
        worst_ci = (math.nan, math.nan)

    if n_valid >= MIN_VALID_N:
        wilcoxon_p = robust_wilcoxon_signed_rank(valid_differences, min_valid_n=MIN_VALID_N)
        sign_p = exact_sign_test(valid_differences, min_valid_n=MIN_VALID_N).p_value
        boot_ci = bootstrap_percentile_ci(valid_differences, min_valid_n=MIN_VALID_N)
    else:
        wilcoxon_p = math.nan
        sign_p = math.nan
        boot_ci = (math.nan, math.nan)

    warnings = []
    if n_valid < 10:
        warnings.append(f"Insufficient sample size for reliable diagnostics: N={n_valid} < 10.")
    if not math.isnan(p) and p < 0.05:
        warnings.append(f"Normality assumption rejected (Shapiro-Wilk p={p:.4f} < 0.05).")

    return SampleDiagnostics(
        n_valid=n_valid,
        n_censored=n_censored,
        n_failed=n_failed,
        shapiro_statistic=stat,
        shapiro_p_value=p,
        skewness=skew,
        kurtosis=kurt,
        sensitivity_scenario_15=best_ci,
        sensitivity_scenario_30=worst_ci,
        wilcoxon_p_value=wilcoxon_p,
        sign_test_p_value=sign_p,
        bootstrap_ci=boot_ci,
        warnings=warnings,
    )


def export_diagnostics_to_dict(diag: SampleDiagnostics) -> dict:
    """Exports diagnostics to a JSON-serializable dictionary."""
    return {
        "metrics": {
            "n_valid": diag.n_valid,
            "n_censored": diag.n_censored,
            "n_failed": diag.n_failed,
        },
        "assumptions": {
            "shapiro_statistic": diag.shapiro_statistic,
            "shapiro_p_value": diag.shapiro_p_value,
            "skewness": diag.skewness,
            "kurtosis": diag.kurtosis,
        },
        "sensitivity": {
            "scenario_TH_15_bounds": list(diag.sensitivity_scenario_15),
            "scenario_TH_30_bounds": list(diag.sensitivity_scenario_30),
        },
        "non_parametric": {
            "wilcoxon_p_value": diag.wilcoxon_p_value,
            "sign_test_p_value": diag.sign_test_p_value,
            "bootstrap_ci": list(diag.bootstrap_ci),
        },
        "warnings": diag.warnings,
    }


def generate_diagnostics_text_summary(diag: SampleDiagnostics) -> str:
    """Generates a human-readable text summary of the statistical diagnostics."""
    lines = [
        "--- Statistical Diagnostics & Assumption Report ---",
        f"Valid pairs: {diag.n_valid} | Censored: {diag.n_censored} | Failed: {diag.n_failed}",
        "",
        f"Assumptions (evaluated on N={diag.n_valid} valid pairs):",
        f"  Shapiro-Wilk p-value : {diag.shapiro_p_value:.4f}",
        f"  Skewness             : {diag.skewness:.4f}",
        f"  Excess Kurtosis      : {diag.kurtosis:.4f}",
        "  (Note: Shapiro-Wilk p >= 0.05 means insufficient evidence to reject normality, not validated)",
        "",
        f"Sensitivity Scenarios (evaluates N={diag.n_valid + diag.n_censored} by imputing censored pairs):",
        f"  Scenario (TH=15) CI  : [{diag.sensitivity_scenario_15[0]:.4f}, {diag.sensitivity_scenario_15[1]:.4f}]",
        f"  Scenario (TH=30) CI  : [{diag.sensitivity_scenario_30[0]:.4f}, {diag.sensitivity_scenario_30[1]:.4f}]",
        "",
        f"Non-Parametric Robustness (evaluated on N={diag.n_valid} valid pairs):",
        f"  Wilcoxon p-value     : {diag.wilcoxon_p_value:.4f}",
        f"  Fisher Sign p-value  : {diag.sign_test_p_value:.4f}",
        f"  Bootstrap CI         : [{diag.bootstrap_ci[0]:.4f}, {diag.bootstrap_ci[1]:.4f}]",
        "",
    ]
    if diag.warnings:
        lines.append("WARNINGS:")
        for w in diag.warnings:
            lines.append(f"  - {w}")
    else:
        lines.append("Warnings: None")
    return "\n".join(lines)


__all__ = [
    "IMPUTATION_DIRECTIONS",
    "FamilyDecision",
    "PairedTTest",
    "SampleDiagnostics",
    "SignTestResult",
    "WilcoxonResult",
    "bootstrap_percentile_ci",
    "calculate_diagnostics",
    "cohen_dz",
    "decide_family",
    "exact_sign_test",
    "exact_wilcoxon_signed_rank",
    "export_diagnostics_to_dict",
    "generate_diagnostics_text_summary",
    "holm_adjust",
    "impute_censored_differences",
    "impute_differences",
    "paired_differences",
    "paired_t_interval",
    "paired_t_test",
    "robust_wilcoxon_signed_rank",
    "sample_kurtosis",
    "sample_skewness",
    "shapiro_wilk",
    "student_t_cdf",
    "student_t_ppf",
]
