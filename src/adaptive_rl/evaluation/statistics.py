"""Statistical summaries for multi-seed evaluation results.

Two distinct kinds of quantity are produced here and must not be confused:

* :class:`DescriptiveMetrics` (``summarize_descriptive_episodes``) pools every
  evaluated episode into a single sample. These are *descriptive* values for
  the episodes that happened to run.
* :class:`MetricStatistics` (``summarize_seed_values``) treats one seed-level
  summary as one independent observation and is the basis for cross-seed
  means, standard deviations, and confidence intervals.

Because the two use different sample units, a pooled mean and the mean of the
per-seed means differ whenever seeds contribute unequal episode counts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Sequence


@dataclass(frozen=True)
class MetricStatistics:
    """Across-seed statistics for one scalar metric."""

    mean: float | None
    std: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    sample_count: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the regularized beta function."""
    max_iterations = 300
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result

    raise ArithmeticError("Incomplete beta continued fraction did not converge.")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in (a, b, x)):
        raise ValueError("Incomplete beta parameters must be real numbers.")
    a, b, x = float(a), float(b), float(x)
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0.0 or b <= 0.0:
        raise ValueError("Incomplete beta parameters a and b must be finite and positive.")
    if not math.isfinite(x) or not 0.0 <= x <= 1.0:
        raise ValueError(f"Incomplete beta x must be in [0, 1], got {x}.")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if isinstance(degrees_of_freedom, bool) or not isinstance(degrees_of_freedom, int):
        raise ValueError("Student-t degrees of freedom must be an integer.")
    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive.")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Student-t value must be a finite real number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Student-t value must be finite.")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)
    return 1.0 - tail if value > 0 else tail


def student_t_critical_value(confidence: float, degrees_of_freedom: int) -> float:
    """Return the positive two-sided Student-t critical value.

    The inverse is computed by bisection over the Student-t CDF, implemented
    using the regularized incomplete beta function; no statistical dependency
    or normal approximation is used.
    """
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("Confidence must be a finite real number.")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}.")
    if isinstance(degrees_of_freedom, bool) or not isinstance(degrees_of_freedom, int):
        raise ValueError("Student-t degrees of freedom must be an integer.")
    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive.")

    target = (1.0 + confidence) / 2.0
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < target:
        upper *= 2.0

    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def summarize_seed_values(values: Sequence[float | None]) -> MetricStatistics:
    """Summarize independent seed-level values using sample standard deviation.

    Missing values are excluded metric-by-metric. CI bounds are unavailable
    unless at least two finite seed-level observations are present.
    """
    observed: list[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Metric values must be real numbers or None, got {value!r}.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Metric values must be finite, got {value!r}.")
        observed.append(numeric_value)

    count = len(observed)
    if count == 0:
        return MetricStatistics(None, None, None, None, 0)

    mean = math.fsum(observed) / count
    if count < 2:
        return MetricStatistics(mean, None, None, None, count)

    variance = math.fsum((value - mean) ** 2 for value in observed) / (count - 1)
    std = math.sqrt(max(0.0, variance))
    critical = student_t_critical_value(0.95, count - 1)
    margin = critical * std / math.sqrt(count)
    return MetricStatistics(mean, std, mean - margin, mean + margin, count)


@dataclass(frozen=True)
class DescriptiveMetrics:
    """Pooled episode-level descriptive statistics for one evaluation run.

    Every evaluated episode contributes exactly one observation, so these
    values describe the sampled episodes rather than an across-seed
    population. They are deliberately distinct from :class:`MetricStatistics`,
    which summarizes independent seed-level observations.
    """

    episodes: int
    success_rate: float | None
    collision_rate: float | None
    timeout_rate: float | None
    mean_reward: float
    std_reward: float | None
    mean_episode_length: float

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _pooled_rate(flags: Sequence[bool | None]) -> float | None:
    observed = [flag for flag in flags if flag is not None]
    if not observed:
        return None
    return float(sum(observed) / len(observed))


def summarize_descriptive_episodes(
    rewards: Sequence[float],
    episode_lengths: Sequence[float],
    successes: Sequence[bool | None],
    collisions: Sequence[bool | None],
    truncations: Sequence[bool],
) -> DescriptiveMetrics:
    """Pool episode-level observations into descriptive statistics.

    Semantics:
      - all inputs must describe the same episodes and be non-empty;
      - rewards must be finite and are summarized with the sample standard
        deviation (``None`` when fewer than two episodes exist);
      - success and collision rates exclude episodes where the outcome is
        unavailable (``None``); the rate itself is ``None`` when no episode
        reports that outcome, never ``0.0``;
      - the timeout rate uses every evaluated episode as its denominator,
        because the Gymnasium truncation flag is always available.
    """
    counts = {
        len(rewards),
        len(episode_lengths),
        len(successes),
        len(collisions),
        len(truncations),
    }
    if len(counts) != 1:
        raise ValueError(f"Episode inputs must be aligned, got lengths {sorted(counts)}.")
    episodes = counts.pop()
    if episodes == 0:
        raise ValueError("At least one episode is required for descriptive metrics.")

    finite_rewards: list[float] = []
    for reward in rewards:
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise ValueError(f"Episode rewards must be real numbers, got {reward!r}.")
        numeric_reward = float(reward)
        if not math.isfinite(numeric_reward):
            raise ValueError(f"Episode rewards must be finite, got {reward!r}.")
        finite_rewards.append(numeric_reward)

    finite_lengths: list[float] = []
    for length in episode_lengths:
        if isinstance(length, bool) or not isinstance(length, Real):
            raise ValueError(f"Episode lengths must be real numbers, got {length!r}.")
        numeric_length = float(length)
        if not math.isfinite(numeric_length) or numeric_length < 0.0:
            raise ValueError(f"Episode lengths must be finite and non-negative, got {length!r}.")
        finite_lengths.append(numeric_length)

    for name, flags in (("successes", successes), ("collisions", collisions)):
        if any(value is not None and not isinstance(value, bool) for value in flags):
            raise ValueError(f"{name} values must be bool or None.")
    if any(not isinstance(value, bool) for value in truncations):
        raise ValueError("Truncation values must be bool.")

    mean_reward = math.fsum(finite_rewards) / episodes
    if episodes > 1:
        variance = math.fsum((reward - mean_reward) ** 2 for reward in finite_rewards) / (
            episodes - 1
        )
        std_reward: float | None = math.sqrt(variance)
    else:
        std_reward = None

    mean_episode_length = math.fsum(finite_lengths) / episodes
    return DescriptiveMetrics(
        episodes=episodes,
        success_rate=_pooled_rate(successes),
        collision_rate=_pooled_rate(collisions),
        timeout_rate=float(sum(truncations) / episodes),
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_episode_length=mean_episode_length,
    )


__all__ = [
    "DescriptiveMetrics",
    "MetricStatistics",
    "student_t_critical_value",
    "summarize_descriptive_episodes",
    "summarize_seed_values",
]
