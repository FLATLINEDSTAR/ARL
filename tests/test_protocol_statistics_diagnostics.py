import math

import pytest

from adaptive_rl.protocol.statistics import (
    calculate_diagnostics,
    impute_censored_differences,
    robust_wilcoxon_signed_rank,
    sample_kurtosis,
    sample_skewness,
    shapiro_wilk,
)


def test_diagnostics_core_metrics_against_scipy() -> None:
    pytest.importorskip("scipy")
    from scipy import stats

    # Normal mock data
    normal_data = [0.1, -0.2, 0.3, -0.1, 0.05, 0.4, -0.3, 0.2, -0.15, 0.0]
    # Skewed mock data
    skewed_data = [0.1, 0.2, 0.15, 0.3, 0.05, 2.5, 3.1, 0.1, 0.2, 0.15]

    for data in [normal_data, skewed_data]:
        # Skewness
        expected_skew = stats.skew(data, bias=True)
        assert sample_skewness(data) == pytest.approx(expected_skew, abs=1e-6)

        # Kurtosis
        expected_kurt = stats.kurtosis(data, fisher=True, bias=True)
        assert sample_kurtosis(data) == pytest.approx(expected_kurt, abs=1e-6)

        # Shapiro-Wilk
        expected_stat, expected_p = stats.shapiro(data)
        stat, p = shapiro_wilk(data)
        assert stat == pytest.approx(expected_stat, abs=1e-6)
        assert p == pytest.approx(expected_p, abs=1e-6)


def test_sensitivity_imputation_bounds() -> None:
    pytest.importorskip("scipy")

    # We will test the lower and upper bounds of imputed censored episodes
    # fixed arm: some missing, some censored (inf)
    # adaptive arm: some missing, some censored (inf)
    fixed = [1.0, 2.0, math.inf, None, 5.0, 6.0, math.inf, 8.0, 9.0, 10.0]
    adaptive = [1.5, 2.5, 3.5, 4.5, math.inf, 6.5, math.inf, 8.5, 9.5, 10.5]

    # Best-case (TH=15)
    best_diffs = impute_censored_differences(fixed, adaptive, 15.0)
    # Expected best_diffs calculation:
    # 0: 1.5 - 1.0 = 0.5
    # 1: 2.5 - 2.0 = 0.5
    # 2: 3.5 - 15.0 = -11.5
    # 3: skip (fixed is None, well wait, fixed is None -> skip)
    # 4: 15.0 - 5.0 = 10.0
    # 5: 6.5 - 6.0 = 0.5
    # 6: 15.0 - 15.0 = 0.0
    # 7: 8.5 - 8.0 = 0.5
    # 8: 9.5 - 9.0 = 0.5
    # 9: 10.5 - 10.0 = 0.5

    assert len(best_diffs) == 9
    assert best_diffs[2] == pytest.approx(-11.5)
    assert best_diffs[3] == pytest.approx(10.0)
    assert best_diffs[5] == pytest.approx(0.0)

    # Worst-case (TH=30)
    worst_diffs = impute_censored_differences(fixed, adaptive, 30.0)
    assert len(worst_diffs) == 9
    assert worst_diffs[2] == pytest.approx(-26.5)  # 3.5 - 30.0
    assert worst_diffs[3] == pytest.approx(25.0)  # 30.0 - 5.0
    assert worst_diffs[5] == pytest.approx(0.0)  # 30.0 - 30.0

    diag = calculate_diagnostics(fixed, adaptive)
    assert diag.n_censored == 3  # (index 2, 4, 6)
    assert diag.n_failed == 1  # (index 3)
    assert diag.n_valid == 6  # (indices 0, 1, 5, 7, 8, 9)

    # Ensure bounds are calculated (should be length 2 tuples)
    assert len(diag.best_case_imputation_bounds) == 2
    assert len(diag.worst_case_imputation_bounds) == 2


def test_non_parametric_robustness_checks() -> None:
    pytest.importorskip("scipy")
    from scipy import stats

    differences = [-1.5, -2.0, -0.5, -3.0, -1.0, -2.5, -0.5, -1.5, -0.1, -0.2]

    # N <= 20 uses exact Wilcoxon
    robust_p = robust_wilcoxon_signed_rank(differences)
    # Should not throw and should be a valid p-value
    assert 0.0 <= robust_p <= 1.0

    # N > 20 uses scipy's asymptotic Wilcoxon
    large_differences = differences * 3  # N = 30
    robust_large_p = robust_wilcoxon_signed_rank(large_differences)
    expected_stat, expected_large_p = stats.wilcoxon(
        large_differences, alternative="less", mode="asymp"
    )
    assert robust_large_p == pytest.approx(expected_large_p, abs=1e-6)


def test_diagnostics_orchestrator_warnings() -> None:
    pytest.importorskip("scipy")

    # Insufficient N
    fixed = [1.0, 2.0, 3.0]
    adaptive = [1.1, 2.1, 3.1]
    diag = calculate_diagnostics(fixed, adaptive)
    assert any("Insufficient sample size" in w for w in diag.warnings)

    # Normality rejected
    # A highly skewed dataset to trigger Shapiro p < 0.05
    fixed_skewed = [0.0] * 20
    adaptive_skewed = [0.1] * 19 + [10.0]  # Skewed
    diag_skewed = calculate_diagnostics(fixed_skewed, adaptive_skewed)
    assert any("Normality assumption rejected" in w for w in diag_skewed.warnings)
