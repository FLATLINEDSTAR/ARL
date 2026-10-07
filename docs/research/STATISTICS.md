# Adaptive RL Statistical Protocol Diagnostics

This document outlines the statistical assumptions, diagnostic checks, and sensitivity analyses used in the Adaptive RL hypothesis testing protocol.

## 1. Core Assumptions Being Checked

The primary preregistered hypothesis test for this project is a **paired one-sided t-test** on the recovery horizon times ($T_H$). Like all parametric tests, it relies on several key assumptions. The diagnostic suite automatically checks these assumptions to ensure the validity of our conclusions.

### Sample Size ($N$)
- **Requirement:** A minimum sample size is required for the Central Limit Theorem to hold and for our statistical tests to have sufficient power.
- **Diagnostic Trigger:** If the number of valid pairs ($N_{valid}$) is strictly less than 10, the diagnostic orchestrator emits a warning: `Insufficient sample size for reliable diagnostics: N < 10`.

### Normality of Paired Differences
- **Requirement:** The paired t-test assumes that the *differences* between the paired observations (Adaptive minus Fixed) are approximately normally distributed.
- **Diagnostic Trigger:** We use the **Shapiro-Wilk test** to formally test the null hypothesis that the data was drawn from a normal distribution. If the resulting $p$-value is less than $0.05$, the assumption of normality is formally rejected, and a warning is emitted: `Normality assumption rejected (Shapiro-Wilk p < 0.05)`.
- **Supplementary Metrics:** We also calculate **Sample Skewness** (Fisher-Pearson coefficient) and **Excess Kurtosis** (Fisher's definition) to provide a descriptive view of *how* the distribution deviates from normality (e.g., heavy tails or asymmetry).

---

## 2. Interpreting Sensitivity Analysis Bounds

In our experimental protocol, some episodes may fail to recover within the designated maximum horizon ($T_H = \infty$). These episodes are considered "censored." Because the true, unobserved recovery time could be anything from just after the horizon to infinity, we must understand how sensitive our primary t-test is to these censored values.

To do this, we compute the bounds of the confidence interval using best-case and worst-case global imputation limits:

- **Best-case Imputation CI (TH = 15):** We temporarily set all censored episodes (where $T_H = \infty$) to a rapid recovery time of 15 seconds. If the upper bound of this interval remains below 0, it suggests that even if all censored episodes recovered almost immediately after the cutoff, the adaptive arm is still significantly better.
- **Worst-case Imputation CI (TH = 30):** We temporarily set all censored episodes to an extended recovery time of 30 seconds. If the upper bound of this interval remains below 0, it suggests our hypothesis holds even if censored episodes took twice as long to recover.

If the bounds span zero, it indicates that our primary conclusion is heavily dependent on the exact unobserved recovery time of the censored episodes.

---

## 3. Interpreting Non-Parametric Checks

When the assumption of normality is violated (e.g., Shapiro-Wilk $p < 0.05$) or when heavy censoring creates extreme outliers, non-parametric tests provide robust secondary validation of our results. 

The diagnostic suite automatically provides three non-parametric checks alongside the primary parametric test:

1. **Wilcoxon Signed-Rank Test:**
   - Evaluates whether the median of the paired differences is less than zero.
   - For small sample sizes ($N \le 20$), we calculate the exact $p$-value using full distribution sign-flip enumeration, ensuring perfectly calibrated results regardless of ties. For larger samples, it defaults to the asymptotic normal approximation.
   - **Interpretation:** If the Wilcoxon $p$-value is $< 0.05$ but the paired t-test is not, outliers or skewness may be artificially inflating the variance in the parametric test.

2. **Fisher's Exact Sign Test:**
   - A highly conservative test that relies solely on the direction (sign) of the difference, entirely ignoring the magnitude.
   - **Interpretation:** This test answers the question: "Did the adaptive arm recover faster than the fixed arm in a statistically significant majority of trials?" This is robust against literally any monotonic transformation of the data.

3. **Bootstrap Percentile Confidence Interval:**
   - Calculates an empirical $95\%$ confidence interval for the mean difference by resampling the paired observations 10,000 times (with replacement).
   - **Interpretation:** Because bootstrapping makes no assumptions about the underlying distribution, this CI is highly reliable for skewed data. If this interval is strictly below zero, it strongly corroborates the primary t-test without relying on the normality assumption.
