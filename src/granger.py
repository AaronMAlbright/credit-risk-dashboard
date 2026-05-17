"""
Granger causality analysis between credit market variables.

Granger causality (Granger 1969): variable X 'Granger-causes' Y if past
values of X contain information that improves the forecast of Y beyond
what past values of Y alone provide. This is NOT true causation — it is
predictive causality (precedence + incremental information).

Test: compare two VAR models:
  Restricted:   Y_t = Σ α_i Y_{t-i} + ε_t
  Unrestricted: Y_t = Σ α_i Y_{t-i} + Σ β_i X_{t-i} + ε_t
F-statistic = ((RSS_r - RSS_u)/p) / (RSS_u/(n-2p-1))
Under H0 (X does not Granger-cause Y), F ~ F(p, n-2p-1)

We implement the F-test WITHOUT scipy/statsmodels using only numpy's
least-squares solver. P-value approximated via the regularised incomplete
beta function using a series expansion.

Pairs tested (bidirectional):
  HY spread ↔ VIX
  HY spread ↔ composite_risk_score_smooth
  VIX ↔ composite_risk_score_smooth
  HY spread ↔ sp500 (levels → returns)

Public API
----------
  LAGS           — [1, 5, 21]  (1-day, 1-week, 1-month)
  granger_test(y, x, lags) → dict  (F-stat, p-value, conclusion)
  run_granger_analysis(df) → dict
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAGS: list[int] = [1, 5, 21]   # 1-day, 1-week, 1-month

# ---------------------------------------------------------------------------
# P-value computation — regularised incomplete beta function
# ---------------------------------------------------------------------------


def _log_beta(a: float, b: float) -> float:
    """Natural log of the Beta function: log B(a,b) = lgamma(a)+lgamma(b)-lgamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, max_iter: int = 200) -> float:
    """Evaluate the continued fraction for the incomplete beta function.

    Uses the Lentz modified continued-fraction algorithm (Press et al.,
    Numerical Recipes, §6.4).  Returns the value of the continued fraction
    component of I_x(a, b).
    """
    FPMIN = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-7:
            break

    return h


def _regularised_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b).

    I_x(a, b) = B(x; a, b) / B(a, b)

    Uses the continued-fraction representation for numerical stability.
    For x > (a+1)/(a+b+2), applies the symmetry relation
    I_x(a,b) = 1 - I_{1-x}(b,a) to ensure fast convergence.

    Parameters
    ----------
    a, b : float   — shape parameters (must be > 0)
    x    : float   — evaluation point in [0, 1]

    Returns
    -------
    float in [0, 1]
    """
    if x < 0.0 or x > 1.0:
        return float("nan")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_factor = a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b)
    factor = math.exp(log_factor)

    # Use symmetry for better convergence
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _betacf(a, b, x) / a
    else:
        return 1.0 - factor * _betacf(b, a, 1.0 - x) / b


def _f_pvalue(f_stat: float, d1: float, d2: float) -> float:
    """Compute the p-value of an F-statistic via the regularised beta function.

    For F ~ F(d1, d2):
      P(F > f) = I_{d2/(d2 + d1*f)}(d2/2, d1/2)

    This is exact, not an approximation.  Falls back to an exponential
    approximation if inputs are degenerate.

    Parameters
    ----------
    f_stat : float   — observed F-statistic (>= 0)
    d1     : float   — numerator degrees of freedom (lags, p)
    d2     : float   — denominator degrees of freedom (n - 2p - 1)
    """
    if not (np.isfinite(f_stat) and np.isfinite(d1) and np.isfinite(d2)):
        return float("nan")
    if f_stat <= 0.0:
        return 1.0
    if d1 <= 0.0 or d2 <= 0.0:
        # Fallback conservative approximation
        return float(np.clip(math.exp(-f_stat / 2.0), 0.0, 1.0))

    try:
        x = d2 / (d2 + d1 * f_stat)
        p = _regularised_beta(d2 / 2.0, d1 / 2.0, x)
        return float(np.clip(p, 0.0, 1.0))
    except (ValueError, ZeroDivisionError, OverflowError):
        # Fallback: conservative exponential approximation
        return float(np.clip(math.exp(-f_stat / 2.0), 0.0, 1.0))


# ---------------------------------------------------------------------------
# OLS building blocks
# ---------------------------------------------------------------------------


def _build_lagged_matrix(
    series: np.ndarray, n_lags: int, start_row: int
) -> np.ndarray:
    """Build a matrix of lagged values for OLS.

    For a series of length T and n_lags lags, returns a matrix of shape
    (T - start_row, n_lags) where column j contains values lagged by (j+1).

    Parameters
    ----------
    series   : 1-D array of length T
    n_lags   : number of lags to include
    start_row: first index of the target (determines how many rows to return)
    """
    T = len(series)
    n_rows = T - start_row
    mat = np.empty((n_rows, n_lags))
    for j in range(n_lags):
        lag = j + 1
        # Row i of output corresponds to time (start_row + i);
        # the lagged value at lag `lag` is at index (start_row + i - lag)
        mat[:, j] = series[start_row - lag: T - lag]
    return mat


def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    """Fit OLS and return the residual sum of squares.

    Uses numpy's least-squares solver which handles rank-deficient cases
    gracefully via the pseudoinverse (SVD-based).

    Parameters
    ----------
    X : design matrix (n, k) — no intercept column added here
    y : response vector (n,)
    """
    # Add intercept
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones, X])

    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    residuals = y - X_aug @ coef
    return float(residuals @ residuals)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def granger_test(
    y: np.ndarray,
    x: np.ndarray,
    lags: int = 5,
) -> dict:
    """Granger causality F-test: does X Granger-cause Y?

    Fits two OLS models:
      Restricted:   Y_t ~ Σ α_i Y_{t-i}
      Unrestricted: Y_t ~ Σ α_i Y_{t-i} + Σ β_i X_{t-i}

    The F-statistic measures whether including lagged X significantly
    reduces the residual variance of Y's forecast.

    Parameters
    ----------
    y    : 1-D numpy array, the dependent (effect) variable
    x    : 1-D numpy array, the candidate Granger-cause
    lags : number of lags p to include (default 5)

    Returns
    -------
    dict with keys:
        f_stat      — float, F-statistic
        p_value     — float, p-value (H0: X does not Granger-cause Y)
        lags        — int
        n           — int, effective sample size
        significant — bool, True if p_value < 0.05
        conclusion  — str
    """
    # ------------------------------------------------------------------
    # Validate and align inputs
    # ------------------------------------------------------------------
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    min_len = max(len(y), len(x))
    if len(y) != len(x):
        # Truncate to shared length
        shared = min(len(y), len(x))
        y = y[:shared]
        x = x[:shared]

    # Remove rows where either series is NaN
    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]
    T = len(y)

    min_required = 2 * lags + 10
    if T < min_required:
        return {
            "f_stat": float("nan"),
            "p_value": float("nan"),
            "lags": lags,
            "n": T,
            "significant": False,
            "conclusion": "Insufficient data",
        }

    start_row = lags  # first usable target row index

    # Response vector: y[lags], y[lags+1], ..., y[T-1]
    y_target = y[start_row:]
    n = len(y_target)   # effective sample size = T - lags

    # ------------------------------------------------------------------
    # Build design matrices
    # ------------------------------------------------------------------
    Y_lags = _build_lagged_matrix(y, lags, start_row)   # (n, lags)
    X_lags = _build_lagged_matrix(x, lags, start_row)   # (n, lags)

    # Restricted model: only lagged Y
    rss_r = _ols_rss(Y_lags, y_target)

    # Unrestricted model: lagged Y + lagged X
    X_unrestricted = np.hstack([Y_lags, X_lags])        # (n, 2*lags)
    rss_u = _ols_rss(X_unrestricted, y_target)

    # ------------------------------------------------------------------
    # F-statistic
    # ------------------------------------------------------------------
    dof_num = lags                   # number of restrictions (β coefficients)
    dof_den = n - 2 * lags - 1      # unrestricted residual dof (incl. intercept)

    if dof_den <= 0 or rss_u <= 0:
        return {
            "f_stat": float("nan"),
            "p_value": float("nan"),
            "lags": lags,
            "n": n,
            "significant": False,
            "conclusion": "Insufficient degrees of freedom",
        }

    # Guard against numerical noise making rss_u > rss_r
    diff = max(rss_r - rss_u, 0.0)
    f_stat = (diff / dof_num) / (rss_u / dof_den)

    p_value = _f_pvalue(f_stat, float(dof_num), float(dof_den))
    significant = bool(np.isfinite(p_value) and p_value < 0.05)

    conclusion = (
        "X Granger-causes Y (p<0.05)"
        if significant
        else "No evidence X Granger-causes Y"
    )

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "lags": lags,
        "n": n,
        "significant": significant,
        "conclusion": conclusion,
    }


def run_granger_analysis(df: pd.DataFrame) -> dict:
    """Run full battery of Granger causality tests across credit market pairs.

    Tests all 4 variable pairs in both directions at 3 lag horizons, yielding
    24 rows in the results DataFrame.  Uses first-differences of spread/VIX
    to induce approximate stationarity before testing.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns (used if present):
            ``hy_spread``                  — HY OAS %
            ``vix``                        — VIX level
            ``composite_risk_score_smooth``— 0–100 composite score
            ``sp500_daily_return``         — S&P 500 daily return (already stationary)

    Returns
    -------
    dict with keys:
        ``results``   — pd.DataFrame with 24 rows
        ``summary``   — str, human-readable summary of significant findings
        ``available`` — bool
    """
    # ------------------------------------------------------------------
    # Extract series (first-differences for level series)
    # ------------------------------------------------------------------
    series: dict[str, np.ndarray] = {}

    # HY spread — first difference for stationarity
    if "hy_spread" in df.columns:
        raw = df["hy_spread"].dropna()
        if len(raw) > 22:
            series["Δ HY Spread"] = np.diff(raw.values)

    # VIX — first difference for stationarity
    if "vix" in df.columns:
        raw = df["vix"].dropna()
        if len(raw) > 22:
            series["Δ VIX"] = np.diff(raw.values)

    # Composite score — already bounded [0,100], use first differences
    if "composite_risk_score_smooth" in df.columns:
        raw = df["composite_risk_score_smooth"].dropna()
        if len(raw) > 22:
            series["Δ Composite Score"] = np.diff(raw.values)

    # SP500 daily return — already a return, no differencing needed
    if "sp500_daily_return" in df.columns:
        raw = df["sp500_daily_return"].dropna()
        if len(raw) > 22:
            series["SP500 Return"] = raw.values

    if len(series) < 2:
        return {
            "results": pd.DataFrame(),
            "summary": "Insufficient data for Granger causality analysis.",
            "available": False,
        }

    # ------------------------------------------------------------------
    # Define test pairs (bidirectional)
    # Pairs tested (cause → effect):
    #   HY Spread ↔ VIX
    #   HY Spread ↔ Composite Score
    #   VIX ↔ Composite Score
    #   HY Spread ↔ SP500
    # ------------------------------------------------------------------
    pairs: list[tuple[str, str]] = []

    candidate_pairs = [
        ("Δ HY Spread", "Δ VIX"),
        ("Δ HY Spread", "Δ Composite Score"),
        ("Δ VIX", "Δ Composite Score"),
        ("Δ HY Spread", "SP500 Return"),
    ]

    for a, b in candidate_pairs:
        if a in series and b in series:
            pairs.append((a, b))   # a → b
            pairs.append((b, a))   # b → a (bidirectional)

    if not pairs:
        return {
            "results": pd.DataFrame(),
            "summary": "Required variable pairs not found in DataFrame.",
            "available": False,
        }

    # ------------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------------
    records: list[dict] = []
    for cause_name, effect_name in pairs:
        # Align arrays to the same length
        x_arr = series[cause_name]
        y_arr = series[effect_name]
        shared_len = min(len(x_arr), len(y_arr))
        x_arr = x_arr[-shared_len:]
        y_arr = y_arr[-shared_len:]

        for lag in LAGS:
            result = granger_test(y_arr, x_arr, lags=lag)
            records.append(
                {
                    "cause": cause_name,
                    "effect": effect_name,
                    "lags": lag,
                    "f_stat": result["f_stat"],
                    "p_value": result["p_value"],
                    "significant": result["significant"],
                    "conclusion": result["conclusion"],
                }
            )

    results_df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    sig_rows = results_df[results_df["significant"] == True]  # noqa: E712
    if sig_rows.empty:
        summary = (
            "No statistically significant Granger causality relationships "
            "found at the 5% significance level across any tested pair or lag horizon."
        )
    else:
        lines = [
            "Statistically significant Granger causality findings (p < 0.05):"
        ]
        for _, row in sig_rows.iterrows():
            lines.append(
                f"  • {row['cause']} → {row['effect']}  "
                f"[lag={row['lags']}d, F={row['f_stat']:.2f}, p={row['p_value']:.4f}]"
            )
        lines.append(
            "\nNote: Granger causality indicates predictive precedence, "
            "NOT structural/economic causation."
        )
        summary = "\n".join(lines)

    return {
        "results": results_df,
        "summary": summary,
        "available": True,
    }
