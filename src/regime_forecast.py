"""
Regime probability forecast.

Uses the historical regime transition matrix (computed fresh from
``df["final_decision"]``, weekly-sampled) to project regime probabilities
1, 4, 8, and 12 weeks forward as a probability cone.

Public API
----------
  REGIMES                   — ordered list of the four regime labels
  FORECAST_WEEKS            — horizon list [1, 4, 8, 12]
  compute_transition_matrix — build 4×4 weekly Markov matrix from df
  project_regime_probabilities — forward probability projection via matrix power
  run_regime_forecast       — top-level orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

REGIMES: list[str] = ["Risk-On", "Neutral", "Caution", "Risk-Off"]
FORECAST_WEEKS: list[int] = [1, 4, 8, 12]

_REGIME_COL: str = "final_decision"
_MIN_ROWS: int = 100          # minimum non-null rows required
_MIN_REGIMES_OBSERVED: int = 3  # at least 3 of 4 regimes must appear
_WEEKLY_STEP: int = 5          # sample every 5 business days ≈ 1 week


# ---------------------------------------------------------------------------
# Public: compute_transition_matrix
# ---------------------------------------------------------------------------

def compute_transition_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute weekly regime transition matrix from ``df["final_decision"]``.

    Samples the series every ``_WEEKLY_STEP`` rows to reduce autocorrelation,
    then counts observed transitions and normalises each row to sum to 1.
    Rows with no observed outgoing transitions are set to uniform (0.25 each).

    Parameters
    ----------
    df : Master DataFrame.  Must contain ``"final_decision"`` column.

    Returns
    -------
    np.ndarray of shape (4, 4).
    Rows and columns are ordered according to ``REGIMES``.
    """
    series = df[_REGIME_COL].dropna()
    # Weekly samples
    weekly = series.iloc[::_WEEKLY_STEP].reset_index(drop=True)

    n = len(REGIMES)
    regime_index = {r: i for i, r in enumerate(REGIMES)}

    counts = np.zeros((n, n), dtype=float)
    for k in range(len(weekly) - 1):
        from_r = weekly.iloc[k]
        to_r = weekly.iloc[k + 1]
        fi = regime_index.get(from_r, -1)
        ti = regime_index.get(to_r, -1)
        if fi >= 0 and ti >= 0:
            counts[fi, ti] += 1

    # Normalise rows
    row_sums = counts.sum(axis=1, keepdims=True)
    uniform_row = np.full((1, n), 1.0 / n)
    # Where row_sum == 0, use uniform; else divide
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(row_sums > 0, counts / row_sums, uniform_row)

    return T


# ---------------------------------------------------------------------------
# Public: project_regime_probabilities
# ---------------------------------------------------------------------------

def project_regime_probabilities(
    current_regime: str,
    transition_matrix: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Project regime probabilities ``n_steps`` weeks forward.

    Starts from a one-hot probability vector for ``current_regime`` and
    iteratively applies the transition matrix.

    Parameters
    ----------
    current_regime    : Label from ``REGIMES``.
    transition_matrix : (4, 4) row-stochastic matrix from
                        ``compute_transition_matrix()``.
    n_steps           : Number of weekly steps to project.

    Returns
    -------
    np.ndarray of shape ``(n_steps + 1, 4)``.
    Row 0 is the one-hot current state; rows 1..n_steps are projected.
    """
    regime_index = {r: i for i, r in enumerate(REGIMES)}
    n = len(REGIMES)

    # One-hot start
    p0 = np.zeros(n, dtype=float)
    idx = regime_index.get(current_regime, 0)
    p0[idx] = 1.0

    rows = np.empty((n_steps + 1, n), dtype=float)
    rows[0] = p0
    p = p0.copy()
    for step in range(1, n_steps + 1):
        p = p @ transition_matrix
        rows[step] = p

    return rows


# ---------------------------------------------------------------------------
# Internal: build sparse forecast table at FORECAST_WEEKS horizons
# ---------------------------------------------------------------------------

def _build_forecast_table(
    current_regime: str,
    transition_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Build a probability table at weeks [0] + FORECAST_WEEKS.

    Returns DataFrame with index [0, 1, 4, 8, 12] and columns = REGIMES.
    """
    max_steps = max(FORECAST_WEEKS)
    # Project all steps up to max_steps
    all_probs = project_regime_probabilities(current_regime, transition_matrix, max_steps)
    # all_probs[i] corresponds to step i (week i)

    horizons = [0] + FORECAST_WEEKS
    rows = []
    for h in horizons:
        rows.append(all_probs[h])

    return pd.DataFrame(rows, index=horizons, columns=REGIMES)


# ---------------------------------------------------------------------------
# Public: run_regime_forecast
# ---------------------------------------------------------------------------

def run_regime_forecast(df: pd.DataFrame) -> dict:
    """Top-level regime forecast orchestrator.

    Computes the weekly transition matrix from ``df["final_decision"]`` and
    projects regime probabilities at 1, 4, 8, and 12 weeks forward.

    Parameters
    ----------
    df : Master DataFrame with DatetimeIndex.  Must contain
         ``"final_decision"`` column.

    Returns
    -------
    dict:
      available              : bool
      current_regime         : str
      transition_matrix      : pd.DataFrame (4×4, labeled with REGIMES)
      forecast               : pd.DataFrame — shape (5, 4),
                               index = [0, 1, 4, 8, 12] weeks,
                               columns = REGIMES, values = probabilities
      most_likely_path       : list[str] — most probable regime at each horizon
                               in FORECAST_WEEKS order
      expected_regime_change : bool — True if most-likely at week 4 != current
      interpretation         : str
      regime_stability       : float — P(stay in current regime next week)
      n_transitions_observed : int — number of weekly transitions used
    """
    try:
        # ---- Validation ----
        if df is None or df.empty:
            return {"available": False}

        if _REGIME_COL not in df.columns:
            return {"available": False}

        clean_series = df[_REGIME_COL].dropna()
        if len(clean_series) < _MIN_ROWS:
            return {"available": False}

        observed_regimes = set(clean_series.unique())
        known_regimes_observed = observed_regimes.intersection(set(REGIMES))
        if len(known_regimes_observed) < _MIN_REGIMES_OBSERVED:
            return {"available": False}

        # ---- Current regime ----
        current_regime = str(clean_series.iloc[-1])
        if current_regime not in REGIMES:
            return {"available": False}

        # ---- Transition matrix ----
        T = compute_transition_matrix(df)

        # Count weekly observations used
        weekly = clean_series.iloc[::_WEEKLY_STEP]
        n_transitions_observed = max(0, len(weekly) - 1)

        # ---- Forecast table ----
        forecast_df = _build_forecast_table(current_regime, T)

        # ---- Most likely path (at each horizon in FORECAST_WEEKS) ----
        most_likely_path: list[str] = []
        for h in FORECAST_WEEKS:
            row = forecast_df.loc[h]
            most_likely_path.append(str(row.idxmax()))

        # ---- Expected regime change ----
        most_likely_week4 = most_likely_path[FORECAST_WEEKS.index(4)]
        expected_regime_change = most_likely_week4 != current_regime

        # ---- Regime stability ----
        regime_index = {r: i for i, r in enumerate(REGIMES)}
        ci = regime_index.get(current_regime, 0)
        regime_stability = float(T[ci, ci])

        # ---- Interpretation ----
        week1_row = forecast_df.loc[1]
        week4_row = forecast_df.loc[4]

        stay_prob_week4 = float(week4_row[current_regime])
        second_regime_week4 = str(
            week4_row.drop(current_regime).idxmax()
        )
        second_prob_week4 = float(week4_row.drop(current_regime).max())

        stability_label = (
            "stable" if regime_stability >= 0.65
            else "moderately stable" if regime_stability >= 0.45
            else "fragile"
        )

        change_note = ""
        if expected_regime_change:
            change_note = (
                f" Most likely regime at week 4 shifts to {most_likely_week4} "
                f"({week4_row[most_likely_week4]:.0%})."
            )
        else:
            change_note = (
                f" 4-week outlook: {stay_prob_week4:.0%} chance of remaining "
                f"{current_regime}, {second_prob_week4:.0%} {second_regime_week4}."
            )

        interpretation = (
            f"Currently in {current_regime}."
            f"{change_note}"
            f" Regime appears {stability_label} "
            f"(week-to-week persistence: {regime_stability:.0%})."
        )

        # ---- Build labeled transition matrix DataFrame ----
        tm_df = pd.DataFrame(
            np.round(T, 4),
            index=REGIMES,
            columns=REGIMES,
        )

        return {
            "available": True,
            "current_regime": current_regime,
            "transition_matrix": tm_df,
            "forecast": forecast_df,
            "most_likely_path": most_likely_path,
            "expected_regime_change": expected_regime_change,
            "interpretation": interpretation,
            "regime_stability": round(regime_stability, 4),
            "n_transitions_observed": n_transitions_observed,
        }

    except Exception:
        return {"available": False}
