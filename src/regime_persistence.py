"""
Regime persistence / dwell-time survival analysis.
Public API: run_persistence_analysis(df) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify consecutive runs of the same ``final_decision`` value.

    Returns a DataFrame with columns:
      regime, start_date, end_date, duration_days
    """
    work = df[["date", "final_decision"]].dropna(subset=["final_decision"]).copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.reset_index(drop=True)

    if work.empty:
        return pd.DataFrame(columns=["regime", "start_date", "end_date", "duration_days"])

    runs: list[dict] = []
    current_regime = work.loc[0, "final_decision"]
    run_start = work.loc[0, "date"]
    run_end = work.loc[0, "date"]

    for i in range(1, len(work)):
        regime = work.loc[i, "final_decision"]
        date = work.loc[i, "date"]
        if regime == current_regime:
            run_end = date
        else:
            duration = (run_end - run_start).days + 1
            runs.append(
                {
                    "regime": current_regime,
                    "start_date": run_start,
                    "end_date": run_end,
                    "duration_days": duration,
                }
            )
            current_regime = regime
            run_start = date
            run_end = date

    # Flush last run
    duration = (run_end - run_start).days + 1
    runs.append(
        {
            "regime": current_regime,
            "start_date": run_start,
            "end_date": run_end,
            "duration_days": duration,
        }
    )

    return pd.DataFrame(runs)


def _build_survival_curve(durations: pd.Series) -> pd.Series:
    """
    Empirical survival curve: P(duration > t) for t in 1..max_duration.

    Returns a Series indexed by t (int).
    """
    if durations.empty:
        return pd.Series(dtype=float)

    max_t = int(durations.max())
    t_values = range(1, max_t + 1)
    n = len(durations)
    survival = pd.Series(
        [float((durations > t).sum()) / n for t in t_values],
        index=list(t_values),
        name="survival_prob",
    )
    return survival


def _survival_at(curve: pd.Series, days_already_in: int, horizon: int) -> float:
    """
    P(regime lasts at least *horizon* more days | already *days_already_in* days in).

    Uses the memoryless-free empirical approach:
      P(T > days_already_in + horizon) / P(T > days_already_in)
    where P(T > t) is read from *curve*.
    """
    if curve.empty:
        return float("nan")

    def _p(t: int) -> float:
        if t <= 0:
            return 1.0
        if t in curve.index:
            return float(curve.loc[t])
        max_t = int(curve.index.max())
        if t > max_t:
            return 0.0
        return float(curve.iloc[-1])

    p_denom = _p(days_already_in)
    if p_denom == 0:
        return 0.0
    return _p(days_already_in + horizon) / p_denom


def run_persistence_analysis(df: pd.DataFrame) -> dict:
    """
    Analyse regime persistence / dwell-time survival.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``date`` and ``final_decision``.

    Returns
    -------
    dict with keys:
      current_regime, days_in_regime, regime_start_date,
      survival_probs ({10, 20, 30, 60} -> float),
      median_dwell, mean_dwell,
      dwell_times (DataFrame), survival_curves (dict of Series)
    """
    runs = _compute_runs(df)

    if runs.empty:
        return {
            "current_regime": None,
            "days_in_regime": 0,
            "regime_start_date": None,
            "survival_probs": {10: float("nan"), 20: float("nan"), 30: float("nan"), 60: float("nan")},
            "median_dwell": float("nan"),
            "mean_dwell": float("nan"),
            "dwell_times": runs,
            "survival_curves": {},
        }

    # Current (last) run — it may still be ongoing so its duration is a lower bound.
    last_run = runs.iloc[-1]
    current_regime: str = str(last_run["regime"])
    regime_start_date: str = str(last_run["start_date"].date())

    # Days-in-regime: use today if the run is still open (end_date == last data row).
    today = pd.Timestamp.today().normalize()
    last_data_date = pd.to_datetime(df["date"]).max()
    if last_run["end_date"] >= last_data_date:
        days_in_regime = int((today - last_run["start_date"]).days) + 1
    else:
        days_in_regime = int(last_run["duration_days"])

    # Build per-regime survival curves (exclude the still-open last run from fitting
    # to avoid censoring bias; include it if we have enough history).
    historical_runs = runs.iloc[:-1]  # exclude the ongoing run

    survival_curves: dict[str, pd.Series] = {}
    for regime in runs["regime"].unique():
        regime_runs = historical_runs[historical_runs["regime"] == regime]["duration_days"]
        if not regime_runs.empty:
            survival_curves[regime] = _build_survival_curve(regime_runs)

    # Survival probabilities for the current regime
    horizons = [10, 20, 30, 60]
    curve = survival_curves.get(current_regime, pd.Series(dtype=float))
    survival_probs = {h: _survival_at(curve, days_in_regime, h) for h in horizons}

    # Dwell statistics for current regime
    current_regime_durations = historical_runs[historical_runs["regime"] == current_regime]["duration_days"]
    median_dwell = float(current_regime_durations.median()) if not current_regime_durations.empty else float("nan")
    mean_dwell = float(current_regime_durations.mean()) if not current_regime_durations.empty else float("nan")

    return {
        "current_regime": current_regime,
        "days_in_regime": days_in_regime,
        "regime_start_date": regime_start_date,
        "survival_probs": survival_probs,
        "median_dwell": median_dwell,
        "mean_dwell": mean_dwell,
        "dwell_times": runs,
        "survival_curves": survival_curves,
    }
