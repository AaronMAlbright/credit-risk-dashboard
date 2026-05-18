"""
Regime duration analysis — spell identification, duration statistics, and fatigue scoring.

Tracks how long the current regime has lasted relative to the historical distribution
of that regime's durations. If the current spell has lasted longer than 75% of
historical instances, it is "overdue" for a change — a concept called regime fatigue.

Key concept:
  duration_percentile = P(historical duration <= current duration)
  fatigue_score       = duration_percentile (0-100 scale)

  0  = brand new regime (shorter than all historical spells of this type)
  90 = current spell is longer than 90% of historical spells → elevated transition risk

Regime age categories:
  Young   : < 25th percentile  — likely to continue
  Mature  : 25th–75th          — typical age
  Aging   : 75th–90th          — approaching typical end
  Overdue : > 90th             — outlier, elevated transition risk

Public API
----------
  REGIMES                  — ordered list of the four regime labels
  identify_regime_spells   — extract all spells from a regime series
  compute_duration_stats   — per-regime empirical duration statistics
  get_current_spell        — current spell metadata + fatigue score
  run_regime_duration      — top-level orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

REGIMES: list[str] = ["Risk-On", "Neutral", "Caution", "Risk-Off"]

_REGIME_COL = "final_decision"
_MIN_ROWS = 100
_MIN_REGIMES = 2        # minimum distinct regimes needed to compute stats

# Age category thresholds (percentile)
_YOUNG_MAX = 25.0
_MATURE_MAX = 75.0
_AGING_MAX = 90.0


# ---------------------------------------------------------------------------
# Public: identify_regime_spells
# ---------------------------------------------------------------------------

def identify_regime_spells(regime_series: pd.Series) -> pd.DataFrame:
    """Identify all regime spells from a series of regime labels.

    A spell is a maximal contiguous run of the same regime label.
    Duration is measured in trading days (row count), not calendar days.

    Parameters
    ----------
    regime_series : pd.Series
        Series of regime labels, indexed by any index (DatetimeIndex preferred).
        NaN values are dropped before processing.

    Returns
    -------
    pd.DataFrame with columns:
      regime        : str   — regime label
      start_idx     : any   — index value at spell start (from regime_series.index)
      end_idx       : any   — index value at spell end
      duration_days : int   — number of rows in the spell (trading days)
    """
    s = regime_series.dropna()
    if s.empty:
        return pd.DataFrame(columns=["regime", "start_idx", "end_idx", "duration_days"])

    # Use positional arrays for speed
    values = s.values
    index_vals = s.index

    spells: list[dict] = []
    current_regime = values[0]
    spell_start_pos = 0

    for i in range(1, len(values)):
        if values[i] != current_regime:
            spells.append(
                {
                    "regime": current_regime,
                    "start_idx": index_vals[spell_start_pos],
                    "end_idx": index_vals[i - 1],
                    "duration_days": i - spell_start_pos,
                }
            )
            current_regime = values[i]
            spell_start_pos = i

    # Flush the last (possibly ongoing) spell
    spells.append(
        {
            "regime": current_regime,
            "start_idx": index_vals[spell_start_pos],
            "end_idx": index_vals[-1],
            "duration_days": len(values) - spell_start_pos,
        }
    )

    return pd.DataFrame(spells)


# ---------------------------------------------------------------------------
# Public: compute_duration_stats
# ---------------------------------------------------------------------------

def compute_duration_stats(spells: pd.DataFrame) -> dict:
    """Compute per-regime duration statistics from completed spells.

    Only uses rows in `spells` (the caller is responsible for excluding the
    current, possibly ongoing spell before passing in).

    Parameters
    ----------
    spells : pd.DataFrame
        Output of identify_regime_spells(), with the last (current) row removed.

    Returns
    -------
    dict mapping regime_label -> {mean, median, p25, p75, p90, p95, max, n_spells}
    Only regimes with at least one completed spell are included.
    """
    stats: dict = {}
    if spells.empty or "regime" not in spells.columns:
        return stats

    for regime in spells["regime"].unique():
        durations = spells.loc[spells["regime"] == regime, "duration_days"].dropna()
        n = len(durations)
        if n == 0:
            continue
        d = durations.values.astype(float)
        stats[str(regime)] = {
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "p25": float(np.percentile(d, 25)),
            "p75": float(np.percentile(d, 75)),
            "p90": float(np.percentile(d, 90)),
            "p95": float(np.percentile(d, 95)),
            "max": float(np.max(d)),
            "n_spells": int(n),
        }

    return stats


# ---------------------------------------------------------------------------
# Internal: empirical percentile rank
# ---------------------------------------------------------------------------

def _empirical_percentile(value: float, historical: np.ndarray) -> float:
    """P(historical <= value) as a percentage (0–100)."""
    if len(historical) == 0:
        return float("nan")
    return float(np.mean(historical <= value) * 100.0)


# ---------------------------------------------------------------------------
# Internal: age category
# ---------------------------------------------------------------------------

def _age_category(percentile: float) -> str:
    """Map a duration percentile (0–100) to an age category label."""
    if np.isnan(percentile):
        return "Unknown"
    if percentile < _YOUNG_MAX:
        return "Young"
    if percentile < _MATURE_MAX:
        return "Mature"
    if percentile < _AGING_MAX:
        return "Aging"
    return "Overdue"


# ---------------------------------------------------------------------------
# Public: get_current_spell
# ---------------------------------------------------------------------------

def get_current_spell(df: pd.DataFrame) -> dict:
    """Find the current (ongoing) regime spell and compute fatigue metrics.

    Parameters
    ----------
    df : Master DataFrame with DatetimeIndex and 'final_decision' column.

    Returns
    -------
    dict with keys:
      regime             : str
      duration_days      : int
      start_date         : str (ISO)
      duration_percentile: float  — percentile vs historical spells of same regime
      age_category       : str    — Young / Mature / Aging / Overdue
      fatigue_score      : float  — same as duration_percentile, 0-100
      days_to_p75        : int or None  — days until regime would be "Aging"
      days_to_p90        : int or None  — days until regime would be "Overdue"
    """
    try:
        if _REGIME_COL not in df.columns:
            return _empty_spell()

        regime_series = df[_REGIME_COL]
        all_spells = identify_regime_spells(regime_series)

        if all_spells.empty:
            return _empty_spell()

        # Current spell is the last one; completed spells are all but the last
        current = all_spells.iloc[-1]
        completed = all_spells.iloc[:-1]

        current_regime = str(current["regime"])
        current_duration = int(current["duration_days"])

        # start_date from the index value
        start_idx = current["start_idx"]
        if hasattr(start_idx, "date"):
            start_date = start_idx.date().isoformat()
        else:
            start_date = str(start_idx)

        # Historical durations for this regime
        same_regime = completed.loc[completed["regime"] == current_regime, "duration_days"]
        hist_durations = same_regime.values.astype(float) if not same_regime.empty else np.array([])

        # Duration percentile
        if len(hist_durations) == 0:
            duration_percentile = float("nan")
        else:
            duration_percentile = _empirical_percentile(float(current_duration), hist_durations)

        age_cat = _age_category(duration_percentile)
        fatigue_score = duration_percentile  # 0-100 same scale

        # Days to p75 and p90 thresholds
        days_to_p75: int | None = None
        days_to_p90: int | None = None

        if len(hist_durations) > 0:
            p75_val = float(np.percentile(hist_durations, 75))
            p90_val = float(np.percentile(hist_durations, 90))
            gap75 = int(np.ceil(p75_val)) - current_duration
            gap90 = int(np.ceil(p90_val)) - current_duration
            days_to_p75 = int(gap75) if gap75 > 0 else None
            days_to_p90 = int(gap90) if gap90 > 0 else None

        return {
            "regime": current_regime,
            "duration_days": current_duration,
            "start_date": start_date,
            "duration_percentile": duration_percentile,
            "age_category": age_cat,
            "fatigue_score": fatigue_score,
            "days_to_p75": days_to_p75,
            "days_to_p90": days_to_p90,
        }

    except Exception:
        return _empty_spell()


def _empty_spell() -> dict:
    return {
        "regime": None,
        "duration_days": 0,
        "start_date": None,
        "duration_percentile": float("nan"),
        "age_category": "Unknown",
        "fatigue_score": float("nan"),
        "days_to_p75": None,
        "days_to_p90": None,
    }


# ---------------------------------------------------------------------------
# Internal: regime sequence (last N transitions)
# ---------------------------------------------------------------------------

def _regime_sequence(all_spells: pd.DataFrame, n: int = 10) -> list[str]:
    """Extract the last n distinct regime labels from the spells DataFrame."""
    if all_spells.empty:
        return []
    recent = all_spells["regime"].tolist()
    return [str(r) for r in recent[-n:]]


# ---------------------------------------------------------------------------
# Internal: interpretation string
# ---------------------------------------------------------------------------

def _build_interpretation(current_spell: dict, duration_stats: dict) -> str:
    """Plain-English interpretation of current regime fatigue."""
    regime = current_spell.get("regime")
    if regime is None:
        return "Insufficient data to compute regime duration statistics."

    duration = current_spell.get("duration_days", 0)
    age_cat = current_spell.get("age_category", "Unknown")
    fatigue = current_spell.get("fatigue_score", float("nan"))
    start = current_spell.get("start_date", "unknown")
    d75 = current_spell.get("days_to_p75")
    d90 = current_spell.get("days_to_p90")

    stats = duration_stats.get(regime, {})
    median_str = f"{stats['median']:.0f}d" if stats.get("median") is not None else "N/A"
    n_spells = stats.get("n_spells", 0)

    fatigue_str = f"{fatigue:.1f}" if not np.isnan(fatigue) else "N/A"

    parts = [
        f"Current regime: {regime}, active since {start} ({duration} trading days).",
        f"Regime fatigue score: {fatigue_str}/100 — age category: {age_cat}.",
    ]

    if n_spells > 0:
        parts.append(
            f"Historical median duration for {regime}: {median_str} "
            f"(based on {n_spells} completed spell{'s' if n_spells != 1 else ''})."
        )

    if age_cat == "Young":
        parts.append(
            f"This {regime} regime is still young relative to history. "
            "Continuation is statistically the base case."
        )
    elif age_cat == "Mature":
        if d75 is not None:
            parts.append(
                f"Regime is at a typical age. It would enter the Aging zone in ~{d75} more trading day{'s' if d75 != 1 else ''}."
            )
        else:
            parts.append("Regime is at a typical age for this label.")
    elif age_cat == "Aging":
        msg = f"This {regime} spell is longer than 75% of historical instances — approaching typical end."
        if d90 is not None:
            msg += f" It would reach the Overdue threshold in ~{d90} more trading day{'s' if d90 != 1 else ''}."
        parts.append(msg)
    elif age_cat == "Overdue":
        parts.append(
            f"This {regime} spell is in the top 10% of historical durations. "
            "Regime transition risk is elevated based on duration alone — "
            "monitor for confirming signals."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public: run_regime_duration
# ---------------------------------------------------------------------------

def run_regime_duration(df: pd.DataFrame) -> dict:
    """Top-level entry point for regime duration analysis.

    Parameters
    ----------
    df : Master DataFrame with DatetimeIndex and 'final_decision' column.

    Returns
    -------
    dict with keys:
      available        : bool          — requires final_decision, >=2 regimes, >=100 rows
      current_spell    : dict          — from get_current_spell()
      duration_stats   : dict          — per-regime statistics
      all_spells       : pd.DataFrame  — all historical spells for visualisation
      regime_sequence  : list[str]     — last 10 regime labels (most recent last)
      interpretation   : str           — plain-English summary
      warning          : str or None   — if age_category in ["Aging", "Overdue"]
    """
    try:
        # Guard: require final_decision column
        if _REGIME_COL not in df.columns:
            return {"available": False}

        # Guard: minimum row count
        valid_rows = df[_REGIME_COL].dropna()
        if len(valid_rows) < _MIN_ROWS:
            return {"available": False}

        # Guard: minimum distinct regimes
        n_regimes_observed = valid_rows.nunique()
        if n_regimes_observed < _MIN_REGIMES:
            return {"available": False}

        # Identify all spells
        all_spells = identify_regime_spells(valid_rows)
        if all_spells.empty:
            return {"available": False}

        # Duration stats: exclude the last (current, possibly ongoing) spell
        completed_spells = all_spells.iloc[:-1]
        duration_stats = compute_duration_stats(completed_spells)

        # Current spell with fatigue metrics
        current_spell = get_current_spell(df)

        # Regime sequence: last 10 spells
        regime_seq = _regime_sequence(all_spells, n=10)

        # Interpretation
        interpretation = _build_interpretation(current_spell, duration_stats)

        # Warning for Aging or Overdue
        age_cat = current_spell.get("age_category", "Unknown")
        warning: str | None = None
        if age_cat == "Aging":
            regime = current_spell.get("regime", "")
            fatigue = current_spell.get("fatigue_score", float("nan"))
            warning = (
                f"{regime} regime is Aging (fatigue score {fatigue:.1f}/100) — "
                "duration exceeds the 75th historical percentile."
            )
        elif age_cat == "Overdue":
            regime = current_spell.get("regime", "")
            fatigue = current_spell.get("fatigue_score", float("nan"))
            warning = (
                f"{regime} regime is Overdue (fatigue score {fatigue:.1f}/100) — "
                "duration exceeds the 90th historical percentile. Elevated transition risk."
            )

        return {
            "available": True,
            "current_spell": current_spell,
            "duration_stats": duration_stats,
            "all_spells": all_spells,
            "regime_sequence": regime_seq,
            "interpretation": interpretation,
            "warning": warning,
        }

    except Exception:
        return {"available": False}
