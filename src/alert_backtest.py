"""
Alert threshold backtest.

For every alert threshold on the composite risk score, shows historically:
  - How often the alert fired (upward crossings above threshold)
  - What HY spreads / SP500 did in the next 30 days after firing
  - Precision (of all alerts fired, what % were followed by stress?)
  - Recall (of all stress events, what % were preceded by an alert?)

Definitions
-----------
  Alert fires       : composite_risk_score_smooth crosses ABOVE threshold
                      (was below on day t-1, is above on day t)
  True positive     : alert fired AND (HY spread widened >25bps OR SP500 fell
                      >3%) in the next 30 days
  Stress event      : any 30-day window where HY spread widened >25bps OR
                      SP500 fell >3%
  Precision         : TP / (TP + FP)
  Recall            : TP / (TP + FN)

Public API
----------
  THRESHOLDS              — list of evaluated alert thresholds
  FORWARD_DAYS            — look-forward window in trading days
  HY_WIDEN_THRESHOLD      — bps widening that defines stress
  SP500_DROP_THRESHOLD    — fractional drop that defines stress
  _find_crossings         — detect upward threshold crossings
  _stress_event_mask      — boolean stress-start flags
  evaluate_threshold      — precision/recall metrics for one threshold
  run_alert_backtest      — orchestrator returning full results dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

THRESHOLDS: list[float] = [30, 40, 50, 60, 70, 80]
FORWARD_DAYS: int = 30
HY_WIDEN_THRESHOLD: float = 25.0   # bps
SP500_DROP_THRESHOLD: float = 0.03  # 3 %

_MIN_ROWS: int = 504   # ~2 trading years
_SCORE_COL = "composite_risk_score_smooth"
_HY_COL = "hy_spread"
_SP500_COL = "sp500"
_ANNUALIZE = 252


# ---------------------------------------------------------------------------
# Helper: upward threshold crossings
# ---------------------------------------------------------------------------

def _find_crossings(series: pd.Series, threshold: float) -> pd.Index:
    """Find indices where series crosses ABOVE threshold.

    A crossing occurs at position i when:
      series.iloc[i-1] < threshold  AND  series.iloc[i] >= threshold

    Returns the integer positional indices (not labels) of crossings.
    """
    values = series.values.astype(float)
    n = len(values)
    if n < 2:
        return pd.Index([], dtype=int)

    # was_below[i] = True if values[i-1] < threshold
    was_below = values[:-1] < threshold      # length n-1, covers positions 0..n-2
    is_above = values[1:] >= threshold       # length n-1, covers positions 1..n-1

    crossing_positions = np.where(was_below & is_above)[0] + 1  # shift to actual index
    return pd.Index(crossing_positions.astype(int))


# ---------------------------------------------------------------------------
# Helper: stress event mask
# ---------------------------------------------------------------------------

def _stress_event_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series indexed like df: True on days that start a stress window.

    A day i is a stress-start if over the next FORWARD_DAYS calendar rows:
      HY spread widens >HY_WIDEN_THRESHOLD bps
      OR SP500 falls >SP500_DROP_THRESHOLD (fractional)

    Only computed for columns that are present; if neither is present the mask
    is all False.
    """
    n = len(df)
    mask = np.zeros(n, dtype=bool)

    has_hy = _HY_COL in df.columns
    has_sp = _SP500_COL in df.columns

    if not has_hy and not has_sp:
        return pd.Series(mask, index=df.index)

    hy_vals = df[_HY_COL].values.astype(float) if has_hy else None
    sp_vals = df[_SP500_COL].values.astype(float) if has_sp else None

    for i in range(n - FORWARD_DAYS):
        end = i + FORWARD_DAYS
        stressed = False

        if has_hy:
            hy_start = hy_vals[i]
            hy_end = hy_vals[end]
            if np.isfinite(hy_start) and np.isfinite(hy_end):
                if (hy_end - hy_start) > HY_WIDEN_THRESHOLD:
                    stressed = True

        if not stressed and has_sp:
            sp_start = sp_vals[i]
            sp_end = sp_vals[end]
            if np.isfinite(sp_start) and np.isfinite(sp_end) and sp_start > 0:
                if (sp_end - sp_start) / sp_start < -SP500_DROP_THRESHOLD:
                    stressed = True

        mask[i] = stressed

    return pd.Series(mask, index=df.index)


# ---------------------------------------------------------------------------
# Single-threshold evaluation
# ---------------------------------------------------------------------------

def evaluate_threshold(
    df: pd.DataFrame,
    threshold: float,
    stress_mask: pd.Series,
) -> dict:
    """Evaluate precision/recall/F1 for a single alert threshold.

    Parameters
    ----------
    df          : master DataFrame with DatetimeIndex
    threshold   : score level for upward crossing detection
    stress_mask : boolean Series from _stress_event_mask (same index as df)

    Returns
    -------
    dict with keys:
      threshold, n_alerts, n_stress_events,
      tp, fp, fn,
      precision (None if n_alerts == 0),
      recall    (None if n_stress_events == 0),
      f1        (None if precision or recall is None),
      avg_hy_change_30d, avg_sp500_change_30d,
      false_alarm_rate
    """
    n = len(df)
    score_series = df[_SCORE_COL]

    crossing_positions = _find_crossings(score_series, threshold)

    has_hy = _HY_COL in df.columns
    has_sp = _SP500_COL in df.columns

    hy_vals = df[_HY_COL].values.astype(float) if has_hy else None
    sp_vals = df[_SP500_COL].values.astype(float) if has_sp else None

    stress_arr = stress_mask.values.astype(bool)
    n_stress_events = int(stress_arr.sum())

    # For each alert firing, check if the following FORWARD_DAYS window is a stress event.
    # We consider an alert at position i to "cover" the stress event starting at position i
    # (same day or within FORWARD_DAYS days).  We match alerts to stress events with a
    # tolerance: alert at position i covers stress events at positions [i, i+FORWARD_DAYS].
    tp = 0
    fp = 0
    hy_changes: list[float] = []
    sp_changes: list[float] = []

    # Track which stress events were preceded by any alert (for recall / FN)
    stress_covered = np.zeros(n, dtype=bool)

    for pos in crossing_positions:
        pos = int(pos)
        end = min(pos + FORWARD_DAYS, n - 1)

        # Compute outcome metrics
        if has_hy:
            hy_s = hy_vals[pos]
            hy_e = hy_vals[end]
            if np.isfinite(hy_s) and np.isfinite(hy_e):
                hy_changes.append(hy_e - hy_s)

        if has_sp:
            sp_s = sp_vals[pos]
            sp_e = sp_vals[end]
            if np.isfinite(sp_s) and np.isfinite(sp_e) and sp_s > 0:
                sp_changes.append((sp_e - sp_s) / sp_s)

        # TP/FP: did stress occur in the window [pos, pos+FORWARD_DAYS]?
        window_slice = slice(pos, min(pos + FORWARD_DAYS + 1, n))
        is_tp = bool(stress_arr[window_slice].any())
        if is_tp:
            tp += 1
            stress_covered[window_slice] = True
        else:
            fp += 1

    fn = int(stress_arr[~stress_covered].sum())

    n_alerts = len(crossing_positions)
    precision = tp / n_alerts if n_alerts > 0 else None
    recall = tp / n_stress_events if n_stress_events > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    avg_hy = float(np.mean(hy_changes)) if hy_changes else float("nan")
    avg_sp = float(np.mean(sp_changes)) if sp_changes else float("nan")
    false_alarm_rate = fp / n_alerts if n_alerts > 0 else float("nan")

    return {
        "threshold":           threshold,
        "n_alerts":            n_alerts,
        "n_stress_events":     n_stress_events,
        "tp":                  tp,
        "fp":                  fp,
        "fn":                  fn,
        "precision":           round(precision, 4) if precision is not None else None,
        "recall":              round(recall, 4) if recall is not None else None,
        "f1":                  round(f1, 4) if f1 is not None else None,
        "avg_hy_change_30d":   round(avg_hy, 2) if np.isfinite(avg_hy) else float("nan"),
        "avg_sp500_change_30d": round(avg_sp, 4) if np.isfinite(avg_sp) else float("nan"),
        "false_alarm_rate":    round(false_alarm_rate, 4) if np.isfinite(false_alarm_rate) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Current alert level helper
# ---------------------------------------------------------------------------

def _current_alert_level(score: float) -> str:
    """Return the alert level string for the given score.

    Returns 'No Alert' if score does not exceed any threshold, otherwise
    'Alert (threshold {T})' for the lowest threshold that score exceeds.
    """
    exceeded = [t for t in sorted(THRESHOLDS) if score >= t]
    if not exceeded:
        return "No Alert"
    return f"Alert (threshold {exceeded[-1]})"


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------

def _build_interpretation(results: list[dict], best: dict | None, current_score: float) -> str:
    if best is None:
        return "Insufficient data to evaluate alert thresholds."

    lines = [
        f"Best threshold: {best['threshold']:.0f} "
        f"(F1={best['f1']:.3f}, precision={best['precision']:.1%}, recall={best['recall']:.1%}).",
        f"At threshold {best['threshold']:.0f}: {best['tp']} true positives, "
        f"{best['fp']} false positives, {best['fn']} missed stress events.",
        f"Current composite score: {current_score:.1f}.",
    ]

    if np.isfinite(best.get("avg_hy_change_30d", float("nan"))):
        lines.append(
            f"Average HY spread change 30 days after alert: "
            f"{best['avg_hy_change_30d']:+.1f} bps."
        )
    if np.isfinite(best.get("avg_sp500_change_30d", float("nan"))):
        lines.append(
            f"Average SP500 return 30 days after alert: "
            f"{best['avg_sp500_change_30d']:+.2%}."
        )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_alert_backtest(df: pd.DataFrame) -> dict:
    """Full alert threshold backtest.

    Minimum requirements
    --------------------
    - 'composite_risk_score_smooth' column must be present
    - At least one of 'hy_spread' / 'sp500' must be present
    - At least MIN_ROWS rows (504)

    Returns
    -------
    dict with keys:
      available         : bool
      results           : list[dict] (one per threshold)
      results_df        : pd.DataFrame
      best_threshold    : float
      best_f1           : float
      interpretation    : str
      current_score     : float
      current_alert_level : str
    """
    try:
        # --- Guard: minimum data requirements ---
        if df is None or df.empty:
            return {"available": False}

        if _SCORE_COL not in df.columns:
            return {"available": False}

        has_hy = _HY_COL in df.columns and df[_HY_COL].notna().sum() > 10
        has_sp = _SP500_COL in df.columns and df[_SP500_COL].notna().sum() > 10
        if not has_hy and not has_sp:
            return {"available": False}

        if len(df) < _MIN_ROWS:
            return {"available": False}

        # --- Stress event mask (computed once, reused for all thresholds) ---
        stress_mask = _stress_event_mask(df)

        # --- Evaluate each threshold ---
        results: list[dict] = []
        for thresh in THRESHOLDS:
            row = evaluate_threshold(df, thresh, stress_mask)
            results.append(row)

        # --- Best threshold by F1 ---
        valid_f1 = [(r["f1"], r) for r in results if r["f1"] is not None]
        if valid_f1:
            best_f1_val, best_row = max(valid_f1, key=lambda x: x[0])
            best_threshold = float(best_row["threshold"])
            best_f1 = float(best_f1_val)
        else:
            best_threshold = float(THRESHOLDS[0])
            best_f1 = float("nan")
            best_row = None

        # --- Current score and alert level ---
        current_score = float(df[_SCORE_COL].iloc[-1])
        alert_level = _current_alert_level(current_score)

        # --- Interpretation ---
        interp = _build_interpretation(results, best_row, current_score)

        return {
            "available":           True,
            "results":             results,
            "results_df":          pd.DataFrame(results),
            "best_threshold":      best_threshold,
            "best_f1":             best_f1,
            "interpretation":      interp,
            "current_score":       current_score,
            "current_alert_level": alert_level,
        }

    except Exception:
        return {"available": False}
