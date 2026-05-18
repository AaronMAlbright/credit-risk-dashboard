"""
Drawdown & Recovery Time Estimator.

For any ongoing drawdown in HY spreads or the composite risk score, shows:
  - Current drawdown depth vs the rolling 252-day extreme
  - Empirical distribution of recovery times from comparable historical episodes
  - Answers: "How long until this normalises?"

Terminology (spread-space conventions):
  "Drawdown" = how far above the 252-day low are spreads right now?
               (i.e. widening from the prior trough = a bad event in credit)
  "Recovery" = return to within RECOVERY_THRESHOLD of the prior trough (low
               point), meaning spreads have tightened back toward prior lows.

For equity / price series (invert=True):
  "Drawdown" = how far below the 252-day high is the price?
  "Recovery" = return to within 10% of the prior peak.

Module constants
----------------
  DRAWDOWN_ASSETS     — list of (col_name, label, invert) tuples
  RECOVERY_THRESHOLD  — within 10% of prior trough = recovered
  MIN_DRAWDOWN_BPS    — minimum 10 bps (or score points) to count

Public API
----------
  identify_drawdown_episodes(series, invert, min_depth) -> pd.DataFrame
  get_current_drawdown(series, invert)                  -> dict
  estimate_recovery_time(current_depth, episodes_df)    -> dict
  run_drawdown_recovery(df)                             -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DRAWDOWN_ASSETS: list[tuple[str, str, bool]] = [
    ("hy_spread",                   "HY Spread",    False),  # higher = worse
    ("composite_risk_score_smooth", "Composite",    False),  # higher = worse
    ("ig_spread",                   "IG Spread",    False),  # higher = worse
    ("sp500",                       "S&P 500",      True),   # lower = worse
]

RECOVERY_THRESHOLD: float = 0.10   # within 10% of prior trough = recovered
MIN_DRAWDOWN_BPS:  float = 10.0    # minimum depth (bps or score points)

_HISTORY_ROWS  = 504   # ~2 trading years for historical slices
_PEAK_WINDOW   = 252   # rolling window for "current drawdown" baseline
_EPISODE_SEP   = 21    # minimum days between distinct trough events
_ROLL_EXTREME_WINDOW = 63   # rolling window for episode detection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float:
    """Convert to float; return nan on failure."""
    try:
        f = float(val)
        return f if np.isfinite(f) else float("nan")
    except Exception:
        return float("nan")


def _percentile_rank(value: float, series: pd.Series) -> float:
    """Fraction of series at or below value (0–100)."""
    clean = series.dropna()
    if clean.empty or np.isnan(value):
        return float("nan")
    return float((clean <= value).mean() * 100.0)


def _index_to_str(idx) -> str:
    """Convert DatetimeIndex element or integer to a readable string."""
    try:
        return str(pd.Timestamp(idx).date())
    except Exception:
        return str(idx)


# ---------------------------------------------------------------------------
# Public: identify_drawdown_episodes
# ---------------------------------------------------------------------------

def identify_drawdown_episodes(
    series: pd.Series,
    invert: bool,
    min_depth: float = MIN_DRAWDOWN_BPS,
) -> pd.DataFrame:
    """
    Identify historical drawdown episodes in a single time series.

    For spread / score series (invert=False):
      - "drawdown" = widening above the rolling 63-day minimum
      - "depth" = trough_val - peak_val  (positive = widening = bad)

    For price series (invert=True):
      - "drawdown" = falling below the rolling 63-day maximum
      - "depth" = peak_val - trough_val  (positive = decline = bad)

    Algorithm:
      1. Compute rolling 63-day extreme (min for spread, max for price).
      2. Define "extreme_deviation" = deviation from that extreme.
      3. An episode starts when extreme_deviation > min_depth.
      4. Track from the start of each run to its resolution.
      5. Enforce minimum separation between troughs of _EPISODE_SEP days.

    Returns
    -------
    pd.DataFrame with columns:
        peak_date, trough_date, peak_val, trough_val,
        depth, recovery_days  (NaN if not yet recovered)
    """
    _empty = pd.DataFrame(columns=[
        "peak_date", "trough_date", "peak_val", "trough_val",
        "depth", "recovery_days",
    ])

    try:
        s = series.dropna()
        if len(s) < _ROLL_EXTREME_WINDOW + _EPISODE_SEP:
            return _empty

        vals = s.values.astype(float)
        idx  = s.index
        n    = len(vals)

        roll_win = _ROLL_EXTREME_WINDOW

        # Rolling extreme via pandas (O(n), vectorised)
        if invert:
            # For prices: rolling max (deviation below = bad)
            roll_ext = s.rolling(window=roll_win, min_periods=1).max().values
        else:
            # For spreads: rolling min (deviation above = bad)
            roll_ext = s.rolling(window=roll_win, min_periods=1).min().values

        if invert:
            # Price: depth = extreme(max) - current  (positive = fell below peak)
            deviation = roll_ext - vals
        else:
            # Spread: depth = current - extreme(min)  (positive = above trough)
            deviation = vals - roll_ext

        # Identify positions where we are "in drawdown" (beyond min_depth)
        in_dd = deviation > min_depth

        # Walk through episodes
        episodes: list[dict] = []
        last_trough_pos: int = -(_EPISODE_SEP + 1)  # force first episode to start clean

        ep_start: int | None = None
        for i in range(n):
            if in_dd[i] and ep_start is None:
                ep_start = i
            elif not in_dd[i] and ep_start is not None:
                # Episode ended — find peak and trough within episode
                ep_end = i - 1
                ep_vals = vals[ep_start : ep_end + 1]

                if invert:
                    trough_local = int(np.argmin(ep_vals))
                    peak_local   = int(np.argmax(ep_vals[:trough_local + 1])) if trough_local > 0 else 0
                else:
                    trough_local = int(np.argmax(ep_vals))
                    peak_local   = int(np.argmin(ep_vals[:trough_local + 1])) if trough_local > 0 else 0

                trough_abs = ep_start + trough_local
                peak_abs   = ep_start + peak_local

                # Enforce minimum separation from last trough
                if trough_abs - last_trough_pos < _EPISODE_SEP:
                    ep_start = None
                    continue

                peak_val   = float(vals[peak_abs])
                trough_val = float(vals[trough_abs])
                depth      = (peak_val - trough_val) if invert else (trough_val - peak_val)

                if depth < min_depth:
                    ep_start = None
                    continue

                # Recovery: find first post-trough point within RECOVERY_THRESHOLD
                recovery_days: float = float("nan")
                threshold_val = trough_val * (1 + RECOVERY_THRESHOLD) if invert else trough_val * (1 - RECOVERY_THRESHOLD)
                for j in range(trough_abs + 1, n):
                    if invert:
                        # Price recovered if back above threshold
                        if vals[j] >= threshold_val:
                            recovery_days = float(j - trough_abs)
                            break
                    else:
                        # Spread recovered if back below threshold
                        if vals[j] <= threshold_val:
                            recovery_days = float(j - trough_abs)
                            break

                episodes.append({
                    "peak_date":      _index_to_str(idx[peak_abs]),
                    "trough_date":    _index_to_str(idx[trough_abs]),
                    "peak_val":       peak_val,
                    "trough_val":     trough_val,
                    "depth":          depth,
                    "recovery_days":  recovery_days,
                })
                last_trough_pos = trough_abs
                ep_start = None

        # Handle open episode at end of series
        if ep_start is not None:
            ep_vals = vals[ep_start:]

            if invert:
                trough_local = int(np.argmin(ep_vals))
                peak_local   = int(np.argmax(ep_vals[:trough_local + 1])) if trough_local > 0 else 0
            else:
                trough_local = int(np.argmax(ep_vals))
                peak_local   = int(np.argmin(ep_vals[:trough_local + 1])) if trough_local > 0 else 0

            trough_abs = ep_start + trough_local
            peak_abs   = ep_start + peak_local

            if trough_abs - last_trough_pos >= _EPISODE_SEP:
                peak_val   = float(vals[peak_abs])
                trough_val = float(vals[trough_abs])
                depth      = (peak_val - trough_val) if invert else (trough_val - peak_val)

                if depth >= min_depth:
                    episodes.append({
                        "peak_date":     _index_to_str(idx[peak_abs]),
                        "trough_date":   _index_to_str(idx[trough_abs]),
                        "peak_val":      peak_val,
                        "trough_val":    trough_val,
                        "depth":         depth,
                        "recovery_days": float("nan"),  # not yet recovered
                    })

        if not episodes:
            return _empty

        return pd.DataFrame(episodes, columns=[
            "peak_date", "trough_date", "peak_val", "trough_val",
            "depth", "recovery_days",
        ])

    except Exception:
        return _empty


# ---------------------------------------------------------------------------
# Public: get_current_drawdown
# ---------------------------------------------------------------------------

def get_current_drawdown(series: pd.Series, invert: bool) -> dict:
    """
    Return the current drawdown state for a single series.

    Parameters
    ----------
    series : pd.Series  — clean time series aligned to DatetimeIndex
    invert : bool       — True for price (lower = bad), False for spread/score

    Returns
    -------
    dict with keys:
        in_drawdown            : bool
        current_val            : float
        drawdown_depth         : float  — bps or points from rolling extreme
        drawdown_pct           : float  — % move from rolling extreme
        drawdown_start_approx  : str    — approx date drawdown began
        percentile_vs_history  : float  — depth vs historical episodes (0-100)
    """
    _empty: dict = {
        "in_drawdown": False,
        "current_val": float("nan"),
        "drawdown_depth": float("nan"),
        "drawdown_pct": float("nan"),
        "drawdown_start_approx": "Unknown",
        "percentile_vs_history": float("nan"),
    }

    try:
        s = series.dropna()
        if len(s) < 2:
            return _empty

        current_val = _safe_float(s.iloc[-1])
        if np.isnan(current_val):
            return _empty

        # Rolling extreme over 252 days
        win = min(_PEAK_WINDOW, len(s))
        recent = s.tail(win)

        if invert:
            extreme_val = float(recent.max())       # price peak
            drawdown_depth = extreme_val - current_val
            drawdown_pct = (drawdown_depth / extreme_val * 100.0) if extreme_val != 0 else float("nan")
        else:
            extreme_val = float(recent.min())       # spread trough (lowest = best)
            drawdown_depth = current_val - extreme_val
            drawdown_pct = (drawdown_depth / abs(extreme_val) * 100.0) if extreme_val != 0 else float("nan")

        in_drawdown = drawdown_depth >= MIN_DRAWDOWN_BPS

        # Approximate drawdown start: last point where depth first exceeded MIN_DRAWDOWN_BPS
        drawdown_start_approx = "Unknown"
        if in_drawdown:
            try:
                if invert:
                    above_thresh = (extreme_val - s) >= MIN_DRAWDOWN_BPS
                else:
                    # Recompute rolling min along the full series to find when deviation started
                    rolling_min = s.rolling(window=_PEAK_WINDOW, min_periods=1).min()
                    above_thresh = (s - rolling_min) >= MIN_DRAWDOWN_BPS

                # Find the first consecutive run of above_thresh ending at the current date
                above_arr = above_thresh.values
                end_pos = len(above_arr) - 1
                start_pos = end_pos
                while start_pos > 0 and above_arr[start_pos - 1]:
                    start_pos -= 1
                drawdown_start_approx = _index_to_str(s.index[start_pos])
            except Exception:
                drawdown_start_approx = "Unknown"

        # Percentile of current depth vs historical episodes
        all_episodes = identify_drawdown_episodes(s, invert, MIN_DRAWDOWN_BPS)
        pct_vs_history: float = float("nan")
        if not all_episodes.empty and in_drawdown:
            pct_vs_history = _percentile_rank(drawdown_depth, all_episodes["depth"])

        return {
            "in_drawdown": bool(in_drawdown),
            "current_val": round(current_val, 4),
            "drawdown_depth": round(drawdown_depth, 4),
            "drawdown_pct": round(drawdown_pct, 2) if not np.isnan(drawdown_pct) else float("nan"),
            "drawdown_start_approx": drawdown_start_approx,
            "percentile_vs_history": round(pct_vs_history, 1) if not np.isnan(pct_vs_history) else float("nan"),
        }

    except Exception:
        return _empty


# ---------------------------------------------------------------------------
# Public: estimate_recovery_time
# ---------------------------------------------------------------------------

def estimate_recovery_time(
    current_depth: float,
    historical_episodes: pd.DataFrame,
) -> dict:
    """
    Estimate recovery time for current drawdown from historical data.

    Finds all historical episodes with depth >= current_depth and returns
    the empirical distribution of their recovery times.

    Parameters
    ----------
    current_depth       : float  — depth of the current drawdown
    historical_episodes : pd.DataFrame  — output of identify_drawdown_episodes()

    Returns
    -------
    dict with keys:
        median_recovery_days : float or None
        p25_recovery_days    : float or None
        p75_recovery_days    : float or None
        n_comparable         : int
        pct_recovered        : float  — fraction of comparable episodes that recovered
    """
    _empty_est: dict = {
        "median_recovery_days": None,
        "p25_recovery_days":    None,
        "p75_recovery_days":    None,
        "n_comparable":         0,
        "pct_recovered":        0.0,
    }

    try:
        if historical_episodes.empty or np.isnan(current_depth) or current_depth <= 0:
            return _empty_est
        if "depth" not in historical_episodes.columns or "recovery_days" not in historical_episodes.columns:
            return _empty_est

        comparable = historical_episodes[historical_episodes["depth"] >= current_depth].copy()
        n_comparable = len(comparable)
        if n_comparable == 0:
            return _empty_est

        recovered = comparable["recovery_days"].dropna()
        pct_recovered = float(len(recovered) / n_comparable)

        if recovered.empty:
            return {
                "median_recovery_days": None,
                "p25_recovery_days":    None,
                "p75_recovery_days":    None,
                "n_comparable":         n_comparable,
                "pct_recovered":        pct_recovered,
            }

        rec_arr = recovered.values.astype(float)
        median_days = float(np.median(rec_arr))
        p25_days    = float(np.percentile(rec_arr, 25))
        p75_days    = float(np.percentile(rec_arr, 75))

        return {
            "median_recovery_days": round(median_days, 1),
            "p25_recovery_days":    round(p25_days, 1),
            "p75_recovery_days":    round(p75_days, 1),
            "n_comparable":         n_comparable,
            "pct_recovered":        round(pct_recovered, 3),
        }

    except Exception:
        return _empty_est


# ---------------------------------------------------------------------------
# Public: run_drawdown_recovery
# ---------------------------------------------------------------------------

def run_drawdown_recovery(df: pd.DataFrame) -> dict:
    """
    Top-level entry point for the drawdown & recovery module.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex.  Requires hy_spread OR
        composite_risk_score_smooth, plus >= 252 rows.

    Returns
    -------
    dict with keys:
        available         : bool
        assets            : dict  — keyed by label, each value:
                            {episodes: pd.DataFrame, current: dict,
                             recovery_estimate: dict}
        primary_asset     : str   — "HY Spread" if available, else "Composite"
        current_depth     : float
        recovery_estimate : dict
        all_episodes      : pd.DataFrame — from primary asset
        interpretation    : str
        warning           : str or None
    """
    _unavail: dict = {"available": False}

    try:
        # ---- availability check -------------------------------------------
        has_hy   = "hy_spread" in df.columns
        has_comp = "composite_risk_score_smooth" in df.columns
        if not has_hy and not has_comp:
            return _unavail
        if len(df) < 252:
            return _unavail

        # ---- per-asset analysis -------------------------------------------
        assets: dict = {}
        for col_name, label, invert in DRAWDOWN_ASSETS:
            if col_name not in df.columns:
                continue
            series = df[col_name].dropna()
            if len(series) < 63:
                continue

            episodes = identify_drawdown_episodes(series, invert, MIN_DRAWDOWN_BPS)
            current  = get_current_drawdown(series, invert)

            depth_for_est = current.get("drawdown_depth", float("nan"))
            if np.isnan(depth_for_est):
                depth_for_est = 0.0
            rec_est = estimate_recovery_time(depth_for_est, episodes)

            assets[label] = {
                "episodes":          episodes,
                "current":           current,
                "recovery_estimate": rec_est,
            }

        if not assets:
            return _unavail

        # ---- primary asset ------------------------------------------------
        if "HY Spread" in assets:
            primary_asset = "HY Spread"
        elif "Composite" in assets:
            primary_asset = "Composite"
        else:
            primary_asset = next(iter(assets))

        primary_data     = assets[primary_asset]
        all_episodes     = primary_data["episodes"]
        current_info     = primary_data["current"]
        current_depth    = current_info.get("drawdown_depth", float("nan"))
        recovery_estimate = primary_data["recovery_estimate"]

        # ---- interpretation -----------------------------------------------
        in_dd = current_info.get("in_drawdown", False)
        if in_dd:
            depth_val = round(current_depth, 1) if not np.isnan(current_depth) else "N/A"
            pct_vs_hist = current_info.get("percentile_vs_history", float("nan"))
            start_date  = current_info.get("drawdown_start_approx", "Unknown")
            med_rec     = recovery_estimate.get("median_recovery_days")
            n_comp      = recovery_estimate.get("n_comparable", 0)
            pct_rec     = recovery_estimate.get("pct_recovered", 0.0)

            pct_str = f"{round(pct_vs_hist, 0):.0f}th" if not np.isnan(pct_vs_hist) else "unknown"
            rec_str = (
                f"Historically, {pct_rec*100:.0f}% of comparable episodes recovered, "
                f"with a median recovery time of {med_rec:.0f} days "
                f"(based on {n_comp} comparable episodes)."
                if med_rec is not None and n_comp > 0
                else f"Insufficient historical episodes at this depth to estimate recovery time."
            )

            interpretation = (
                f"{primary_asset} is currently in a drawdown of {depth_val} bps/points "
                f"(approx. since {start_date}), ranking at the {pct_str} percentile of "
                f"historical depth. {rec_str}"
            )
        else:
            depth_val = round(current_depth, 1) if not np.isnan(current_depth) else 0
            interpretation = (
                f"{primary_asset} is not in a significant drawdown. "
                f"Current deviation from the 252-day extreme is {depth_val} bps/points, "
                f"below the {MIN_DRAWDOWN_BPS} bps minimum threshold."
            )

        # ---- warning ------------------------------------------------------
        warning: str | None = None
        if not has_hy:
            warning = (
                "HY Spread not available — drawdown analysis is based on "
                f"{primary_asset} only, which may behave differently."
            )
        n_assets_available = len(assets)
        if n_assets_available < len(DRAWDOWN_ASSETS):
            miss = [
                label
                for _, label, _ in DRAWDOWN_ASSETS
                if label not in assets
            ]
            miss_str = ", ".join(miss)
            w2 = f"Missing columns for: {miss_str}."
            warning = f"{warning}  {w2}" if warning else w2

        return {
            "available":          True,
            "assets":             assets,
            "primary_asset":      primary_asset,
            "current_depth":      float(current_depth) if not np.isnan(current_depth) else float("nan"),
            "recovery_estimate":  recovery_estimate,
            "all_episodes":       all_episodes,
            "interpretation":     interpretation,
            "warning":            warning,
        }

    except Exception:
        return _unavail
