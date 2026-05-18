"""
Rolling correlation heatmap — cross-asset correlation stress detector.

When correlations across asset classes spike toward 1.0 in a downturn,
diversification has failed: every position moves against you at once.
This is a hallmark of credit crises (2008, 2020).

Monitoring rolling 90-day correlations across 8 core market variables
provides an early warning of diversification breakdown.

Stress thresholds (average absolute off-diagonal correlation):
  Normal   < 0.45 : diversification is working
  Elevated 0.45–0.60 : correlations rising, watch closely
  Stress   0.60–0.75 : high correlation regime, reduce gross exposure
  Crisis   > 0.75 : near-perfect co-movement, systemic risk

Public API
----------
  HEATMAP_SIGNALS  : list of (col, label) tuples — module constant
  DEFAULT_WINDOW   : int = 90
  compute_rolling_correlation_matrix(df, window) → dict
  detect_correlation_stress(matrix_current)       → dict
  run_correlation_heatmap_analysis(df, window)    → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEATMAP_SIGNALS: list[tuple[str, str]] = [
    ("hy_spread",  "HY Spread"),
    ("ig_spread",  "IG Spread"),
    ("vix",        "VIX"),
    ("yield_10y",  "10y Yield"),
    ("sp500",      "SP500"),
    ("yield_2y",   "2y Yield"),
    ("dxy",        "DXY"),
    ("oil",        "Oil"),
]

DEFAULT_WINDOW: int = 90

_MIN_SIGNAL_COUNT = 4   # minimum signals needed to call analysis "available"
_MIN_OBS = 20           # minimum joint observations in a window


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _available_signals(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return (available_cols, available_labels) for signals that exist in df
    and have at least some non-NaN values.
    """
    avail_cols: list[str] = []
    avail_labels: list[str] = []
    for col, label in HEATMAP_SIGNALS:
        if col in df.columns and df[col].notna().any():
            avail_cols.append(col)
            avail_labels.append(label)
    return avail_cols, avail_labels


def _avg_abs_offdiag(matrix: pd.DataFrame) -> float:
    """
    Compute the mean absolute value of all strictly off-diagonal elements
    in a square correlation matrix.

    Returns NaN if the matrix has fewer than 2 signals.
    """
    n = len(matrix)
    if n < 2:
        return float("nan")
    vals = matrix.values.copy()
    # Mask diagonal with NaN
    np.fill_diagonal(vals, float("nan"))
    flat = vals.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) == 0:
        return float("nan")
    return float(np.mean(np.abs(flat)))


def _stress_regime(avg_abs: float) -> str:
    """Map average absolute off-diagonal correlation to a regime label."""
    if pd.isna(avg_abs):
        return "Unknown"
    if avg_abs > 0.75:
        return "Crisis"
    if avg_abs > 0.60:
        return "Stress"
    if avg_abs > 0.45:
        return "Elevated"
    return "Normal"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rolling_correlation_matrix(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> dict:
    """
    Compute a rolling correlation matrix for the available HEATMAP_SIGNALS.

    The ``matrix_current`` uses the last ``window`` rows of price-return
    data (percent changes) to give a snapshot of the current correlation
    regime.  Display labels (not raw column names) are used as the
    DataFrame index and columns so the matrix renders cleanly in
    ``st.dataframe()``.

    Signals that are entirely NaN over the last window are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex.
    window : int
        Rolling window in trading days (default 90).

    Returns
    -------
    dict with keys:
        matrix_current    : pd.DataFrame — square correlation matrix (labels)
        matrix_labels     : list[str] — display labels for available signals
        available_signals : list[str] — column names present in df
        n_signals         : int
    """
    avail_cols, avail_labels = _available_signals(df)

    if len(avail_cols) < 2:
        empty = pd.DataFrame()
        return {
            "matrix_current": empty,
            "matrix_labels": avail_labels,
            "available_signals": avail_cols,
            "n_signals": len(avail_cols),
        }

    # Percent-change returns for correlation (stationarises series)
    returns = df[avail_cols].pct_change()

    # Last-window slice for current snapshot
    window_data = returns.iloc[-window:]

    # Drop signals that are all-NaN over the last window
    valid_mask = window_data.notna().sum() >= _MIN_OBS
    valid_cols = [c for c, ok in zip(avail_cols, valid_mask) if ok]
    valid_labels = [lbl for c, lbl, ok in zip(avail_cols, avail_labels, valid_mask) if ok]

    if len(valid_cols) < 2:
        empty = pd.DataFrame()
        return {
            "matrix_current": empty,
            "matrix_labels": valid_labels,
            "available_signals": valid_cols,
            "n_signals": len(valid_cols),
        }

    # Correlation matrix for current window, with display labels
    corr_raw = window_data[valid_cols].corr()
    corr_raw.index = valid_labels
    corr_raw.columns = valid_labels

    return {
        "matrix_current": corr_raw,
        "matrix_labels": valid_labels,
        "available_signals": valid_cols,
        "n_signals": len(valid_cols),
    }


def detect_correlation_stress(matrix_current: pd.DataFrame) -> dict:
    """
    Identify correlation stress from a square correlation matrix.

    Parameters
    ----------
    matrix_current : pd.DataFrame
        Square correlation matrix with display labels as both index and columns
        (as returned by ``compute_rolling_correlation_matrix``).

    Returns
    -------
    dict with keys:
        avg_abs_correlation  : float — mean abs off-diagonal correlation
        stress_regime        : str — "Normal" / "Elevated" / "Stress" / "Crisis"
        max_pair             : tuple (label_a, label_b, corr) — most correlated pair
        min_pair             : tuple (label_a, label_b, corr) — least correlated pair
        diversification_ratio : float — 1 - avg_abs_correlation
    """
    if matrix_current.empty or len(matrix_current) < 2:
        return {
            "avg_abs_correlation": float("nan"),
            "stress_regime": "Unknown",
            "max_pair": (None, None, float("nan")),
            "min_pair": (None, None, float("nan")),
            "diversification_ratio": float("nan"),
        }

    labels = list(matrix_current.index)
    n = len(labels)
    avg_abs = _avg_abs_offdiag(matrix_current)
    stress_regime = _stress_regime(avg_abs)
    diversification_ratio = (1.0 - avg_abs) if not pd.isna(avg_abs) else float("nan")

    # Find max and min off-diagonal pairs (upper triangle only to avoid duplicates)
    max_corr = -2.0
    min_corr = 2.0
    max_pair: tuple = (None, None, float("nan"))
    min_pair: tuple = (None, None, float("nan"))

    for i in range(n):
        for j in range(i + 1, n):
            val = matrix_current.iloc[i, j]
            if pd.isna(val):
                continue
            abs_val = abs(val)
            if abs_val > max_corr:
                max_corr = abs_val
                max_pair = (labels[i], labels[j], float(val))
            if abs_val < min_corr:
                min_corr = abs_val
                min_pair = (labels[i], labels[j], float(val))

    return {
        "avg_abs_correlation": avg_abs,
        "stress_regime": stress_regime,
        "max_pair": max_pair,
        "min_pair": min_pair,
        "diversification_ratio": diversification_ratio,
    }


def run_correlation_heatmap_analysis(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> dict:
    """
    Full rolling correlation heatmap analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Main market data frame with DatetimeIndex.
    window : int
        Rolling window in trading days (default 90).

    Returns
    -------
    dict with keys:
        matrix_current     : pd.DataFrame — square correlation matrix (labels)
        stress             : dict from detect_correlation_stress()
        rolling_avg_corr   : pd.Series — rolling average abs off-diagonal
                             correlation, last 504 rows, DatetimeIndex
        available          : bool — True if >= 4 signals have data
        n_signals          : int
        interpretation     : one-sentence string
    """
    avail_cols, avail_labels = _available_signals(df)
    available = len(avail_cols) >= _MIN_SIGNAL_COUNT

    # ---- Current correlation matrix -----------------------------------------
    matrix_result = compute_rolling_correlation_matrix(df, window=window)
    matrix_current = matrix_result["matrix_current"]
    stress = detect_correlation_stress(matrix_current)

    # ---- Rolling average abs off-diagonal correlation time series -----------
    # Strategy: use pandas multivariate rolling corr, then average abs off-diag
    # per date.  This is the canonical pandas approach for a rolling N×N matrix.
    rolling_avg_corr = _compute_rolling_avg_corr(df, avail_cols, window)

    # ---- Interpretation string ----------------------------------------------
    regime = stress.get("stress_regime", "Unknown")
    avg_c = stress.get("avg_abs_correlation", float("nan"))
    max_pair = stress.get("max_pair", (None, None, float("nan")))
    div_ratio = stress.get("diversification_ratio", float("nan"))

    if available and not pd.isna(avg_c):
        max_a, max_b, max_c = max_pair
        max_str = (
            f"{max_a}/{max_b} at {max_c:.2f}"
            if max_a is not None and not pd.isna(max_c)
            else "N/A"
        )
        div_str = f"{div_ratio:.2f}" if not pd.isna(div_ratio) else "N/A"
        if regime == "Crisis":
            implication = (
                "diversification has nearly failed — reduce gross exposure and "
                "avoid adding new risk positions"
            )
        elif regime == "Stress":
            implication = (
                "portfolio diversification is significantly impaired — "
                "correlation hedges may be less effective"
            )
        elif regime == "Elevated":
            implication = (
                "cross-asset correlations are elevated — monitor for further "
                "convergence before adding spread duration"
            )
        else:
            implication = (
                "diversification is broadly intact — normal cross-asset "
                "relationships are holding"
            )
        interpretation = (
            f"90-day average abs correlation = {avg_c:.2f} ({regime} regime; "
            f"diversification ratio {div_str}); most correlated pair is {max_str}; "
            f"{implication}."
        )
    else:
        interpretation = (
            "Insufficient signal coverage to compute correlation stress regime "
            f"(need >= {_MIN_SIGNAL_COUNT} signals, found {len(avail_cols)})."
        )

    return {
        "matrix_current": matrix_current,
        "stress": stress,
        "rolling_avg_corr": rolling_avg_corr,
        "available": available,
        "n_signals": len(avail_cols),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Rolling average correlation helper (kept private to this module)
# ---------------------------------------------------------------------------

def _compute_rolling_avg_corr(
    df: pd.DataFrame,
    avail_cols: list[str],
    window: int,
) -> pd.Series:
    """
    Compute a time series of the average absolute off-diagonal pairwise
    correlation, using a rolling ``window``-period window.

    Implementation
    --------------
    pandas ``rolling(window).corr()`` on pct_change returns produces a
    MultiIndex Series (date, variable) × variable.  We extract the mean
    absolute off-diagonal value per date.

    Returns the last 504 rows as a pd.Series with DatetimeIndex.
    Returns an empty Series if fewer than 2 signals are available.
    """
    if len(avail_cols) < 2:
        return pd.Series(dtype=float, name="rolling_avg_corr")

    returns = df[avail_cols].pct_change()

    # pandas rolling().corr() returns a MultiIndex (outer=date, inner=col)
    # MultiIndex Series with columns = col names
    try:
        rolling_corr = returns.rolling(window, min_periods=_MIN_OBS).corr()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float, name="rolling_avg_corr")

    # rolling_corr has MultiIndex: (date, col_i) × col_j
    # We want one scalar per date: mean abs off-diagonal
    n_signals = len(avail_cols)
    dates = df.index

    avg_series_values = np.full(len(dates), float("nan"))

    for idx_pos, date in enumerate(dates):
        try:
            # Slice the cross-section for this date
            mat = rolling_corr.loc[date]
            if isinstance(mat, pd.DataFrame) and mat.shape == (n_signals, n_signals):
                avg_series_values[idx_pos] = _avg_abs_offdiag(mat)
            # If mat is not the right shape (e.g. some NaN columns), skip
        except KeyError:
            pass

    result = pd.Series(avg_series_values, index=dates, name="rolling_avg_corr")
    return result.tail(504)
