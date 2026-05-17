"""
Stress contagion analysis — rolling cross-score correlations.
Public API:
  compute_contagion_matrix(df) -> pd.DataFrame  (current corr matrix)
  compute_contagion_index(df, window) -> pd.Series  (mean off-diagonal over time)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_CONTAGION_SIGNALS = {
    "treasury_stress_score_smooth":         "Treasury",
    "credit_market_risk_score_smooth":      "Credit",
    "macro_risk_score_smooth":              "Macro",
    "complacency_score_smooth":             "Complacency",
    "liquidity_regime_score_smooth":        "Liquidity",
    "enhanced_funding_stress_score_smooth": "Funding",
    "fx_commodity_score_smooth":            "FX/Cmdty",
}


def _available_signals(df: pd.DataFrame) -> dict[str, str]:
    """Return subset of _CONTAGION_SIGNALS whose columns are present in df."""
    return {col: label for col, label in _CONTAGION_SIGNALS.items() if col in df.columns}


def compute_contagion_matrix(df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix across available signal columns using
    the last *window* rows of *df*.

    Columns and index are renamed to short labels from ``_CONTAGION_SIGNALS``.
    Returns an empty DataFrame if fewer than 2 signals are available or there
    is insufficient data.
    """
    avail = _available_signals(df)
    if len(avail) < 2:
        return pd.DataFrame()

    subset = df[list(avail.keys())].tail(window)
    if len(subset) < 2:
        return pd.DataFrame()

    corr = subset.corr(method="pearson")
    corr = corr.rename(columns=avail, index=avail)
    return corr


def compute_contagion_index(df: pd.DataFrame, window: int = 90) -> pd.Series:
    """
    Rolling *window*-day mean of absolute off-diagonal correlations across all
    available signal columns.

    Returns a Series indexed by date (using ``df["date"]`` when present).
    High values indicate stress spreading across asset classes (contagion).
    """
    avail = _available_signals(df)
    if len(avail) < 2:
        return pd.Series(dtype=float)

    cols = list(avail.keys())
    data = df[cols].copy()

    # Build index
    if "date" in df.columns:
        data.index = pd.to_datetime(df["date"])
    else:
        data.index = df.index

    def _mean_abs_offdiag(window_data: pd.DataFrame) -> float:
        if len(window_data) < 2:
            return np.nan
        corr = window_data.corr(method="pearson").values
        n = corr.shape[0]
        if n < 2:
            return np.nan
        mask = ~np.eye(n, dtype=bool)
        return float(np.nanmean(np.abs(corr[mask])))

    index_values = [
        _mean_abs_offdiag(data.iloc[max(0, i - window + 1) : i + 1])
        for i in range(len(data))
    ]

    result = pd.Series(index_values, index=data.index, name="contagion_index")
    # First (window-1) values are based on fewer rows; mask them as NaN for clarity.
    result.iloc[: window - 1] = np.nan
    return result


def run_contagion_analysis(df: pd.DataFrame, window: int = 90) -> dict:
    """
    Convenience wrapper.

    Returns
    -------
    dict with keys:
      ``matrix``        — current correlation matrix (DataFrame)
      ``index``         — rolling contagion index over time (Series)
      ``current_index`` — most recent non-NaN index value (float, or NaN)
      ``window``        — window used
    """
    matrix = compute_contagion_matrix(df, window=window)
    index = compute_contagion_index(df, window=window)
    current_index = float(index.dropna().iloc[-1]) if not index.dropna().empty else float("nan")

    return {
        "matrix": matrix,
        "index": index,
        "current_index": current_index,
        "window": window,
    }
