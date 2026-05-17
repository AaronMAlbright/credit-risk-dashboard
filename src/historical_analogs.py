"""historical_analogs.py — Find closest historical periods to current market conditions.

Uses cosine similarity across composite sub-score signals to identify past dates
that most closely resemble the current market environment, then computes forward
outcomes for those analog periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ANALOG_SIGNALS = [
    "composite_risk_score_smooth",
    "treasury_stress_score_smooth",
    "credit_market_risk_score_smooth",
    "macro_risk_score_smooth",
    "complacency_score_smooth",
    "liquidity_regime_score_smooth",
    "enhanced_funding_stress_score_smooth",
    "fx_commodity_score_smooth",
]


def find_historical_analogs(
    df: pd.DataFrame,
    n: int = 5,
    exclude_recent_days: int = 252,
) -> pd.DataFrame:
    """Find the n historical periods most similar to the current market environment.

    Similarity is measured via cosine similarity across all composite sub-score
    signals (``_ANALOG_SIGNALS``), after min-max normalizing each signal over the
    full history.  The most recent ``exclude_recent_days`` rows are excluded from
    the candidate pool so that near-identical recent dates are not returned as
    analogs.

    Forward outcomes are computed on the raw ``sp500`` and ``hy_spread`` columns
    before any filtering so that they reflect actual subsequent market moves.

    Parameters
    ----------
    df:
        Full historical DataFrame.  Must contain at minimum the columns listed in
        ``_ANALOG_SIGNALS`` plus ``sp500``, ``hy_spread``, ``date``, and
        ``final_decision``.
    n:
        Number of analog periods to return (default 5).
    exclude_recent_days:
        Number of trailing rows to exclude from the candidate pool (default 252,
        approximately one trading year).

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``final_decision``, ``similarity``,
        ``composite_risk_score_smooth``, ``sp500_fwd_30d``, ``sp500_fwd_60d``,
        ``hy_fwd_30d``, ``hy_fwd_60d``.  Rows are sorted by ``similarity``
        descending.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "final_decision",
                "similarity",
                "composite_risk_score_smooth",
                "sp500_fwd_30d",
                "sp500_fwd_60d",
                "hy_fwd_30d",
                "hy_fwd_60d",
            ]
        )

    # --- Step 1: compute forward outcomes on the full df before any filtering ---
    work = df.copy()

    if "sp500" in work.columns:
        sp500_fwd_30 = work["sp500"].shift(-30)
        sp500_fwd_60 = work["sp500"].shift(-60)
        work["sp500_fwd_30d"] = ((sp500_fwd_30 - work["sp500"]) / work["sp500"] * 100).round(2)
        work["sp500_fwd_60d"] = ((sp500_fwd_60 - work["sp500"]) / work["sp500"] * 100).round(2)
    else:
        work["sp500_fwd_30d"] = np.nan
        work["sp500_fwd_60d"] = np.nan

    if "hy_spread" in work.columns:
        hy_fwd_30 = work["hy_spread"].shift(-30)
        hy_fwd_60 = work["hy_spread"].shift(-60)
        work["hy_fwd_30d"] = (hy_fwd_30 - work["hy_spread"]).round(2)
        work["hy_fwd_60d"] = (hy_fwd_60 - work["hy_spread"]).round(2)
    else:
        work["hy_fwd_30d"] = np.nan
        work["hy_fwd_60d"] = np.nan

    # --- Step 2: drop rows with any NaN in signal columns ---
    clean = work.dropna(subset=_ANALOG_SIGNALS).copy()

    if clean.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "final_decision",
                "similarity",
                "composite_risk_score_smooth",
                "sp500_fwd_30d",
                "sp500_fwd_60d",
                "hy_fwd_30d",
                "hy_fwd_60d",
            ]
        )

    # --- Step 3: min-max normalize each signal over the full clean history ---
    signal_data = clean[_ANALOG_SIGNALS].copy()
    col_min = signal_data.min()
    col_max = signal_data.max()
    denom = col_max - col_min
    # Avoid division by zero for constant columns
    denom = denom.where(denom != 0, other=1.0)
    normalized = (signal_data - col_min) / denom

    # --- Step 4: current vector = last row after normalization ---
    current_vec = normalized.iloc[-1].values.astype(float)

    # --- Step 5: exclude the most recent exclude_recent_days rows from candidates ---
    if len(normalized) <= exclude_recent_days:
        # Not enough history; fall back to requiring at least n candidates
        cutoff = max(0, len(normalized) - n - 1)
    else:
        cutoff = len(normalized) - exclude_recent_days

    candidate_norm = normalized.iloc[:cutoff]
    candidate_rows = clean.iloc[:cutoff]

    if candidate_norm.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "final_decision",
                "similarity",
                "composite_risk_score_smooth",
                "sp500_fwd_30d",
                "sp500_fwd_60d",
                "hy_fwd_30d",
                "hy_fwd_60d",
            ]
        )

    # --- Step 6: cosine similarity between current vector and every candidate ---
    candidate_matrix = candidate_norm.values.astype(float)

    current_norm = np.linalg.norm(current_vec)
    candidate_norms = np.linalg.norm(candidate_matrix, axis=1)

    # Dot products
    dots = candidate_matrix @ current_vec

    # Guard against zero-norm vectors
    denom_sim = candidate_norms * current_norm
    denom_sim = np.where(denom_sim == 0, 1.0, denom_sim)

    similarities = (dots / denom_sim).round(3)

    # --- Step 7: select top-n by similarity ---
    top_indices = np.argsort(similarities)[::-1][:n]

    result = candidate_rows.iloc[top_indices][
        ["date", "final_decision", "composite_risk_score_smooth",
         "sp500_fwd_30d", "sp500_fwd_60d", "hy_fwd_30d", "hy_fwd_60d"]
    ].copy()

    result.insert(2, "similarity", similarities[top_indices])

    result = result.reset_index(drop=True)

    return result


def get_analog_summary(analogs_df: pd.DataFrame) -> dict:
    """Compute mean and median forward outcomes across the top analog periods.

    Parameters
    ----------
    analogs_df:
        DataFrame returned by :func:`find_historical_analogs`.

    Returns
    -------
    dict
        Keys for each forward outcome column (``sp500_fwd_30d``, ``sp500_fwd_60d``,
        ``hy_fwd_30d``, ``hy_fwd_60d``) with nested ``mean`` and ``median`` values,
        both rounded to 2 decimal places.  Returns an empty dict if ``analogs_df``
        is empty.
    """
    outcome_cols = ["sp500_fwd_30d", "sp500_fwd_60d", "hy_fwd_30d", "hy_fwd_60d"]

    if analogs_df.empty:
        return {}

    summary: dict = {}
    for col in outcome_cols:
        if col not in analogs_df.columns:
            continue
        series = analogs_df[col].dropna()
        summary[col] = {
            "mean": round(float(series.mean()), 2) if not series.empty else None,
            "median": round(float(series.median()), 2) if not series.empty else None,
        }

    return summary
