"""
Cross-asset signal scoring module.

Each function takes pre-computed scalar inputs and returns a score in [0, 100]
where higher = more risk. The scoring logic is deliberately simple, rule-based,
and interpretable — no ML, no lookahead, no parameter fitting to returns.

Signal groups
-------------
  rates_stress        — yield curve inversion severity + Fed tightening pace
  funding_stress      — NFCI/ANFCI/STLFSI + initial claims spike
  fx_commodity        — USD strength + oil crash as risk-off / supply-shock signals
  banking_stress      — deposit outflows + loan contraction + delinquency trends

These scores are ADDITIVE to the existing composite — they do not replace or
re-weight the current macro/credit/liquidity structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rates stress score
# ---------------------------------------------------------------------------

def compute_rates_stress_score(
    spread_10y3m: float,
    fed_funds_rate: float,
    fed_funds_change_90d: float,
    fed_funds_change_360d: float,
) -> float:
    """
    Combines yield curve inversion and Fed tightening pace into 0-100.

    Inputs
    ------
    spread_10y3m         : 10Y minus 3M yield (bps if negative = inversion)
    fed_funds_rate       : effective fed funds rate level (%)
    fed_funds_change_90d : 90-day change in fed funds rate (percentage points)
    fed_funds_change_360d: 360-day change — captures the full tightening cycle

    Economic rationale
    ------------------
    - 10Y-3M inversion is empirically the most reliable recession predictor
      among yield curve shapes (Estrella & Mishkin 1996; NY Fed model).
    - Rapid Fed tightening compresses corporate margins, increases rollover
      risk, and historically precedes credit stress (2000, 2006-07, 2022).
    """
    score = 0.0

    # Inversion depth: each 25bp of inversion adds ~8 points
    if pd.isna(spread_10y3m):
        pass
    elif spread_10y3m < -1.5:
        score += 40
    elif spread_10y3m < -1.0:
        score += 30
    elif spread_10y3m < -0.5:
        score += 20
    elif spread_10y3m < 0.0:
        score += 10
    elif spread_10y3m < 0.5:
        score += 3     # very flat — mild risk signal

    # Fed funds level (restrictive territory = elevated risk)
    if not pd.isna(fed_funds_rate):
        if fed_funds_rate >= 5.0:
            score += 25
        elif fed_funds_rate >= 4.0:
            score += 15
        elif fed_funds_rate >= 3.0:
            score += 8

    # Tightening pace: large 90-day hike = stress signal
    if not pd.isna(fed_funds_change_90d):
        if fed_funds_change_90d >= 1.5:    # ≥ 6 hikes of 25bp in 90 days
            score += 25
        elif fed_funds_change_90d >= 0.75:
            score += 15
        elif fed_funds_change_90d >= 0.25:
            score += 8

    # Full-cycle tightening (360d): captures late-cycle stress
    if not pd.isna(fed_funds_change_360d):
        if fed_funds_change_360d >= 4.0:   # >400bp in a year — 2022 era
            score += 10
        elif fed_funds_change_360d >= 2.0:
            score += 5

    return round(min(max(score, 0.0), 100.0), 1)


# ---------------------------------------------------------------------------
# Enhanced funding / liquidity stress score
# ---------------------------------------------------------------------------

def compute_enhanced_funding_stress_score(
    nfci: float,
    anfci: float,
    stlfsi: float,
    initial_claims_zscore: float,
) -> float:
    """
    Combines three financial conditions indices with initial claims to produce
    a broader funding-stress signal than NFCI alone.

    Inputs
    ------
    nfci                  : Chicago NFCI level (normal ≈ 0, elevated > 0)
    anfci                 : Adjusted NFCI (removes expected macro deterioration;
                            reflects purely financial/funding stress)
    stlfsi                : St. Louis FSI (normal ≈ 0, elevated > 0)
    initial_claims_zscore : z-score of weekly initial jobless claims vs. 2Y rolling
                            history (> 2 = rapid claims spike, classic stress signal)

    ANFCI note
    ----------
    ANFCI strips out the contribution of expected economic conditions from NFCI.
    A high ANFCI therefore signals financial tightening beyond what macro
    fundamentals would predict — a purer liquidity/funding stress read.
    """
    score = 0.0

    # NFCI level
    if not pd.isna(nfci):
        if nfci > 0.5:
            score += 30
        elif nfci > 0.2:
            score += 18
        elif nfci > 0.0:
            score += 8
        elif nfci > -0.3:
            score += 2

    # ANFCI adds incremental signal beyond NFCI level
    if not pd.isna(anfci):
        if anfci > 0.5:
            score += 20
        elif anfci > 0.2:
            score += 12
        elif anfci > 0.0:
            score += 5

    # St. Louis FSI (orthogonal to NFCI — different sub-components)
    if not pd.isna(stlfsi):
        if stlfsi > 1.0:
            score += 20
        elif stlfsi > 0.5:
            score += 12
        elif stlfsi > 0.0:
            score += 5

    # Initial claims z-score — rapid spike signals labour / credit stress
    if not pd.isna(initial_claims_zscore):
        if initial_claims_zscore > 3.0:
            score += 20
        elif initial_claims_zscore > 2.0:
            score += 12
        elif initial_claims_zscore > 1.5:
            score += 6

    return round(min(max(score, 0.0), 100.0), 1)


# ---------------------------------------------------------------------------
# FX / commodity risk-off score
# ---------------------------------------------------------------------------

def compute_fx_commodity_score(
    eurusd_change_30d: float,
    oil_change_30d: float,
    usdjpy_change_30d: float,
) -> float:
    """
    Detects cross-asset risk-off signals from FX and commodity markets.

    Inputs (all as % change or fractional return)
    ------
    eurusd_change_30d  : 30-day change in EUR/USD (negative = USD strengthening)
    oil_change_30d     : 30-day % change in WTI crude oil
    usdjpy_change_30d  : 30-day change in USD/JPY (negative = JPY strengthening = risk-off)

    Economic rationale
    ------------------
    - USD strength (EUR/USD falls) correlates with global risk-off; EM credit stress
      is amplified by dollar appreciation (Avdjiev et al. 2019).
    - Sharp oil crashes (>15% in 30d) historically coincide with demand fears and
      often trigger HY energy-sector spread widening (2015-16, 2020).
    - JPY strengthening (USD/JPY falling) is a classic safe-haven flight signal
      driven by yen carry-trade unwind.
    """
    score = 0.0

    # EUR/USD decline = USD strengthening = risk-off signal
    if not pd.isna(eurusd_change_30d):
        if eurusd_change_30d < -0.04:   # > 4% USD appreciation
            score += 25
        elif eurusd_change_30d < -0.02:
            score += 12
        elif eurusd_change_30d < -0.01:
            score += 5

    # Oil crash
    if not pd.isna(oil_change_30d):
        if oil_change_30d < -0.20:      # > 20% crash
            score += 30
        elif oil_change_30d < -0.10:
            score += 18
        elif oil_change_30d < -0.05:
            score += 8

    # JPY appreciation (USD/JPY falling)
    if not pd.isna(usdjpy_change_30d):
        if usdjpy_change_30d < -5.0:    # >5 yen appreciation
            score += 25
        elif usdjpy_change_30d < -2.5:
            score += 12
        elif usdjpy_change_30d < -1.0:
            score += 5

    return round(min(max(score, 0.0), 100.0), 1)


# ---------------------------------------------------------------------------
# Banking / credit channel stress score
# ---------------------------------------------------------------------------

def compute_banking_stress_score(
    deposit_growth_90d: float,
    loan_growth_90d: float,
    fed_balance_sheet_change_90d: float,
) -> float:
    """
    Captures bank balance-sheet stress: deposit flight, credit contraction,
    and emergency Fed balance sheet expansion.

    Inputs (all as fractional % change over 90 trading days)
    ------
    deposit_growth_90d            : 90d change in aggregate bank deposits
    loan_growth_90d               : 90d change in total bank loans outstanding
    fed_balance_sheet_change_90d  : 90d change in Fed balance sheet
                                    (large spike = emergency facility usage)

    Note on lag
    -----------
    FRED weekly banking data (H.8) is released with ~1 week delay and is often
    revised. These signals are best treated as confirming, lagging indicators
    of stress — not early-warning signals. Treat this score as a 'reality check'
    on the primary NFCI/HY signals rather than a leading indicator.
    """
    score = 0.0

    # Deposit outflow (negative growth = withdrawal flight risk)
    if not pd.isna(deposit_growth_90d):
        if deposit_growth_90d < -0.03:    # >3% deposit decline in 90 days
            score += 35
        elif deposit_growth_90d < -0.015:
            score += 18
        elif deposit_growth_90d < 0.0:
            score += 5

    # Loan contraction (negative = credit crunch)
    if not pd.isna(loan_growth_90d):
        if loan_growth_90d < -0.03:
            score += 30
        elif loan_growth_90d < -0.015:
            score += 15
        elif loan_growth_90d < 0.0:
            score += 5

    # Fed balance sheet emergency expansion (rapid large expansion = crisis mode)
    if not pd.isna(fed_balance_sheet_change_90d):
        if fed_balance_sheet_change_90d > 0.10:  # >10% in 90 days (2020-level)
            score += 30
        elif fed_balance_sheet_change_90d > 0.05:
            score += 15
        elif fed_balance_sheet_change_90d > 0.02:
            score += 5

    return round(min(max(score, 0.0), 100.0), 1)


# ---------------------------------------------------------------------------
# Rolling z-score helper
# ---------------------------------------------------------------------------

def rolling_zscore(s: pd.Series, window: int = 504) -> pd.Series:
    """
    Per-row rolling z-score: (value - rolling_mean) / rolling_std.
    Uses min_periods=window//2 to provide partial values during warmup.
    Returns NaN where std is zero.
    """
    mu  = s.rolling(window, min_periods=window // 2).mean()
    sig = s.rolling(window, min_periods=window // 2).std()
    z   = (s - mu) / sig.replace(0, np.nan)
    return z
