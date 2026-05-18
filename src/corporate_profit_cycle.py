"""
Corporate profit margin and earnings cycle monitor.

Profit margins compressing → corporate leverage rising → default probability
increasing → HY spread widening. Profit cycle leads credit spreads by
roughly 2-4 quarters.

Methodology
-----------
profit_margin = CP / GDP  (or NFCPATAX / GDP when available — nonfinancial
corporate profits strip out financial-sector volatility and are more
directly linked to debt service capacity across the HY universe).

profit_stress_score = 100 - profit_cycle_score
  where profit_cycle_score is an expanding percentile rank (0-100).
  High stress = compressed margins = deteriorating coverage = wider spreads.

interest_coverage_proxy = profit_margin / effective_rate
  Numerator = margins as a share of GDP; denominator = a rate proxy that
  approximates the average coupon on outstanding debt (short rate + spread
  buffer).  This is a relative indicator, not an absolute coverage ratio.

Lead analysis: corr(profit_stress_score today, hy_spread 126 days forward)

FRED series (quarterly, forward-filled to daily)
------------------------------------------------
  CP       — Corporate Profits After Tax (seasonally adjusted annual rate, $B)
  GDP      — Nominal GDP ($B, quarterly)
  CPATAX   — Corporate Profits After Tax with IVA and CCAdj
  NFCPATAX — Nonfinancial Corporate Profits After Tax (preferred for credit)

Public API
----------
  FRED_PROFIT_SERIES   — dict of key → FRED ID
  PROFIT_REGIMES       — dict of label → (lo, hi) score range
  compute_profit_cycle(df)       -> pd.DataFrame
  get_current_profit_cycle(df)   -> dict
  run_corporate_profit_cycle(df) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_PROFIT_SERIES: dict[str, str] = {
    "corporate_profits": "CP",
    "nfc_profits":       "NFCPATAX",
    "gdp":               "GDP",
}

PROFIT_REGIMES: dict[str, tuple[int, int]] = {
    "Boom":           (0,  25),
    "Healthy":        (25, 40),
    "Neutral":        (40, 60),
    "Compression":    (60, 75),
    "Recession Risk": (75, 100),
}

_ZSCORE_BASELINE_ROWS: int = 252 * 10   # 10-year rolling z-score
_MIN_PERIODS_ZSCORE: int   = 20
_HIST_ROWS: int            = 504
_MIN_ROWS: int             = 40          # quarterly series → 10 years of data

# Long-run average profit margin (CP/GDP), 1970-2023
_LONG_RUN_MARGIN_AVG: float = 0.09      # ~9% of GDP

# Rate buffer added to short-rate proxy so effective rate ≈ all-in coupon
_RATE_BUFFER: float = 0.02

# Profit momentum thresholds (YoY %)
_MOMENTUM_EXPANDING: float =  3.0
_MOMENTUM_CONTRACTING: float = -3.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_fred_profit_data() -> dict[str, pd.Series]:
    """Attempt to pull quarterly profit series from FRED. Returns {} on failure."""
    try:
        import os
        from fredapi import Fred
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            return {}
        fred = Fred(api_key=api_key)
        result: dict[str, pd.Series] = {}
        for key, sid in FRED_PROFIT_SERIES.items():
            try:
                s = fred.get_series(sid)
                s.index = pd.to_datetime(s.index)
                result[key] = s.dropna()
            except Exception:
                pass
        return result
    except Exception:
        return {}


def _to_daily(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a low-frequency series onto an arbitrary daily index."""
    combined = series.reindex(series.index.union(idx)).ffill()
    return combined.reindex(idx)


def _expanding_percentile_rank(s: pd.Series) -> pd.Series:
    """
    For each point, compute percentile rank within all preceding observations.
    Returns 0-100 scale.  min_periods=_MIN_PERIODS_ZSCORE.
    """
    out = pd.Series(np.nan, index=s.index, dtype=float)
    arr = s.values
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        window = arr[:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < _MIN_PERIODS_ZSCORE:
            continue
        rank = float(np.sum(valid <= arr[i]) - 1) / float(len(valid) - 1) * 100.0
        out.iloc[i] = float(np.clip(rank, 0.0, 100.0))
    return out


def _rolling_zscore_expanding(s: pd.Series) -> pd.Series:
    """Expanding-window z-score (mean and std grow with sample size)."""
    mu  = s.expanding(min_periods=_MIN_PERIODS_ZSCORE).mean()
    sig = s.expanding(min_periods=_MIN_PERIODS_ZSCORE).std(ddof=1)
    return (s - mu) / sig.replace(0.0, np.nan)


def _profit_regime_label(score: float) -> str:
    if np.isnan(score):
        return "Unknown"
    for label, (lo, hi) in PROFIT_REGIMES.items():
        if lo <= score < hi:
            return label
    return "Recession Risk"


def _profit_momentum_label(yoy: float) -> str:
    if np.isnan(yoy):
        return "Unknown"
    if yoy > _MOMENTUM_EXPANDING:
        return "Expanding"
    if yoy < _MOMENTUM_CONTRACTING:
        return "Contracting"
    return "Stable"


def _effective_rate(df: pd.DataFrame) -> pd.Series:
    """
    Best available rate proxy for coverage denominator.
    Priority: yield_3m > yield_2y > fed_funds_rate fallback > constant 0.05.
    """
    for col in ("yield_3m", "yield_2y", "fed_funds_rate"):
        if col in df.columns:
            s = df[col].copy()
            if s.notna().sum() > 10:
                # Series stored as % (e.g. 5.0 = 5%); convert to decimal
                if s.median() > 1.0:
                    s = s / 100.0
                return s.clip(lower=0.001)
    return pd.Series(0.05, index=df.index)


# ---------------------------------------------------------------------------
# compute_profit_cycle
# ---------------------------------------------------------------------------

def compute_profit_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a copy of df with profit cycle columns.

    New columns
    -----------
    profit_margin              : CP / GDP (decimal, e.g. 0.112)
    profit_margin_yoy          : YoY % change (4-quarter lag, i.e. 252-day proxy)
    profit_margin_zscore       : expanding z-score
    profit_cycle_score         : expanding percentile rank 0-100 (high = good)
    profit_stress_score        : 100 - profit_cycle_score (high = bad)
    profit_cycle_regime        : str regime label
    profit_momentum            : Expanding / Stable / Contracting
    interest_coverage_proxy    : margin / effective_rate (higher = better)
    coverage_stress_score      : expanding pct rank of (1/coverage) × 100
    leverage_flag              : bool series — compression + rising yields
    """
    out = df.copy()

    try:
        fred_data = _fetch_fred_profit_data()
        daily_idx = out.index

        # ---- resolve profit series ------------------------------------------
        if "nfc_profits" in fred_data and "gdp" in fred_data:
            profits_q = fred_data["nfc_profits"]
            gdp_q     = fred_data["gdp"]
        elif "corporate_profits" in fred_data and "gdp" in fred_data:
            profits_q = fred_data["corporate_profits"]
            gdp_q     = fred_data["gdp"]
        elif "nfc_profits" in fred_data:
            profits_q = fred_data["nfc_profits"]
            gdp_q     = None
        else:
            profits_q = None
            gdp_q     = None

        # ---- fallback: look for pre-built columns in df ----------------------
        if profits_q is None and "corporate_profits" in out.columns:
            profits_q = out["corporate_profits"]
        if gdp_q is None and "gdp" in out.columns:
            gdp_q = out["gdp"]

        # ---- if we have a pre-built profit_margin column, use it -------------
        if profits_q is None or gdp_q is None:
            if "profit_margin" in out.columns:
                margin_daily = out["profit_margin"].ffill()
            else:
                for col in [
                    "profit_margin",
                    "profit_margin_proxy",
                    "corporate_profit_margin",
                ]:
                    if col in out.columns:
                        margin_daily = out[col].ffill()
                        break
                else:
                    out["profit_margin"]           = np.nan
                    out["profit_margin_yoy"]        = np.nan
                    out["profit_margin_zscore"]     = np.nan
                    out["profit_cycle_score"]       = np.nan
                    out["profit_stress_score"]      = np.nan
                    out["profit_cycle_regime"]      = "Unknown"
                    out["profit_momentum"]          = "Unknown"
                    out["interest_coverage_proxy"]  = np.nan
                    out["coverage_stress_score"]    = np.nan
                    out["leverage_flag"]            = False
                    return out
        else:
            profits_daily = _to_daily(profits_q, daily_idx)
            gdp_daily     = _to_daily(gdp_q, daily_idx)
            margin_daily  = (profits_daily / gdp_daily.replace(0, np.nan)).clip(lower=0)

        out["profit_margin"] = margin_daily

        # ---- YoY % change (252-day proxy for 4-quarter lag) -----------------
        yoy = (margin_daily / margin_daily.shift(252) - 1.0) * 100.0
        out["profit_margin_yoy"] = yoy

        # ---- z-score and percentile rank ------------------------------------
        out["profit_margin_zscore"] = _rolling_zscore_expanding(margin_daily)

        pct_rank = _expanding_percentile_rank(margin_daily)
        out["profit_cycle_score"]  = pct_rank
        out["profit_stress_score"] = (100.0 - pct_rank).clip(0.0, 100.0)

        out["profit_cycle_regime"] = out["profit_stress_score"].apply(
            lambda v: _profit_regime_label(float(v)) if pd.notna(v) else "Unknown"
        )

        out["profit_momentum"] = out["profit_margin_yoy"].apply(
            lambda v: _profit_momentum_label(float(v)) if pd.notna(v) else "Unknown"
        )

        # ---- coverage proxy -------------------------------------------------
        rate = _effective_rate(out)
        effective_rate = (rate + _RATE_BUFFER).clip(lower=0.001)
        coverage = (margin_daily / effective_rate) * 10.0
        out["interest_coverage_proxy"] = coverage.clip(lower=0)

        inv_coverage = (1.0 / coverage.replace(0, np.nan))
        out["coverage_stress_score"] = _expanding_percentile_rank(inv_coverage).clip(0.0, 100.0)

        # ---- leverage flag --------------------------------------------------
        stress_s = out["profit_stress_score"]
        if "yield_10y" in out.columns:
            y10 = out["yield_10y"]
        else:
            y10 = pd.Series(np.nan, index=out.index)

        y10_roll_mean = y10.rolling(252, min_periods=63).mean()
        rate_rising = y10 > y10_roll_mean
        out["leverage_flag"] = (stress_s > 60) & rate_rising.fillna(False)

    except Exception:
        for col in [
            "profit_margin", "profit_margin_yoy", "profit_margin_zscore",
            "profit_cycle_score", "profit_stress_score", "profit_cycle_regime",
            "profit_momentum", "interest_coverage_proxy", "coverage_stress_score",
            "leverage_flag",
        ]:
            if col not in out.columns:
                out[col] = np.nan if col != "leverage_flag" else False

    return out


# ---------------------------------------------------------------------------
# get_current_profit_cycle
# ---------------------------------------------------------------------------

def get_current_profit_cycle(df: pd.DataFrame) -> dict:
    """
    Return the current profit cycle state as a flat dict.

    Parameters
    ----------
    df : master DataFrame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        available, profit_margin, profit_margin_yoy, profit_stress_score,
        profit_cycle_regime, profit_momentum, interest_coverage_proxy,
        coverage_stress_score, leverage_flag, lead_signal, interpretation,
        warning
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        enriched = compute_profit_cycle(df)
        row = enriched.iloc[-1]

        def _f(col: str):
            if col not in enriched.columns:
                return None
            v = row[col]
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            try:
                fv = float(v)
                return None if np.isnan(fv) else fv
            except Exception:
                return None

        margin       = _f("profit_margin")
        yoy          = _f("profit_margin_yoy")
        stress_score = _f("profit_stress_score")
        regime       = str(row.get("profit_cycle_regime", "Unknown")) if "profit_cycle_regime" in enriched.columns else "Unknown"
        momentum     = str(row.get("profit_momentum", "Unknown")) if "profit_momentum" in enriched.columns else "Unknown"
        coverage     = _f("interest_coverage_proxy")
        cov_stress   = _f("coverage_stress_score")
        lev_flag     = bool(row.get("leverage_flag", False)) if "leverage_flag" in enriched.columns else False

        if stress_score is None:
            return {"available": False}

        # ---- lead signal ----------------------------------------------------
        if stress_score >= 75:
            lead_signal = "Severe margin compression — HY spread widening risk is high over the next 2-4 quarters."
        elif stress_score >= 60:
            lead_signal = "Margin compression underway — watch HY quality deterioration."
        elif stress_score >= 40:
            lead_signal = "Near-average margins — neutral credit signal."
        elif stress_score >= 25:
            lead_signal = "Above-average margins — supportive for debt service capacity."
        else:
            lead_signal = "Margins near cycle highs — strong debt service coverage, spreads likely to remain contained."

        # ---- interpretation -------------------------------------------------
        margin_str = f"{margin * 100:.1f}%" if margin is not None else "N/A"
        yoy_str    = f"{yoy:+.1f}%" if yoy is not None else "N/A"
        interpretation = (
            f"Corporate profit margin: {margin_str} of GDP ({yoy_str} YoY). "
            f"Profit stress score: {stress_score:.0f}/100 — regime: '{regime}'. "
            f"Momentum: {momentum}. {lead_signal}"
        )

        # ---- warning --------------------------------------------------------
        warning: str | None = None
        if stress_score >= 75:
            warning = (
                f"Profit stress score ({stress_score:.0f}/100) in 'Recession Risk' zone. "
                "Severe margin compression historically precedes HY spread widening by 2-3 quarters."
            )
        elif lev_flag:
            warning = (
                "Leverage flag active: margin compression AND rising long rates detected. "
                "Debt service stress risk is elevated."
            )

        return {
            "available":               True,
            "profit_margin":           round(margin, 4) if margin is not None else None,
            "profit_margin_yoy":       round(yoy, 2)   if yoy is not None    else None,
            "profit_stress_score":     round(stress_score, 1),
            "profit_cycle_regime":     regime,
            "profit_momentum":         momentum,
            "interest_coverage_proxy": round(coverage, 2)   if coverage is not None  else None,
            "coverage_stress_score":   round(cov_stress, 1) if cov_stress is not None else None,
            "leverage_flag":           lev_flag,
            "lead_signal":             lead_signal,
            "interpretation":          interpretation,
            "warning":                 warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# run_corporate_profit_cycle
# ---------------------------------------------------------------------------

def run_corporate_profit_cycle(df: pd.DataFrame) -> dict:
    """
    Top-level entry point for corporate profit cycle analysis.

    Parameters
    ----------
    df : master DataFrame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        available, current, df, profit_margin_history, profit_stress_history,
        coverage_proxy_history, lead_correlation, regime_history
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        enriched = compute_profit_cycle(df)
        current  = get_current_profit_cycle(df)

        if not current.get("available"):
            return {"available": False}

        # ---- history slices -------------------------------------------------
        profit_margin_history = (
            enriched["profit_margin"].tail(_HIST_ROWS).copy()
            if "profit_margin" in enriched.columns
            else pd.Series(dtype=float)
        )

        profit_stress_history = (
            enriched["profit_stress_score"].tail(_HIST_ROWS).copy()
            if "profit_stress_score" in enriched.columns
            else pd.Series(dtype=float)
        )

        coverage_proxy_history = (
            enriched["interest_coverage_proxy"].tail(_HIST_ROWS).copy()
            if "interest_coverage_proxy" in enriched.columns
            else pd.Series(dtype=float)
        )

        regime_history = (
            enriched["profit_cycle_regime"].tail(_HIST_ROWS).copy()
            if "profit_cycle_regime" in enriched.columns
            else pd.Series(dtype=str)
        )

        # ---- lead correlation: stress_score today vs hy_spread 126d forward --
        lead_correlation: float | None = None
        if (
            "profit_stress_score" in enriched.columns
            and "hy_spread" in enriched.columns
        ):
            try:
                fwd_hy = enriched["hy_spread"].shift(-126)
                pair   = pd.concat(
                    [enriched["profit_stress_score"], fwd_hy], axis=1
                ).dropna()
                if len(pair) >= 50:
                    corr_mat = np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])
                    lead_correlation = round(float(corr_mat[0, 1]), 3)
            except Exception:
                lead_correlation = None

        return {
            "available":               True,
            "current":                 current,
            "df":                      enriched,
            "profit_margin_history":   profit_margin_history,
            "profit_stress_history":   profit_stress_history,
            "coverage_proxy_history":  coverage_proxy_history,
            "lead_correlation":        lead_correlation,
            "regime_history":          regime_history,
        }

    except Exception:
        return {"available": False}
