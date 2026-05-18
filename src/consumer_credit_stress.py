"""
Consumer credit stress monitor.

Consumer credit quality links to HY spreads via retail, auto, and consumer
finance sectors (~30% of HY index). Rising delinquencies lead corporate
credit stress by 2–3 quarters.

Data sourcing (FRED via fredapi, lazy-import):
  DRCCLACBS   — Credit card delinquency rate (quarterly, %)
  DRAUTOACBS  — Auto/consumer loan delinquency rate (quarterly, %)
  DRSFRMACBS  — Mortgage delinquency rate (quarterly, %)
  REVOLSL     — Revolving consumer credit (monthly, $B)

Scoring:
  cc_delinquency   40% weight
  auto_delinquency 25%
  mortgage_delinq  20%
  revolving_growth 15% (inverted — slower growth = higher stress)

All quarterly series forward-filled to daily frequency.

Public API
----------
  FRED_CONSUMER_SERIES  — dict of component → FRED series ID
  COMPONENT_WEIGHTS     — dict of component → weight
  compute_consumer_stress(df) → pd.DataFrame
  get_current_consumer_stress(df) → dict
  run_consumer_credit_stress(df) → dict
"""

import numpy as np
import pandas as pd

FRED_CONSUMER_SERIES = {
    "cc_delinquency":       "DRCCLACBS",
    "auto_delinquency":     "DRAUTOACBS",
    "mortgage_delinquency": "DRSFRMACBS",
    "revolving_credit":     "REVOLSL",
}

COMPONENT_WEIGHTS = {
    "cc_delinquency":       0.40,
    "auto_delinquency":     0.25,
    "mortgage_delinquency": 0.20,
    "revolving_growth":     0.15,
}

REGIME_THRESHOLDS = [
    ("Strong",   0,  25),
    ("Moderate", 25, 40),
    ("Elevated", 40, 60),
    ("Stress",   60, 80),
    ("Crisis",   80, 101),
]

_ZSCORE_WINDOW = 252 * 10  # 10yr rolling baseline for delinquency z-scores


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_fred_data() -> dict:
    """Fetch FRED consumer series. Returns dict of {key: pd.Series} or {}."""
    try:
        import os
        from fredapi import Fred
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            return {}
        fred = Fred(api_key=api_key)
        result = {}
        for key, sid in FRED_CONSUMER_SERIES.items():
            try:
                s = fred.get_series(sid)
                s.index = pd.to_datetime(s.index)
                result[key] = s
            except Exception:
                pass
        return result
    except Exception:
        return {}


def _to_daily(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Forward-fill a low-frequency series to business-day daily index."""
    idx = pd.bdate_range(start=start, end=end)
    return series.reindex(idx, method="ffill")


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = series.rolling(window, min_periods=max(20, window // 4)).std()
    return (series - mu) / sd.replace(0, np.nan)


def _zscore_to_score(z: pd.Series, invert: bool = False) -> pd.Series:
    """Map z-score to 0-100 via sigmoid-like percentile rank."""
    score = z.expanding(min_periods=20).rank(pct=True).mul(100)
    return (100 - score) if invert else score


def _regime(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    for name, lo, hi in REGIME_THRESHOLDS:
        if lo <= score < hi:
            return name
    return "Crisis"


# ---------------------------------------------------------------------------
# compute_consumer_stress
# ---------------------------------------------------------------------------

def compute_consumer_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with consumer_stress_* columns added.

    Pulls FRED data when available; falls back to df columns or NaN.
    """
    out = df.copy()

    start = df.index.min()
    end   = df.index.max()

    fred_data = _fetch_fred_data()

    component_scores = {}

    # --- credit card delinquency -------------------------------------------
    if "cc_delinquency" in fred_data:
        cc = _to_daily(fred_data["cc_delinquency"], start, end)
        cc = cc.reindex(df.index, method="ffill")
        cc_z = _rolling_zscore(cc, _ZSCORE_WINDOW)
        component_scores["cc_delinquency"] = _zscore_to_score(cc_z)
        out["consumer_cc_delinquency"] = cc
    elif "cc_delinquency_rate" in df.columns:
        cc_z = _rolling_zscore(df["cc_delinquency_rate"], _ZSCORE_WINDOW)
        component_scores["cc_delinquency"] = _zscore_to_score(cc_z)

    # --- auto delinquency ---------------------------------------------------
    if "auto_delinquency" in fred_data:
        auto = _to_daily(fred_data["auto_delinquency"], start, end)
        auto = auto.reindex(df.index, method="ffill")
        auto_z = _rolling_zscore(auto, _ZSCORE_WINDOW)
        component_scores["auto_delinquency"] = _zscore_to_score(auto_z)
        out["consumer_auto_delinquency"] = auto
    elif "auto_delinquency_rate" in df.columns:
        auto_z = _rolling_zscore(df["auto_delinquency_rate"], _ZSCORE_WINDOW)
        component_scores["auto_delinquency"] = _zscore_to_score(auto_z)

    # --- mortgage delinquency -----------------------------------------------
    if "mortgage_delinquency" in fred_data:
        mort = _to_daily(fred_data["mortgage_delinquency"], start, end)
        mort = mort.reindex(df.index, method="ffill")
        mort_z = _rolling_zscore(mort, _ZSCORE_WINDOW)
        component_scores["mortgage_delinquency"] = _zscore_to_score(mort_z)
        out["consumer_mortgage_delinquency"] = mort
    elif "mortgage_delinquency_rate" in df.columns:
        mort_z = _rolling_zscore(df["mortgage_delinquency_rate"], _ZSCORE_WINDOW)
        component_scores["mortgage_delinquency"] = _zscore_to_score(mort_z)

    # --- revolving credit growth (inverted — slower = more stress) ----------
    if "revolving_credit" in fred_data:
        rev = _to_daily(fred_data["revolving_credit"], start, end)
        rev = rev.reindex(df.index, method="ffill")
        rev_growth = rev.pct_change(63)
        rev_z = _rolling_zscore(rev_growth, _ZSCORE_WINDOW)
        component_scores["revolving_growth"] = _zscore_to_score(rev_z, invert=True)
        out["consumer_revolving_growth"] = rev_growth
    elif "revolving_credit" in df.columns:
        rev_growth = df["revolving_credit"].pct_change(63)
        rev_z = _rolling_zscore(rev_growth, _ZSCORE_WINDOW)
        component_scores["revolving_growth"] = _zscore_to_score(rev_z, invert=True)

    # --- weighted composite -------------------------------------------------
    if component_scores:
        total_weight = sum(
            COMPONENT_WEIGHTS.get(k, 0) for k in component_scores
        )
        if total_weight > 0:
            composite = sum(
                component_scores[k] * COMPONENT_WEIGHTS.get(k, 0)
                for k in component_scores
            ) / total_weight
        else:
            composite = pd.Series(np.nan, index=df.index)
        out["consumer_stress_score"] = composite.clip(0, 100)
    else:
        out["consumer_stress_score"] = pd.Series(np.nan, index=df.index)

    for key, series in component_scores.items():
        out[f"consumer_{key}_score"] = series.clip(0, 100)

    out["consumer_stress_regime"] = out["consumer_stress_score"].apply(_regime)

    return out


# ---------------------------------------------------------------------------
# get_current_consumer_stress
# ---------------------------------------------------------------------------

def get_current_consumer_stress(df: pd.DataFrame) -> dict:
    """Return dict describing current consumer credit stress conditions."""
    try:
        enriched = compute_consumer_stress(df)
        if enriched.empty:
            return {"available": False}

        row = enriched.iloc[-1]

        def _safe(col):
            val = row.get(col) if col in enriched.columns else None
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return val

        score  = _safe("consumer_stress_score")
        regime = _safe("consumer_stress_regime") or "Unknown"

        component_scores = {}
        for key in COMPONENT_WEIGHTS:
            col = f"consumer_{key}_score"
            v = _safe(col)
            component_scores[key] = round(float(v), 1) if v is not None else None

        hy_spread = _safe("hy_spread")
        at_risk_fraction = 0.30
        at_risk_bps = round(at_risk_fraction * float(hy_spread), 1) if hy_spread is not None else None

        if score is not None:
            if score >= 60:
                lead_signal = "HY Retail/Auto Spread Widening Risk (2–3 quarter lead)"
            elif score >= 40:
                lead_signal = "Watch Lower-Quality HY Consumer Sectors"
            elif score >= 25:
                lead_signal = "Mild Consumer Strain — Monitor Trends"
            else:
                lead_signal = "Consumer Balance Sheets Healthy — Supportive for HY"
        else:
            lead_signal = "Insufficient data"

        if score is not None and regime not in ("Unknown",):
            interpretation = (
                f"Consumer credit stress is in the **{regime}** zone (score: {score:.0f}/100). "
                f"Consumer-linked HY sectors (~{at_risk_fraction:.0%} of index) "
                + (f"account for ~{at_risk_bps:.0f}bps of current spread. " if at_risk_bps else "")
                + lead_signal + "."
            )
        else:
            interpretation = "Insufficient consumer credit data — requires FRED API key."

        warning = None
        if score is not None and score >= 60:
            warning = f"Consumer stress elevated ({score:.0f}/100) — retail/auto HY sectors at risk of spread widening."

        return {
            "available": score is not None,
            "consumer_stress_score": round(float(score), 1) if score is not None else None,
            "regime": regime,
            "component_scores": component_scores,
            "at_risk_fraction": at_risk_fraction,
            "at_risk_spread_contribution": at_risk_bps,
            "lead_signal": lead_signal,
            "interpretation": interpretation,
            "warning": warning,
        }
    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# run_consumer_credit_stress
# ---------------------------------------------------------------------------

def run_consumer_credit_stress(df: pd.DataFrame) -> dict:
    """Top-level entry point — returns everything needed to render the UI.

    Keys
    ----
    current             : get_current_consumer_stress result
    df                  : compute_consumer_stress result (full history)
    stress_history      : pd.Series 0-100 (DatetimeIndex)
    regime_history      : pd.Series of regime strings
    component_histories : dict of {component: pd.Series}
    hy_lead_correlation : float (correlation of consumer stress with forward hy_spread)
    available           : bool
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        enriched = compute_consumer_stress(df)
        current  = get_current_consumer_stress(df)

        stress_history = enriched.get("consumer_stress_score", pd.Series(dtype=float))
        if hasattr(stress_history, "dropna") and stress_history.dropna().empty:
            return {"available": False, "reason": "No consumer credit data — FRED API key required."}

        regime_history = enriched.get("consumer_stress_regime", pd.Series(dtype=str))

        component_histories = {}
        for key in COMPONENT_WEIGHTS:
            col = f"consumer_{key}_score"
            if col in enriched.columns:
                component_histories[key] = enriched[col]

        # Lead correlation: consumer stress today vs hy_spread 189 days (9 months) forward
        hy_lead_correlation = None
        if "hy_spread" in enriched.columns and "consumer_stress_score" in enriched.columns:
            fwd_hy = enriched["hy_spread"].shift(-189)
            aligned = pd.concat([enriched["consumer_stress_score"], fwd_hy], axis=1).dropna()
            if len(aligned) > 50:
                corr_matrix = np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])
                hy_lead_correlation = round(float(corr_matrix[0, 1]), 3)

        return {
            "available": current.get("available", False),
            "current": current,
            "df": enriched,
            "stress_history": stress_history,
            "regime_history": regime_history,
            "component_histories": component_histories,
            "hy_lead_correlation": hy_lead_correlation,
        }
    except Exception:
        return {"available": False}
