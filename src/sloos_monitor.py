"""
Bank Senior Loan Officer Opinion Survey (SLOOS) monitor.

Net tightening = % banks tightening − % banks easing. Positive values mean
more banks are tightening lending standards. C&I net tightening leads HY
spread widening by 2-4 quarters.

Historical benchmarks:
  GFC 2008:   net tightening ~80%
  COVID 2020: net tightening ~70%
  Dot-com:    net tightening ~50%

FRED series (quarterly, forward-filled to daily):
  DRTSCILM    — Net % tightening C&I loans, large/medium firms
  DRTSCIS     — Net % tightening C&I loans, small firms
  DRTSCLCC    — Net % tightening credit card loans
  SUBLPPDCNOT — Net % tightening commercial real estate loans

Composite score normalization: [-80, +80] net tightening → [0, 100]
  50 = zero net tightening (neutral)

Public API
----------
  FRED_SLOOS_SERIES   : dict
  SLOOS_REGIMES       : dict
  compute_sloos(df)           -> pd.DataFrame
  get_current_sloos(df)       -> dict
  run_sloos_monitor(df)       -> dict
"""

from __future__ import annotations

import datetime
import os

import numpy as np
import pandas as pd

FRED_SLOOS_SERIES: dict[str, str] = {
    "ci_large":    "DRTSCILM",
    "ci_small":    "DRTSCIS",
    "credit_card": "DRTSCLCC",
    "cre":         "SUBLPPDCNOT",
}

SLOOS_REGIMES: dict[str, tuple[int, int]] = {
    "Easing":            (0,  35),
    "Neutral":           (35, 50),
    "Mild Tightening":   (50, 65),
    "Tightening":        (65, 80),
    "Severe Tightening": (80, 100),
}

_COMPOSITE_WEIGHTS: dict[str, float] = {
    "ci_large": 0.50,
    "ci_small": 0.30,
    "cre":      0.20,
}

_NET_TIGHTENING_MIN = -80.0
_NET_TIGHTENING_MAX =  80.0
_NET_TIGHTENING_RANGE = _NET_TIGHTENING_MAX - _NET_TIGHTENING_MIN

_FRED_START = "1990-01-01"
_HIST_ROWS  = 504


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_sloos_fred() -> dict[str, pd.Series]:
    try:
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            return {}
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        result: dict[str, pd.Series] = {}
        for key, sid in FRED_SLOOS_SERIES.items():
            try:
                s = fred.get_series(sid, observation_start=_FRED_START)
                if s is not None and len(s) > 0:
                    s.index = pd.to_datetime(s.index)
                    result[key] = s.astype(float)
            except Exception:
                pass
        if not result:
            try:
                s = fred.get_series("STDSAUTO", observation_start=_FRED_START)
                if s is not None and len(s) > 0:
                    s.index = pd.to_datetime(s.index)
                    result["ci_large"] = s.astype(float)
            except Exception:
                pass
        return result
    except Exception:
        return {}


def _net_tightening_to_score(series: pd.Series) -> pd.Series:
    """Map net tightening percentage [-80, +80] to score [0, 100]. 50 = neutral."""
    normalized = (series - _NET_TIGHTENING_MIN) / _NET_TIGHTENING_RANGE * 100.0
    return normalized.clip(0.0, 100.0)


def _score_to_net_tightening(score: float) -> float:
    return score / 100.0 * _NET_TIGHTENING_RANGE + _NET_TIGHTENING_MIN


def _to_daily(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    combined = series.reindex(index.union(series.index)).ffill()
    return combined.reindex(index)


def _regime_from_score(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    for name, lo, hi in SLOOS_REGIMES.items():
        if lo <= score < hi:
            return name
    return "Severe Tightening"


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except Exception:
        return None


# ---------------------------------------------------------------------------
# compute_sloos
# ---------------------------------------------------------------------------

def compute_sloos(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with sloos_* columns added.

    Fetches FRED data when available. Falls back to any sloos_* columns
    already present in df. New columns:
        sloos_ci_large        : net tightening % (raw) — C&I large/med firms
        sloos_ci_small        : net tightening % (raw) — C&I small firms
        sloos_credit_card     : net tightening % (raw) — credit card
        sloos_cre             : net tightening % (raw) — CRE
        sloos_ci_large_score  : component score 0-100
        sloos_ci_small_score  : component score 0-100
        sloos_cre_score       : component score 0-100
        sloos_score           : composite 0-100
        sloos_regime          : str
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    idx = out.index

    fred_data = _fetch_sloos_fred()

    component_raw: dict[str, pd.Series] = {}
    component_scores: dict[str, pd.Series] = {}

    for key in ("ci_large", "ci_small", "cre", "credit_card"):
        raw_col = f"sloos_{key}"
        if key in fred_data:
            daily = _to_daily(fred_data[key], idx)
            out[raw_col] = daily
            component_raw[key] = daily
            component_scores[key] = _net_tightening_to_score(daily)
            out[f"sloos_{key}_score"] = component_scores[key]
        elif raw_col in out.columns:
            component_raw[key] = out[raw_col].astype(float)
            component_scores[key] = _net_tightening_to_score(component_raw[key])
            out[f"sloos_{key}_score"] = component_scores[key]

    if component_scores:
        available_weights = {
            k: _COMPOSITE_WEIGHTS[k]
            for k in _COMPOSITE_WEIGHTS
            if k in component_scores
        }
        if available_weights:
            total_w = sum(available_weights.values())
            composite = sum(
                component_scores[k] * w
                for k, w in available_weights.items()
            ) / total_w
        elif "credit_card" in component_scores:
            composite = component_scores["credit_card"]
        else:
            first_key = next(iter(component_scores))
            composite = component_scores[first_key]
        out["sloos_score"] = composite.clip(0.0, 100.0)
    else:
        out["sloos_score"] = pd.Series(np.nan, index=idx)

    out["sloos_regime"] = out["sloos_score"].apply(_regime_from_score)

    return out


# ---------------------------------------------------------------------------
# get_current_sloos
# ---------------------------------------------------------------------------

def get_current_sloos(df: pd.DataFrame) -> dict:
    """Return flat dict describing current SLOOS conditions."""
    try:
        if df is None or df.empty:
            return {"available": False}

        if "sloos_score" not in df.columns:
            enriched = compute_sloos(df)
        else:
            enriched = df

        if enriched["sloos_score"].notna().sum() < 2:
            return {"available": False}

        last = enriched.iloc[-1]

        def _col(name):
            if name not in enriched.columns:
                return None
            v = last.get(name)
            return _safe_float(v)

        score  = _col("sloos_score")
        regime = _regime_from_score(score if score is not None else float("nan"))

        net_ci_large  = _col("sloos_ci_large")
        net_ci_small  = _col("sloos_ci_small")
        net_cre       = _col("sloos_cre")

        sloos_direction = "Stable"
        if "sloos_score" in enriched.columns:
            score_series = enriched["sloos_score"].dropna()
            if len(score_series) >= 63:
                one_quarter_ago = float(score_series.iloc[-63])
                current_score   = float(score_series.iloc[-1])
                delta = current_score - one_quarter_ago
                if delta > 3:
                    sloos_direction = "Tightening Further"
                elif delta < -3:
                    sloos_direction = "Easing"
                else:
                    sloos_direction = "Stable"

        if score is not None:
            if score > 65:
                lead_signal = (
                    "High probability of HY spread widening within 2 quarters. "
                    "C&I credit contraction typically precedes defaults by 3-4 quarters."
                )
            elif score > 50:
                lead_signal = (
                    "Mild tightening underway. Watch for spread drift wider over next 2 quarters."
                )
            elif score < 35:
                lead_signal = (
                    "Banks easing standards — credit availability expanding. "
                    "Supportive for spread compression over the next 2-4 quarters."
                )
            else:
                lead_signal = "Neutral lending standards. No strong directional spread signal."
        else:
            lead_signal = "Insufficient data"

        if score is not None:
            raw_nt = _score_to_net_tightening(score)
            interpretation = (
                f"SLOOS composite score: {score:.0f}/100 ({regime}). "
                f"Net tightening equivalent: {raw_nt:+.1f}%. "
                f"Direction trend: {sloos_direction}. {lead_signal}"
            )
        else:
            interpretation = "SLOOS data unavailable — requires FRED API key."

        warning: str | None = None
        if score is not None and score > 65:
            warning = (
                f"SLOOS score {score:.0f}/100 — tightening regime. "
                "Historical precedent: spread widening typically occurs within 2 quarters "
                "of sustained net tightening above 30%."
            )

        return {
            "available":             score is not None,
            "sloos_score":           round(score, 1) if score is not None else None,
            "sloos_regime":          regime,
            "net_tightening_ci_large": round(net_ci_large, 1) if net_ci_large is not None else None,
            "net_tightening_ci_small": round(net_ci_small, 1) if net_ci_small is not None else None,
            "net_tightening_cre":    round(net_cre, 1) if net_cre is not None else None,
            "sloos_direction":       sloos_direction,
            "lead_signal":           lead_signal,
            "interpretation":        interpretation,
            "warning":               warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# run_sloos_monitor
# ---------------------------------------------------------------------------

def run_sloos_monitor(df: pd.DataFrame) -> dict:
    """Top-level SLOOS analysis entry point.

    Returns
    -------
    dict with keys:
        available            : bool
        current              : dict from get_current_sloos()
        df                   : pd.DataFrame enriched with sloos_* columns
        sloos_history        : pd.Series 0-100 (DatetimeIndex, daily)
        component_histories  : dict of {component: pd.Series}
        lead_correlation_6m  : float — corr(sloos_score, hy_spread shifted -126d)
        regime_history       : pd.Series of regime strings
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        enriched = compute_sloos(df)

        if enriched["sloos_score"].notna().sum() < 4:
            return {"available": False, "reason": "No SLOOS data — FRED API key required."}

        current = get_current_sloos(enriched)

        sloos_history = enriched["sloos_score"].tail(_HIST_ROWS).copy()
        sloos_history.name = "sloos_score"

        regime_history = enriched["sloos_regime"].tail(_HIST_ROWS).copy()
        regime_history.name = "sloos_regime"

        component_histories: dict[str, pd.Series] = {}
        for key in ("ci_large", "ci_small", "cre", "credit_card"):
            col = f"sloos_{key}_score"
            if col in enriched.columns:
                s = enriched[col].tail(_HIST_ROWS).copy()
                s.name = key
                component_histories[key] = s

        lead_correlation_6m: float = float("nan")
        try:
            if "hy_spread" in enriched.columns and enriched["sloos_score"].notna().sum() >= 126:
                fwd_spread = enriched["hy_spread"].shift(-126)
                aligned = pd.concat(
                    {"sloos": enriched["sloos_score"], "fwd": fwd_spread},
                    axis=1,
                ).dropna()
                if len(aligned) >= 63:
                    lead_correlation_6m = float(
                        np.corrcoef(aligned["sloos"].values, aligned["fwd"].values)[0, 1]
                    )
        except Exception:
            pass

        return {
            "available":           current.get("available", False),
            "current":             current,
            "df":                  enriched,
            "sloos_history":       sloos_history,
            "component_histories": component_histories,
            "lead_correlation_6m": lead_correlation_6m,
            "regime_history":      regime_history,
        }

    except Exception:
        return {"available": False}
