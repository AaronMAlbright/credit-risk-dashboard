"""
Sovereign Credit Stress Monitor.

Tracks how EU peripheral and EM sovereign stress spills over into corporate
credit markets. Sovereign stress is a leading indicator for corporate spread
widening.

Transmission mechanism:
  EU peripherals → bank holdings of sovereign debt → bank funding stress
                 → corporate credit tightening
  EM sovereigns  → dollar funding pressure → EM corporate credit stress
                 → broad EM HY widening (2-4 week lead)

EU peripheral proxy: EWI/EWG 63d relative return z-score.
  Negative = Italy underperforming Germany = peripheral stress.

EM sovereign proxy: EMB/TLT 21d relative return z-score.
  Negative = EM bond underperforming US Treasury = EM sovereign stress.

Sovereign stress score (0-100):
  sovereign_stress_score = 0.55 × EU_stress + 0.45 × EM_stress

Public API
----------
  SOVEREIGN_WEIGHTS      — dict of component weights
  CONTAGION_THRESHOLD    — float; score above this triggers contagion_flag
  fetch_sovereign_data(period)      -> dict  (yfinance, cached by date)
  compute_historical_sovereign(df)  -> pd.DataFrame
  get_sovereign_snapshot(df)        -> dict
  run_sovereign_contagion(df)       -> dict
"""
from __future__ import annotations

import datetime
import functools

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOVEREIGN_WEIGHTS: dict[str, float] = {
    "eu_peripheral": 0.55,
    "em_sovereign":  0.45,
}
CONTAGION_THRESHOLD: float = 65.0

_LIVE_PERIOD = "2y"
_ZSCORE_WINDOW = 252
_MIN_PERIODS = 63
_EU_REL_WINDOW = 63     # 63d relative return for EU peripheral proxy
_EM_REL_WINDOW = 21     # 21d relative return for EM sovereign proxy
_CORR_WINDOW   = 63     # rolling correlation window (sovereign ↔ HY spread)
_HIST_ROWS     = 504    # ~2 trading years to surface

# Regime thresholds
_CONTAINED_MAX    = 30.0
_ELEVATED_MAX     = 55.0
_CONTAGION_MAX    = 75.0

# Score normalization: z-score clipped to [-3, 3], then mapped [0, 100]
# Invert because negative z-score (underperformance) = high stress
_Z_FLOOR = -3.0
_Z_CAP   =  3.0

# HY spread column candidates in master DataFrame
_HY_SPREAD_CANDIDATES = (
    "hy_spread", "hy_oas", "us_hy_spread", "credit_spread", "hy_option_adj_spread"
)

# EU peripheral column candidates in master DataFrame
_EU_PERIPHERAL_CANDIDATES = (
    "italy_spread", "peripheral_spread", "eu_peripheral_stress", "btp_bund_spread"
)

# EM sovereign column candidates in master DataFrame
_EM_SOVEREIGN_CANDIDATES = (
    "em_spread", "em_sovereign_stress", "sovereign_stress", "emb_spread"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _rolling_zscore(s: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    rm = s.rolling(window=window, min_periods=_MIN_PERIODS).mean()
    rs = s.rolling(window=window, min_periods=_MIN_PERIODS).std(ddof=1)
    return ((s - rm) / rs.replace(0, np.nan)).clip(_Z_FLOOR, _Z_CAP)


def _zscore_to_stress(z: float) -> float:
    """Map a z-score where negative = stress (underperformance) to [0, 100]."""
    if np.isnan(z):
        return float("nan")
    # Invert: z = -3 → score 100; z = +3 → score 0
    score = (-z - _Z_FLOOR) / (_Z_CAP - _Z_FLOOR) * 100.0
    return float(np.clip(score, 0.0, 100.0))


def _series_zscore_to_stress(z_series: pd.Series) -> pd.Series:
    """Vectorised version of _zscore_to_stress for a full Series."""
    score = (-z_series - _Z_FLOOR) / (_Z_CAP - _Z_FLOOR) * 100.0
    return score.clip(0.0, 100.0)


def _sovereign_regime(score: float) -> str:
    if np.isnan(score):
        return "Unknown"
    if score < _CONTAINED_MAX:
        return "Contained"
    if score < _ELEVATED_MAX:
        return "Elevated"
    if score < _CONTAGION_MAX:
        return "Contagion Risk"
    return "Crisis Mode"


def _spillover_risk(score: float) -> str:
    if np.isnan(score):
        return "Unknown"
    if score >= CONTAGION_THRESHOLD:
        return "High"
    if score >= 40.0:
        return "Moderate"
    return "Low"


def _last_valid(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    v = float(clean.iloc[-1])
    return v if np.isfinite(v) else None


def _rel_return(s: pd.Series, window: int) -> pd.Series:
    """Rolling window relative return: price[t] / price[t-window] - 1."""
    if len(s) < window + 1:
        return pd.Series(float("nan"), index=s.index)
    return s / s.shift(window) - 1.0


# ---------------------------------------------------------------------------
# Live yfinance fetch (cached per calendar day)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _cached_sovereign_history(date_key: str) -> dict:  # noqa: ARG001
    """Download sovereign-proxy ETF closes. Cached per day."""
    try:
        import yfinance as yf
        tickers = ["EWI", "EWG", "EMB", "TLT", "PCY", "VWOB", "IEF"]
        raw = yf.download(
            tickers, period=_LIVE_PERIOD, auto_adjust=True, progress=False
        )
        if raw.empty:
            return {"available": False}

        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].copy()
        else:
            closes = raw.copy()

        closes = closes.dropna(how="all")
        if closes.empty:
            return {"available": False}

        return {"available": True, "closes": closes}
    except Exception:
        return {"available": False}


def fetch_sovereign_data(period: str = "2y") -> dict:
    """Fetch sovereign-proxy ETF data via yfinance. Cached by calendar date.

    Parameters
    ----------
    period : yfinance period string (default "2y")

    Returns
    -------
    dict:
        available : bool
        closes    : pd.DataFrame or None — daily closes for sovereign tickers
        as_of     : str
    """
    try:
        date_key = datetime.date.today().isoformat()
        result = _cached_sovereign_history(date_key)
        return {
            "available": result.get("available", False),
            "closes": result.get("closes"),
            "as_of": datetime.datetime.utcnow().isoformat(),
        }
    except Exception:
        return {
            "available": False,
            "closes": None,
            "as_of": datetime.datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# EU and EM stress series builders
# ---------------------------------------------------------------------------

def _build_eu_stress_from_etfs(closes: pd.DataFrame) -> pd.Series | None:
    """Build EU peripheral stress series from EWI/EWG relative performance."""
    try:
        if "EWI" not in closes.columns or "EWG" not in closes.columns:
            return None

        ewi = closes["EWI"].dropna()
        ewg = closes["EWG"].dropna()

        common = ewi.index.intersection(ewg.index)
        if len(common) < _EU_REL_WINDOW + _MIN_PERIODS:
            return None

        ewi_a = ewi.reindex(common)
        ewg_a = ewg.reindex(common)

        ewi_ret = _rel_return(ewi_a, _EU_REL_WINDOW)
        ewg_ret = _rel_return(ewg_a, _EU_REL_WINDOW)
        rel = ewi_ret - ewg_ret

        z = _rolling_zscore(rel, window=_ZSCORE_WINDOW)
        stress = _series_zscore_to_stress(z)
        stress.name = "eu_stress"
        return stress
    except Exception:
        return None


def _build_em_stress_from_etfs(closes: pd.DataFrame) -> pd.Series | None:
    """Build EM sovereign stress series from EMB/TLT relative performance."""
    try:
        emb_col = None
        for candidate in ("EMB", "PCY", "VWOB"):
            if candidate in closes.columns:
                emb_col = candidate
                break

        if emb_col is None or "TLT" not in closes.columns:
            return None

        emb = closes[emb_col].dropna()
        tlt = closes["TLT"].dropna()

        common = emb.index.intersection(tlt.index)
        if len(common) < _EM_REL_WINDOW + _MIN_PERIODS:
            return None

        emb_a = emb.reindex(common)
        tlt_a = tlt.reindex(common)

        emb_ret = _rel_return(emb_a, _EM_REL_WINDOW)
        tlt_ret = _rel_return(tlt_a, _EM_REL_WINDOW)
        rel = emb_ret - tlt_ret

        z = _rolling_zscore(rel, window=_ZSCORE_WINDOW)
        stress = _series_zscore_to_stress(z)
        stress.name = "em_stress"
        return stress
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public: compute_historical_sovereign
# ---------------------------------------------------------------------------

def compute_historical_sovereign(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical sovereign stress series and add to a copy of df.

    Checks df columns first, then falls back to live ETF data.

    Appended columns:
      sovereign_eu_stress   — EU peripheral stress score [0, 100]
      sovereign_em_stress   — EM sovereign stress score [0, 100]
      sovereign_stress_score — composite [0, 100]
      sovereign_regime       — categorical string

    Returns a copy of df (never raises).
    """
    try:
        out = df.copy()

        eu_series: pd.Series | None = None
        em_series: pd.Series | None = None

        # ---- Check df for pre-existing sovereign columns ---------------------
        eu_col = _resolve_column(out, _EU_PERIPHERAL_CANDIDATES)
        em_col = _resolve_column(out, _EM_SOVEREIGN_CANDIDATES)

        if eu_col is not None:
            z = _rolling_zscore(out[eu_col].copy())
            eu_series = _series_zscore_to_stress(z)
            eu_series.index = out.index

        if em_col is not None:
            z = _rolling_zscore(out[em_col].copy())
            em_series = _series_zscore_to_stress(z)
            em_series.index = out.index

        # ---- Fall back to live ETF data if needed ----------------------------
        if eu_series is None or em_series is None:
            data = fetch_sovereign_data()
            if data.get("available") and data.get("closes") is not None:
                closes: pd.DataFrame = data["closes"]

                if eu_series is None:
                    eu_live = _build_eu_stress_from_etfs(closes)
                    if eu_live is not None:
                        eu_series = eu_live.reindex(out.index, method="ffill")

                if em_series is None:
                    em_live = _build_em_stress_from_etfs(closes)
                    if em_live is not None:
                        em_series = em_live.reindex(out.index, method="ffill")

        # ---- Assemble composite ----------------------------------------------
        if eu_series is not None:
            out["sovereign_eu_stress"] = eu_series.values if len(eu_series) == len(out) else eu_series.reindex(out.index)
        else:
            out["sovereign_eu_stress"] = pd.Series(float("nan"), index=out.index)

        if em_series is not None:
            out["sovereign_em_stress"] = em_series.values if len(em_series) == len(out) else em_series.reindex(out.index)
        else:
            out["sovereign_em_stress"] = pd.Series(float("nan"), index=out.index)

        eu_w = SOVEREIGN_WEIGHTS["eu_peripheral"]
        em_w = SOVEREIGN_WEIGHTS["em_sovereign"]

        eu_s = out["sovereign_eu_stress"]
        em_s = out["sovereign_em_stress"]

        both_valid = eu_s.notna() & em_s.notna()
        eu_only    = eu_s.notna() & em_s.isna()
        em_only    = eu_s.isna()  & em_s.notna()

        composite = pd.Series(float("nan"), index=out.index)
        composite[both_valid] = (
            eu_w * eu_s[both_valid] + em_w * em_s[both_valid]
        )
        composite[eu_only] = eu_s[eu_only]
        composite[em_only] = em_s[em_only]
        out["sovereign_stress_score"] = composite.clip(0.0, 100.0)

        out["sovereign_regime"] = out["sovereign_stress_score"].apply(
            lambda v: _sovereign_regime(float(v)) if not np.isnan(float(v)) else "Unknown"
        )

        return out
    except Exception:
        return df.copy()


# ---------------------------------------------------------------------------
# Public: get_sovereign_snapshot
# ---------------------------------------------------------------------------

def get_sovereign_snapshot(df: pd.DataFrame) -> dict:
    """Compute current sovereign stress snapshot from df + live ETF data.

    Parameters
    ----------
    df : Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict:
        sovereign_stress_score : float 0-100
        eu_peripheral_score    : float 0-100
        em_sovereign_score     : float 0-100
        regime                 : str
        contagion_flag         : bool
        spillover_risk         : str
        interpretation         : str
        warning                : str or None
    """
    _unavail = {
        "sovereign_stress_score": float("nan"),
        "eu_peripheral_score":    float("nan"),
        "em_sovereign_score":     float("nan"),
        "regime":                 "Unknown",
        "contagion_flag":         False,
        "spillover_risk":         "Unknown",
        "interpretation":         "Insufficient data for sovereign stress analysis.",
        "warning":                None,
    }

    try:
        enriched = compute_historical_sovereign(df)

        eu_score  = float("nan")
        em_score  = float("nan")
        composite = float("nan")

        if "sovereign_eu_stress" in enriched.columns:
            v = _last_valid(enriched["sovereign_eu_stress"])
            if v is not None:
                eu_score = v

        if "sovereign_em_stress" in enriched.columns:
            v = _last_valid(enriched["sovereign_em_stress"])
            if v is not None:
                em_score = v

        if "sovereign_stress_score" in enriched.columns:
            v = _last_valid(enriched["sovereign_stress_score"])
            if v is not None:
                composite = v

        if np.isnan(composite) and not (np.isnan(eu_score) and np.isnan(em_score)):
            eu_w = SOVEREIGN_WEIGHTS["eu_peripheral"]
            em_w = SOVEREIGN_WEIGHTS["em_sovereign"]
            if not np.isnan(eu_score) and not np.isnan(em_score):
                composite = eu_w * eu_score + em_w * em_score
            elif not np.isnan(eu_score):
                composite = eu_score
            else:
                composite = em_score

        if np.isnan(composite):
            return _unavail

        regime      = _sovereign_regime(composite)
        spill_risk  = _spillover_risk(composite)

        # contagion_flag: high sovereign stress AND HY spread elevated vs 21d avg
        contagion_flag = False
        hy_col = _resolve_column(df, _HY_SPREAD_CANDIDATES)
        if composite >= CONTAGION_THRESHOLD and hy_col is not None:
            hy = df[hy_col].dropna()
            if len(hy) >= 21:
                hy_last = float(hy.iloc[-1])
                hy_21d_avg = float(hy.iloc[-21:].mean())
                if np.isfinite(hy_last) and np.isfinite(hy_21d_avg):
                    contagion_flag = bool(hy_last > hy_21d_avg)

        eu_str = f"{eu_score:.0f}" if not np.isnan(eu_score) else "N/A"
        em_str = f"{em_score:.0f}" if not np.isnan(em_score) else "N/A"

        regime_desc = {
            "Contained":      "Sovereign stress is isolated; limited corporate spillover risk.",
            "Elevated":       "Sovereign stress elevated; watch for corporate spread widening.",
            "Contagion Risk": "Corporate credit likely to widen as sovereign stress transmits via banks.",
            "Crisis Mode":    "Systematic sovereign-to-corporate transmission underway.",
            "Unknown":        "Cannot determine sovereign stress regime.",
        }

        interpretation = (
            f"Sovereign stress: {composite:.0f}/100 → '{regime}'. "
            f"EU peripheral: {eu_str}/100, EM sovereign: {em_str}/100. "
            f"Spillover risk: {spill_risk}. "
            f"{regime_desc.get(regime, '')}"
        )

        warning: str | None = None
        if contagion_flag:
            warning = (
                f"CONTAGION FLAG ACTIVE: Sovereign stress score {composite:.0f} exceeds "
                f"threshold ({CONTAGION_THRESHOLD:.0f}) and HY spreads are above their "
                "21-day average. Sovereign-to-corporate spread transmission is likely."
            )
        elif composite >= _ELEVATED_MAX:
            warning = (
                f"Sovereign stress elevated ({composite:.0f}/100). "
                "Monitor EU peripheral spreads and EM bond flows for corporate spillover."
            )

        return {
            "sovereign_stress_score": round(composite, 1),
            "eu_peripheral_score":    round(eu_score, 1) if not np.isnan(eu_score) else float("nan"),
            "em_sovereign_score":     round(em_score, 1) if not np.isnan(em_score) else float("nan"),
            "regime":                 regime,
            "contagion_flag":         contagion_flag,
            "spillover_risk":         spill_risk,
            "interpretation":         interpretation,
            "warning":                warning,
        }
    except Exception:
        return _unavail


# ---------------------------------------------------------------------------
# Public: run_sovereign_contagion
# ---------------------------------------------------------------------------

def run_sovereign_contagion(df: pd.DataFrame) -> dict:
    """Top-level entry point for the sovereign contagion module.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict:
        available            : bool
        current              : dict  (from get_sovereign_snapshot)
        df                   : pd.DataFrame  (enriched with sovereign_ columns)
        stress_history       : pd.Series  (sovereign_stress_score, last 504 rows)
        eu_stress_history    : pd.Series  (sovereign_eu_stress, last 504 rows)
        em_stress_history    : pd.Series  (sovereign_em_stress, last 504 rows)
        spillover_correlation: float or None  (63d rolling corr, latest value)
        regime_history       : pd.Series  (sovereign_regime, last 504 rows)
    """
    _unavail: dict = {
        "available":             False,
        "current":               {"sovereign_stress_score": float("nan")},
        "df":                    df.copy() if df is not None else pd.DataFrame(),
        "stress_history":        pd.Series(dtype=float),
        "eu_stress_history":     pd.Series(dtype=float),
        "em_stress_history":     pd.Series(dtype=float),
        "spillover_correlation": None,
        "regime_history":        pd.Series(dtype=str),
    }

    try:
        enriched = compute_historical_sovereign(df)

        has_stress = (
            "sovereign_stress_score" in enriched.columns
            and enriched["sovereign_stress_score"].notna().any()
        )

        if not has_stress:
            return _unavail

        current = get_sovereign_snapshot(df)

        tail = enriched.tail(_HIST_ROWS)

        stress_history = (
            tail["sovereign_stress_score"].copy()
            if "sovereign_stress_score" in tail.columns
            else pd.Series(dtype=float)
        )
        eu_history = (
            tail["sovereign_eu_stress"].copy()
            if "sovereign_eu_stress" in tail.columns
            else pd.Series(dtype=float)
        )
        em_history = (
            tail["sovereign_em_stress"].copy()
            if "sovereign_em_stress" in tail.columns
            else pd.Series(dtype=float)
        )
        regime_history = (
            tail["sovereign_regime"].copy()
            if "sovereign_regime" in tail.columns
            else pd.Series(dtype=str)
        )

        # ---- Spillover correlation (sovereign stress vs HY spread) -----------
        spillover_corr: float | None = None
        hy_col = _resolve_column(enriched, _HY_SPREAD_CANDIDATES)
        if hy_col is not None and "sovereign_stress_score" in enriched.columns:
            try:
                stress_s = enriched["sovereign_stress_score"].dropna()
                hy_s     = enriched[hy_col].dropna()
                common   = stress_s.index.intersection(hy_s.index)
                if len(common) >= _CORR_WINDOW:
                    s_aligned = stress_s.reindex(common)
                    h_aligned = hy_s.reindex(common)
                    rolling_corr = s_aligned.rolling(
                        window=_CORR_WINDOW, min_periods=_MIN_PERIODS
                    ).corr(h_aligned)
                    last_corr = rolling_corr.dropna()
                    if not last_corr.empty:
                        spillover_corr = round(float(last_corr.iloc[-1]), 4)
            except Exception:
                spillover_corr = None

        return {
            "available":             True,
            "current":               current,
            "df":                    enriched,
            "stress_history":        stress_history,
            "eu_stress_history":     eu_history,
            "em_stress_history":     em_history,
            "spillover_correlation": spillover_corr,
            "regime_history":        regime_history,
        }

    except Exception:
        return _unavail
