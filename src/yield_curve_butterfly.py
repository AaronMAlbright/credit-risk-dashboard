"""
Yield curve butterfly (2s5s10s) — measures curvature of the yield curve.

Butterfly spread = 2 × yield_5y − yield_2y − yield_10y.

Negative butterfly (belly cheap relative to wings) = curve flattening.
This pattern tends to precede credit spread widening by 3-6 months.

Key thresholds:
  fly < -50 bps  → historically associated with recession within 6-12 months.
  2s10s ≤ -10    → curve inverted (classic recession signal).

Data sourcing priority:
  1. df columns: yield_2y, yield_5y, yield_10y
  2. FRED API: DGS2, DGS5, DGS10
  3. yfinance: ^IRX (proxy 3m), ^FVX (5y), ^TNX (10y), ^TYX (30y)

Public API
----------
  CURVE_REGIMES             — module constant
  compute_butterfly(df)     -> pd.DataFrame
  get_current_butterfly(df) -> dict
  run_butterfly_analysis(df) -> dict
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CURVE_REGIMES: dict[str, tuple[int, int]] = {
    "Steep":           (150,  999),
    "Normal":          ( 50,  150),
    "Flat":            (-10,   50),
    "Inverted":        (-50,  -10),
    "Deeply Inverted": (-999, -50),
}

_ZSCORE_WINDOW   = 252
_MIN_PERIODS     = 63
_HIST_ROWS       = 504
_INVERT_THRESH   = -10.0   # bps, 2s10s threshold for "Inverted"
_FLY_STRESS_THRESH = -50.0  # bps, butterfly below this = high-stress signal

# FRED series IDs
_FRED_2Y  = "DGS2"
_FRED_5Y  = "DGS5"
_FRED_10Y = "DGS10"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(s: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    mu  = s.rolling(window, min_periods=_MIN_PERIODS).mean()
    sig = s.rolling(window, min_periods=_MIN_PERIODS).std(ddof=1)
    return (s - mu) / sig.replace(0.0, np.nan)


def _percentile_rank(s: pd.Series) -> pd.Series:
    out   = pd.Series(np.nan, index=s.index, dtype=float)
    s_arr = s.values
    for i in range(len(s_arr)):
        if np.isnan(s_arr[i]):
            continue
        window_vals = s_arr[: i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) < 2:
            continue
        rank = float(np.sum(valid <= s_arr[i]) - 1) / float(len(valid) - 1) * 100.0
        out.iloc[i] = float(np.clip(rank, 0.0, 100.0))
    return out


def _fetch_fred_series(series_id: str, start: str = "2000-01-01") -> pd.Series | None:
    try:
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            return None
        from fredapi import Fred  # type: ignore
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id, observation_start=start)
        if data is None or len(data) == 0:
            return None
        s = pd.Series(data, dtype=float)
        s.index = pd.to_datetime(s.index)
        return s.dropna()
    except Exception:
        return None


def _fetch_yf_yield(ticker: str) -> pd.Series | None:
    """Fetch a yield proxy from yfinance; returns % (not decimal)."""
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].iloc[:, 0]
        else:
            close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        s = close.dropna()
        return s if not s.empty else None
    except Exception:
        return None


def _resolve_yield(df: pd.DataFrame, col_name: str, fred_id: str, yf_ticker: str | None = None) -> pd.Series | None:
    """Resolve a yield series: df column → FRED → yfinance."""
    if col_name in df.columns:
        s = df[col_name].dropna()
        if not s.empty:
            return df[col_name]

    fred_s = _fetch_fred_series(fred_id)
    if fred_s is not None and not fred_s.empty:
        return fred_s.reindex(df.index).ffill()

    if yf_ticker is not None:
        yf_s = _fetch_yf_yield(yf_ticker)
        if yf_s is not None and not yf_s.empty:
            return yf_s.reindex(df.index).ffill()

    return None


def _curve_regime(slope_bps: float) -> str:
    if np.isnan(slope_bps):
        return "Unknown"
    if slope_bps > 150:
        return "Steep"
    if slope_bps > 50:
        return "Normal"
    if slope_bps > -10:
        return "Flat"
    if slope_bps > -50:
        return "Inverted"
    return "Deeply Inverted"


def _butterfly_credit_implication(butterfly_bps: float, slope_bps: float, regime: str, direction: str) -> str:
    inverted = slope_bps < _INVERT_THRESH if not np.isnan(slope_bps) else False
    fly_stress = butterfly_bps < _FLY_STRESS_THRESH if not np.isnan(butterfly_bps) else False

    if inverted and fly_stress:
        base = (
            f"Both the 2s10s slope ({slope_bps:.0f} bps) and the butterfly "
            f"({butterfly_bps:.0f} bps) are deeply negative — a historically high-conviction "
            "recession signal. HY spreads have widened 150-400 bps in the 6-12 months "
            "following this configuration in prior cycles."
        )
    elif inverted:
        base = (
            f"The yield curve is inverted ({slope_bps:.0f} bps 2s10s) — a classic recession "
            "precursor. Credit spread widening typically lags curve inversion by 6-12 months. "
            "IG duration is at risk; HY spreads are the more sensitive indicator to watch."
        )
    elif fly_stress:
        base = (
            f"The butterfly has flattened significantly ({butterfly_bps:.0f} bps) even without "
            "full inversion — this curvature compression historically precedes spread widening "
            "by 3-6 months. The belly of the curve is cheap relative to wings."
        )
    elif regime == "Flat":
        base = (
            f"The yield curve is flat ({slope_bps:.0f} bps 2s10s) — a late-cycle configuration "
            "that compresses net interest margins and often precedes credit quality deterioration. "
            "Monitor for inversion over the next 1-3 months."
        )
    elif regime in ("Steep", "Normal"):
        base = (
            f"The yield curve slope ({slope_bps:.0f} bps 2s10s) is {regime.lower()}, consistent "
            "with an expansion or early-cycle environment. Historically supportive of credit "
            "fundamentals as bank profitability and lending appetite are healthy."
        )
    else:
        base = f"Yield curve regime: {regime}. Butterfly: {butterfly_bps:.0f} bps."

    if direction == "Flattening":
        base += " The curve is actively flattening — a developing headwind for credit."
    elif direction == "Steepening":
        base += " Active steepening is a near-term positive for credit risk appetite."

    return base


# ---------------------------------------------------------------------------
# Public: compute_butterfly
# ---------------------------------------------------------------------------

def compute_butterfly(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a copy of df with butterfly_* and curve_* columns.

    New columns
    -----------
    curve_yield_2y           : 2y yield used (%)
    curve_yield_5y           : 5y yield used (%)
    curve_yield_10y          : 10y yield used (%)
    butterfly_bps            : 2×5y − 2y − 10y, in bps (×100)
    curve_slope_2s10s_bps    : (10y − 2y) × 100
    curve_slope_2s5s_bps     : (5y − 2y) × 100
    curve_slope_5s10s_bps    : (10y − 5y) × 100
    curve_regime             : str regime label (driven by 2s10s)
    curve_inversion_depth_bps: max(0, -(2s10s)) in bps
    curve_inversion_days     : rolling count of consecutive inverted days
    butterfly_score          : 0-100 stress score (higher = more negative butterfly)
    butterfly_direction      : "Flattening" / "Steepening" / "Stable"
    """
    out = df.copy()

    try:
        y2  = _resolve_yield(out, "yield_2y",  _FRED_2Y,  None)
        y5  = _resolve_yield(out, "yield_5y",  _FRED_5Y,  "^FVX")
        y10 = _resolve_yield(out, "yield_10y", _FRED_10Y, "^TNX")

        if y2 is None or y5 is None or y10 is None:
            for col in (
                "curve_yield_2y", "curve_yield_5y", "curve_yield_10y",
                "butterfly_bps", "curve_slope_2s10s_bps", "curve_slope_2s5s_bps",
                "curve_slope_5s10s_bps", "curve_regime", "curve_inversion_depth_bps",
                "curve_inversion_days", "butterfly_score", "butterfly_direction",
            ):
                out[col] = np.nan
            out["curve_regime"]        = "Unknown"
            out["butterfly_direction"] = "Unknown"
            return out

        out["curve_yield_2y"]  = y2
        out["curve_yield_5y"]  = y5
        out["curve_yield_10y"] = y10

        # Butterfly = 2×5y − 2y − 10y, expressed in bps
        butterfly_pct = 2.0 * y5 - y2 - y10
        out["butterfly_bps"]         = butterfly_pct * 100.0
        out["curve_slope_2s10s_bps"] = (y10 - y2)  * 100.0
        out["curve_slope_2s5s_bps"]  = (y5  - y2)  * 100.0
        out["curve_slope_5s10s_bps"] = (y10 - y5)  * 100.0

        out["curve_regime"] = out["curve_slope_2s10s_bps"].apply(
            lambda v: _curve_regime(float(v)) if pd.notna(v) else "Unknown"
        )

        slope = out["curve_slope_2s10s_bps"]
        out["curve_inversion_depth_bps"] = np.maximum(0.0, -slope).where(slope.notna(), np.nan)

        # Consecutive days inverted (rolling count resets on non-inversion)
        inverted_flag = (slope < _INVERT_THRESH).astype(float)
        inverted_flag[slope.isna()] = np.nan
        consec = []
        count  = 0
        for v in inverted_flag:
            if np.isnan(v):
                consec.append(np.nan)
                count = 0
            elif v == 1.0:
                count += 1
                consec.append(float(count))
            else:
                count = 0
                consec.append(0.0)
        out["curve_inversion_days"] = pd.array(consec, dtype=float)

        # Butterfly stress score: higher = more negative butterfly (more stress)
        # Invert sign so that more-negative butterfly → higher percentile rank score
        neg_fly = -out["butterfly_bps"]
        out["butterfly_score"] = _percentile_rank(neg_fly.dropna()).reindex(out.index)

        # Direction: 21d change in 2s10s slope
        slope_21d_chg = slope - slope.shift(21)
        direction_s = pd.Series("Stable", index=out.index, dtype=object)
        direction_s[slope_21d_chg < -5.0] = "Flattening"
        direction_s[slope_21d_chg >  5.0] = "Steepening"
        out["butterfly_direction"] = direction_s

    except Exception:
        for col in (
            "curve_yield_2y", "curve_yield_5y", "curve_yield_10y",
            "butterfly_bps", "curve_slope_2s10s_bps", "curve_slope_2s5s_bps",
            "curve_slope_5s10s_bps", "curve_regime", "curve_inversion_depth_bps",
            "curve_inversion_days", "butterfly_score", "butterfly_direction",
        ):
            if col not in out.columns:
                out[col] = np.nan
        for col in ("curve_regime", "butterfly_direction"):
            if col in out.columns:
                out[col] = out[col].fillna("Unknown")
            else:
                out[col] = "Unknown"

    return out


# ---------------------------------------------------------------------------
# Public: get_current_butterfly
# ---------------------------------------------------------------------------

def get_current_butterfly(df: pd.DataFrame) -> dict:
    """Return current yield curve butterfly state.

    Returns
    -------
    dict with keys:
        available, butterfly_bps, slope_2s10s_bps, slope_2s5s_bps,
        slope_5s10s_bps, curve_regime, inversion_depth_bps,
        inversion_duration_days, butterfly_score, butterfly_direction,
        credit_implication, warning
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        enriched = compute_butterfly(df)

        fly_valid = enriched["butterfly_bps"].dropna() if "butterfly_bps" in enriched.columns else pd.Series(dtype=float)
        if fly_valid.empty:
            return {"available": False}

        last = enriched.iloc[-1]

        def _fv(col: str) -> float | None:
            if col not in enriched.columns:
                return None
            v = last[col]
            try:
                fv = float(v)
                return None if np.isnan(fv) else fv
            except Exception:
                return None

        butterfly_bps    = _fv("butterfly_bps")
        slope_2s10s      = _fv("curve_slope_2s10s_bps")
        slope_2s5s       = _fv("curve_slope_2s5s_bps")
        slope_5s10s      = _fv("curve_slope_5s10s_bps")
        inv_depth        = _fv("curve_inversion_depth_bps")
        inv_days_raw     = _fv("curve_inversion_days")
        fly_score        = _fv("butterfly_score")

        if butterfly_bps is None or fly_score is None:
            return {"available": False}

        inv_days  = int(inv_days_raw) if inv_days_raw is not None else 0
        inv_depth = inv_depth if inv_depth is not None else 0.0

        regime    = str(last.get("curve_regime",        "Unknown"))
        direction = str(last.get("butterfly_direction", "Unknown"))

        slope_for_impl = slope_2s10s if slope_2s10s is not None else float("nan")
        credit_impl = _butterfly_credit_implication(butterfly_bps, slope_for_impl, regime, direction)

        warning: str | None = None
        if slope_2s10s is not None and slope_2s10s < _INVERT_THRESH and butterfly_bps < _FLY_STRESS_THRESH:
            warning = (
                f"High-conviction stress signal: 2s10s is {slope_2s10s:.0f} bps (inverted) "
                f"and butterfly is {butterfly_bps:.0f} bps. This dual-negative configuration "
                "has preceded recessions in 1989, 2000, 2006, and 2019. Credit spread "
                "widening of 100-400+ bps historically follows within 6-12 months."
            )
        elif slope_2s10s is not None and slope_2s10s < _INVERT_THRESH:
            warning = (
                f"Yield curve inverted ({slope_2s10s:.0f} bps 2s10s). Credit spread "
                "widening typically follows with a 6-12 month lag."
            )
        elif butterfly_bps < _FLY_STRESS_THRESH:
            warning = (
                f"Butterfly at {butterfly_bps:.0f} bps — below the -50 bps stress threshold. "
                "Curvature compression at this level has historically led HY spread widening "
                "by 3-6 months."
            )

        return {
            "available":               True,
            "butterfly_bps":           round(butterfly_bps, 1),
            "slope_2s10s_bps":         round(slope_2s10s, 1) if slope_2s10s is not None else None,
            "slope_2s5s_bps":          round(slope_2s5s,  1) if slope_2s5s  is not None else None,
            "slope_5s10s_bps":         round(slope_5s10s, 1) if slope_5s10s is not None else None,
            "curve_regime":            regime,
            "inversion_depth_bps":     round(inv_depth, 1),
            "inversion_duration_days": inv_days,
            "butterfly_score":         round(fly_score, 1),
            "butterfly_direction":     direction,
            "credit_implication":      credit_impl,
            "warning":                 warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Public: run_butterfly_analysis
# ---------------------------------------------------------------------------

def run_butterfly_analysis(df: pd.DataFrame) -> dict:
    """Top-level butterfly analysis entry point.

    Returns
    -------
    dict with keys:
        available, current, df, butterfly_history, slope_2s10s_history,
        slope_2s5s_history, butterfly_score_history, regime_history,
        credit_lead_correlation
    """
    _unavail: dict = {"available": False}

    try:
        if df is None or df.empty:
            return _unavail

        enriched = compute_butterfly(df)
        current  = get_current_butterfly(df)

        if not current.get("available"):
            return _unavail

        def _hist(col: str) -> pd.Series:
            if col not in enriched.columns:
                return pd.Series(dtype=float)
            return enriched[col].tail(_HIST_ROWS).copy()

        butterfly_history       = _hist("butterfly_bps")
        slope_2s10s_history     = _hist("curve_slope_2s10s_bps")
        slope_2s5s_history      = _hist("curve_slope_2s5s_bps")
        butterfly_score_history = _hist("butterfly_score")
        regime_history          = _hist("curve_regime")

        # credit_lead_correlation: corr(butterfly_score, hy_spread.shift(-126))
        credit_lead_corr = float("nan")
        if "butterfly_score" in enriched.columns and "hy_spread" in enriched.columns:
            try:
                fly_sc   = enriched["butterfly_score"]
                hy_fwd   = enriched["hy_spread"].shift(-126)
                paired   = pd.concat([fly_sc, hy_fwd], axis=1).dropna()
                paired.columns = ["fly", "hy"]
                if len(paired) >= 63:
                    corr_val = float(paired["fly"].corr(paired["hy"]))
                    if not np.isnan(corr_val):
                        credit_lead_corr = round(corr_val, 4)
            except Exception:
                pass

        return {
            "available":              True,
            "current":                current,
            "df":                     enriched,
            "butterfly_history":      butterfly_history,
            "slope_2s10s_history":    slope_2s10s_history,
            "slope_2s5s_history":     slope_2s5s_history,
            "butterfly_score_history": butterfly_score_history,
            "regime_history":         regime_history,
            "credit_lead_correlation": credit_lead_corr,
        }

    except Exception:
        return _unavail
