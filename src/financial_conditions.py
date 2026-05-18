"""
Financial Conditions Index (FCI) — credit-tilted single composite score.

The FCI aggregates HY spreads, IG spreads, VIX, long-term yields, USD
strength, and equity returns into a normalized 0–100 score where higher
= tighter conditions. Credit inputs receive the dominant weight (0.50
total) because this is a credit dashboard, not an equity monitor.

Methodology follows Goldman Sachs / Bloomberg FCI conventions:
  1. Z-score each input against a 252-day rolling window
  2. Sign-adjust so that tightening always = positive contribution
  3. Weighted average of z-scores
  4. Map to 0–100 via percentile rank against full sample history

A 100-bps tightening in Goldman's FCI historically adds ~0.3–0.5% to
HY default rates over the subsequent four quarters.

Public API
----------
  FCI_WEIGHTS           — module constant: component weights
  FCI_REGIMES           — module constant: regime score ranges
  ZSCORE_WINDOW         — 252
  compute_fci(df)       -> pd.DataFrame  (adds fci_score, fci_regime, fci_* columns)
  get_current_fci(df)   -> dict
  run_fci_analysis(df)  -> dict
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FCI_WEIGHTS: dict[str, float] = {
    "hy_spread":  0.30,
    "ig_spread":  0.20,
    "vix":        0.15,
    "yield_10y":  0.15,
    "usd":        0.10,
    "sp500":      0.10,
}

FCI_REGIMES: dict[str, tuple[int, int]] = {
    "Very Loose": (0,  25),
    "Loose":      (25, 40),
    "Neutral":    (40, 60),
    "Tight":      (60, 75),
    "Very Tight": (75, 100),
}

ZSCORE_WINDOW = 252

_HIST_ROWS    = 504

# Candidate column names for USD input (first match wins)
_USD_CANDIDATES = ["dxy", "usd_index", "dx_y_nyb", "dollar_index"]

# eurusd column names (if found, USD proxy = inverse)
_EURUSD_CANDIDATES = ["eurusd", "eur_usd", "eurusd_spot"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score with min_periods=63 to avoid early NaN explosion."""
    mu  = s.rolling(window, min_periods=63).mean()
    sig = s.rolling(window, min_periods=63).std(ddof=1)
    return (s - mu) / sig.replace(0.0, np.nan)


def _percentile_rank(s: pd.Series) -> pd.Series:
    """
    Map each observation to its percentile rank within all preceding + current
    observations (expanding window) on a 0–100 scale.
    """
    out = pd.Series(np.nan, index=s.index, dtype=float)
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


@functools.lru_cache(maxsize=2)
def _fetch_eurusd_yf(date_key: str) -> pd.Series | None:
    """Fetch EURUSD=X via yfinance, cached by date."""
    try:
        import yfinance as yf
        raw = yf.download("EURUSD=X", period="1y", auto_adjust=True, progress=False)
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


def _resolve_usd_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Find a USD-strength series from df:
      1. Direct DXY-style column → use as-is
      2. EURUSD column → invert (USD = 1 / EURUSD)
      3. Fallback yfinance fetch of EURUSD=X
    Returns a Series or None.
    """
    cols = set(df.columns)

    for cand in _USD_CANDIDATES:
        if cand in cols:
            s = df[cand].dropna()
            if not s.empty:
                return df[cand].rename("usd")

    for cand in _EURUSD_CANDIDATES:
        if cand in cols:
            s = df[cand].dropna()
            if not s.empty:
                return (1.0 / df[cand].replace(0, np.nan)).rename("usd")

    # Try yfinance fallback — invert EURUSD so stronger USD = higher value
    try:
        import datetime
        date_key = datetime.date.today().isoformat()
        eur_series = _fetch_eurusd_yf(date_key)
        if eur_series is not None:
            inv = (1.0 / eur_series.replace(0, np.nan)).rename("usd")
            return inv
    except Exception:
        pass

    return None


def _regime_from_score(score: float) -> str:
    """Map FCI score to regime label."""
    if np.isnan(score):
        return "Unknown"
    if score < 25:
        return "Very Loose"
    if score < 40:
        return "Loose"
    if score < 60:
        return "Neutral"
    if score < 75:
        return "Tight"
    return "Very Tight"


# ---------------------------------------------------------------------------
# Public: compute_fci
# ---------------------------------------------------------------------------

def compute_fci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a copy of df with FCI columns.

    New columns:
        fci_z_hy_spread      : rolling z-score of hy_spread
        fci_z_ig_spread      : rolling z-score of ig_spread
        fci_z_vix            : rolling z-score of vix
        fci_z_yield_10y      : rolling z-score of yield_10y
        fci_z_usd            : rolling z-score of usd proxy
        fci_z_sp500          : rolling NEGATIVE z-score of sp500 (lower = tighter)
        fci_raw              : weighted sum of z-scores (positive = tighter)
        fci_score            : 0-100 percentile rank of fci_raw
        fci_regime           : str regime label

    Parameters
    ----------
    df : master DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame (copy with new fci_ columns appended)
    """
    out = df.copy()

    try:
        cols = set(out.columns)
        component_series: dict[str, pd.Series] = {}
        weights_used: dict[str, float] = {}

        # ---- HY spread -------------------------------------------------------
        if "hy_spread" in cols:
            z = _rolling_zscore(out["hy_spread"])
            component_series["hy_spread"] = z
            weights_used["hy_spread"] = FCI_WEIGHTS["hy_spread"]
            out["fci_z_hy_spread"] = z

        # ---- IG spread -------------------------------------------------------
        if "ig_spread" in cols:
            z = _rolling_zscore(out["ig_spread"])
            component_series["ig_spread"] = z
            weights_used["ig_spread"] = FCI_WEIGHTS["ig_spread"]
            out["fci_z_ig_spread"] = z

        # ---- VIX -------------------------------------------------------------
        if "vix" in cols:
            z = _rolling_zscore(out["vix"])
            component_series["vix"] = z
            weights_used["vix"] = FCI_WEIGHTS["vix"]
            out["fci_z_vix"] = z

        # ---- yield_10y -------------------------------------------------------
        if "yield_10y" in cols:
            z = _rolling_zscore(out["yield_10y"])
            component_series["yield_10y"] = z
            weights_used["yield_10y"] = FCI_WEIGHTS["yield_10y"]
            out["fci_z_yield_10y"] = z

        # ---- USD proxy -------------------------------------------------------
        usd_series = _resolve_usd_series(out)
        if usd_series is not None:
            usd_aligned = usd_series.reindex(out.index).ffill()
            z = _rolling_zscore(usd_aligned)
            component_series["usd"] = z
            weights_used["usd"] = FCI_WEIGHTS["usd"]
            out["fci_z_usd"] = z

        # ---- SP500 (inverted: lower prices = tighter) -----------------------
        if "sp500" in cols:
            z = _rolling_zscore(out["sp500"])
            z_inv = -z
            component_series["sp500"] = z_inv
            weights_used["sp500"] = FCI_WEIGHTS["sp500"]
            out["fci_z_sp500"] = z_inv

        if not component_series:
            out["fci_raw"]    = np.nan
            out["fci_score"]  = np.nan
            out["fci_regime"] = "Unknown"
            return out

        # ---- Reweight to available components --------------------------------
        total_weight = sum(weights_used.values())
        fci_raw = pd.Series(0.0, index=out.index, dtype=float)
        for key, z_series in component_series.items():
            w = weights_used[key] / total_weight
            fci_raw = fci_raw.add(z_series.multiply(w), fill_value=0.0)

        fci_raw = fci_raw.where(
            pd.concat(list(component_series.values()), axis=1).notna().any(axis=1)
        )
        out["fci_raw"] = fci_raw

        # ---- Percentile rank → 0-100 ----------------------------------------
        out["fci_score"] = _percentile_rank(fci_raw.dropna()).reindex(out.index)

        # ---- Regime ----------------------------------------------------------
        out["fci_regime"] = out["fci_score"].apply(
            lambda v: _regime_from_score(float(v)) if pd.notna(v) else "Unknown"
        )

    except Exception:
        for col in ("fci_raw", "fci_score", "fci_regime"):
            if col not in out.columns:
                out[col] = np.nan
        if "fci_regime" in out.columns:
            out["fci_regime"] = out["fci_regime"].fillna("Unknown")

    return out


# ---------------------------------------------------------------------------
# Public: get_current_fci
# ---------------------------------------------------------------------------

def get_current_fci(df: pd.DataFrame) -> dict:
    """
    Compute and return the current FCI state.

    Parameters
    ----------
    df : master DataFrame with DatetimeIndex. Must contain at least one of
         the component columns.

    Returns
    -------
    dict with keys:
        available              : bool
        fci_score              : float 0-100
        fci_regime             : str
        component_contributions: dict {component: contribution}
        fci_1m_change          : float (delta vs 21 days ago)
        fci_3m_change          : float (delta vs 63 days ago)
        tightest_component     : str
        interpretation         : str
        warning                : str or None
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        enriched = compute_fci(df)

        if "fci_score" not in enriched.columns:
            return {"available": False}

        fci_valid = enriched["fci_score"].dropna()
        if fci_valid.empty:
            return {"available": False}

        last_row = enriched.iloc[-1]

        fci_score  = float(last_row.get("fci_score", float("nan")))
        fci_regime = str(last_row.get("fci_regime", "Unknown"))

        if np.isnan(fci_score):
            return {"available": False}

        # ---- Component contributions -----------------------------------------
        z_col_map = {
            "hy_spread":  "fci_z_hy_spread",
            "ig_spread":  "fci_z_ig_spread",
            "vix":        "fci_z_vix",
            "yield_10y":  "fci_z_yield_10y",
            "usd":        "fci_z_usd",
            "sp500":      "fci_z_sp500",
        }

        total_weight = sum(
            FCI_WEIGHTS[k]
            for k, v in z_col_map.items()
            if v in enriched.columns and pd.notna(last_row.get(v))
        )

        component_contributions: dict[str, float] = {}
        tightest_component = None
        tightest_val = -float("inf")

        for comp, z_col in z_col_map.items():
            if z_col in enriched.columns:
                z_val = last_row.get(z_col)
                if pd.notna(z_val):
                    w = FCI_WEIGHTS[comp] / total_weight if total_weight > 0 else 0.0
                    contribution = float(z_val) * w
                    component_contributions[comp] = round(contribution, 3)
                    if contribution > tightest_val:
                        tightest_val = contribution
                        tightest_component = comp

        # ---- Changes over time -----------------------------------------------
        fci_series = enriched["fci_score"]

        def _change_vs_lag(lag: int) -> float:
            clean = fci_series.dropna()
            if len(clean) < lag + 1:
                return float("nan")
            return float(clean.iloc[-1]) - float(clean.iloc[-lag - 1])

        fci_1m_change = _change_vs_lag(21)
        fci_3m_change = _change_vs_lag(63)

        # ---- Interpretation --------------------------------------------------
        change_str = ""
        if not np.isnan(fci_1m_change):
            direction = "tightened" if fci_1m_change > 0 else "loosened"
            change_str = f"Conditions have {direction} by {abs(fci_1m_change):.1f} points over the past month. "

        tightest_str = ""
        if tightest_component:
            tightest_str = f"The primary tightening driver is {tightest_component}. "

        interpretation = (
            f"FCI: {fci_score:.0f}/100 — {fci_regime}. "
            f"{change_str}"
            f"{tightest_str}"
        )

        # ---- Warning ---------------------------------------------------------
        warning: str | None = None
        if fci_score >= 75:
            warning = (
                f"Financial conditions are Very Tight ({fci_score:.0f}/100). "
                "A 100-bps FCI tightening historically adds ~0.3–0.5% to HY default "
                "rates over the subsequent 4 quarters. Credit origination stress "
                "is likely already emerging."
            )
        elif fci_score >= 60 and not np.isnan(fci_1m_change) and fci_1m_change > 5:
            warning = (
                f"FCI is Tight and tightening rapidly (+{fci_1m_change:.1f} pts/month). "
                "Accelerating conditions tightening can front-run fundamental credit "
                "deterioration by 1–2 quarters."
            )

        return {
            "available":               True,
            "fci_score":               round(fci_score, 1),
            "fci_regime":              fci_regime,
            "component_contributions": component_contributions,
            "fci_1m_change":           round(fci_1m_change, 1) if not np.isnan(fci_1m_change) else None,
            "fci_3m_change":           round(fci_3m_change, 1) if not np.isnan(fci_3m_change) else None,
            "tightest_component":      tightest_component,
            "interpretation":          interpretation,
            "warning":                 warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Public: run_fci_analysis
# ---------------------------------------------------------------------------

def run_fci_analysis(df: pd.DataFrame) -> dict:
    """
    Top-level FCI analysis entry point.

    Parameters
    ----------
    df : master DataFrame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        available              : bool
        current                : dict from get_current_fci()
        df                     : pd.DataFrame (enriched with fci_ columns)
        fci_history            : pd.Series (DatetimeIndex, last _HIST_ROWS rows)
        component_contributions: dict (same as current['component_contributions'])
        fci_vs_spreads_corr    : float (correlation of fci_score with hy_spread)
        regime_history         : pd.Series of str (DatetimeIndex)
        component_df           : pd.DataFrame (all component z-scores over time)
    """
    _unavail: dict = {"available": False}

    try:
        if df is None or df.empty:
            return _unavail

        # ---- Enrich and compute current state --------------------------------
        enriched = compute_fci(df)
        current  = get_current_fci(df)

        if not current.get("available"):
            return _unavail

        # ---- FCI history slice -----------------------------------------------
        fci_history = (
            enriched["fci_score"].tail(_HIST_ROWS).copy()
            if "fci_score" in enriched.columns
            else pd.Series(dtype=float)
        )

        # ---- Regime history --------------------------------------------------
        regime_history = (
            enriched["fci_regime"].tail(_HIST_ROWS).copy()
            if "fci_regime" in enriched.columns
            else pd.Series(dtype=str)
        )

        # ---- Component z-score DataFrame ------------------------------------
        z_cols = [c for c in enriched.columns if c.startswith("fci_z_")]
        component_df = enriched[z_cols].tail(_HIST_ROWS).copy() if z_cols else pd.DataFrame()

        # ---- FCI vs HY spread correlation ------------------------------------
        fci_vs_spreads_corr = float("nan")
        if "fci_score" in enriched.columns and "hy_spread" in enriched.columns:
            paired = enriched[["fci_score", "hy_spread"]].dropna()
            if len(paired) >= 30:
                fci_vs_spreads_corr = float(paired["fci_score"].corr(paired["hy_spread"]))
                if not np.isnan(fci_vs_spreads_corr):
                    fci_vs_spreads_corr = round(fci_vs_spreads_corr, 3)

        return {
            "available":               True,
            "current":                 current,
            "df":                      enriched,
            "fci_history":             fci_history,
            "component_contributions": current.get("component_contributions", {}),
            "fci_vs_spreads_corr":     fci_vs_spreads_corr,
            "regime_history":          regime_history,
            "component_df":            component_df,
        }

    except Exception:
        return _unavail
