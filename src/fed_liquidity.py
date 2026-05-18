"""
Fed balance sheet liquidity analysis.

Tracks Fed total assets, reverse repo (RRP), bank reserve balances, and SOFR —
the upstream liquidity variables that lead credit spreads by 4-8 weeks.

Fed balance sheet expansion (QE) floods the system with reserves, compresses
credit spreads, and supports risk assets. QT does the opposite. The
net-liquidity measure (fed_assets - rrp - reserves) captures the share of
central bank money that is actively deployed into markets rather than
sitting idle at the Fed.

Public API
----------
  FRED_SERIES        : dict — FRED series IDs used
  fetch_fed_data     — pulls raw series from FRED API (falls back to df columns)
  compute_liquidity_metrics  — derives regime / signals from raw data
  run_fed_liquidity  — top-level entry point
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

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

FRED_SERIES: dict[str, str] = {
    "fed_assets":   "WALCL",      # Fed total assets, weekly, billions
    "rrp":          "WLRRAL",     # Reverse repo (weekly) — primary
    "rrp_daily":    "RRPONTSYD",  # Reverse repo (daily)  — fallback
    "reserves":     "WRESBAL",    # Bank reserve balances, weekly, billions
    "sofr":         "SOFR",       # SOFR rate, daily
    "fed_funds":    "DFF",        # Effective fed funds rate, daily
}

# Column names to check in df when FRED API is unavailable
_DF_COLUMNS: dict[str, list[str]] = {
    "fed_assets": ["fed_assets", "WALCL"],
    "rrp":        ["rrp", "WLRRAL", "RRPONTSYD"],
    "reserves":   ["reserves", "WRESBAL"],
    "sofr":       ["sofr", "SOFR"],
    "fed_funds":  ["fed_funds", "DFF", "fed_funds_eff"],
}

_RRP_LARGE       = 1_000.0   # $1 trillion threshold, billions
_RRP_DRAIN_RATE  =   100.0   # >$100B/month drain = cash flowing to markets
_LEAD_WEEKS      = 6         # central estimate; range 4-8

# ---------------------------------------------------------------------------
# FRED fetch helpers
# ---------------------------------------------------------------------------

def _get_fred_series(fred_client, series_id: str, start: str = "2015-01-01") -> pd.Series | None:
    """Fetch one FRED series. Returns pd.Series or None on failure."""
    try:
        data = fred_client.get_series(series_id, observation_start=start)
        if data is None or len(data) == 0:
            return None
        s = pd.Series(data, dtype=float)
        s.index = pd.to_datetime(s.index)
        s.name = series_id
        return s.dropna()
    except Exception:
        return None


def _column_from_df(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Return the first matching non-empty column from df, or None."""
    for col in candidates:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                return s.rename(col)
    return None


# ---------------------------------------------------------------------------
# Public: fetch_fed_data
# ---------------------------------------------------------------------------

def fetch_fed_data(df: pd.DataFrame | None = None) -> dict:
    """Fetch Fed liquidity data from FRED API, falling back to df columns.

    Tries FRED API first (requires FRED_API_KEY env var and fredapi package).
    If that fails for any series, falls back to looking for the series in df.

    Parameters
    ----------
    df : optional master DataFrame with DatetimeIndex; used as fallback source.

    Returns
    -------
    dict with keys:
        available    : bool — True if at least one series was retrieved
        fed_assets   : pd.Series or None
        rrp          : pd.Series or None
        reserves     : pd.Series or None
        sofr         : pd.Series or None
        fed_funds    : pd.Series or None
        as_of        : str — ISO date of most recent observation across series
        source       : str — "FRED API", "DataFrame columns", or "Mixed"
    """
    try:
        result: dict = {
            "available":  False,
            "fed_assets": None,
            "rrp":        None,
            "reserves":   None,
            "sofr":       None,
            "fed_funds":  None,
            "as_of":      "unknown",
            "source":     "none",
        }

        api_key = os.environ.get("FRED_API_KEY", "")
        fred_client = None
        fred_available = False

        if api_key:
            try:
                from fredapi import Fred  # type: ignore
                fred_client = Fred(api_key=api_key)
                fred_available = True
            except Exception:
                fred_available = False

        from_api: set[str]  = set()
        from_df:  set[str]  = set()

        # ---- fed_assets ----
        if fred_available:
            s = _get_fred_series(fred_client, "WALCL")
            if s is not None:
                result["fed_assets"] = s
                from_api.add("fed_assets")
        if result["fed_assets"] is None and df is not None:
            s = _column_from_df(df, _DF_COLUMNS["fed_assets"])
            if s is not None:
                result["fed_assets"] = s
                from_df.add("fed_assets")

        # ---- rrp (try weekly first, then daily) ----
        if fred_available:
            s = _get_fred_series(fred_client, "WLRRAL")
            if s is None:
                s = _get_fred_series(fred_client, "RRPONTSYD")
            if s is not None:
                result["rrp"] = s
                from_api.add("rrp")
        if result["rrp"] is None and df is not None:
            s = _column_from_df(df, _DF_COLUMNS["rrp"])
            if s is not None:
                result["rrp"] = s
                from_df.add("rrp")

        # ---- reserves ----
        if fred_available:
            s = _get_fred_series(fred_client, "WRESBAL")
            if s is not None:
                result["reserves"] = s
                from_api.add("reserves")
        if result["reserves"] is None and df is not None:
            s = _column_from_df(df, _DF_COLUMNS["reserves"])
            if s is not None:
                result["reserves"] = s
                from_df.add("reserves")

        # ---- sofr ----
        if fred_available:
            s = _get_fred_series(fred_client, "SOFR")
            if s is not None:
                result["sofr"] = s
                from_api.add("sofr")
        if result["sofr"] is None and df is not None:
            s = _column_from_df(df, _DF_COLUMNS["sofr"])
            if s is not None:
                result["sofr"] = s
                from_df.add("sofr")

        # ---- fed_funds ----
        if fred_available:
            s = _get_fred_series(fred_client, "DFF")
            if s is not None:
                result["fed_funds"] = s
                from_api.add("fed_funds")
        if result["fed_funds"] is None and df is not None:
            s = _column_from_df(df, _DF_COLUMNS["fed_funds"])
            if s is not None:
                result["fed_funds"] = s
                from_df.add("fed_funds")

        # ---- availability & metadata ----
        series_available = [k for k in ("fed_assets", "rrp", "reserves", "sofr", "fed_funds")
                            if result[k] is not None]
        result["available"] = len(series_available) > 0

        if from_api and from_df:
            result["source"] = "Mixed"
        elif from_api:
            result["source"] = "FRED API"
        elif from_df:
            result["source"] = "DataFrame columns"
        else:
            result["source"] = "none"

        # most recent observation date across all series
        latest_dates = []
        for k in series_available:
            s = result[k]
            if s is not None and len(s) > 0:
                latest_dates.append(s.index.max())
        if latest_dates:
            result["as_of"] = str(max(latest_dates).date())

        return result

    except Exception:
        return {
            "available":  False,
            "fed_assets": None,
            "rrp":        None,
            "reserves":   None,
            "sofr":       None,
            "fed_funds":  None,
            "as_of":      "unknown",
            "source":     "error",
        }


# ---------------------------------------------------------------------------
# Public: compute_liquidity_metrics
# ---------------------------------------------------------------------------

def compute_liquidity_metrics(fed_data: dict) -> dict:
    """Derive regime classifications and derived metrics from raw Fed series.

    Parameters
    ----------
    fed_data : dict from fetch_fed_data()

    Returns
    -------
    dict with keys:
        fed_assets_latest        : float (billions)
        fed_assets_13w_chg_pct   : float
        fed_assets_regime        : str
        rrp_latest               : float or None
        rrp_4w_chg               : float or None
        rrp_signal               : str
        reserves_latest          : float or None
        sofr_latest              : float or None
        net_liquidity            : float or None
        net_liquidity_4w_chg     : float or None
        liquidity_composite_signal : str
    """
    try:
        def _latest(s: pd.Series | None) -> float | None:
            if s is None or len(s) == 0:
                return None
            return float(s.iloc[-1])

        def _nw_chg(s: pd.Series | None, weeks: int) -> float | None:
            """N-week change in absolute units."""
            if s is None or len(s) == 0:
                return None
            # try weekly spacing first; fall back to business day count
            try:
                s_sorted = s.sort_index()
                approx_days = weeks * 7
                cutoff = s_sorted.index[-1] - pd.Timedelta(days=approx_days)
                prior = s_sorted[s_sorted.index <= cutoff]
                if len(prior) == 0:
                    return None
                return float(s_sorted.iloc[-1]) - float(prior.iloc[-1])
            except Exception:
                return None

        # ---- fed assets ----
        fa = fed_data.get("fed_assets")
        fa_latest = _latest(fa)
        fa_latest_nn = fa_latest if fa_latest is not None else float("nan")

        fa_13w_chg_pct: float | None = None
        if fa is not None and len(fa) >= 2:
            chg_13w = _nw_chg(fa, 13)
            if chg_13w is not None:
                base_val = fa_latest_nn - chg_13w
                if base_val != 0:
                    fa_13w_chg_pct = chg_13w / abs(base_val) * 100.0

        # fed assets regime
        fa_regime = "Unknown"
        if fa_13w_chg_pct is not None:
            if fa_13w_chg_pct > 5.0:
                fa_regime = "Expanding (QE)"
            elif fa_13w_chg_pct > 0.0:
                fa_regime = "Stable"
            elif fa_13w_chg_pct > -5.0:
                fa_regime = "Tapering"
            else:
                fa_regime = "QT (Quantitative Tightening)"

        # ---- RRP ----
        rrp_s      = fed_data.get("rrp")
        rrp_latest = _latest(rrp_s)

        rrp_4w_chg: float | None = None
        if rrp_s is not None and len(rrp_s) >= 2:
            rrp_4w_chg = _nw_chg(rrp_s, 4)

        rrp_signal = "Unknown"
        if rrp_latest is not None:
            if rrp_latest > _RRP_LARGE:
                # large balance — cash parked at Fed, not deployed
                if rrp_4w_chg is not None and rrp_4w_chg < -_RRP_DRAIN_RATE:
                    rrp_signal = "Draining rapidly — cash deploying into markets (credit positive)"
                else:
                    rrp_signal = "Elevated — abundant liquidity parked at Fed (neutral/positive)"
            elif rrp_latest > 200.0:
                if rrp_4w_chg is not None and rrp_4w_chg < -_RRP_DRAIN_RATE:
                    rrp_signal = "Draining — late-cycle deployment (mildly positive)"
                else:
                    rrp_signal = "Moderate — liquidity partially deployed"
            else:
                rrp_signal = "Near zero — liquidity fully deployed (late cycle / neutral)"
        elif rrp_s is None:
            rrp_signal = "RRP data unavailable"

        # ---- reserves ----
        res_s      = fed_data.get("reserves")
        res_latest = _latest(res_s)

        # ---- SOFR ----
        sofr_s      = fed_data.get("sofr")
        sofr_latest = _latest(sofr_s)

        # ---- net liquidity: fed_assets - rrp - reserves ----
        net_liq: float | None = None
        net_liq_4w: float | None = None

        if fa is not None and len(fa) > 0:
            # build aligned series for net liquidity over time
            fa_sorted = fa.sort_index()
            combined = pd.DataFrame({"fa": fa_sorted})

            if rrp_s is not None and len(rrp_s) > 0:
                combined = combined.join(rrp_s.rename("rrp").sort_index(), how="left")
            else:
                combined["rrp"] = 0.0

            if res_s is not None and len(res_s) > 0:
                combined = combined.join(res_s.rename("res").sort_index(), how="left")
            else:
                combined["res"] = 0.0

            combined = combined.ffill().dropna(subset=["fa"])
            if len(combined) > 0:
                net_series = combined["fa"] - combined.get("rrp", 0.0).fillna(0.0) \
                             - combined.get("res", 0.0).fillna(0.0)
                net_liq = float(net_series.iloc[-1])

                if len(net_series) >= 2:
                    chg = _nw_chg(net_series, 4)
                    if chg is not None:
                        net_liq_4w = chg

        # ---- composite liquidity signal ----
        composite = "Neutral"
        if fa_regime in ("Expanding (QE)",):
            composite = "Expanding"
        elif fa_regime in ("QT (Quantitative Tightening)",):
            composite = "Contracting"
        elif fa_regime == "Tapering":
            # downgrade slightly if RRP also near zero
            if rrp_latest is not None and rrp_latest < 200.0:
                composite = "Contracting"
            else:
                composite = "Neutral"
        else:
            composite = "Neutral"

        # net liquidity trend overrides if strong
        if net_liq_4w is not None and fa_latest_nn > 0:
            net_chg_pct = net_liq_4w / abs(fa_latest_nn) * 100.0
            if net_chg_pct > 3.0 and composite != "Expanding":
                composite = "Expanding"
            elif net_chg_pct < -3.0 and composite != "Contracting":
                composite = "Contracting"

        return {
            "fed_assets_latest":          fa_latest,
            "fed_assets_13w_chg_pct":     round(fa_13w_chg_pct, 2) if fa_13w_chg_pct is not None else None,
            "fed_assets_regime":          fa_regime,
            "rrp_latest":                 rrp_latest,
            "rrp_4w_chg":                 round(rrp_4w_chg, 1) if rrp_4w_chg is not None else None,
            "rrp_signal":                 rrp_signal,
            "reserves_latest":            res_latest,
            "sofr_latest":                sofr_latest,
            "net_liquidity":              round(net_liq, 1) if net_liq is not None else None,
            "net_liquidity_4w_chg":       round(net_liq_4w, 1) if net_liq_4w is not None else None,
            "liquidity_composite_signal": composite,
        }

    except Exception:
        return {
            "fed_assets_latest":          None,
            "fed_assets_13w_chg_pct":     None,
            "fed_assets_regime":          "Unknown",
            "rrp_latest":                 None,
            "rrp_4w_chg":                 None,
            "rrp_signal":                 "Unknown",
            "reserves_latest":            None,
            "sofr_latest":                None,
            "net_liquidity":              None,
            "net_liquidity_4w_chg":       None,
            "liquidity_composite_signal": "Unknown",
        }


# ---------------------------------------------------------------------------
# Internal: build historical dict for charting (last 104 weeks)
# ---------------------------------------------------------------------------

def _build_historical(fed_data: dict, weeks: int = 104) -> dict:
    """Return dict of {series_name: pd.Series} trimmed to last `weeks` weeks."""
    hist: dict[str, pd.Series] = {}
    cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(weeks=weeks)

    for key in ("fed_assets", "rrp", "reserves", "sofr", "fed_funds"):
        s = fed_data.get(key)
        if s is not None and len(s) > 0:
            s_sorted = s.sort_index()
            hist[key] = s_sorted[s_sorted.index >= cutoff]

    # net liquidity historical series
    if "fed_assets" in hist:
        fa_h = hist["fed_assets"]
        combined = pd.DataFrame({"fa": fa_h})
        if "rrp" in hist:
            combined = combined.join(hist["rrp"].rename("rrp"), how="left")
        else:
            combined["rrp"] = 0.0
        if "reserves" in hist:
            combined = combined.join(hist["reserves"].rename("res"), how="left")
        else:
            combined["res"] = 0.0
        combined = combined.ffill().dropna(subset=["fa"])
        if len(combined) > 0:
            hist["net_liquidity"] = (
                combined["fa"]
                - combined.get("rrp", pd.Series(0.0, index=combined.index)).fillna(0.0)
                - combined.get("res", pd.Series(0.0, index=combined.index)).fillna(0.0)
            )

    return hist


# ---------------------------------------------------------------------------
# Internal: credit implication text
# ---------------------------------------------------------------------------

def _credit_implication(metrics: dict) -> str:
    """One-sentence credit spread implication from current liquidity regime."""
    regime    = metrics.get("fed_assets_regime", "Unknown")
    composite = metrics.get("liquidity_composite_signal", "Neutral")
    rrp_sig   = metrics.get("rrp_signal", "")
    nl_4w     = metrics.get("net_liquidity_4w_chg")

    if composite == "Expanding":
        base = (
            "Fed balance sheet expansion is adding reserves — historically a 4-8 week "
            "tailwind for credit spreads as excess liquidity seeks yield."
        )
    elif composite == "Contracting":
        base = (
            "Balance sheet contraction (QT) is draining reserves — a 4-8 week headwind "
            "for credit as funding costs rise and risk appetite narrows."
        )
    else:
        base = (
            "Liquidity is in a neutral/stable regime — credit spread direction will be "
            "driven more by fundamental credit dynamics than Fed mechanics near-term."
        )

    if nl_4w is not None and abs(nl_4w) > 50:
        direction = "expanding" if nl_4w > 0 else "contracting"
        base += (
            f" Net liquidity (assets minus RRP minus reserves) is {direction} at "
            f"${abs(nl_4w):.0f}B over 4 weeks, amplifying the signal."
        )

    return base


# ---------------------------------------------------------------------------
# Public: run_fed_liquidity
# ---------------------------------------------------------------------------

def run_fed_liquidity(df: pd.DataFrame) -> dict:
    """Top-level entry point for Fed liquidity analysis.

    Attempts FRED API fetch first, then falls back to df columns.

    Parameters
    ----------
    df : master DataFrame with DatetimeIndex (used as fallback data source).

    Returns
    -------
    dict with keys:
        available          : bool
        fed_data           : dict from fetch_fed_data
        metrics            : dict from compute_liquidity_metrics
        historical         : dict — {series_name: pd.Series} last 104 weeks
        credit_implication : str
        interpretation     : str
        lead_weeks         : int
    """
    try:
        # pass df so fallback column search can use it
        fed_data = fetch_fed_data(df=df)

        if not fed_data.get("available", False):
            return {"available": False}

        metrics    = compute_liquidity_metrics(fed_data)
        historical = _build_historical(fed_data, weeks=104)

        # ---- build interpretation ----
        regime    = metrics.get("fed_assets_regime", "Unknown")
        composite = metrics.get("liquidity_composite_signal", "Neutral")
        fa_latest = metrics.get("fed_assets_latest")
        fa_chg    = metrics.get("fed_assets_13w_chg_pct")
        rrp_l     = metrics.get("rrp_latest")
        nl        = metrics.get("net_liquidity")
        sofr      = metrics.get("sofr_latest")

        fa_str = f"${fa_latest:,.0f}B" if fa_latest is not None else "N/A"
        chg_str = (
            f"{fa_chg:+.1f}% over 13 weeks"
            if fa_chg is not None else "change N/A"
        )
        rrp_str = f"${rrp_l:,.0f}B" if rrp_l is not None else "unavailable"
        nl_str  = f"${nl:,.0f}B" if nl is not None else "N/A"
        sofr_str = f"{sofr:.2f}%" if sofr is not None else "N/A"

        interpretation = (
            f"Fed total assets: {fa_str} ({chg_str}) — regime: {regime}. "
            f"RRP outstanding: {rrp_str}. "
            f"Net deployed liquidity: {nl_str}. "
            f"SOFR: {sofr_str}. "
            f"Composite liquidity signal: {composite}."
        )

        credit_impl = _credit_implication(metrics)

        return {
            "available":           True,
            "fed_data":            fed_data,
            "metrics":             metrics,
            "historical":          historical,
            "credit_implication":  credit_impl,
            "interpretation":      interpretation,
            "lead_weeks":          _LEAD_WEEKS,
        }

    except Exception:
        return {"available": False}
