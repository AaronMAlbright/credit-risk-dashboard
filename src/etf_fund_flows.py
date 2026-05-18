"""
Credit ETF fund flow monitor.

Sustained outflows from credit ETFs create technical selling pressure as
authorized participants unwind creations, forcing ETFs to sell underlying
bonds. Sustained inflows provide a technical bid. HYG is the world's most
liquid HY vehicle — its flows move first and matter most.

True flow = ΔAUM − (price_return × prior_AUM), which strips the price
appreciation effect from the AUM change. When shares_outstanding is
unavailable, volume × sign(Δprice) gives the direction proxy.

Composite score (0-100, percentile rank of 21-day flow z-scores):
  HYG 0.40 + LQD 0.30 + JNK 0.30

Public API
----------
  FLOW_ETFS     : list
  FLOW_WEIGHTS  : dict
  FLOW_REGIMES  : dict
  fetch_flow_data(period)           -> dict of {ticker: pd.DataFrame}
  compute_etf_flows(df)             -> pd.DataFrame
  get_current_flows()               -> dict
  run_etf_fund_flows(df)            -> dict
"""

from __future__ import annotations

import datetime
import functools

import numpy as np
import pandas as pd

FLOW_ETFS: list[str] = ["HYG", "LQD", "JNK", "BKLN", "EMB"]

FLOW_WEIGHTS: dict[str, float] = {
    "HYG": 0.40,
    "LQD": 0.30,
    "JNK": 0.30,
}

FLOW_REGIMES: dict[str, tuple[int, int]] = {
    "Heavy Outflow": (0,  30),
    "Mild Outflow":  (30, 45),
    "Neutral":       (45, 55),
    "Mild Inflow":   (55, 70),
    "Heavy Inflow":  (70, 100),
}

_FETCH_PERIOD    = "1y"
_FLOW_WINDOW     = 21
_ZSCORE_WINDOW   = 63
_HIST_ROWS       = 504
_LARGE_OUTFLOW_Z = 2.0
_HYG_LARGE_OUTFLOW_USD = 500_000_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except Exception:
        return None


def _regime_from_score(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    for name, (lo, hi) in FLOW_REGIMES.items():
        if lo <= score < hi:
            return name
    return "Heavy Inflow"


def _rolling_zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    roll = series.rolling(window, min_periods=window // 3)
    mu   = roll.mean()
    sig  = roll.std(ddof=1).replace(0, np.nan)
    return (series - mu) / sig


def _shares_outstanding(ticker_obj) -> float | None:
    try:
        so = ticker_obj.fast_info.shares_outstanding
        if so is not None and not np.isnan(float(so)) and float(so) > 0:
            return float(so)
    except Exception:
        pass
    try:
        info = ticker_obj.info
        so = info.get("sharesOutstanding")
        if so is not None and float(so) > 0:
            return float(so)
    except Exception:
        pass
    return None


def _compute_true_flow(close: pd.Series, shares: float | None) -> pd.Series:
    """Compute daily net flow series.

    Primary: shares × close = AUM; flow = ΔAUM − return × prior_AUM.
    Fallback: signed_volume proxy (requires 'volume' attribute — passed separately).
    """
    if shares is not None:
        aum = close * shares
        delta_aum   = aum.diff()
        price_return = close.pct_change()
        price_effect = price_return * aum.shift(1)
        return (delta_aum - price_effect).fillna(0.0)
    return pd.Series(dtype=float)


def _signed_volume_flow(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff())
    return (volume * sign).fillna(0.0)


# ---------------------------------------------------------------------------
# fetch_flow_data  (date-cached to avoid repeated same-day yfinance calls)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _cached_fetch(period: str, date_key: str) -> dict[str, pd.DataFrame]:
    """Inner fetch function cached by (period, date)."""
    try:
        import yfinance as yf
        result: dict[str, pd.DataFrame] = {}
        for ticker in FLOW_ETFS:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period=period, auto_adjust=True)
                if hist is None or hist.empty:
                    continue
                hist.index = pd.to_datetime(hist.index).tz_localize(None)
                so = _shares_outstanding(t)

                close = hist["Close"].astype(float)
                volume = hist["Volume"].astype(float) if "Volume" in hist.columns else pd.Series(0.0, index=hist.index)

                if so is not None:
                    flow = _compute_true_flow(close, so)
                    flow_source = "aum"
                else:
                    flow = _signed_volume_flow(close, volume)
                    flow_source = "volume_proxy"

                flow_21d = flow.rolling(_FLOW_WINDOW, min_periods=_FLOW_WINDOW // 2).sum()
                flow_z   = _rolling_zscore(flow_21d, _ZSCORE_WINDOW)

                df_t = pd.DataFrame({
                    "close":       close,
                    "volume":      volume,
                    "flow_daily":  flow,
                    "flow_21d":    flow_21d,
                    "flow_z":      flow_z,
                    "flow_source": flow_source,
                }, index=hist.index)

                result[ticker] = df_t

            except Exception:
                pass
        return result
    except Exception:
        return {}


def fetch_flow_data(period: str = _FETCH_PERIOD) -> dict[str, pd.DataFrame]:
    """Fetch ETF flow data for all FLOW_ETFS via yfinance. Cached by calendar date."""
    date_key = datetime.date.today().isoformat()
    return _cached_fetch(period, date_key)


# ---------------------------------------------------------------------------
# compute_etf_flows
# ---------------------------------------------------------------------------

def compute_etf_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich df with flow_* columns derived from live ETF data or df columns.

    Adds:
        flow_hyg_z, flow_lqd_z, flow_jnk_z  : 21d flow z-scores
        flow_score                            : composite 0-100
        flow_regime                           : str
        flow_hyg_21d                          : HYG 21d cumulative flow proxy
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)

    etf_data = fetch_flow_data()

    z_series: dict[str, pd.Series] = {}

    for ticker in FLOW_WEIGHTS:
        col_z   = f"flow_{ticker.lower()}_z"
        col_21d = f"flow_{ticker.lower()}_21d"

        if ticker in etf_data and not etf_data[ticker].empty:
            etf_df  = etf_data[ticker]
            fz      = etf_df["flow_z"].reindex(out.index.union(etf_df.index)).ffill(limit=5).reindex(out.index)
            f21d    = etf_df["flow_21d"].reindex(out.index.union(etf_df.index)).ffill(limit=5).reindex(out.index)
            out[col_z]   = fz
            out[col_21d] = f21d
            z_series[ticker] = fz
        elif col_z in out.columns:
            z_series[ticker] = out[col_z].astype(float)

    if z_series:
        total_w = sum(FLOW_WEIGHTS[t] for t in z_series)
        composite_z = sum(
            z_series[t] * FLOW_WEIGHTS[t] for t in z_series
        ) / total_w
        pct_rank = composite_z.expanding(min_periods=20).rank(pct=True).mul(100)
        out["flow_score"]  = pct_rank.clip(0.0, 100.0)
    else:
        out["flow_score"]  = pd.Series(np.nan, index=out.index)

    out["flow_regime"] = out["flow_score"].apply(_regime_from_score)

    return out


# ---------------------------------------------------------------------------
# get_current_flows
# ---------------------------------------------------------------------------

def get_current_flows() -> dict:
    """Return flat dict describing current ETF flow conditions (live data)."""
    try:
        etf_data = fetch_flow_data()
        if not etf_data:
            return {"available": False}

        z_last: dict[str, float] = {}
        f21d_last: dict[str, float] = {}
        etf_flows: dict[str, dict] = {}

        for ticker in FLOW_ETFS:
            if ticker not in etf_data or etf_data[ticker].empty:
                continue
            row  = etf_data[ticker].iloc[-1]
            fz   = _safe_float(row.get("flow_z"))
            f21d = _safe_float(row.get("flow_21d"))
            src  = row.get("flow_source", "volume_proxy")

            if fz is not None:
                z_last[ticker] = fz
            if f21d is not None:
                f21d_last[ticker] = f21d

            flow_direction = "Neutral"
            if fz is not None:
                if fz > 0.5:
                    flow_direction = "Inflow"
                elif fz < -0.5:
                    flow_direction = "Outflow"

            etf_flows[ticker] = {
                "flow_21d_proxy":  round(f21d, 0) if f21d is not None else None,
                "flow_z":          round(fz, 2)  if fz is not None else None,
                "flow_direction":  flow_direction,
                "flow_source":     src,
            }

        if not z_last:
            return {"available": False}

        weighted_tickers = [t for t in FLOW_WEIGHTS if t in z_last]
        if not weighted_tickers:
            return {"available": False}

        total_w   = sum(FLOW_WEIGHTS[t] for t in weighted_tickers)
        composite = sum(z_last[t] * FLOW_WEIGHTS[t] for t in weighted_tickers) / total_w

        all_z = list(z_last.values())
        all_z_arr = np.array(all_z)
        z_min, z_max = all_z_arr.min(), all_z_arr.max()
        z_range = z_max - z_min if z_max != z_min else 1.0
        flow_score = float(np.clip((composite - z_min) / z_range * 100.0, 0.0, 100.0))
        if len(all_z) < 3:
            flow_score = float(np.clip((composite + 3.0) / 6.0 * 100.0, 0.0, 100.0))

        flow_regime = _regime_from_score(flow_score)

        largest_outflow_etf = min(z_last, key=z_last.get) if z_last else None
        largest_inflow_etf  = max(z_last, key=z_last.get) if z_last else None

        hyg_flow_21d = f21d_last.get("HYG")
        hyg_z        = z_last.get("HYG")

        if flow_score < 30:
            interpretation = (
                f"ETF flows signal heavy outflows (score: {flow_score:.0f}/100). "
                f"Technical selling pressure elevated — credit bond demand is broadly impaired. "
                f"Largest outflow ETF: {largest_outflow_etf}."
            )
        elif flow_score < 45:
            interpretation = (
                f"Mild net outflows from credit ETFs (score: {flow_score:.0f}/100). "
                "Monitor for follow-through selling in underlying credit markets."
            )
        elif flow_score > 70:
            interpretation = (
                f"Strong inflows into credit ETFs (score: {flow_score:.0f}/100). "
                f"Technical demand supporting spread compression. "
                f"Largest inflow ETF: {largest_inflow_etf}."
            )
        elif flow_score > 55:
            interpretation = (
                f"Mild net inflows (score: {flow_score:.0f}/100). "
                "Credit technicals supportive at the margin."
            )
        else:
            interpretation = (
                f"Credit ETF flows are neutral (score: {flow_score:.0f}/100). "
                "No strong technical directional signal."
            )

        warning: str | None = None
        hyg_large_outflow = (
            hyg_z is not None and hyg_z < -_LARGE_OUTFLOW_Z
        ) or (
            hyg_flow_21d is not None and hyg_flow_21d < -_HYG_LARGE_OUTFLOW_USD
        )
        if flow_score < 30:
            warning = (
                f"Credit ETF flow score {flow_score:.0f}/100 — heavy outflow regime. "
                "Sustained ETF outflows historically precede spread widening by days to weeks."
            )
        if hyg_large_outflow:
            hyg_warn = (
                f"HYG flow z-score {hyg_z:.2f} — 2σ+ outflow event. "
                "Large single-channel HYG outflows often precede bond market dislocations."
            )
            warning = f"{warning}  {hyg_warn}" if warning else hyg_warn

        return {
            "available":           True,
            "flow_score":          round(flow_score, 1),
            "flow_regime":         flow_regime,
            "etf_flows":           etf_flows,
            "largest_outflow_etf": largest_outflow_etf,
            "largest_inflow_etf":  largest_inflow_etf,
            "hyg_flow_21d":        round(hyg_flow_21d, 0) if hyg_flow_21d is not None else None,
            "interpretation":      interpretation,
            "warning":             warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# run_etf_fund_flows
# ---------------------------------------------------------------------------

def run_etf_fund_flows(df: pd.DataFrame) -> dict:
    """Top-level ETF fund flow analysis entry point.

    Returns
    -------
    dict with keys:
        available           : bool
        current             : dict from get_current_flows()
        df_enriched         : pd.DataFrame with flow_* columns
        flow_history        : pd.Series 0-100 (DatetimeIndex)
        etf_flow_details    : dict of {ticker: pd.DataFrame} from fetch_flow_data()
        lead_correlation    : float — corr(flow_score, hy_spread shifted -21d)
        regime_history      : pd.Series of regime strings
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        enriched = compute_etf_flows(df)

        current = get_current_flows()

        if not current.get("available") and enriched["flow_score"].notna().sum() < 4:
            return {"available": False}

        flow_history = enriched["flow_score"].tail(_HIST_ROWS).copy()
        flow_history.name = "flow_score"

        regime_history = enriched["flow_regime"].tail(_HIST_ROWS).copy()
        regime_history.name = "flow_regime"

        etf_flow_details = fetch_flow_data()

        lead_correlation: float = float("nan")
        try:
            if "hy_spread" in enriched.columns and enriched["flow_score"].notna().sum() >= 42:
                fwd_spread = enriched["hy_spread"].shift(-_FLOW_WINDOW)
                aligned = pd.concat(
                    {"flow": enriched["flow_score"], "fwd": fwd_spread},
                    axis=1,
                ).dropna()
                if len(aligned) >= 42:
                    lead_correlation = float(
                        np.corrcoef(aligned["flow"].values, aligned["fwd"].values)[0, 1]
                    )
        except Exception:
            pass

        return {
            "available":        current.get("available", flow_history.notna().any()),
            "current":          current,
            "df_enriched":      enriched,
            "flow_history":     flow_history,
            "etf_flow_details": etf_flow_details,
            "lead_correlation": lead_correlation,
            "regime_history":   regime_history,
        }

    except Exception:
        return {"available": False}
