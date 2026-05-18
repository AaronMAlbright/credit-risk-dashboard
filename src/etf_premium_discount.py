"""
Credit ETF premium/discount to NAV monitor.

During market stress, HYG/LQD/JNK trade at discounts to their intraday NAV —
signalling forced redemption pressure and liquidity dislocation in the underlying
bond market. The ETF arbitrage mechanism (authorised participant creation/redemption)
normally keeps price within ±10 bps of NAV. Under stress this breaks down.

March 2020 benchmark: HYG reached -5.5% discount. Energy 2016: -1.5% to -2%.
Monitoring discounts in real time is the fastest indicator of credit stress because
ETF prices reprice intraday while bond indices are stale.

Proxy methodology (true iNAV not freely available via yfinance)
---------------------------------------------------------------
Primary proxy  : price_vs_5d_avg = (close / rolling(5).mean()) - 1
                 5d average smooths the ETF itself; deviation = short-term dislocation.
Range proxy    : intraday_position = (close - low) / (high - low)
                 Close near low on high volume = selling pressure = discount signal.
Volume anomaly : volume > 2× 21d avg AND price_vs_5d_avg < -30bps = stress_day.

Composite dislocation score (0–100)
  avg_abs_bps = mean abs(price_vs_5d_avg × 10000) across HYG, LQD, JNK
  dislocation_score = min(100, avg_abs_bps / 150 × 100)

Regimes
  0–20  : Normal — ETF mechanism working
  20–40 : Elevated — minor dislocations, watch
  40–70 : Stress — redemption pressure building
  70–100: Dislocation — forced selling, liquidity crisis

Public API
----------
  ETF_TICKERS                : list[str]
  DISLOCATION_THRESHOLD_BPS  : int
  fetch_etf_data(period)     -> dict of {ticker: pd.DataFrame}
  compute_etf_premium_discount(df) -> pd.DataFrame
  get_current_etf_dislocation()    -> dict
  run_etf_premium_discount(df)     -> dict
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ETF_TICKERS: list[str] = ["HYG", "LQD", "JNK", "BKLN", "EMB"]

DISLOCATION_THRESHOLD_BPS: int = 50

_STRESS_THRESHOLD_BPS: int = 30       # price_vs_5d_avg < -30bps AND vol spike = stress_day
_VOL_SPIKE_RATIO: float = 2.0
_VOL_MA_WINDOW: int = 21
_PRICE_MA_WINDOW: int = 5
_COMPOSITE_TICKERS: list[str] = ["HYG", "LQD", "JNK"]   # used for composite score
_SCORE_SCALE_BPS: float = 150.0       # 150 bps → score 100
_HIST_ROWS: int = 504
_FETCH_PERIOD: str = "1y"

# Score regime thresholds
_REGIME_ELEVATED: float = 20.0
_REGIME_STRESS: float = 40.0
_REGIME_DISLOCATION: float = 70.0

# Cache: store fetch result for the current calendar day to avoid redundant calls
_cache: dict = {}


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


def _dislocation_regime(score: float) -> str:
    if np.isnan(score):
        return "Normal"
    if score >= _REGIME_DISLOCATION:
        return "Dislocation"
    if score >= _REGIME_STRESS:
        return "Stress"
    if score >= _REGIME_ELEVATED:
        return "Elevated"
    return "Normal"


def _compute_ticker_metrics(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute dislocation metrics for a single ETF's OHLCV DataFrame.

    Input columns expected (case-insensitive match attempted): Open, High, Low,
    Close, Volume.

    Returns DataFrame with columns:
        close, volume, price_vs_5d_avg, price_vs_5d_avg_bps,
        intraday_position, vol_ratio_21d, dislocation_flag, stress_day
    """
    out = price_df.copy()

    # Normalise column names
    out.columns = [c.lower() for c in out.columns]

    required = {"close", "volume"}
    if not required.issubset(out.columns):
        return pd.DataFrame()

    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    ma5 = close.rolling(_PRICE_MA_WINDOW, min_periods=2).mean()
    out["price_vs_5d_avg"] = (close / ma5) - 1.0
    out["price_vs_5d_avg_bps"] = out["price_vs_5d_avg"] * 10_000.0

    # Intraday position (0 = closed at low, 1 = closed at high)
    if "high" in out.columns and "low" in out.columns:
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        rng = (high - low).replace(0, np.nan)
        out["intraday_position"] = (close - low) / rng
    else:
        out["intraday_position"] = np.nan

    vol_ma21 = volume.rolling(_VOL_MA_WINDOW, min_periods=5).mean()
    out["vol_ratio_21d"] = volume / vol_ma21.replace(0, np.nan)

    out["dislocation_flag"] = (
        out["price_vs_5d_avg"].abs() > (DISLOCATION_THRESHOLD_BPS / 10_000.0)
    )

    out["stress_day"] = (
        (out["vol_ratio_21d"] >= _VOL_SPIKE_RATIO) &
        (out["price_vs_5d_avg"] < -(_STRESS_THRESHOLD_BPS / 10_000.0))
    )

    return out


# ---------------------------------------------------------------------------
# fetch_etf_data
# ---------------------------------------------------------------------------

def fetch_etf_data(period: str = _FETCH_PERIOD) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for all ETF_TICKERS via yfinance.

    Results are cached for the current calendar day to avoid re-downloading
    within a single session.

    Returns
    -------
    dict of {ticker: pd.DataFrame with DatetimeIndex}, empty on failure.
    Each DataFrame has columns: Open, High, Low, Close, Volume.
    Missing tickers are omitted rather than raising.
    """
    global _cache

    cache_key = f"etf_ohlcv_{period}_{date.today()}"
    if cache_key in _cache:
        return _cache[cache_key]

    result: dict[str, pd.DataFrame] = {}

    try:
        import yfinance as yf

        raw = yf.download(
            ETF_TICKERS,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if raw.empty:
            return result

        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in ETF_TICKERS:
                try:
                    ticker_df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                    if not ticker_df.empty:
                        result[ticker] = ticker_df
                except Exception:
                    continue
        else:
            # Single-ticker fallback (yfinance sometimes returns flat columns for one ticker)
            if len(ETF_TICKERS) == 1:
                result[ETF_TICKERS[0]] = raw.dropna(how="all")

        _cache[cache_key] = result
        return result

    except Exception:
        return result


# ---------------------------------------------------------------------------
# compute_etf_premium_discount
# ---------------------------------------------------------------------------

def compute_etf_premium_discount(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ETF dislocation metrics from a pre-fetched or pre-loaded DataFrame.

    If df already contains ETF price columns named like hyg_close / lqd_close /
    jnk_close (plus optional _high, _low, _volume variants), metrics are computed
    from those columns. Otherwise the function calls fetch_etf_data() for live data.

    Returns
    -------
    pd.DataFrame with new columns for each available ticker:
        {ticker}_price_vs_5d_avg_bps, {ticker}_dislocation_flag,
        {ticker}_stress_day, {ticker}_vol_ratio_21d, {ticker}_intraday_position
    Plus composite columns:
        etf_dislocation_score, etf_dislocation_regime
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)

    ticker_frames: dict[str, pd.DataFrame] = {}

    # Check if df already has etf columns (lower-case convention: hyg_close, etc.)
    for ticker in ETF_TICKERS:
        t = ticker.lower()
        close_col = f"{t}_close"
        if close_col in out.columns:
            sub = pd.DataFrame({"close": out[close_col]}, index=out.index)
            for suffix in ["high", "low", "volume"]:
                col = f"{t}_{suffix}"
                if col in out.columns:
                    sub[suffix] = out[col]
            ticker_frames[ticker] = sub

    # If none found via df columns, fetch live
    if not ticker_frames:
        fetched = fetch_etf_data()
        for ticker, tdf in fetched.items():
            ticker_frames[ticker] = tdf

    if not ticker_frames:
        out["etf_dislocation_score"] = np.nan
        out["etf_dislocation_regime"] = "Normal"
        return out

    # Compute metrics per ticker and join back to out
    metrics_frames: dict[str, pd.DataFrame] = {}
    for ticker, tdf in ticker_frames.items():
        m = _compute_ticker_metrics(tdf)
        if m.empty:
            continue
        metrics_frames[ticker] = m

    if not metrics_frames:
        out["etf_dislocation_score"] = np.nan
        out["etf_dislocation_regime"] = "Normal"
        return out

    for ticker, m in metrics_frames.items():
        t = ticker.lower()
        m_aligned = m.reindex(out.index)
        for col in ["price_vs_5d_avg_bps", "dislocation_flag", "stress_day", "vol_ratio_21d", "intraday_position"]:
            if col in m.columns:
                out[f"{t}_{col}"] = m_aligned[col] if col in m_aligned.columns else np.nan

    # Composite dislocation score from HYG, LQD, JNK (use whichever are available)
    abs_bps_cols = [
        f"{t.lower()}_price_vs_5d_avg_bps"
        for t in _COMPOSITE_TICKERS
        if f"{t.lower()}_price_vs_5d_avg_bps" in out.columns
    ]

    if abs_bps_cols:
        avg_abs_bps = out[abs_bps_cols].abs().mean(axis=1, skipna=True)
        out["etf_dislocation_score"] = (avg_abs_bps / _SCORE_SCALE_BPS * 100.0).clip(0.0, 100.0)
    else:
        out["etf_dislocation_score"] = np.nan

    out["etf_dislocation_regime"] = out["etf_dislocation_score"].apply(
        lambda v: _dislocation_regime(float(v)) if pd.notna(v) else "Normal"
    )

    return out


# ---------------------------------------------------------------------------
# get_current_etf_dislocation
# ---------------------------------------------------------------------------

def get_current_etf_dislocation() -> dict:
    """Fetch live ETF data and return the current dislocation state.

    Returns
    -------
    dict — see module docstring for full key list.
    """
    try:
        fetched = fetch_etf_data()
        if not fetched:
            return {"available": False}

        metrics_frames: dict[str, pd.DataFrame] = {}
        for ticker, tdf in fetched.items():
            m = _compute_ticker_metrics(tdf)
            if not m.empty and "price_vs_5d_avg_bps" in m.columns:
                metrics_frames[ticker] = m

        if not metrics_frames:
            return {"available": False}

        etf_readings: dict[str, dict] = {}
        for ticker, m in metrics_frames.items():
            last = m.iloc[-1]
            pdb = _safe_float(last.get("price_vs_5d_avg_bps"))
            vol_ratio = _safe_float(last.get("vol_ratio_21d"))
            disloc = bool(last.get("dislocation_flag", False))
            stress = bool(last.get("stress_day", False))
            etf_readings[ticker] = {
                "premium_discount_bps": pdb,
                "dislocation_flag": disloc,
                "stress_day": stress,
                "volume_ratio": vol_ratio,
            }

        # Composite score from core tickers
        abs_bps_values = [
            abs(etf_readings[t]["premium_discount_bps"])
            for t in _COMPOSITE_TICKERS
            if t in etf_readings and etf_readings[t]["premium_discount_bps"] is not None
        ]

        if abs_bps_values:
            avg_abs = float(np.mean(abs_bps_values))
            dislocation_score = float(min(100.0, avg_abs / _SCORE_SCALE_BPS * 100.0))
        else:
            avg_abs = float("nan")
            dislocation_score = float("nan")

        composite_pd_bps = float("nan")
        core_pds = [
            etf_readings[t]["premium_discount_bps"]
            for t in _COMPOSITE_TICKERS
            if t in etf_readings and etf_readings[t]["premium_discount_bps"] is not None
        ]
        if core_pds:
            composite_pd_bps = float(np.mean(core_pds))

        regime = _dislocation_regime(dislocation_score if not np.isnan(dislocation_score) else 0.0)

        # Most dislocated ticker (largest abs deviation)
        most_dislocated = ""
        if etf_readings:
            max_abs = -1.0
            for t, rd in etf_readings.items():
                v = rd["premium_discount_bps"]
                if v is not None and abs(v) > max_abs:
                    max_abs = abs(v)
                    most_dislocated = t

        # Warning
        warning: str | None = None
        if not np.isnan(dislocation_score) and dislocation_score > _REGIME_STRESS:
            pd_str = f"{composite_pd_bps:+.0f} bps" if not np.isnan(composite_pd_bps) else "unknown"
            warning = (
                f"ETF dislocation score {dislocation_score:.0f}/100 ({regime}). "
                f"Average premium/discount: {pd_str}. "
                f"Most dislocated: {most_dislocated}. "
                "This level indicates redemption pressure and underlying bond market illiquidity."
            )

        if not np.isnan(dislocation_score) and not np.isnan(composite_pd_bps):
            pd_sign = "premium" if composite_pd_bps > 0 else "discount"
            interp = (
                f"Credit ETF composite {pd_sign} of {composite_pd_bps:+.1f} bps "
                f"(score {dislocation_score:.0f}/100, regime: {regime}). "
            )
            if regime == "Normal":
                interp += "ETF arbitrage mechanism is functioning normally."
            elif regime == "Elevated":
                interp += "Minor dislocations present — monitor for widening."
            elif regime == "Stress":
                interp += "Redemption pressure building; underlying bond liquidity deteriorating."
            else:
                interp += "Forced selling / liquidity crisis indicators active. March-2020-style dynamics."
        else:
            interp = f"ETF dislocation regime: {regime}."

        return {
            "available": True,
            "dislocation_score": dislocation_score,
            "regime": regime,
            "etf_readings": etf_readings,
            "most_dislocated": most_dislocated,
            "composite_premium_discount_bps": composite_pd_bps,
            "interpretation": interp,
            "warning": warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# run_etf_premium_discount
# ---------------------------------------------------------------------------

def run_etf_premium_discount(df: pd.DataFrame) -> dict:
    """Top-level entry point for ETF premium/discount analysis.

    Attempts live fetch via yfinance; if that fails, falls back to ETF price
    columns in df (hyg_close, lqd_close, jnk_close etc.).

    Returns
    -------
    dict with keys:
        available           : bool
        current             : dict from get_current_etf_dislocation()
        df_enriched         : pd.DataFrame — df with dislocation columns appended
        etf_details         : dict of {ticker: pd.Series of premium_discount_bps}
        dislocation_history : pd.Series (DatetimeIndex, 0–100)
        regime_history      : pd.Series (DatetimeIndex, regime strings)
        interpretation      : str
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        # Try live current state first (fail gracefully)
        current = get_current_etf_dislocation()

        # Enrich the master df — uses live fetch OR df etf columns
        df_enriched = compute_etf_premium_discount(df)

        etf_score_valid = (
            "etf_dislocation_score" in df_enriched.columns
            and df_enriched["etf_dislocation_score"].notna().any()
        )

        if not current.get("available") and not etf_score_valid:
            return {"available": False}

        # Historical score series
        if etf_score_valid:
            dislocation_history = df_enriched["etf_dislocation_score"].tail(_HIST_ROWS).copy()
            dislocation_history.name = "etf_dislocation_score"
        else:
            dislocation_history = pd.Series(dtype=float, name="etf_dislocation_score")

        # Historical regime series
        if "etf_dislocation_regime" in df_enriched.columns:
            regime_history = df_enriched["etf_dislocation_regime"].tail(_HIST_ROWS).copy()
            regime_history.name = "etf_dislocation_regime"
        else:
            regime_history = pd.Series(dtype=str, name="etf_dislocation_regime")

        # Per-ticker premium/discount history
        etf_details: dict[str, pd.Series] = {}
        for ticker in ETF_TICKERS:
            col = f"{ticker.lower()}_price_vs_5d_avg_bps"
            if col in df_enriched.columns:
                s = df_enriched[col].tail(_HIST_ROWS).copy()
                s.name = ticker
                etf_details[ticker] = s

        # Also populate etf_details from live fetch if not yet in df_enriched
        if not etf_details and current.get("available"):
            fetched = fetch_etf_data()
            for ticker, tdf in fetched.items():
                m = _compute_ticker_metrics(tdf)
                if not m.empty and "price_vs_5d_avg_bps" in m.columns:
                    s = m["price_vs_5d_avg_bps"].tail(_HIST_ROWS).copy()
                    s.name = ticker
                    etf_details[ticker] = s

        # Synthesise available flag
        available = current.get("available", False) or etf_score_valid

        # Interpretation
        if current.get("available"):
            interpretation = current.get("interpretation", "")
        elif etf_score_valid:
            last_score = _safe_float(df_enriched["etf_dislocation_score"].dropna().iloc[-1])
            regime = _dislocation_regime(last_score) if last_score is not None else "Normal"
            interpretation = (
                f"ETF dislocation score (from df columns): {last_score:.0f}/100, regime: {regime}."
                if last_score is not None else
                "ETF dislocation analysis based on df columns — live fetch unavailable."
            )
        else:
            interpretation = "ETF premium/discount data unavailable."

        # Use current from live or build a minimal one from df_enriched
        if not current.get("available") and etf_score_valid:
            last_row = df_enriched.iloc[-1]
            last_score = _safe_float(last_row.get("etf_dislocation_score"))
            regime = _dislocation_regime(last_score if last_score is not None else 0.0)
            current = {
                "available": True,
                "dislocation_score": last_score,
                "regime": regime,
                "etf_readings": {},
                "most_dislocated": "",
                "composite_premium_discount_bps": float("nan"),
                "interpretation": interpretation,
                "warning": (
                    f"Dislocation score {last_score:.0f}/100 ({regime}) — exceeds stress threshold."
                    if last_score is not None and last_score > _REGIME_STRESS else None
                ),
            }

        return {
            "available": available,
            "current": current,
            "df_enriched": df_enriched,
            "etf_details": etf_details,
            "dislocation_history": dislocation_history,
            "regime_history": regime_history,
            "interpretation": interpretation,
        }

    except Exception:
        return {"available": False}
