"""
EM Credit Stress Monitor.

EM sovereign spread (EMBI+) and EM vs DM HY differential signal global risk
appetite. Dollar strength (DXY) is the key transmission mechanism: a rising
DXY tightens financial conditions for EM dollar-debt borrowers, widening EM
spreads. EM credit stress often LEADS DM HY by 4-6 weeks.

Implementation uses the EMB/HYG price ratio as a proxy for EM vs US HY
relative performance when a direct EMBI OAS series is not available in df.

Public API
----------
  get_em_snapshot()          -> dict
  compute_em_credit(df)      -> pd.DataFrame
  run_em_credit_analysis(df) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMB_TICKER = "EMB"
_HYG_TICKER = "HYG"
_DXY_TICKER = "DX-Y.NYB"

# Column name candidates already present in the scored df
_EM_SPREAD_CANDIDATES = ("em_spread", "embi_spread", "bamlemhbhycrpioas")
_DXY_CANDIDATES = ("dxy", "dx_y_nyb", "dollar_index")

_SNAPSHOT_PERIOD = "90d"
_HISTORY_PERIOD = "5y"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column present in df, or None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _fetch_price_series(ticker: str, period: str) -> pd.Series | None:
    """
    Fetch daily Close prices for a single yfinance ticker.

    Returns a pd.Series indexed by normalised date (midnight UTC), or None
    on any failure.
    """
    if not _YF_AVAILABLE:
        return None
    try:
        raw = yf.Ticker(ticker).history(period=period)
        if raw is None or raw.empty:
            return None
        closes = raw["Close"].dropna()
        if closes.empty:
            return None
        closes.index = pd.to_datetime(closes.index).normalize()
        closes.name = ticker
        return closes
    except Exception:
        return None


def _classify_em_stress_regime(
    ratio_30d_chg: float,
    dxy_30d_chg: float | None,
) -> str:
    """
    Derive a qualitative EM stress regime from the EMB/HYG 30d change and DXY.

    Parameters
    ----------
    ratio_30d_chg : float
        30-day percent change in the EMB/HYG price ratio.
    dxy_30d_chg : float or None
        30-day percent change in DXY (positive = dollar strengthening).

    Returns
    -------
    str — "Risk-On" / "Neutral" / "Risk-Off" / "Crisis"
    """
    if pd.isna(ratio_30d_chg):
        return "Unknown"

    dxy_stress = (
        not pd.isna(dxy_30d_chg) and dxy_30d_chg > 3.0
    )

    if ratio_30d_chg > 2.0 and not dxy_stress:
        return "Risk-On"
    if ratio_30d_chg < -7.0 or (ratio_30d_chg < -5.0 and dxy_stress):
        return "Crisis"
    if ratio_30d_chg < -2.0 or dxy_stress:
        return "Risk-Off"
    return "Neutral"


def _ratio_to_stress_score(ratio_30d_chg: float) -> float:
    """
    Map 30-day EMB/HYG ratio change to a 0-100 stress score.

    Higher score = more EM stress (worse for EM credit).
        ratio rising (> +2%)      →  30 (EM outperforming, risk-on)
        -2% to +2%                →  50
        -5% to -2%                →  65
        < -5%                     →  85
    """
    if pd.isna(ratio_30d_chg):
        return float("nan")
    if ratio_30d_chg > 2.0:
        return 30.0
    if ratio_30d_chg >= -2.0:
        return 50.0
    if ratio_30d_chg >= -5.0:
        return 65.0
    return 85.0


def _rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std over `window` periods."""
    roll = series.rolling(window, min_periods=max(window // 4, 20))
    return (series - roll.mean()) / roll.std()


# ---------------------------------------------------------------------------
# Public API — snapshot
# ---------------------------------------------------------------------------

def get_em_snapshot() -> dict:
    """
    Fetch live 90-day data for EMB, HYG, and DXY via yfinance.

    Returns
    -------
    dict with keys:
        emb_price           : float — latest EMB price
        hyg_price           : float — latest HYG price
        em_vs_dm_ratio      : float — EMB/HYG (higher = EM outperforming)
        em_vs_dm_30d_chg    : float — 30-day percent change in ratio
        dxy                 : float — latest DXY level
        dxy_30d_chg         : float — 30-day percent change in DXY
        em_stress_regime    : str   — "Risk-On"/"Neutral"/"Risk-Off"/"Crisis"
        available           : bool
        error               : str (empty string on success)
    """
    _nan = float("nan")
    empty: dict = {
        "emb_price": _nan,
        "hyg_price": _nan,
        "em_vs_dm_ratio": _nan,
        "em_vs_dm_30d_chg": _nan,
        "dxy": _nan,
        "dxy_30d_chg": _nan,
        "em_stress_regime": "Unknown",
        "available": False,
        "error": "",
    }

    if not _YF_AVAILABLE:
        empty["error"] = "yfinance not installed"
        return empty

    errors: list[str] = []

    # --- EMB -----------------------------------------------------------------
    emb_series = _fetch_price_series(_EMB_TICKER, _SNAPSHOT_PERIOD)
    if emb_series is None or emb_series.empty:
        errors.append("EMB fetch failed")
        emb_latest = _nan
    else:
        emb_latest = float(emb_series.iloc[-1])

    # --- HYG -----------------------------------------------------------------
    hyg_series = _fetch_price_series(_HYG_TICKER, _SNAPSHOT_PERIOD)
    if hyg_series is None or hyg_series.empty:
        errors.append("HYG fetch failed")
        hyg_latest = _nan
        ratio_latest = _nan
        ratio_30d_chg = _nan
    else:
        hyg_latest = float(hyg_series.iloc[-1])
        if not pd.isna(emb_latest) and hyg_latest != 0:
            # Compute ratio from the aligned history
            if emb_series is not None and not emb_series.empty:
                combined = pd.DataFrame({"emb": emb_series, "hyg": hyg_series}).dropna()
                if not combined.empty and combined["hyg"].ne(0).all():
                    ratio_series = combined["emb"] / combined["hyg"]
                    ratio_latest = float(ratio_series.iloc[-1])
                    # 30d pct change (approx 21 trading days; use calendar 30d index)
                    if len(ratio_series) >= 21:
                        ratio_30d_chg = float(
                            (ratio_series.iloc[-1] / ratio_series.iloc[-21] - 1) * 100
                        )
                    else:
                        ratio_30d_chg = _nan
                else:
                    ratio_latest = _nan
                    ratio_30d_chg = _nan
            else:
                ratio_latest = _nan
                ratio_30d_chg = _nan
        else:
            ratio_latest = _nan
            ratio_30d_chg = _nan

    # --- DXY -----------------------------------------------------------------
    dxy_series = _fetch_price_series(_DXY_TICKER, _SNAPSHOT_PERIOD)
    if dxy_series is None or dxy_series.empty:
        errors.append("DXY fetch failed")
        dxy_latest = _nan
        dxy_30d_chg = _nan
    else:
        dxy_latest = float(dxy_series.iloc[-1])
        if len(dxy_series) >= 21:
            dxy_30d_chg = float(
                (dxy_series.iloc[-1] / dxy_series.iloc[-21] - 1) * 100
            )
        else:
            dxy_30d_chg = _nan

    # --- Regime --------------------------------------------------------------
    regime = _classify_em_stress_regime(ratio_30d_chg, dxy_30d_chg)

    available = bool(not (pd.isna(emb_latest) and pd.isna(hyg_latest)))

    return {
        "emb_price": emb_latest,
        "hyg_price": hyg_latest,
        "em_vs_dm_ratio": ratio_latest,
        "em_vs_dm_30d_chg": ratio_30d_chg,
        "dxy": dxy_latest,
        "dxy_30d_chg": dxy_30d_chg,
        "em_stress_regime": regime,
        "available": available,
        "error": "; ".join(errors),
    }


# ---------------------------------------------------------------------------
# Public API — DataFrame enrichment
# ---------------------------------------------------------------------------

def compute_em_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EM credit stress columns to a copy of df by aligning yfinance EMB/HYG
    history to df's DatetimeIndex.

    New columns
    -----------
    em_hyg_ratio          : EMB/HYG price ratio (left-joined to df index)
    em_hyg_ratio_30d_chg  : 30-day percent change in ratio (pct_change(21))
    em_vs_dm_signal       : 0-100 stress score derived from ratio_30d_chg
    em_dxy_zscore         : rolling 252d z-score of DXY (from df['dxy'] if
                            present; NaN column otherwise)

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex.

    Returns
    -------
    pd.DataFrame — copy of df with columns added (NaN where data unavailable).
    """
    out = df.copy()

    _nan = float("nan")

    # Ensure DatetimeIndex
    out.index = pd.to_datetime(out.index)

    # ---- EMB / HYG ratio from yfinance --------------------------------------
    emb_series = _fetch_price_series(_EMB_TICKER, _HISTORY_PERIOD)
    hyg_series = _fetch_price_series(_HYG_TICKER, _HISTORY_PERIOD)

    if (
        emb_series is not None and not emb_series.empty
        and hyg_series is not None and not hyg_series.empty
    ):
        try:
            combined = pd.DataFrame(
                {"emb": emb_series, "hyg": hyg_series}
            ).dropna()
            combined = combined[combined["hyg"] != 0]
            combined["em_hyg_ratio"] = combined["emb"] / combined["hyg"]

            # Left-join onto df index (date only, normalise)
            combined.index = pd.to_datetime(combined.index).normalize()
            out.index = pd.to_datetime(out.index).normalize()
            out = out.join(combined[["em_hyg_ratio"]], how="left")

            out["em_hyg_ratio_30d_chg"] = out["em_hyg_ratio"].pct_change(21) * 100
            out["em_vs_dm_signal"] = out["em_hyg_ratio_30d_chg"].apply(
                _ratio_to_stress_score
            )
        except Exception:
            out["em_hyg_ratio"] = _nan
            out["em_hyg_ratio_30d_chg"] = _nan
            out["em_vs_dm_signal"] = _nan
    else:
        out["em_hyg_ratio"] = _nan
        out["em_hyg_ratio_30d_chg"] = _nan
        out["em_vs_dm_signal"] = _nan

    # ---- DXY z-score from df if available -----------------------------------
    dxy_col = _resolve_column(out, _DXY_CANDIDATES)
    if dxy_col is not None:
        out["em_dxy_zscore"] = _rolling_zscore(out[dxy_col], window=252)
    else:
        out["em_dxy_zscore"] = _nan

    return out


# ---------------------------------------------------------------------------
# Public API — full analysis
# ---------------------------------------------------------------------------

def run_em_credit_analysis(df: pd.DataFrame) -> dict:
    """
    Full EM credit stress analysis combining live snapshot and df enrichment.

    Parameters
    ----------
    df : pd.DataFrame
        Main market data frame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        snapshot         : dict from get_em_snapshot()
        df               : enriched DataFrame
        current          : dict of latest values
        interpretation   : one-sentence plain-English string
        lead_signal      : str — note about EM leading DM if stress elevated
        available        : bool
    """
    snapshot = get_em_snapshot()
    enriched = compute_em_credit(df)

    # ---- Current row snapshot -----------------------------------------------
    def _last_valid(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return float("nan")
        return float(s.iloc[-1])

    em_hyg_ratio = _last_valid(enriched["em_hyg_ratio"])
    em_hyg_ratio_30d_chg = _last_valid(enriched["em_hyg_ratio_30d_chg"])
    em_vs_dm_signal = _last_valid(enriched["em_vs_dm_signal"])
    em_dxy_zscore = _last_valid(enriched["em_dxy_zscore"])

    # DXY: prefer df value, fall back to snapshot
    dxy_col = _resolve_column(enriched, _DXY_CANDIDATES)
    if dxy_col is not None:
        dxy_val = _last_valid(enriched[dxy_col])
    else:
        dxy_val = snapshot.get("dxy", float("nan"))

    current: dict = {
        "em_hyg_ratio": em_hyg_ratio,
        "em_hyg_ratio_30d_chg": em_hyg_ratio_30d_chg,
        "em_vs_dm_signal": em_vs_dm_signal,
        "dxy": dxy_val,
        "em_dxy_zscore": em_dxy_zscore,
    }

    # ---- Lead signal --------------------------------------------------------
    if not pd.isna(em_vs_dm_signal) and em_vs_dm_signal > 65:
        lead_signal = (
            "EM leading DM stress by ~4-6 weeks: elevated EM underperformance "
            "historically precedes DM HY spread widening."
        )
    else:
        lead_signal = ""

    # ---- Interpretation -----------------------------------------------------
    regime = snapshot.get("em_stress_regime", "Unknown")
    ratio_chg = snapshot.get("em_vs_dm_30d_chg", float("nan"))
    dxy_chg = snapshot.get("dxy_30d_chg", float("nan"))

    if not pd.isna(ratio_chg) and not pd.isna(dxy_chg):
        direction_em = "outperforming" if ratio_chg > 0 else "underperforming"
        interpretation = (
            f"EM credit is {direction_em} DM HY by {abs(ratio_chg):.1f}% over the "
            f"past 30 days (DXY {'+' if dxy_chg >= 0 else ''}{dxy_chg:.1f}%), "
            f"placing the EM stress regime in '{regime}'."
        )
    elif not pd.isna(ratio_chg):
        direction_em = "outperforming" if ratio_chg > 0 else "underperforming"
        interpretation = (
            f"EM credit is {direction_em} DM HY by {abs(ratio_chg):.1f}% over 30 "
            f"days; EM stress regime: '{regime}'."
        )
    else:
        interpretation = (
            "EM credit data is unavailable; cannot assess relative EM vs DM HY "
            "performance or dollar transmission stress."
        )

    available = bool(snapshot.get("available", False) or (
        enriched["em_hyg_ratio"].notna().any()
    ))

    return {
        "snapshot": snapshot,
        "df": enriched,
        "current": current,
        "interpretation": interpretation,
        "lead_signal": lead_signal,
        "available": available,
    }
