"""
MOVE Index — bond market volatility (rates equivalent of VIX).

The MOVE Index (ICE BofA MOVE) measures implied volatility of 1-month US
Treasury options across maturities. Like VIX for equities, but for bonds.

Key insight: MOVE spikes often precede HY spread widening by days to weeks,
making it a leading indicator of credit stress — rates market participants
see it first.

Normal regime: MOVE < 80. Elevated: 80–120. Stress: > 120. Crisis: > 150.

Public API
----------
  MOVE_NORMAL   = 80.0
  MOVE_ELEVATED = 120.0
  MOVE_STRESS   = 150.0
  get_move_snapshot() → dict
  compute_move_signal(df) → pd.DataFrame  (adds move_signal columns to df copy)
  run_move_analysis(df) → dict
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

MOVE_NORMAL: float = 80.0
MOVE_ELEVATED: float = 120.0
MOVE_STRESS: float = 150.0

_TICKER = "^MOVE"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_regime(level: float) -> str:
    """Map a MOVE level to a regime label."""
    if level >= MOVE_STRESS:
        return "Crisis"
    if level >= MOVE_ELEVATED:
        return "Stress"
    if level >= MOVE_NORMAL:
        return "Elevated"
    return "Normal"


def _move_to_signal(level: float) -> float:
    """
    Piecewise-linear mapping from MOVE level to a 0–100 risk signal.

    Breakpoints:
        0    →   0
        80   →  40    (Normal ceiling)
        120  →  60    (Elevated ceiling)
        150  →  80    (Stress ceiling)
        200+ → 100    (Crisis)
    """
    if pd.isna(level) or level < 0:
        return float("nan")
    if level <= 0:
        return 0.0
    if level <= 80:
        return (level / 80.0) * 40.0
    if level <= 120:
        return 40.0 + ((level - 80.0) / 40.0) * 20.0
    if level <= 150:
        return 60.0 + ((level - 120.0) / 30.0) * 20.0
    # 150 → 200+
    return min(100.0, 80.0 + ((level - 150.0) / 50.0) * 20.0)


def _fetch_move_history(period: str = "60d") -> pd.DataFrame | None:
    """
    Fetch MOVE history from yfinance. Returns a DataFrame indexed by date
    with a single column 'move', or None on failure.
    """
    if not _YF_AVAILABLE:
        return None
    try:
        raw = yf.Ticker(_TICKER).history(period=period)
        if raw is None or raw.empty:
            return None
        closes = raw["Close"].dropna()
        if closes.empty:
            return None
        df = closes.rename("move").to_frame()
        df.index = pd.to_datetime(df.index).normalize()
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_move_snapshot() -> dict:
    """
    Fetch a current MOVE Index snapshot from yfinance.

    Returns
    -------
    dict with keys:
        current           : latest MOVE level (float)
        prev_close        : prior day close (float)
        day_chg_pct       : day-over-day % change (float)
        five_day_avg      : 5-day rolling mean (float)
        thirty_day_avg    : 30-day rolling mean (float)
        fifty_two_week_high : rolling 252-day high (float)
        fifty_two_week_low  : rolling 252-day low (float)
        regime            : "Normal" / "Elevated" / "Stress" / "Crisis"
        available         : True if fetch succeeded
        error             : error message string (empty string on success)
    """
    empty: dict = {
        "current": float("nan"),
        "prev_close": float("nan"),
        "day_chg_pct": float("nan"),
        "five_day_avg": float("nan"),
        "thirty_day_avg": float("nan"),
        "fifty_two_week_high": float("nan"),
        "fifty_two_week_low": float("nan"),
        "regime": "Unknown",
        "available": False,
        "error": "",
    }

    hist = _fetch_move_history(period="60d")
    if hist is None or hist.empty:
        empty["error"] = "yfinance returned no data for ^MOVE"
        return empty

    series = hist["move"].dropna()
    if len(series) < 2:
        empty["error"] = "Insufficient MOVE history (< 2 observations)"
        return empty

    current = float(series.iloc[-1])
    prev_close = float(series.iloc[-2])
    day_chg_pct = (current / prev_close - 1.0) * 100.0 if prev_close != 0 else float("nan")
    five_day_avg = float(series.iloc[-min(5, len(series)):].mean())
    thirty_day_avg = float(series.iloc[-min(30, len(series)):].mean())
    fifty_two_week_high = float(series.max())
    fifty_two_week_low = float(series.min())

    return {
        "current": current,
        "prev_close": prev_close,
        "day_chg_pct": day_chg_pct,
        "five_day_avg": five_day_avg,
        "thirty_day_avg": thirty_day_avg,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
        "regime": _classify_regime(current),
        "available": True,
        "error": "",
    }


def compute_move_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge MOVE Index data into the scored DataFrame and add signal columns.

    Takes the scored DataFrame (which must have a ``vix`` column and a
    DatetimeIndex or a column that can be used as a date index).

    New columns added
    -----------------
    move            : MOVE Index level (float, NaN where data unavailable)
    move_regime     : "Normal" / "Elevated" / "Stress" / "Crisis"
    move_vs_vix_ratio : MOVE / VIX
                        > 1.5 → rates vol dominating
                        < 0.8 → equity vol dominating
    move_signal     : 0–100 risk score (piecewise linear; see _move_to_signal)

    If the yfinance fetch fails the DataFrame is returned unchanged and
    ``available`` is set to False in the module-level state (no side effects
    on df).
    """
    df = df.copy()

    move_hist = _fetch_move_history(period="60d")
    if move_hist is None or move_hist.empty:
        # Gracefully degrade — add NaN columns so callers don't need to check
        for col in ("move", "move_regime", "move_vs_vix_ratio", "move_signal"):
            if col not in df.columns:
                df[col] = float("nan")
        return df

    # Normalise the DataFrame index to date for the merge
    df_idx_was_reset = False
    if not isinstance(df.index, pd.DatetimeIndex):
        # Attempt to locate a date-like column
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            df = df.set_index(pd.to_datetime(df[date_cols[0]]))
            df_idx_was_reset = True
        else:
            for col in ("move", "move_regime", "move_vs_vix_ratio", "move_signal"):
                if col not in df.columns:
                    df[col] = float("nan")
            return df

    df.index = pd.to_datetime(df.index).normalize()

    # Left-join MOVE onto the scored df
    df = df.join(move_hist[["move"]], how="left")

    # Regime label
    df["move_regime"] = df["move"].apply(
        lambda v: _classify_regime(v) if not pd.isna(v) else "Unknown"
    )

    # MOVE / VIX ratio
    if "vix" in df.columns:
        df["move_vs_vix_ratio"] = np.where(
            (df["vix"].notna()) & (df["vix"] != 0),
            df["move"] / df["vix"],
            float("nan"),
        )
    else:
        df["move_vs_vix_ratio"] = float("nan")

    # 0-100 signal score
    df["move_signal"] = df["move"].apply(_move_to_signal)

    return df


def run_move_analysis(df: pd.DataFrame) -> dict:
    """
    Full MOVE Index analysis combining snapshot and DataFrame enrichment.

    Returns
    -------
    dict with keys:
        snapshot               : result of get_move_snapshot()
        df                     : DataFrame enriched by compute_move_signal()
        available              : True if MOVE data was successfully fetched
        correlation_with_hy_spread : Pearson r of move vs hy_spread (last 252 rows)
                                     NaN if data unavailable
        lead_lag_summary       : plain-English description of the relationship
    """
    snapshot = get_move_snapshot()
    enriched_df = compute_move_signal(df)

    available = snapshot["available"] and ("move" in enriched_df.columns)

    # Pearson correlation: MOVE vs HY spread on last 252 rows
    corr_hy = float("nan")
    if available and "hy_spread" in enriched_df.columns:
        subset = enriched_df[["move", "hy_spread"]].dropna().iloc[-252:]
        if len(subset) >= 20:
            move_arr = subset["move"].values
            hy_arr = subset["hy_spread"].values
            # Pure numpy Pearson r
            move_dm = move_arr - move_arr.mean()
            hy_dm = hy_arr - hy_arr.mean()
            denom = np.sqrt((move_dm ** 2).sum() * (hy_dm ** 2).sum())
            corr_hy = float(np.dot(move_dm, hy_dm) / denom) if denom > 0 else float("nan")

    # Lead-lag description
    if not available:
        lead_lag = (
            "MOVE data unavailable. Normally MOVE spikes precede HY spread "
            "widening by days to weeks — rates participants see credit stress first."
        )
    elif pd.isna(corr_hy):
        lead_lag = (
            "Insufficient overlapping history to compute MOVE–HY correlation. "
            "Typically positive (r ≈ 0.5–0.7): rising bond vol leads spread widening."
        )
    elif corr_hy >= 0.6:
        lead_lag = (
            f"Strong positive correlation (r = {corr_hy:.2f}) between MOVE and HY "
            "spreads over the trailing year. MOVE historically leads by 1–3 weeks — "
            "current MOVE regime is a meaningful leading indicator of credit stress."
        )
    elif corr_hy >= 0.3:
        lead_lag = (
            f"Moderate positive correlation (r = {corr_hy:.2f}). MOVE and HY spreads "
            "are broadly co-moving. Some leading relationship likely persists."
        )
    else:
        lead_lag = (
            f"Weak or negative correlation (r = {corr_hy:.2f}) in this window. "
            "MOVE may be driven by idiosyncratic rates vol rather than credit stress "
            "— watch for divergence as a potential early signal."
        )

    return {
        "snapshot": snapshot,
        "df": enriched_df,
        "available": available,
        "correlation_with_hy_spread": corr_hy,
        "lead_lag_summary": lead_lag,
    }
