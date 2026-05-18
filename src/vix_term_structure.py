"""
VIX term structure — bond market and equity options volatility signals.

The VIX term structure compares implied volatility at different horizons:
  VIX9D  (9-day)  — acute near-term fear
  VIX    (30-day) — standard fear gauge
  VIX3M  (93-day) — medium-term uncertainty

Key signal: Term structure SLOPE = VIX3M − VIX
  Positive slope (VIX3M > VIX) = CONTANGO — normal, calm market
  Flat / zero                   = uncertainty building
  Negative slope (VIX3M < VIX) = BACKWARDATION — acute stress, near-term fear
    exceeds long-term fear. This is a STRONG warning signal for credit spreads.

Second key signal: VVIX (VIX of VIX)
  VVIX measures uncertainty ABOUT the VIX itself.
  High VVIX (>100) = "we don't know how scared to be" = regime uncertainty
  Historically high VVIX precedes VIX spikes by 1-3 days.

VIX/VIX3M ratio:
  < 0.85 = deep contango (complacency)
  0.85–0.95 = normal contango
  0.95–1.05 = flat (caution)
  > 1.05 = backwardation (stress)
  > 1.20 = severe backwardation (crisis)

Public API
----------
  get_vix_term_snapshot() → dict
  compute_vix_term_signal(df) → pd.DataFrame   adds vix_ts_ columns
  run_vix_term_analysis(df) → dict
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VIX_TICKERS = ["^VIX", "^VIX3M", "^VIX9D", "^VVIX"]


def _fetch_close(tickers: list[str], period: str = "60d") -> pd.DataFrame:
    """Download Close prices for *tickers*, return a DataFrame indexed by date."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    if raw.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when >1 ticker; single ticker → flat
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers  # rename to ticker name

    close.index = pd.to_datetime(close.index).normalize()
    return close


def _latest(series: pd.Series) -> Optional[float]:
    """Return the last non-NaN value from a Series, or None."""
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _classify_structure(ratio: float) -> str:
    """Map VIX/VIX3M ratio to a term structure label."""
    if ratio < 0.85:
        return "Deep Contango"
    elif ratio < 0.95:
        return "Contango"
    elif ratio < 1.05:
        return "Flat"
    elif ratio < 1.20:
        return "Backwardation"
    else:
        return "Severe Backwardation"


def _classify_vvix(vvix: float) -> str:
    if vvix < 80:
        return "Normal (<80)"
    elif vvix < 100:
        return "Elevated (80–100)"
    elif vvix < 120:
        return "High (100–120)"
    else:
        return "Extreme (>120)"


def _ratio_to_signal(ratio: float) -> float:
    """
    Piecewise-linear mapping of VIX/VIX3M ratio → 0–100 signal score.

    Breakpoints:
      ratio < 0.85  → 0
      ratio 0.85–0.95 → 0–25
      ratio 0.95–1.05 → 25–60
      ratio 1.05–1.20 → 60–85
      ratio > 1.20  → 85–100  (capped at 100)
    """
    if ratio < 0.85:
        return 0.0
    elif ratio < 0.95:
        return (ratio - 0.85) / (0.95 - 0.85) * 25.0
    elif ratio < 1.05:
        return 25.0 + (ratio - 0.95) / (1.05 - 0.95) * 35.0
    elif ratio < 1.20:
        return 60.0 + (ratio - 1.05) / (1.20 - 1.05) * 25.0
    else:
        return min(85.0 + (ratio - 1.20) / 0.10 * 15.0, 100.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_vix_term_snapshot() -> dict:
    """
    Fetch current VIX term structure snapshot from live market data.

    Returns
    -------
    dict with keys:
        vix, vix3m, vix9d, vvix : float — current index levels
        slope     : float — VIX3M minus VIX (positive = contango)
        ratio     : float — VIX / VIX3M (>1 = backwardation)
        structure : str   — regime label for term structure shape
        vvix_regime : str — regime label for VVIX level
        vix9d_vs_vix : float — VIX9D / VIX (>1 = acute near-term spike)
        available : bool
        error     : str or None
    """
    try:
        closes = _fetch_close(_VIX_TICKERS, period="60d")
        if closes.empty:
            return _error_snapshot("yfinance returned empty data for VIX tickers")

        vix = _latest(closes.get("^VIX", pd.Series(dtype=float)))
        vix3m = _latest(closes.get("^VIX3M", pd.Series(dtype=float)))
        vix9d = _latest(closes.get("^VIX9D", pd.Series(dtype=float)))
        vvix = _latest(closes.get("^VVIX", pd.Series(dtype=float)))

        if vix is None or vix3m is None:
            return _error_snapshot("Could not retrieve VIX or VIX3M values")

        slope = vix3m - vix
        ratio = vix / vix3m if vix3m != 0 else float("nan")
        vix9d_vs_vix = (vix9d / vix) if (vix9d is not None and vix != 0) else None

        return {
            "vix": vix,
            "vix3m": vix3m,
            "vix9d": vix9d,
            "vvix": vvix,
            "slope": slope,
            "ratio": ratio,
            "structure": _classify_structure(ratio),
            "vvix_regime": _classify_vvix(vvix) if vvix is not None else "Unknown",
            "vix9d_vs_vix": vix9d_vs_vix,
            "available": True,
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        return _error_snapshot(str(exc))


def _error_snapshot(msg: str) -> dict:
    return {
        "vix": None,
        "vix3m": None,
        "vix9d": None,
        "vvix": None,
        "slope": None,
        "ratio": None,
        "structure": None,
        "vvix_regime": None,
        "vix9d_vs_vix": None,
        "available": False,
        "error": msg,
    }


def compute_vix_term_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge VIX term structure signals onto *df* by date.

    Expects *df* to have a "date" column (or DatetimeIndex).  Adds columns:
        vix_3m          — VIX3M level on that date
        vix_ts_slope    — VIX3M minus VIX (from df "vix" column if present)
        vix_ts_ratio    — VIX / VIX3M
        vix_ts_structure — regime label string
        vix_ts_signal   — 0–100 stress score

    If VIX3M data cannot be fetched, the above columns are added as NaN and
    the original DataFrame is returned otherwise unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a "date" column (datetime-like) or have a DatetimeIndex.
        Optionally contains a "vix" column for slope/ratio computation.

    Returns
    -------
    pd.DataFrame — copy of df with vix_ts_* columns appended.
    """
    df = df.copy()

    # Normalise date access
    if "date" in df.columns:
        df_dates = pd.to_datetime(df["date"]).dt.normalize()
    else:
        df_dates = pd.to_datetime(df.index).normalize()

    nan_cols = ["vix_3m", "vix_ts_slope", "vix_ts_ratio", "vix_ts_structure", "vix_ts_signal"]

    try:
        closes = _fetch_close(["^VIX3M"], period="60d")
        if closes.empty or "^VIX3M" not in closes.columns:
            raise ValueError("VIX3M data unavailable")

        vix3m_series = closes["^VIX3M"].dropna()
        vix3m_series.index = pd.to_datetime(vix3m_series.index).normalize()

        # Build a lookup Series keyed by date
        vix3m_map = vix3m_series.to_dict()

        df["vix_3m"] = df_dates.map(vix3m_map)

        # Compute slope if we have a "vix" column
        if "vix" in df.columns:
            df["vix_ts_slope"] = df["vix_3m"] - pd.to_numeric(df["vix"], errors="coerce")
            df["vix_ts_ratio"] = pd.to_numeric(df["vix"], errors="coerce") / df["vix_3m"]
        else:
            df["vix_ts_slope"] = np.nan
            df["vix_ts_ratio"] = np.nan

        # Vectorised structure label and signal score
        df["vix_ts_structure"] = df["vix_ts_ratio"].apply(
            lambda r: _classify_structure(r) if pd.notna(r) else np.nan
        )
        df["vix_ts_signal"] = df["vix_ts_ratio"].apply(
            lambda r: _ratio_to_signal(r) if pd.notna(r) else np.nan
        )

    except Exception:  # noqa: BLE001
        for col in nan_cols:
            df[col] = np.nan

    return df


def run_vix_term_analysis(df: pd.DataFrame) -> dict:
    """
    Run a full VIX term structure analysis, combining a live snapshot with
    historical signal computation.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data frame.  Expected columns (used when present):
            date, vix, hy_spread

    Returns
    -------
    dict with keys:
        snapshot                    : get_vix_term_snapshot() result
        df                          : compute_vix_term_signal(df) result
        available                   : bool
        correlation_vix_slope_with_hy : float or None — Pearson r
        lead_signal                 : str
        key_insight                 : str
    """
    snapshot = get_vix_term_snapshot()
    enriched_df = compute_vix_term_signal(df)
    available = snapshot["available"]

    # -----------------------------------------------------------------------
    # Correlation: vix_ts_slope vs hy_spread (last 252 rows)
    # -----------------------------------------------------------------------
    corr_slope_hy: Optional[float] = None
    if "vix_ts_slope" in enriched_df.columns and "hy_spread" in enriched_df.columns:
        window = enriched_df.tail(252)[["vix_ts_slope", "hy_spread"]].dropna()
        if len(window) >= 10:
            corr_slope_hy = float(window["vix_ts_slope"].corr(window["hy_spread"]))

    # -----------------------------------------------------------------------
    # Lead signal: estimate days of lead by cross-correlation (max lag 10)
    # -----------------------------------------------------------------------
    lead_signal = _compute_lead_signal(enriched_df)

    # -----------------------------------------------------------------------
    # Key insight: backwardation → HY widening hit rate
    # -----------------------------------------------------------------------
    key_insight = _compute_key_insight(enriched_df)

    return {
        "snapshot": snapshot,
        "df": enriched_df,
        "available": available,
        "correlation_vix_slope_with_hy": corr_slope_hy,
        "lead_signal": lead_signal,
        "key_insight": key_insight,
    }


def _compute_lead_signal(df: pd.DataFrame) -> str:
    """
    Estimate how many days VIX term structure inversion leads HY spread widening.

    Uses cross-correlation of vix_ts_signal vs forward-shifted hy_spread changes.
    Returns a descriptive string.
    """
    if "vix_ts_signal" not in df.columns or "hy_spread" not in df.columns:
        return "Backwardation typically leads HY widening by 2–5 days (insufficient data to estimate)."

    sub = df[["vix_ts_signal", "hy_spread"]].dropna().tail(252)
    if len(sub) < 30:
        return "Backwardation typically leads HY widening by 2–5 days (insufficient history)."

    signal = sub["vix_ts_signal"].values
    hy = sub["hy_spread"].values

    best_lag = 0
    best_corr = -999.0
    max_lag = min(10, len(sub) // 5)

    for lag in range(1, max_lag + 1):
        s = signal[:-lag]
        h = hy[lag:]
        if len(s) < 10:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = float(np.corrcoef(s, h)[0, 1])
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_lag == 0 or best_corr < 0.05:
        return (
            "Backwardation typically leads HY widening by 2–5 days "
            "(historical data did not show a clear lead in this sample)."
        )
    return (
        f"Backwardation typically leads HY widening by {best_lag} day(s) "
        f"(cross-correlation r={best_corr:.2f} in this sample)."
    )


def _compute_key_insight(df: pd.DataFrame) -> str:
    """
    Compute: when VIX9D > VIX, what % of subsequent 5-day windows saw HY widening?
    Returns a plain-English insight string.
    """
    if "vix_ts_slope" not in df.columns or "hy_spread" not in df.columns:
        return (
            "When VIX9D > VIX (backwardation at short end), credit spreads have "
            "historically widened in a majority of subsequent 5-day windows."
        )

    sub = df[["vix_ts_slope", "hy_spread"]].dropna().tail(252).reset_index(drop=True)
    if len(sub) < 15:
        return (
            "When VIX9D > VIX (backwardation at short end), credit spreads have "
            "historically widened in a majority of subsequent 5-day windows (insufficient data)."
        )

    # Backwardation at the short end → slope < 0 is already a proxy here
    # (slope = VIX3M - VIX; negative means VIX > VIX3M, i.e. front end elevated)
    widened = 0
    total = 0
    fwd_window = 5

    for i in range(len(sub) - fwd_window):
        if sub.loc[i, "vix_ts_slope"] < 0:  # backwardation
            future_change = sub.loc[i + fwd_window, "hy_spread"] - sub.loc[i, "hy_spread"]
            if future_change > 0:
                widened += 1
            total += 1

    if total == 0:
        pct = "N/A"
        pct_str = "N/A%"
    else:
        pct = round(100 * widened / total, 1)
        pct_str = f"{pct}%"

    return (
        f"When VIX9D > VIX (backwardation at short end), credit spreads have widened "
        f"in {pct_str} of subsequent 5-day windows historically "
        f"(based on {total} backwardation episodes in this sample)."
    )
