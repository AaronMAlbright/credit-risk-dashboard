"""
Put/Call sentiment — CBOE equity put/call ratio + VIX as a contrarian
positioning signal.

Extreme put buying = crowded short = squeeze risk.
Extreme complacency (low P/C) = insufficient hedging = asymmetric downside.

Sentiment composite score (0-100):
  Higher score = more bearish market sentiment = more contrarian bullish.
  0.6 * pc_score + 0.4 * vix_score

Put/Call interpretation (contrarian):
  P/C > 1.2  : Extreme fear   → contrarian bullish
  P/C 0.9-1.2: Elevated fear  → mildly bullish
  P/C 0.7-0.9: Neutral
  P/C 0.5-0.7: Complacent     → mildly bearish
  P/C < 0.5  : Extreme complacency → contrarian bearish

Fallback: When ^PCCE is unavailable, synthetic P/C = VIX / 20.

Public API
----------
  get_put_call_snapshot()           -> dict
  compute_historical_sentiment(df)  -> pd.DataFrame
  run_put_call_sentiment(df)        -> dict
"""

from __future__ import annotations

import datetime
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# P/C ratio normalization: 0.4 -> 0, 1.5 -> 100
_PC_SCORE_MIN = 0.40
_PC_SCORE_MAX = 1.50

# VIX score window for percentile
_VIX_PCT_WINDOW = 252

# Composite weights
_PC_WEIGHT = 0.6
_VIX_WEIGHT = 0.4

# Minimum rows required to compute meaningful historical sentiment
_MIN_HIST_ROWS = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_yf_close(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download adjusted Close prices via yfinance. Returns empty DF on error."""
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

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers[:1]

    close.index = pd.to_datetime(close.index).normalize()
    return close.sort_index()


def _pc_to_score(pc_ratio: float) -> float:
    """Normalize a P/C ratio to 0-100 (linear clamp).

    P/C of _PC_SCORE_MAX (1.5) or above maps to 100.
    P/C of _PC_SCORE_MIN (0.4) or below maps to 0.
    """
    if pc_ratio >= _PC_SCORE_MAX:
        return 100.0
    if pc_ratio <= _PC_SCORE_MIN:
        return 0.0
    return (pc_ratio - _PC_SCORE_MIN) / (_PC_SCORE_MAX - _PC_SCORE_MIN) * 100.0


def _vix_to_score(vix_level: float, vix_series: pd.Series) -> float:
    """Convert a VIX level to a 0-100 percentile score vs *vix_series*.

    Higher VIX = higher score (more fear = higher contrarian-bullish reading).
    Uses a 252-day window from vix_series if available; falls back to
    a simple absolute mapping if the series is too short.
    """
    s = vix_series.dropna()
    if len(s) >= 20:
        window = s.iloc[-_VIX_PCT_WINDOW:] if len(s) >= _VIX_PCT_WINDOW else s
        pct = float(np.mean(window.values <= vix_level) * 100.0)
        return pct
    # Absolute fallback: VIX 10 -> 0, VIX 40 -> 100
    return float(np.clip((vix_level - 10.0) / 30.0 * 100.0, 0.0, 100.0))


def _composite_score(pc_score: float, vix_score: float) -> float:
    """Combine P/C and VIX scores into a 0-100 sentiment composite."""
    return round(_PC_WEIGHT * pc_score + _VIX_WEIGHT * vix_score, 2)


def _signal_from_composite(score: float) -> str:
    """Map a 0-100 composite score to a sentiment signal string."""
    if score > 70:
        return "Contrarian Bullish (Extreme Fear)"
    elif score > 55:
        return "Mildly Bullish Contrarian"
    elif score >= 45:
        return "Neutral"
    elif score >= 30:
        return "Mildly Bearish Contrarian"
    else:
        return "Contrarian Bearish (Extreme Complacency)"


def _latest_non_nan(series: pd.Series) -> Optional[float]:
    """Return the last non-NaN value, or None."""
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_put_call_snapshot() -> dict:
    """Fetch live put/call ratio and VIX data and compute sentiment composite.

    Tries to retrieve ^PCCE (CBOE Equity Put/Call Ratio) first. If that
    fails or returns empty data, falls back to a synthetic P/C proxy derived
    from VIX (synthetic P/C = VIX / 20).  If VIX also fails, returns
    ``{"available": False}``.

    Returns
    -------
    dict with keys:
        available           : bool
        pc_ratio            : float or None
        pc_source           : str  — "CBOE" or "Synthetic (VIX proxy)"
        vix                 : float or None
        pc_score            : float  — 0-100 normalized P/C
        vix_score           : float  — 0-100 VIX percentile (1-year window)
        sentiment_composite : float  — 0-100
        sentiment_signal    : str
        as_of               : str
    """
    try:
        as_of = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        # Step 1: Fetch VIX (always needed — fallback base)
        vix_close = _fetch_yf_close(["^VIX"], period="1y")
        vix_series: pd.Series = pd.Series(dtype=float)
        vix: Optional[float] = None

        if not vix_close.empty and "^VIX" in vix_close.columns:
            vix_series = vix_close["^VIX"].dropna()
            vix = _latest_non_nan(vix_series)

        if vix is None:
            # VIX completely unavailable — nothing to work with
            return {"available": False}

        # Step 2: Attempt CBOE Equity Put/Call Ratio (^PCCE)
        pc_ratio: Optional[float] = None
        pc_source = "Synthetic (VIX proxy)"

        try:
            pc_close = _fetch_yf_close(["^PCCE"], period="1y")
            if not pc_close.empty and "^PCCE" in pc_close.columns:
                pc_series = pc_close["^PCCE"].dropna()
                candidate = _latest_non_nan(pc_series)
                # Sanity-check: valid P/C ratio is in [0.1, 5.0]
                if candidate is not None and 0.1 <= candidate <= 5.0:
                    pc_ratio = candidate
                    pc_source = "CBOE"
        except Exception:
            pass  # Fall through to synthetic proxy

        # Step 3: If CBOE P/C unavailable, build synthetic proxy from VIX
        if pc_ratio is None:
            pc_ratio = vix / 20.0
            pc_source = "Synthetic (VIX proxy)"

        # Step 4: Score both components
        pc_score = _pc_to_score(pc_ratio)
        vix_score = _vix_to_score(vix, vix_series)

        # Step 5: Composite
        composite = _composite_score(pc_score, vix_score)
        signal = _signal_from_composite(composite)

        return {
            "available": True,
            "pc_ratio": pc_ratio,
            "pc_source": pc_source,
            "vix": vix,
            "pc_score": pc_score,
            "vix_score": vix_score,
            "sentiment_composite": composite,
            "sentiment_signal": signal,
            "as_of": as_of,
        }

    except Exception:
        return {"available": False}


def compute_historical_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Build historical sentiment scores from columns in *df*.

    Looks for a ``vix`` column (required) and optionally ``put_call_ratio``
    or ``pc_ratio``.  When neither P/C column exists, VIX alone drives the
    composite via the synthetic proxy (VIX / 20).

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex and at minimum a ``vix`` column.

    Returns
    -------
    pd.DataFrame — copy of *df* with added columns:
        sentiment_vix_score    : float  — 0-100 rolling VIX percentile
        sentiment_pc_score     : float  — 0-100 P/C score (or NaN)
        sentiment_composite    : float  — 0-100
        sentiment_signal       : str
    """
    out = df.copy()

    # Initialize output columns as NaN
    out["sentiment_vix_score"] = np.nan
    out["sentiment_pc_score"] = np.nan
    out["sentiment_composite"] = np.nan
    out["sentiment_signal"] = np.nan

    if "vix" not in out.columns:
        return out

    vix_col = pd.to_numeric(out["vix"], errors="coerce")
    if vix_col.notna().sum() < _MIN_HIST_ROWS:
        return out

    # --- Detect P/C column ---
    pc_col_name: Optional[str] = None
    for candidate in ("put_call_ratio", "pc_ratio"):
        if candidate in out.columns:
            pc_col_name = candidate
            break

    # --- Rolling VIX percentile (252-day window) ---
    vix_scores = pd.Series(np.nan, index=out.index, dtype=float)
    for i in range(len(out)):
        v = vix_col.iloc[i]
        if pd.isna(v):
            continue
        start = max(0, i - _VIX_PCT_WINDOW + 1)
        window_vals = vix_col.iloc[start : i + 1].dropna().values
        if len(window_vals) < _MIN_HIST_ROWS:
            continue
        vix_scores.iloc[i] = float(np.mean(window_vals <= v) * 100.0)

    out["sentiment_vix_score"] = vix_scores

    # --- P/C score ---
    if pc_col_name is not None:
        pc_numeric = pd.to_numeric(out[pc_col_name], errors="coerce")
        out["sentiment_pc_score"] = pc_numeric.apply(
            lambda x: _pc_to_score(x) if pd.notna(x) else np.nan
        )
    else:
        # Synthetic proxy: VIX / 20 as P/C approximation
        synthetic_pc = vix_col / 20.0
        out["sentiment_pc_score"] = synthetic_pc.apply(
            lambda x: _pc_to_score(x) if pd.notna(x) else np.nan
        )

    # --- Composite ---
    pc_s = out["sentiment_pc_score"]
    vix_s = out["sentiment_vix_score"]
    composite_mask = pc_s.notna() & vix_s.notna()

    out.loc[composite_mask, "sentiment_composite"] = (
        _PC_WEIGHT * pc_s[composite_mask] + _VIX_WEIGHT * vix_s[composite_mask]
    ).round(2)

    out["sentiment_signal"] = out["sentiment_composite"].apply(
        lambda x: _signal_from_composite(x) if pd.notna(x) else np.nan
    )

    return out


def run_put_call_sentiment(df: pd.DataFrame) -> dict:
    """Top-level entry point for the put/call sentiment module.

    Combines a live market snapshot with a historical sentiment series built
    from *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex.  Requires a ``vix`` column for
        historical sentiment.  Optionally contains ``put_call_ratio`` or
        ``pc_ratio``.

    Returns
    -------
    dict with keys:
        available          : bool
        snapshot           : dict          — from get_put_call_snapshot()
        historical         : pd.DataFrame  — last 504 rows from
                                             compute_historical_sentiment()
        signal_series      : pd.Series     — sentiment_composite last 504 rows
        current_historical : dict          — most-recent row of historical df
                                             as a dict (for consistency checks)
        contrarian_signal  : str           — actionable contrarian interpretation
        interpretation     : str           — narrative summary

    ``available`` is True when either the live snapshot succeeded or the
    historical sentiment is computable from *df* (requires a ``vix`` column).
    """
    try:
        snapshot = get_put_call_snapshot()

        # Historical sentiment
        historical_full = compute_historical_sentiment(df)
        historical = historical_full.tail(504).copy()
        signal_series = historical["sentiment_composite"].copy()

        hist_available = (
            "vix" in df.columns
            and pd.to_numeric(df["vix"], errors="coerce").notna().sum() >= _MIN_HIST_ROWS
        )

        available = snapshot.get("available", False) or hist_available

        if not available:
            return {"available": False}

        # Current historical state (last row with valid composite)
        valid_hist = historical.dropna(subset=["sentiment_composite"])
        if not valid_hist.empty:
            last_row = valid_hist.iloc[-1]
            current_historical = {
                "sentiment_composite": float(last_row["sentiment_composite"]),
                "sentiment_vix_score": float(last_row["sentiment_vix_score"])
                if pd.notna(last_row["sentiment_vix_score"])
                else None,
                "sentiment_pc_score": float(last_row["sentiment_pc_score"])
                if pd.notna(last_row["sentiment_pc_score"])
                else None,
                "sentiment_signal": last_row["sentiment_signal"],
                "date": str(last_row.name)[:10] if hasattr(last_row, "name") else None,
            }
        else:
            current_historical = {}

        # Contrarian signal — prefer live snapshot, fall back to historical
        if snapshot.get("available"):
            contrarian_signal = snapshot.get("sentiment_signal", "Unknown")
            source_composite = snapshot.get("sentiment_composite")
        elif current_historical:
            contrarian_signal = current_historical.get("sentiment_signal", "Unknown")
            source_composite = current_historical.get("sentiment_composite")
        else:
            contrarian_signal = "Unknown"
            source_composite = None

        interpretation = _build_pc_interpretation(snapshot, current_historical, contrarian_signal, source_composite)

        return {
            "available": available,
            "snapshot": snapshot,
            "historical": historical,
            "signal_series": signal_series,
            "current_historical": current_historical,
            "contrarian_signal": contrarian_signal,
            "interpretation": interpretation,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Internal narrative builder
# ---------------------------------------------------------------------------


def _build_pc_interpretation(
    snapshot: dict,
    current_historical: dict,
    contrarian_signal: str,
    composite: Optional[float],
) -> str:
    """Compose a narrative summary of the current put/call sentiment state."""
    lines: list[str] = []

    if snapshot.get("available"):
        pc = snapshot.get("pc_ratio")
        pc_src = snapshot.get("pc_source", "Unknown")
        vix = snapshot.get("vix")
        comp = snapshot.get("sentiment_composite")
        as_of = snapshot.get("as_of", "")

        if as_of:
            lines.append(f"Put/call snapshot as of {as_of[:10]}.")

        if pc is not None:
            lines.append(f"P/C ratio: {pc:.2f} ({pc_src}).")
            if pc > 1.2:
                lines.append(
                    "Extreme put buying indicates crowded short positioning — "
                    "squeeze risk elevated; contrarian bullish for risk assets."
                )
            elif pc > 0.9:
                lines.append("Elevated put/call ratio signals broad hedging activity — mildly bullish contrarian.")
            elif pc < 0.5:
                lines.append(
                    "P/C ratio at extreme complacency — insufficient hedging in the market; "
                    "historically precedes drawdowns as downside protection is underpriced."
                )
            elif pc < 0.7:
                lines.append("P/C below neutral; market is under-hedged — mildly bearish contrarian signal.")

        if vix is not None:
            lines.append(f"VIX: {vix:.1f}.")

        if comp is not None:
            lines.append(f"Sentiment composite: {comp:.0f}/100.")

    elif current_historical:
        lines.append(
            f"Live P/C data unavailable. "
            f"Historical sentiment composite (last observation): "
            f"{current_historical.get('sentiment_composite', 'N/A')}."
        )

    # Actionable summary
    if contrarian_signal == "Contrarian Bullish (Extreme Fear)":
        lines.append(
            "CONTRARIAN BULLISH: Fear is at extremes — crowded short positioning "
            "historically resolves via short squeeze; credit spreads may compress."
        )
    elif contrarian_signal == "Mildly Bullish Contrarian":
        lines.append(
            "Elevated fear supports a mild contrarian-bullish bias. "
            "Credit risk appetite may improve as positioning unwinds."
        )
    elif contrarian_signal == "Neutral":
        lines.append("Positioning is balanced. No strong contrarian signal for credit.")
    elif contrarian_signal == "Mildly Bearish Contrarian":
        lines.append(
            "Market is under-hedged relative to norms. "
            "Credit spreads face upside (widening) risk if a catalyst emerges."
        )
    elif contrarian_signal == "Contrarian Bearish (Extreme Complacency)":
        lines.append(
            "CONTRARIAN BEARISH: Extreme complacency — put/call ratios near multi-year lows. "
            "Historically precedes sharp credit spread widening when complacency breaks."
        )

    return " ".join(lines) if lines else "Put/call sentiment data unavailable."
