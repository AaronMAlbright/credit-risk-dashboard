"""
Sector divergence — detect which sectors are leading or lagging the composite
risk signal using live ETF prices.

Key insight: Financials (XLF) stress that precedes the composite signal by
3-4 weeks is the most reliable early warning for credit spread widening.
Energy (XLE) leading down signals rising default cycle risk; Utilities (XLU)
leading up confirms risk-off rotation. When >=4 sectors simultaneously show
"Leading Down," systematic credit stress is likely.

Public API
----------
  SECTOR_ETFS : list[tuple[str, str]]
  get_sector_snapshot()           -> dict
  run_sector_divergence(df)       -> dict
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTOR_ETFS: list[tuple[str, str]] = [
    ("XLF", "Financials"),
    ("XLE", "Energy"),
    ("XBI", "Biotech/Healthcare"),
    ("XLU", "Utilities"),
    ("XHB", "Homebuilders"),
    ("XLI", "Industrials"),
    ("XLK", "Technology"),
    ("XLP", "Staples"),
]

# All tickers fetched in one call (sectors + SPY benchmark)
_ALL_TICKERS: list[str] = [t for t, _ in SECTOR_ETFS] + ["SPY"]

# Z-score thresholds for signal classification
_Z_LEADING_DOWN_THRESHOLD = -1.5
_Z_LAGGING_THRESHOLD = -0.5
_Z_OUTPERFORMING_THRESHOLD = 0.5
_Z_LEADING_UP_THRESHOLD = 1.5

# Minimum bars needed to produce a valid 252-day z-score
_MIN_HISTORY = 63

# Systematic stress trigger
_SYSTEMATIC_STRESS_N = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_etf_data() -> pd.DataFrame:
    """Download 1 year of adjusted Close prices for all sector ETFs + SPY.

    Returns a DataFrame with tickers as columns, dates as DatetimeIndex.
    Empty DataFrame on failure.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            _ALL_TICKERS,
            period="1y",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        # Single-ticker fallback (shouldn't happen with a list, but be safe)
        close = raw[["Close"]].copy()
        close.columns = _ALL_TICKERS[:1]

    close.index = pd.to_datetime(close.index).normalize()
    return close.sort_index()


def _safe_return(prices: pd.Series, window: int) -> Optional[float]:
    """Compute the *window*-day return for the last observation in *prices*.

    Returns None if there are insufficient non-NaN values.
    """
    s = prices.dropna()
    if len(s) < window + 1:
        return None
    current = float(s.iloc[-1])
    past = float(s.iloc[-(window + 1)])
    if past == 0:
        return None
    return (current / past) - 1.0


def _safe_zscore(prices: pd.Series, window: int = 21, hist_window: int = 252) -> Optional[float]:
    """Compute the z-score of the most recent *window*-day return vs the
    rolling *hist_window*-day distribution of such returns.

    Uses all available observations up to *hist_window* days.
    Returns None if not enough data.
    """
    s = prices.dropna()
    if len(s) < hist_window + window:
        # Relax: use whatever history is available down to _MIN_HISTORY
        min_needed = _MIN_HISTORY + window
        if len(s) < min_needed:
            return None

    # Build full history of window-day rolling returns
    rolling_returns = s.pct_change(window).dropna()
    if len(rolling_returns) < 20:
        return None

    # Use the trailing hist_window observations of the rolling-return series
    hist_slice = rolling_returns.iloc[-hist_window:] if len(rolling_returns) >= hist_window else rolling_returns
    current_ret = float(rolling_returns.iloc[-1])
    mean_ret = float(hist_slice.mean())
    std_ret = float(hist_slice.std(ddof=1))

    if std_ret == 0 or np.isnan(std_ret):
        return 0.0

    return (current_ret - mean_ret) / std_ret


def _classify_signal(z: float) -> str:
    """Map a z-score to a categorical signal string."""
    if z < _Z_LEADING_DOWN_THRESHOLD:
        return "Leading Down"
    elif z < _Z_LAGGING_THRESHOLD:
        return "Lagging"
    elif z <= _Z_OUTPERFORMING_THRESHOLD:
        return "Neutral"
    elif z <= _Z_LEADING_UP_THRESHOLD:
        return "Outperforming"
    else:
        return "Leading Up"


def _build_lead_signal(
    financials_signal: str,
    energy_signal: str,
    utilities_signal: str,
    systematic_stress: bool,
) -> str:
    """Derive the top-level lead signal for credit from sector signals."""
    if systematic_stress:
        return "Credit Warning"
    if financials_signal == "Leading Down":
        return "Credit Warning"
    if energy_signal == "Leading Down":
        return "Elevated Risk"
    if utilities_signal == "Leading Up":
        return "Elevated Risk"
    # Count warning sectors
    warning_signals = {"Leading Down", "Lagging"}
    if financials_signal in warning_signals or energy_signal in warning_signals:
        return "Elevated Risk"
    if utilities_signal in {"Outperforming", "Leading Up"}:
        return "Neutral"
    return "Risk-On"


def _build_credit_implication(lead_signal: str, snapshot: dict) -> str:
    """Return a plain-English credit spread implication."""
    fin_sig = snapshot.get("financials_signal", "Unknown")
    n_down = snapshot.get("n_leading_down", 0)

    if lead_signal == "Credit Warning":
        if snapshot.get("systematic_stress"):
            return (
                f"{n_down} sectors simultaneously leading down — systematic stress signal; "
                "credit spreads historically widen 30-80 bps within 4-6 weeks."
            )
        return (
            "Financials (XLF) leading the composite lower — "
            "bank/financial sector weakness typically precedes IG/HY spread widening by 3-4 weeks."
        )
    elif lead_signal == "Elevated Risk":
        if fin_sig in ("Leading Down", "Lagging"):
            return (
                "Financials underperforming; credit risk appetite is deteriorating — "
                "watch for spread widening in lower-rated credit."
            )
        return (
            "Defensive rotation (Utilities outperforming) signals risk-off sentiment; "
            "credit spreads may face modest widening pressure."
        )
    elif lead_signal == "Neutral":
        return (
            "Sector signals are mixed with no clear directional bias; "
            "credit spreads likely range-bound near term."
        )
    else:  # Risk-On
        return (
            "Cyclical sectors are leading higher with no stress signals — "
            "credit conditions supportive; spread compression bias."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_sector_snapshot() -> dict:
    """Fetch live sector ETF data via yfinance and compute divergence signals.

    Returns
    -------
    dict with keys:
        available        : bool
        as_of            : str  — ISO timestamp of snapshot
        sectors          : list[dict]  — one entry per ETF:
                               ticker, name, price, return_1m, return_3m,
                               rel_perf_1m, rel_perf_3m, z_score_1m, signal
        n_leading_down   : int
        n_leading_up     : int
        stress_sectors   : list[str]  — names of "Leading Down" sectors
        financials_signal: str        — XLF signal specifically
        systematic_stress: bool       — True if n_leading_down >= 4
        spy_return_1m    : float      — SPY 1M return (benchmark)
        spy_return_3m    : float      — SPY 3M return (benchmark)

    Returns {"available": False} on any error.
    """
    try:
        close = _fetch_etf_data()
        if close.empty:
            return {"available": False}

        # Verify SPY is present (required benchmark)
        if "SPY" not in close.columns:
            return {"available": False}

        import datetime as _dt
        as_of = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()

        spy_prices = close["SPY"]
        spy_ret_1m = _safe_return(spy_prices, 21)
        spy_ret_3m = _safe_return(spy_prices, 63)

        sectors_out: list[dict] = []
        n_leading_down = 0
        n_leading_up = 0
        stress_sectors: list[str] = []
        financials_signal = "Unknown"

        for ticker, name in SECTOR_ETFS:
            if ticker not in close.columns:
                sectors_out.append(
                    {
                        "ticker": ticker,
                        "name": name,
                        "price": None,
                        "return_1m": None,
                        "return_3m": None,
                        "rel_perf_1m": None,
                        "rel_perf_3m": None,
                        "z_score_1m": None,
                        "signal": "N/A",
                    }
                )
                continue

            prices = close[ticker]
            s = prices.dropna()

            # Current price
            price = float(s.iloc[-1]) if not s.empty else None

            # Absolute returns
            ret_1m = _safe_return(prices, 21)
            ret_3m = _safe_return(prices, 63)

            # Relative performance vs SPY
            rel_1m: Optional[float] = None
            rel_3m: Optional[float] = None
            if ret_1m is not None and spy_ret_1m is not None:
                rel_1m = ret_1m - spy_ret_1m
            if ret_3m is not None and spy_ret_3m is not None:
                rel_3m = ret_3m - spy_ret_3m

            # Z-score of 21-day return vs 252-day rolling history
            z = _safe_zscore(prices, window=21, hist_window=252)

            # Signal classification
            if z is not None:
                signal = _classify_signal(z)
            else:
                signal = "Insufficient Data"

            # Aggregate counts
            if signal == "Leading Down":
                n_leading_down += 1
                stress_sectors.append(name)
            elif signal == "Leading Up":
                n_leading_up += 1

            if ticker == "XLF":
                financials_signal = signal

            sectors_out.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "price": price,
                    "return_1m": ret_1m,
                    "return_3m": ret_3m,
                    "rel_perf_1m": rel_1m,
                    "rel_perf_3m": rel_3m,
                    "z_score_1m": z,
                    "signal": signal,
                }
            )

        systematic_stress = n_leading_down >= _SYSTEMATIC_STRESS_N

        return {
            "available": True,
            "as_of": as_of,
            "sectors": sectors_out,
            "n_leading_down": n_leading_down,
            "n_leading_up": n_leading_up,
            "stress_sectors": stress_sectors,
            "financials_signal": financials_signal,
            "systematic_stress": systematic_stress,
            "spy_return_1m": spy_ret_1m,
            "spy_return_3m": spy_ret_3m,
        }

    except Exception:
        return {"available": False}


def run_sector_divergence(df: pd.DataFrame) -> dict:
    """Top-level entry point for the sector divergence module.

    Fetches a live sector ETF snapshot and optionally augments with any
    sector columns already present in *df* (e.g. xlf, xlf_price,
    financials_etf).

    Parameters
    ----------
    df : pd.DataFrame
        Historical master DataFrame with a DatetimeIndex.  Optional columns
        like ``xlf``, ``xlf_price``, ``financials_etf`` are detected but
        never required.

    Returns
    -------
    dict with keys:
        available          : bool
        snapshot           : dict    — from get_sector_snapshot()
        lead_signal        : str     — "Credit Warning" / "Elevated Risk" /
                                       "Neutral" / "Risk-On"
        credit_implication : str     — plain-English credit spread implication
        interpretation     : str     — narrative summary
    """
    try:
        snapshot = get_sector_snapshot()

        # Check df for any sector-related columns (case-insensitive)
        df_sector_cols: list[str] = []
        sector_col_patterns = [
            "xlf", "xle", "xbi", "xlu", "xhb", "xli", "xlk", "xlp",
            "financials_etf", "financials", "energy_etf",
        ]
        if df is not None and isinstance(df, pd.DataFrame):
            lower_cols = {c.lower(): c for c in df.columns}
            for pat in sector_col_patterns:
                for lc, orig in lower_cols.items():
                    if pat in lc:
                        df_sector_cols.append(orig)

        df_has_sectors = len(df_sector_cols) > 0
        available = snapshot.get("available", False) or df_has_sectors

        if not available:
            return {"available": False}

        # Derive lead signal from snapshot when available
        if snapshot.get("available"):
            fin_sig = snapshot.get("financials_signal", "Unknown")
            # Retrieve XLE and XLU signals
            energy_signal = "Unknown"
            utilities_signal = "Unknown"
            for sec in snapshot.get("sectors", []):
                if sec["ticker"] == "XLE":
                    energy_signal = sec.get("signal", "Unknown")
                elif sec["ticker"] == "XLU":
                    utilities_signal = sec.get("signal", "Unknown")

            systematic_stress = snapshot.get("systematic_stress", False)
            lead_signal = _build_lead_signal(
                fin_sig, energy_signal, utilities_signal, systematic_stress
            )
        else:
            # Live data failed; try to derive a basic signal from df columns
            lead_signal = "Neutral"
            fin_sig = "Unknown"
            systematic_stress = False

        credit_implication = _build_credit_implication(lead_signal, snapshot if snapshot.get("available") else {})
        interpretation = _build_interpretation(snapshot, lead_signal, df_sector_cols)

        return {
            "available": available,
            "snapshot": snapshot,
            "lead_signal": lead_signal,
            "credit_implication": credit_implication,
            "interpretation": interpretation,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Internal narrative builder
# ---------------------------------------------------------------------------


def _build_interpretation(
    snapshot: dict,
    lead_signal: str,
    df_sector_cols: list[str],
) -> str:
    """Compose a plain-English narrative of the current sector setup."""
    lines: list[str] = []

    if snapshot.get("available"):
        n_down = snapshot.get("n_leading_down", 0)
        n_up = snapshot.get("n_leading_up", 0)
        stress = snapshot.get("stress_sectors", [])
        fin_sig = snapshot.get("financials_signal", "Unknown")
        systematic = snapshot.get("systematic_stress", False)
        as_of = snapshot.get("as_of", "")

        if as_of:
            lines.append(f"Sector snapshot as of {as_of[:10]}.")

        if systematic:
            lines.append(
                f"SYSTEMATIC STRESS: {n_down} of 8 tracked sectors are Leading Down "
                f"({', '.join(stress)}). Broad sector weakness of this magnitude is "
                "historically consistent with credit spread widening of 50+ bps."
            )
        elif lead_signal == "Credit Warning":
            lines.append(
                f"Financials (XLF) signal: {fin_sig}. "
                "Financial sector leading the composite lower is the most reliable "
                "3-4 week early warning for credit spread widening."
            )
        elif n_down > 0:
            lines.append(
                f"{n_down} sector(s) are Leading Down: {', '.join(stress)}. "
                "Monitor for broadening weakness."
            )
        else:
            lines.append(
                f"No sectors are Leading Down. {n_up} sector(s) Leading Up. "
                f"Financials signal: {fin_sig}."
            )

        # Specific sector notes
        for sec in snapshot.get("sectors", []):
            if sec["ticker"] == "XLE" and sec.get("signal") == "Leading Down":
                lines.append(
                    "Energy (XLE) is Leading Down — historically associated with rising "
                    "default cycle risk in leveraged credit."
                )
            if sec["ticker"] == "XLU" and sec.get("signal") in ("Leading Up", "Outperforming"):
                lines.append(
                    "Utilities (XLU) are outperforming — classic risk-off rotation "
                    "that confirms defensive positioning in credit."
                )

    elif df_sector_cols:
        lines.append(
            f"Live sector ETF data unavailable. "
            f"Historical sector columns found in df: {', '.join(df_sector_cols)}."
        )
    else:
        lines.append("Sector divergence data unavailable.")

    lines.append(f"Overall lead signal: {lead_signal}.")
    return " ".join(lines)
