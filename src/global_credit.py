"""
Global Credit Divergence Monitor — US HY vs International HY.

When US and European/international credit diverge, one is leading the other
into stress. US credit leads in rate-driven stress episodes; European/global
credit leads in sovereign or banking-sector stress. The HYG/HYXU price ratio
is the primary signal; wider divergence in the ratio precedes contagion.

Tickers used:
  HYG  — iShares iBoxx $ High Yield Corporate Bond ETF (US HY)
  HYXU — iShares International High Yield Bond ETF (primary international)
  IHY  — VanEck International High Yield Bond ETF (fallback)
  IGOV — iShares International Treasury Bond ETF (safe-haven reference)

Public API
----------
  get_global_credit_snapshot()      -> dict
  compute_global_credit(df)         -> pd.DataFrame
  run_global_credit_analysis(df)    -> dict
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

_HYG_TICKER: str = "HYG"
_HYXU_TICKER: str = "HYXU"
_IHY_TICKER: str = "IHY"          # fallback for international HY
_IGOV_TICKER: str = "IGOV"

_SNAPSHOT_PERIOD: str = "90d"
_HISTORY_PERIOD: str = "5y"

_HISTORY_ROWS: int = 504           # ~2 trading years for historical slice

# Divergence thresholds (percentage points)
_DIV_US_LEADING: float = 2.0
_DIV_EUR_LEADING: float = -2.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_price_series(ticker: str, period: str) -> pd.Series | None:
    """
    Fetch daily Close prices for a single yfinance ticker.

    Returns a pd.Series with a date-normalised DatetimeIndex, or None on any
    failure (network error, empty data, etc.).
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


def _pct_change_n(series: pd.Series, n: int) -> float:
    """
    Return the n-period percent change for the last two valid points at least
    n positions apart. Returns NaN if not enough data.
    """
    s = series.dropna()
    if len(s) < n + 1:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[-n - 1] - 1) * 100)


def _classify_divergence_regime(divergence_30d: float) -> str:
    """
    Classify US vs international HY divergence into a qualitative regime.

    divergence_30d = us_hy_30d_return - intl_hy_30d_return (percentage points)

    > +2%  -> "US Leading"
    -2% to +2% -> "Aligned"
    < -2%  -> "Europe Leading"
    """
    if pd.isna(divergence_30d):
        return "Unknown"
    if divergence_30d > _DIV_US_LEADING:
        return "US Leading"
    if divergence_30d < _DIV_EUR_LEADING:
        return "Europe Leading"
    return "Aligned"


def _classify_stress_origin(divergence_regime: str, us_30d: float, intl_30d: float) -> str:
    """
    Determine the likely stress origin from regime and absolute return levels.

    "US-driven"     : US underperforming, international roughly stable
    "Europe-driven" : International underperforming, US roughly stable
    "Global"        : Both markets moving in the same direction (stress or rally)
    """
    if pd.isna(us_30d) or pd.isna(intl_30d):
        return "Unknown"

    both_negative = us_30d < -2.0 and intl_30d < -2.0
    both_positive = us_30d > 1.0 and intl_30d > 1.0

    if both_negative or both_positive:
        return "Global"

    if divergence_regime == "US Leading" and us_30d < 0:
        # US underperforming a positive or less-negative international market
        return "US-driven"
    if divergence_regime == "Europe Leading" and intl_30d < 0:
        return "Europe-driven"
    if divergence_regime == "US Leading":
        return "US-driven"
    if divergence_regime == "Europe Leading":
        return "Europe-driven"
    return "Global"


def _divergence_to_stress_signal(divergence_63d: float) -> float:
    """
    Map 63-day US/international ratio divergence (% change) to a 0-100 stress score.

    divergence_63d < -5%     -> 75  (Europe/global leading US into stress)
    -5% to -2%               -> 55
    -2% to +2%               -> 40
    > +2%                    -> 25  (US outperforming; global stress lower)
    """
    if pd.isna(divergence_63d):
        return float("nan")
    if divergence_63d < -5.0:
        return 75.0
    if divergence_63d < -2.0:
        return 55.0
    if divergence_63d <= 2.0:
        return 40.0
    return 25.0


def _last_valid(series: pd.Series) -> float:
    """Return the last non-NaN value as a Python float, or NaN."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])


# ---------------------------------------------------------------------------
# Public API — snapshot
# ---------------------------------------------------------------------------


def get_global_credit_snapshot() -> dict:
    """
    Fetch live 90-day data for HYG and HYXU (with IHY fallback) via yfinance.

    Returns
    -------
    dict with keys:
        hyg_price            : float — latest HYG close
        hyxu_price           : float — latest HYXU (or IHY) close
        us_hy_30d_return     : float — 30-day pct change in HYG (%)
        intl_hy_30d_return   : float — 30-day pct change in HYXU/IHY (%)
        divergence_30d       : float — us_hy_30d_return - intl_hy_30d_return
        divergence_regime    : str   — "US Leading" / "Aligned" / "Europe Leading"
        stress_origin        : str   — "US-driven" / "Global" / "Europe-driven"
        available            : bool
        error                : str (empty on success)
    """
    _nan = float("nan")
    errors: list[str] = []

    empty: dict = {
        "hyg_price": _nan,
        "hyxu_price": _nan,
        "us_hy_30d_return": _nan,
        "intl_hy_30d_return": _nan,
        "divergence_30d": _nan,
        "divergence_regime": "Unknown",
        "stress_origin": "Unknown",
        "available": False,
        "error": "",
    }

    if not _YF_AVAILABLE:
        empty["error"] = "yfinance not installed"
        return empty

    # --- HYG (US HY) ---------------------------------------------------------
    hyg_series = _fetch_price_series(_HYG_TICKER, _SNAPSHOT_PERIOD)
    if hyg_series is None or hyg_series.empty:
        errors.append("HYG fetch failed")
        hyg_latest = _nan
        us_30d = _nan
    else:
        hyg_latest = float(hyg_series.iloc[-1])
        us_30d = _pct_change_n(hyg_series, 21)

    # --- HYXU (International HY, primary) ------------------------------------
    hyxu_series = _fetch_price_series(_HYXU_TICKER, _SNAPSHOT_PERIOD)
    intl_ticker_used = _HYXU_TICKER

    if hyxu_series is None or hyxu_series.empty:
        errors.append(f"{_HYXU_TICKER} fetch failed; trying {_IHY_TICKER}")
        # Fallback: IHY
        hyxu_series = _fetch_price_series(_IHY_TICKER, _SNAPSHOT_PERIOD)
        intl_ticker_used = _IHY_TICKER
        if hyxu_series is None or hyxu_series.empty:
            errors.append(f"{_IHY_TICKER} fallback also failed")
            hyxu_latest = _nan
            intl_30d = _nan
        else:
            hyxu_latest = float(hyxu_series.iloc[-1])
            intl_30d = _pct_change_n(hyxu_series, 21)
    else:
        hyxu_latest = float(hyxu_series.iloc[-1])
        intl_30d = _pct_change_n(hyxu_series, 21)

    # --- Divergence ----------------------------------------------------------
    if pd.isna(us_30d) or pd.isna(intl_30d):
        divergence_30d = _nan
    else:
        divergence_30d = us_30d - intl_30d

    regime = _classify_divergence_regime(divergence_30d)
    stress_origin = _classify_stress_origin(regime, us_30d, intl_30d)

    available = not (pd.isna(hyg_latest) and pd.isna(hyxu_latest))

    return {
        "hyg_price": hyg_latest,
        "hyxu_price": hyxu_latest,
        "us_hy_30d_return": us_30d,
        "intl_hy_30d_return": intl_30d,
        "divergence_30d": divergence_30d,
        "divergence_regime": regime,
        "stress_origin": stress_origin,
        "available": available,
        "error": "; ".join(errors),
    }


# ---------------------------------------------------------------------------
# Public API — DataFrame enrichment
# ---------------------------------------------------------------------------


def compute_global_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join 5-year HYG and HYXU price history to df's DatetimeIndex and
    compute US vs international HY divergence signals.

    New columns
    -----------
    hyg_price              : HYG daily close (left-joined)
    hyxu_price             : HYXU (or IHY fallback) daily close (left-joined)
    us_intl_ratio          : HYG / HYXU — higher = US outperforming
    us_intl_divergence_63d : 63-day pct_change of us_intl_ratio (%)
    global_credit_signal   : 0-100 stress score from divergence_63d

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex.

    Returns
    -------
    pd.DataFrame — copy of df with new columns appended (NaN where data unavailable).
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()

    _nan = float("nan")

    hyg_series = _fetch_price_series(_HYG_TICKER, _HISTORY_PERIOD)

    hyxu_series = _fetch_price_series(_HYXU_TICKER, _HISTORY_PERIOD)
    if hyxu_series is None or hyxu_series.empty:
        hyxu_series = _fetch_price_series(_IHY_TICKER, _HISTORY_PERIOD)

    if (
        hyg_series is not None and not hyg_series.empty
        and hyxu_series is not None and not hyxu_series.empty
    ):
        try:
            combined = pd.DataFrame(
                {"hyg_price": hyg_series, "hyxu_price": hyxu_series}
            ).dropna()
            combined = combined[combined["hyxu_price"] != 0]
            combined["us_intl_ratio"] = combined["hyg_price"] / combined["hyxu_price"]

            combined.index = pd.to_datetime(combined.index).normalize()

            out = out.join(combined[["hyg_price", "hyxu_price", "us_intl_ratio"]], how="left")

            out["us_intl_divergence_63d"] = out["us_intl_ratio"].pct_change(63) * 100

            out["global_credit_signal"] = out["us_intl_divergence_63d"].apply(
                _divergence_to_stress_signal
            )
        except Exception:
            for col in ("hyg_price", "hyxu_price", "us_intl_ratio",
                        "us_intl_divergence_63d", "global_credit_signal"):
                out[col] = _nan
    else:
        for col in ("hyg_price", "hyxu_price", "us_intl_ratio",
                    "us_intl_divergence_63d", "global_credit_signal"):
            out[col] = _nan

    return out


# ---------------------------------------------------------------------------
# Public API — full analysis
# ---------------------------------------------------------------------------


def run_global_credit_analysis(df: pd.DataFrame) -> dict:
    """
    Full global credit divergence analysis combining live snapshot and df enrichment.

    Parameters
    ----------
    df : pd.DataFrame
        Main market data frame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        snapshot             : dict from get_global_credit_snapshot()
        df                   : enriched DataFrame
        current              : dict(us_intl_ratio, us_intl_divergence_63d,
                               global_credit_signal)
        historical           : last 504 rows — index + us_intl_ratio +
                               us_intl_divergence_63d + global_credit_signal
        available            : bool
        interpretation       : one-sentence plain-English summary
        lead_signal          : str — non-empty if global_credit_signal > 65
    """
    snapshot = get_global_credit_snapshot()
    enriched = compute_global_credit(df)

    # ---- Current scalars ----------------------------------------------------
    us_intl_ratio = _last_valid(enriched["us_intl_ratio"])
    divergence_63d = _last_valid(enriched["us_intl_divergence_63d"])
    global_signal = _last_valid(enriched["global_credit_signal"])

    current: dict = {
        "us_intl_ratio": us_intl_ratio,
        "us_intl_divergence_63d": divergence_63d,
        "global_credit_signal": global_signal,
    }

    # ---- Historical slice ---------------------------------------------------
    hist_cols = ["us_intl_ratio", "us_intl_divergence_63d", "global_credit_signal"]
    hist = enriched[hist_cols].tail(_HISTORY_ROWS)

    # ---- Lead signal --------------------------------------------------------
    if not pd.isna(global_signal) and global_signal > 65:
        lead_signal = (
            "Europe/global credit leading US into stress — monitor for contagion."
        )
    else:
        lead_signal = ""

    # ---- Available ----------------------------------------------------------
    available = (
        snapshot.get("available", False)
        or enriched["us_intl_ratio"].notna().any()
    )

    # ---- Interpretation -----------------------------------------------------
    div_30d = snapshot.get("divergence_30d", float("nan"))
    regime = snapshot.get("divergence_regime", "Unknown")
    stress_origin = snapshot.get("stress_origin", "Unknown")

    if not pd.isna(div_30d):
        direction = "outperforming" if div_30d > 0 else "underperforming"
        interpretation = (
            f"US HY is {direction} international HY by {abs(div_30d):.1f}% over "
            f"the past 30 days, placing global credit in the '{regime}' regime "
            f"with '{stress_origin}' stress dynamics."
        )
    elif available:
        interpretation = (
            "Global credit divergence data is partially available; "
            "30-day return differential could not be computed."
        )
    else:
        interpretation = (
            "Global credit divergence analysis unavailable: HYG and HYXU/IHY "
            "price data could not be fetched."
        )

    return {
        "snapshot": snapshot,
        "df": enriched,
        "current": current,
        "historical": hist,
        "available": available,
        "interpretation": interpretation,
        "lead_signal": lead_signal,
    }
