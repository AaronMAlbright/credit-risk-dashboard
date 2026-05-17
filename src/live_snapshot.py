"""
Live intraday market snapshot via yfinance.
Public API: get_live_snapshot() -> dict
"""
from __future__ import annotations

import datetime
import functools

import yfinance as yf

_TICKERS = {
    "vix":   "^VIX",
    "sp500": "^GSPC",
    "hyg":   "HYG",
    "lqd":   "LQD",
}


def _fetch_ticker(sym: str) -> dict | None:
    """Fetch a single ticker and return price metrics, or None on failure."""
    try:
        hist = yf.Ticker(sym).history(period="5d", interval="1d")
        if hist.empty or len(hist) < 2:
            return None

        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None

        current = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])
        first_close = float(closes.iloc[0])

        day_chg_pct = (current / prev_close - 1) * 100
        five_day_chg_pct = (current / first_close - 1) * 100

        return {
            "symbol": sym,
            "current": current,
            "prev_close": prev_close,
            "day_chg_pct": day_chg_pct,
            "five_day_chg_pct": five_day_chg_pct,
        }
    except Exception:
        return None


# Cache key includes today's date so the cache refreshes daily.
@functools.lru_cache(maxsize=1)
def _cached_snapshot(date_key: str) -> dict:  # noqa: ARG001  (date_key used only as cache discriminator)
    result: dict = {}
    for key, sym in _TICKERS.items():
        result[key] = _fetch_ticker(sym)
    result["as_of"] = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    return result


def get_live_snapshot() -> dict:
    """
    Pull current intraday market data for VIX, S&P 500, HYG, and LQD.

    Returns a dict with keys ``vix``, ``sp500``, ``hyg``, ``lqd`` (each a
    sub-dict of price metrics, or None on failure) and ``as_of`` (ISO
    timestamp).  On catastrophic failure the dict contains only ``error``.
    """
    try:
        date_key = datetime.date.today().isoformat()
        return _cached_snapshot(date_key)
    except Exception as e:
        return {"error": str(e)}
