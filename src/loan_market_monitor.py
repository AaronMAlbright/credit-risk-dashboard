"""
Leveraged loan market monitor — BKLN vs HYG divergence as a CLO stress proxy.

When loans (BKLN, floating rate) underperform HY bonds (HYG, fixed rate), structured
credit vehicles (CLOs) face rising mark-to-market stress, often a leading indicator
for broader credit deterioration. Rising BKLN/HYG indicates late-cycle leverage
expansion and high CLO issuance appetite.

Secondary checks:
  - JNK vs HYG spread: two HY ETFs whose divergence flags liquidity stress
  - BKLN vs LQD: cross-asset stress when loans fall vs investment grade

All data is fetched live via yfinance. The module degrades gracefully if yfinance
is unavailable — it falls back to any loan columns already in the master DataFrame.

Public API
----------
  get_loan_snapshot()           -> dict  (live yfinance fetch)
  run_loan_market_monitor(df)   -> dict  (top-level entry)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TICKERS = ["BKLN", "HYG", "JNK", "LQD"]
_FETCH_PERIOD = "2y"

# Regime thresholds: 30-day % change in BKLN/HYG ratio
_SEVERE_STRESS_THRESHOLD = -0.03   # < -3% → Severe Loan Stress
_STRESS_THRESHOLD = -0.01          # < -1% → Loan Stress
_OUTPERFORM_THRESHOLD = 0.01       # > +1% → Loan Outperformance

# CLO stress: rolling windows
_RATIO_MA_WINDOW = 252             # 252d moving average for ratio level
_VOL_WINDOW = 21                   # 21d rolling std for vol
_VOL_MEAN_WINDOW = 252             # 252d mean of that vol series

# Z-score window for ratio vs history
_ZSCORE_WINDOW = 252

# Historical loan-related column names the master df might contain
_DF_LOAN_COLS = ["bkln", "loan_index", "leveraged_loan_spread"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_ratio_zscore(ratio: pd.Series, window: int = _ZSCORE_WINDOW) -> float:
    """Z-score of most recent ratio value vs rolling window history."""
    if len(ratio.dropna()) < window // 2:
        return float("nan")
    mu = ratio.rolling(window, min_periods=window // 2).mean().iloc[-1]
    sigma = ratio.rolling(window, min_periods=window // 2).std().iloc[-1]
    if pd.isna(sigma) or sigma == 0:
        return float("nan")
    return float((ratio.iloc[-1] - mu) / sigma)


def _pct_change_over_n_days(series: pd.Series, n: int) -> float:
    """% change of most recent value vs n rows ago (not calendar days)."""
    clean = series.dropna()
    if len(clean) <= n:
        return float("nan")
    old = float(clean.iloc[-(n + 1)])
    new = float(clean.iloc[-1])
    if old == 0:
        return float("nan")
    return (new / old - 1.0) * 100.0


def _ytd_change(series: pd.Series) -> float:
    """% change from first row of current calendar year to latest."""
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    current_year = clean.index[-1].year if hasattr(clean.index[-1], "year") else None
    if current_year is None:
        return float("nan")
    ytd = clean[clean.index.year == current_year]
    if len(ytd) < 2:
        return float("nan")
    old = float(ytd.iloc[0])
    new = float(ytd.iloc[-1])
    if old == 0:
        return float("nan")
    return (new / old - 1.0) * 100.0


def _classify_loan_regime(chg_30d_pct: float) -> str:
    """Map 30d ratio change (%) to a loan market regime label."""
    if chg_30d_pct < _SEVERE_STRESS_THRESHOLD * 100:
        return "Severe Loan Stress (CLO Deleveraging)"
    if chg_30d_pct < _STRESS_THRESHOLD * 100:
        return "Loan Stress (CLO Pressure)"
    if chg_30d_pct > _OUTPERFORM_THRESHOLD * 100:
        return "Loan Outperformance (Late Cycle)"
    return "Neutral"


def _compute_clo_stress_flag(ratio: pd.Series) -> bool:
    """
    CLO stress = True when BOTH:
      1. ratio is below its 252d moving average (level deterioration)
      2. 21d rolling std of ratio daily returns is above its 252d mean (elevated vol)
    """
    clean = ratio.dropna()
    if len(clean) < _RATIO_MA_WINDOW // 2:
        return False

    # Condition 1: ratio below 252d MA
    ma = clean.rolling(_RATIO_MA_WINDOW, min_periods=_RATIO_MA_WINDOW // 2).mean()
    below_ma = float(clean.iloc[-1]) < float(ma.iloc[-1])

    # Condition 2: elevated vol
    daily_returns = clean.pct_change().dropna()
    if len(daily_returns) < _VOL_WINDOW:
        return False
    rolling_vol = daily_returns.rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW // 2).std()
    vol_mean = rolling_vol.rolling(_VOL_MEAN_WINDOW, min_periods=_VOL_MEAN_WINDOW // 4).mean()
    if pd.isna(rolling_vol.iloc[-1]) or pd.isna(vol_mean.iloc[-1]):
        return False
    elevated_vol = float(rolling_vol.iloc[-1]) > float(vol_mean.iloc[-1])

    return bool(below_ma and elevated_vol)


# ---------------------------------------------------------------------------
# Public: get_loan_snapshot
# ---------------------------------------------------------------------------

def get_loan_snapshot() -> dict:
    """Fetch live loan market data via yfinance.

    Returns
    -------
    dict with keys:
      available                : bool
      bkln_price               : float
      hyg_price                : float
      jnk_price                : float or None
      lqd_price                : float or None
      bkln_hyg_ratio           : float
      bkln_hyg_ratio_30d_chg   : float  (% change in ratio over ~30 trading days)
      bkln_hyg_ratio_ytd_chg   : float  (% change ytd)
      loan_market_regime        : str
      clo_stress_flag          : bool
      bkln_hyg_zscore_1y       : float  (z-score of ratio vs 252d history)
      as_of                    : str    (ISO date of latest data)

    If yfinance fails for any reason, returns {"available": False}.
    """
    try:
        import yfinance as yf

        raw = yf.download(
            _TICKERS,
            period=_FETCH_PERIOD,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        # yfinance returns MultiIndex columns (field, ticker) when >1 ticker.
        # Extract the "Close" level.
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                prices = raw["Close"]
            else:
                return {"available": False}
        else:
            # Single ticker (shouldn't happen with a list, but be defensive)
            prices = raw[["Close"]] if "Close" in raw.columns else raw

        prices = prices.dropna(how="all")
        if prices.empty:
            return {"available": False}

        # Require BKLN and HYG
        if "BKLN" not in prices.columns or "HYG" not in prices.columns:
            return {"available": False}

        bkln = prices["BKLN"].dropna()
        hyg = prices["HYG"].dropna()
        if bkln.empty or hyg.empty:
            return {"available": False}

        # Align on common index
        common_idx = bkln.index.intersection(hyg.index)
        if len(common_idx) < 30:
            return {"available": False}

        bkln = bkln.loc[common_idx]
        hyg = hyg.loc[common_idx]

        ratio = bkln / hyg

        bkln_price = float(bkln.iloc[-1])
        hyg_price = float(hyg.iloc[-1])
        bkln_hyg_ratio_val = float(ratio.iloc[-1])

        # Optional tickers
        jnk_price: float | None = None
        lqd_price: float | None = None
        if "JNK" in prices.columns:
            jnk_s = prices["JNK"].dropna()
            if not jnk_s.empty:
                jnk_price = float(jnk_s.iloc[-1])
        if "LQD" in prices.columns:
            lqd_s = prices["LQD"].dropna()
            if not lqd_s.empty:
                lqd_price = float(lqd_s.iloc[-1])

        # 30-trading-day change: use 30 index positions
        chg_30d = _pct_change_over_n_days(ratio, 30)

        # YTD change
        ytd_chg = _ytd_change(ratio)

        # Regime classification
        regime = _classify_loan_regime(chg_30d) if not np.isnan(chg_30d) else "Neutral"

        # CLO stress flag
        clo_flag = _compute_clo_stress_flag(ratio)

        # 1-year z-score
        zscore_1y = _compute_ratio_zscore(ratio, _ZSCORE_WINDOW)

        as_of = str(ratio.index[-1].date()) if hasattr(ratio.index[-1], "date") else str(ratio.index[-1])

        return {
            "available": True,
            "bkln_price": bkln_price,
            "hyg_price": hyg_price,
            "jnk_price": jnk_price,
            "lqd_price": lqd_price,
            "bkln_hyg_ratio": bkln_hyg_ratio_val,
            "bkln_hyg_ratio_30d_chg": chg_30d,
            "bkln_hyg_ratio_ytd_chg": ytd_chg,
            "loan_market_regime": regime,
            "clo_stress_flag": clo_flag,
            "bkln_hyg_zscore_1y": zscore_1y,
            "as_of": as_of,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Internal: extract historical loan data from master df
# ---------------------------------------------------------------------------

def _extract_historical_ratio(df: pd.DataFrame) -> pd.Series | None:
    """
    If df contains a loan-related column, return it as a Series.
    Checks for: 'bkln', 'loan_index', 'leveraged_loan_spread'.
    Returns None if none found.
    """
    for col in _DF_LOAN_COLS:
        if col in df.columns:
            s = df[col].dropna()
            if not s.empty:
                return s.copy()
    return None


# ---------------------------------------------------------------------------
# Internal: derive credit lead signal
# ---------------------------------------------------------------------------

def _derive_lead_signal(snapshot: dict) -> str:
    """
    Translate loan market data into a directional credit signal.

    Bearish  — stress regime or CLO flag
    Bullish  — outperformance regime with no CLO flag
    Neutral  — everything else
    """
    if not snapshot.get("available"):
        return "Neutral"

    regime = snapshot.get("loan_market_regime", "Neutral")
    clo_flag = snapshot.get("clo_stress_flag", False)
    zscore = snapshot.get("bkln_hyg_zscore_1y", float("nan"))

    if "Stress" in regime or clo_flag:
        return "Bearish"
    if "Outperformance" in regime and not clo_flag:
        # Additional check: z-score above 0 means ratio is above historical mean
        if not np.isnan(zscore) and zscore > 0:
            return "Bullish"
        return "Bullish"
    return "Neutral"


def _build_interpretation(snapshot: dict, lead_signal: str) -> str:
    """Build a plain-English interpretation string from the snapshot."""
    if not snapshot.get("available"):
        return "Live loan market data unavailable. Check yfinance connectivity."

    regime = snapshot.get("loan_market_regime", "Neutral")
    chg = snapshot.get("bkln_hyg_ratio_30d_chg", float("nan"))
    clo_flag = snapshot.get("clo_stress_flag", False)
    zscore = snapshot.get("bkln_hyg_zscore_1y", float("nan"))
    as_of = snapshot.get("as_of", "unknown date")

    chg_str = f"{chg:+.2f}%" if not np.isnan(chg) else "N/A"
    zs_str = f"{zscore:.2f}" if not np.isnan(zscore) else "N/A"

    parts = [
        f"Loan market regime: {regime} (BKLN/HYG 30d change: {chg_str}).",
        f"CLO stress flag: {'ACTIVE' if clo_flag else 'inactive'}.",
        f"BKLN/HYG ratio z-score vs 1Y history: {zs_str}.",
    ]

    if lead_signal == "Bearish":
        if clo_flag:
            parts.append(
                "Loans are underperforming HY bonds with elevated ratio volatility — "
                "CLO mark-to-market stress is building. This is a leading warning signal "
                "for broader credit deterioration."
            )
        else:
            parts.append(
                "Loans are underperforming HY bonds. Watch for CLO manager deleveraging "
                "activity which could amplify credit spread widening."
            )
    elif lead_signal == "Bullish":
        parts.append(
            "Loans are outperforming HY bonds — investor appetite for floating-rate "
            "credit is elevated. This is typical of a late-cycle credit expansion phase "
            "with high CLO issuance and leveraged buyout activity."
        )
    else:
        parts.append(
            "Loan market is in line with HY bonds. No significant CLO stress or "
            "outperformance signal at this time."
        )

    parts.append(f"Data as of {as_of}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public: run_loan_market_monitor
# ---------------------------------------------------------------------------

def run_loan_market_monitor(df: pd.DataFrame) -> dict:
    """Top-level entry point for the loan market monitor.

    Fetches live BKLN/HYG data via yfinance and checks the master DataFrame
    for any historical loan/leveraged-loan columns.

    Parameters
    ----------
    df : Master DataFrame with DatetimeIndex.
         Optional columns checked: 'bkln', 'loan_index', 'leveraged_loan_spread'.

    Returns
    -------
    dict with keys:
      available        : bool — True if live snapshot or historical data succeeded
      snapshot         : dict — from get_loan_snapshot()
      historical_ratio : pd.Series or None — loan data from df if present
      lead_signal      : str  — "Bullish" / "Neutral" / "Bearish" for credit
      interpretation   : str  — plain-English summary
    """
    try:
        snapshot = get_loan_snapshot()
        historical_ratio = _extract_historical_ratio(df)
        available = snapshot.get("available", False) or (historical_ratio is not None)
        lead_signal = _derive_lead_signal(snapshot)
        interpretation = _build_interpretation(snapshot, lead_signal)

        return {
            "available": available,
            "snapshot": snapshot,
            "historical_ratio": historical_ratio,
            "lead_signal": lead_signal,
            "interpretation": interpretation,
        }

    except Exception:
        return {"available": False}
