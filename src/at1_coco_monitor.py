"""
AT1 / CoCo (Additional Tier 1 / Contingent Convertible) Monitor.

AT1 bonds are the most subordinated debt in a bank's capital stack. They are
written down or converted to equity when a bank's CET1 ratio falls below a
trigger (typically 5.125% or 7%). The Credit Suisse AT1 wipeout in March 2023
— where ~$17bn of AT1 bonds were zeroed while equity received some value —
was the defining credit event of the cycle and rewrote investor perception of
AT1 bail-in hierarchy.

This module tracks AT1/CoCo spread behaviour relative to senior high-yield
debt and bank equity proxies. When AT1s underperform HY, bank capital stress
is being priced in by sophisticated institutional investors ahead of equity
market recognition.

Live data via yfinance (ETF proxies):
  CSHI  — iShares Contingent Capital ETF (AT1/CoCo bonds)   [primary]
  AT1   — Invesco AT1 Capital Bond ETF                       [secondary]
  HYG   — iShares HY Bond ETF                                [basis]
  KBE   — SPDR Bank ETF                                      [equity proxy]
  KRE   — SPDR Regional Bank ETF                             [regional stress]
  XLF   — Financial Select Sector SPDR                       [broad fallback]
  SPY   — S&P 500 ETF                                        [market reference]

Fallback: XLF/SPY ratio as a crude bank stress proxy when AT1 ETFs are absent.

Public API
----------
  AT1_ETFS              — module constant (list of tickers fetched)
  get_at1_snapshot()    -> dict
  run_at1_coco_monitor(df) -> dict
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AT1_ETFS: list[str] = ["CSHI", "AT1", "HYG", "KBE", "KRE", "XLF", "SPY"]

_FETCH_PERIOD = "1y"
_RATIO_WINDOW = 21          # 21 trading days for relative return windows
_52W_HIGH_WINDOW = 252      # approx. 1 trading year

# AT1/HYG ratio signal thresholds (21d % change in ratio)
_AT1_SEVERE_UNDERPERF = -0.03   # < -3%: AT1 Underperformance — Bank Stress
_AT1_MILD_UNDERPERF   = -0.01   # < -1%: Mild AT1 Stress
_AT1_OUTPERF          =  0.01   # >  1%: AT1 Outperformance — Bank Health

# CoCo trigger distance thresholds (% below 52-week high)
_TRIGGER_NEAR      = -0.15   # KBE > 15% below 52w high: Near Trigger Zone
_TRIGGER_ELEVATED  = -0.25   # KBE > 25% below 52w high: Trigger Risk Elevated

# Normalisation window for bank stress composite
_NORM_WINDOW = 252
_NORM_MIN_PERIODS = 63

# Historical rows to include in historical_banking_stress output
_HISTORY_ROWS = 504

# AT1 ETF preference order
_AT1_TICKER_PREFERENCE = ["CSHI", "AT1"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct_change_21d(series: pd.Series) -> float:
    """Return the 21-day percentage change in the last value vs 21 days ago."""
    clean = series.dropna()
    if len(clean) < _RATIO_WINDOW + 1:
        return float("nan")
    val_now = float(clean.iloc[-1])
    val_21d_ago = float(clean.iloc[-_RATIO_WINDOW - 1])
    if val_21d_ago == 0:
        return float("nan")
    return (val_now / val_21d_ago - 1.0) * 100.0


def _relative_return_21d(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Return 21-day relative return: return_a - return_b (percentage points).

    Uses the last available observation in each series and the observation
    21 trading days prior, aligned by position (not calendar date).
    """
    a_clean = series_a.dropna()
    b_clean = series_b.dropna()
    if len(a_clean) < _RATIO_WINDOW + 1 or len(b_clean) < _RATIO_WINDOW + 1:
        return float("nan")
    a_ret = (float(a_clean.iloc[-1]) / float(a_clean.iloc[-_RATIO_WINDOW - 1]) - 1.0) * 100.0
    b_ret = (float(b_clean.iloc[-1]) / float(b_clean.iloc[-_RATIO_WINDOW - 1]) - 1.0) * 100.0
    return a_ret - b_ret


def _normalize_to_0_100(value: float, series: pd.Series, invert: bool = False) -> float:
    """
    Map a scalar value to [0, 100] based on min/max of series.

    invert=True means low series values map to high stress scores (e.g. falling
    relative return = rising stress).
    """
    clean = series.dropna()
    if clean.empty or len(clean) < 5:
        return float("nan")
    s_min = float(clean.min())
    s_max = float(clean.max())
    if s_max == s_min:
        return 50.0
    norm = (value - s_min) / (s_max - s_min) * 100.0
    if invert:
        norm = 100.0 - norm
    return float(np.clip(norm, 0.0, 100.0))


def _at1_signal_label(ratio_21d_chg: float) -> str:
    """Classify 21-day AT1/HYG ratio change into an AT1 stress signal."""
    if pd.isna(ratio_21d_chg):
        return "Neutral"
    if ratio_21d_chg < _AT1_SEVERE_UNDERPERF * 100:
        return "AT1 Underperformance — Bank Stress"
    if ratio_21d_chg < _AT1_MILD_UNDERPERF * 100:
        return "Mild AT1 Stress"
    if ratio_21d_chg > _AT1_OUTPERF * 100:
        return "AT1 Outperformance — Bank Health"
    return "Neutral"


def _coco_trigger_severity(kbe_from_52w: float) -> str:
    """Map KBE drawdown from 52-week high to a CoCo trigger severity label."""
    if pd.isna(kbe_from_52w):
        return "None"
    if kbe_from_52w <= _TRIGGER_ELEVATED * 100:
        return "Trigger Risk Elevated"
    if kbe_from_52w <= _TRIGGER_NEAR * 100:
        return "Near Trigger Zone"
    return "None"


def _fetch_etf_history() -> dict:
    """
    Download 1-year daily close history for all AT1_ETFS tickers.

    Returns a dict mapping ticker -> pd.Series of closes, or {} on failure.
    """
    try:
        import yfinance as yf
        raw = yf.download(
            AT1_ETFS,
            period=_FETCH_PERIOD,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return {}
        # Handle multi-level columns from yf.download
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw
        closes = closes.dropna(how="all")
        result: dict[str, pd.Series] = {}
        for ticker in AT1_ETFS:
            if ticker in closes.columns:
                s = closes[ticker].dropna()
                if not s.empty:
                    result[ticker] = s
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public: get_at1_snapshot
# ---------------------------------------------------------------------------

def get_at1_snapshot() -> dict:
    """
    Fetch live AT1/bank ETF data via yfinance (period="1y") and compute
    real-time AT1 stress metrics.

    Returns
    -------
    dict with keys:
        available             : bool
        at1_etf               : str or None   — which AT1 ETF found (CSHI/AT1/None)
        at1_price             : float or None
        hyg_price             : float
        kbe_price             : float
        kre_price             : float
        at1_hyg_ratio         : float or None — current AT1/HYG price ratio
        at1_hyg_21d_chg       : float or None — % change in ratio over 21 days
        kbe_spy_21d_rel       : float or None — KBE 21d return minus SPY 21d return
        kre_spy_21d_rel       : float or None — KRE 21d return minus SPY 21d return
        bank_stress_score     : float         — 0-100 composite (100 = max stress)
        at1_signal            : str
        coco_trigger_flag     : bool          — True if KBE > 15% below 52w high
        coco_trigger_severity : str           — "None" / "Near Trigger Zone" / "Trigger Risk Elevated"
        kbe_from_52w_high     : float         — % below 52w high (negative = below)
        as_of                 : str           — ISO timestamp
    """
    _unavail: dict = {
        "available": False,
        "at1_etf": None,
        "at1_price": None,
        "hyg_price": float("nan"),
        "kbe_price": float("nan"),
        "kre_price": float("nan"),
        "at1_hyg_ratio": None,
        "at1_hyg_21d_chg": None,
        "kbe_spy_21d_rel": None,
        "kre_spy_21d_rel": None,
        "bank_stress_score": 50.0,
        "at1_signal": "Neutral",
        "coco_trigger_flag": False,
        "coco_trigger_severity": "None",
        "kbe_from_52w_high": float("nan"),
        "as_of": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    }

    try:
        etf_data = _fetch_etf_history()
        if not etf_data:
            return _unavail

        as_of = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        # ---- Identify which AT1 ETF is available ----------------------------
        at1_etf: str | None = None
        at1_series: pd.Series | None = None
        for candidate in _AT1_TICKER_PREFERENCE:
            if candidate in etf_data and len(etf_data[candidate]) >= _RATIO_WINDOW + 1:
                at1_etf = candidate
                at1_series = etf_data[candidate]
                break

        hyg_series = etf_data.get("HYG")
        kbe_series = etf_data.get("KBE")
        kre_series = etf_data.get("KRE")
        spy_series = etf_data.get("SPY")
        xlf_series = etf_data.get("XLF")

        # ---- Latest prices --------------------------------------------------
        def _last(s: pd.Series | None) -> float:
            if s is None or s.empty:
                return float("nan")
            return float(s.iloc[-1])

        at1_price = _last(at1_series)
        hyg_price  = _last(hyg_series)
        kbe_price  = _last(kbe_series)
        kre_price  = _last(kre_series)

        # ---- AT1/HYG ratio --------------------------------------------------
        at1_hyg_ratio: float | None = None
        at1_hyg_21d_chg: float | None = None

        if at1_series is not None and hyg_series is not None:
            # Align on index
            aligned = pd.DataFrame({
                "at1": at1_series,
                "hyg": hyg_series,
            }).dropna()
            if len(aligned) >= _RATIO_WINDOW + 1:
                ratio = aligned["at1"] / aligned["hyg"]
                at1_hyg_ratio = float(ratio.iloc[-1])
                at1_hyg_21d_chg = _pct_change_21d(ratio)

        # ---- AT1 signal label -----------------------------------------------
        at1_signal = _at1_signal_label(at1_hyg_21d_chg if at1_hyg_21d_chg is not None else float("nan"))

        # ---- Bank equity relative returns -----------------------------------
        kbe_spy_21d_rel: float | None = None
        kre_spy_21d_rel: float | None = None

        if kbe_series is not None and spy_series is not None:
            rel = _relative_return_21d(kbe_series, spy_series)
            kbe_spy_21d_rel = rel if not np.isnan(rel) else None

        if kre_series is not None and spy_series is not None:
            rel = _relative_return_21d(kre_series, spy_series)
            kre_spy_21d_rel = rel if not np.isnan(rel) else None

        # ---- Bank stress composite (0-100) ----------------------------------
        # Each component: falling relative return = higher stress (invert=True)
        # We use the 21d relative return history to build normalisation ranges,
        # then take the latest normalised value.
        stress_components: list[float] = []

        def _rel_return_series(s_a: pd.Series, s_b: pd.Series) -> pd.Series:
            """Rolling 21d relative return of a vs b as a series."""
            aligned = pd.DataFrame({"a": s_a, "b": s_b}).dropna()
            if len(aligned) < _RATIO_WINDOW + 2:
                return pd.Series(dtype=float)
            a_ret = aligned["a"].pct_change(_RATIO_WINDOW) * 100.0
            b_ret = aligned["b"].pct_change(_RATIO_WINDOW) * 100.0
            return (a_ret - b_ret).dropna()

        if kbe_series is not None and spy_series is not None and kbe_spy_21d_rel is not None:
            rel_hist = _rel_return_series(kbe_series, spy_series)
            if not rel_hist.empty:
                score = _normalize_to_0_100(kbe_spy_21d_rel, rel_hist, invert=True)
                if not np.isnan(score):
                    stress_components.append(score)

        if kre_series is not None and spy_series is not None and kre_spy_21d_rel is not None:
            rel_hist = _rel_return_series(kre_series, spy_series)
            if not rel_hist.empty:
                score = _normalize_to_0_100(kre_spy_21d_rel, rel_hist, invert=True)
                if not np.isnan(score):
                    stress_components.append(score)

        if at1_series is not None and hyg_series is not None and at1_hyg_21d_chg is not None:
            at1_hyg_aligned = pd.DataFrame({
                "at1": at1_series, "hyg": hyg_series
            }).dropna()
            if len(at1_hyg_aligned) >= _RATIO_WINDOW + 2:
                ratio_hist = (at1_hyg_aligned["at1"] / at1_hyg_aligned["hyg"])
                ratio_ret_hist = ratio_hist.pct_change(_RATIO_WINDOW).dropna() * 100.0
                if not ratio_ret_hist.empty:
                    score = _normalize_to_0_100(at1_hyg_21d_chg, ratio_ret_hist, invert=True)
                    if not np.isnan(score):
                        stress_components.append(score)

        # Fallback: XLF/SPY if no components resolved
        if not stress_components and xlf_series is not None and spy_series is not None:
            rel = _relative_return_21d(xlf_series, spy_series)
            if not np.isnan(rel):
                xlf_rel_hist = _rel_return_series(xlf_series, spy_series)
                if not xlf_rel_hist.empty:
                    score = _normalize_to_0_100(rel, xlf_rel_hist, invert=True)
                    if not np.isnan(score):
                        stress_components.append(score)

        bank_stress_score = float(np.mean(stress_components)) if stress_components else 50.0

        # ---- CoCo trigger distance (KBE from 52-week high) ------------------
        kbe_from_52w_high = float("nan")
        coco_trigger_flag = False
        coco_trigger_severity = "None"

        if kbe_series is not None and len(kbe_series) >= 10:
            lookback = min(_52W_HIGH_WINDOW, len(kbe_series))
            kbe_52w_high = float(kbe_series.iloc[-lookback:].max())
            kbe_current = float(kbe_series.iloc[-1])
            if kbe_52w_high > 0:
                kbe_from_52w_high = (kbe_current / kbe_52w_high - 1.0) * 100.0
                coco_trigger_flag = kbe_from_52w_high <= _TRIGGER_NEAR * 100
                coco_trigger_severity = _coco_trigger_severity(kbe_from_52w_high)

        return {
            "available": True,
            "at1_etf": at1_etf,
            "at1_price": at1_price if not np.isnan(at1_price) else None,
            "hyg_price": hyg_price,
            "kbe_price": kbe_price,
            "kre_price": kre_price,
            "at1_hyg_ratio": at1_hyg_ratio,
            "at1_hyg_21d_chg": at1_hyg_21d_chg,
            "kbe_spy_21d_rel": kbe_spy_21d_rel,
            "kre_spy_21d_rel": kre_spy_21d_rel,
            "bank_stress_score": round(bank_stress_score, 1),
            "at1_signal": at1_signal,
            "coco_trigger_flag": coco_trigger_flag,
            "coco_trigger_severity": coco_trigger_severity,
            "kbe_from_52w_high": kbe_from_52w_high,
            "as_of": as_of,
        }

    except Exception:
        return _unavail


# ---------------------------------------------------------------------------
# Public: run_at1_coco_monitor
# ---------------------------------------------------------------------------

def run_at1_coco_monitor(df: pd.DataFrame) -> dict:
    """
    Top-level AT1/CoCo analysis entry point.

    Combines a live yfinance snapshot with any bank/AT1 columns present in
    the master DataFrame. Columns checked in df (by name, case-insensitive):
      - xlf, kbe        : bank equity proxies
      - banking_stress_score_smooth : smoothed composite banking stress score

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex. No "date" column assumed.

    Returns
    -------
    dict with keys:
        available                : bool
        snapshot                 : dict  — from get_at1_snapshot()
        historical_banking_stress: pd.Series or None
                                   banking_stress_score_smooth last 504 rows
        lead_signal              : str  — "Bank Stress" / "Neutral" / "Bank Health"
        systemic_risk_flag       : bool — True if coco_trigger_flag AND AT1 signal
                                          contains "Stress"
        interpretation           : str
        warning                  : str or None
    """
    _unavail: dict = {"available": False}

    try:
        # ---- Live snapshot --------------------------------------------------
        snapshot = get_at1_snapshot()

        # ---- Historical banking stress from df ------------------------------
        df_cols_lower = {c.lower(): c for c in df.columns}

        historical_banking_stress: pd.Series | None = None
        stress_col_key = "banking_stress_score_smooth"
        if stress_col_key in df_cols_lower:
            actual_col = df_cols_lower[stress_col_key]
            historical_banking_stress = df[actual_col].tail(_HISTORY_ROWS).copy()
        # If smoothed not available, fall back to any banking_stress column
        elif "banking_stress_score" in df_cols_lower:
            actual_col = df_cols_lower["banking_stress_score"]
            historical_banking_stress = df[actual_col].tail(_HISTORY_ROWS).copy()

        # ---- Lead signal (synthesise from snapshot + df) --------------------
        at1_signal = snapshot.get("at1_signal", "Neutral") if snapshot.get("available") else "Neutral"

        # Derive lead signal bucket from AT1 signal
        if "Stress" in at1_signal:
            lead_signal = "Bank Stress"
        elif "Outperformance" in at1_signal or "Health" in at1_signal:
            lead_signal = "Bank Health"
        else:
            lead_signal = "Neutral"

        # ---- Systemic risk flag ---------------------------------------------
        coco_trigger_flag = snapshot.get("coco_trigger_flag", False) if snapshot.get("available") else False
        systemic_risk_flag = bool(coco_trigger_flag and "Stress" in at1_signal)

        # ---- Interpretation -------------------------------------------------
        if not snapshot.get("available"):
            interpretation = (
                "Live AT1/CoCo data unavailable (yfinance fetch failed). "
                "Bank stress cannot be assessed in real time. "
                "Check df for historical banking_stress_score_smooth if available."
            )
        else:
            at1_etf = snapshot.get("at1_etf")
            bank_score = snapshot.get("bank_stress_score", 50.0)
            kbe_from_high = snapshot.get("kbe_from_52w_high", float("nan"))
            coco_severity = snapshot.get("coco_trigger_severity", "None")

            if at1_etf is not None:
                at1_detail = (
                    f"The {at1_etf}/HYG ratio is "
                    f"{'falling' if (snapshot.get('at1_hyg_21d_chg') or 0) < 0 else 'rising'} "
                    f"({snapshot.get('at1_hyg_21d_chg', 0):.1f}% over 21 days), signalling {at1_signal}. "
                )
            else:
                at1_detail = "No direct AT1 ETF data available; bank stress is proxied via equity ratios. "

            kbe_detail = (
                f"KBE is {abs(kbe_from_high):.1f}% below its 52-week high "
                f"(trigger severity: {coco_severity}). "
                if not np.isnan(kbe_from_high)
                else ""
            )

            interpretation = (
                f"{at1_detail}"
                f"{kbe_detail}"
                f"Bank stress composite: {bank_score:.0f}/100. "
                f"Systemic risk flag: {'ACTIVE' if systemic_risk_flag else 'inactive'}."
            )

        # ---- Warning --------------------------------------------------------
        warning: str | None = None

        if snapshot.get("available") and snapshot.get("at1_etf") is None:
            warning = (
                "Neither CSHI nor AT1 ETF returned data from yfinance. "
                "Bank stress composite is derived from KBE/KRE vs SPY only — "
                "no direct AT1/CoCo price signal available."
            )

        if systemic_risk_flag:
            trigger_warn = (
                "SYSTEMIC RISK FLAG: CoCo trigger zone breach (KBE) coincides with "
                "AT1 underperformance. This combination preceded the CS AT1 event. "
                "Monitor bank CET1 disclosures and ECB/Fed supervisory statements."
            )
            warning = f"{warning}  {trigger_warn}" if warning else trigger_warn

        return {
            "available": True,
            "snapshot": snapshot,
            "historical_banking_stress": historical_banking_stress,
            "lead_signal": lead_signal,
            "systemic_risk_flag": systemic_risk_flag,
            "interpretation": interpretation,
            "warning": warning,
        }

    except Exception:
        return _unavail
