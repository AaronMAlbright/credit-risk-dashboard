"""
Term premium decomposition — splits the 10y Treasury yield into:
  1. Expected short rate  (where markets expect short rates to average over 10y)
  2. Term premium         (extra yield demanded for bearing duration/uncertainty risk)

Why this matters for credit:
  - Rising yield from rising term premium → risk-off (bad for IG duration, bad for credit)
  - Rising yield from rising rate expectations → growth optimism (mildly positive for HY)
  - Term premium near zero or negative = Fed-suppressed rates = artificially easy conditions
  - Term premium spike = rate volatility regime = credit spread widening risk

Methodology (pure numpy — no ACM Kalman filter required):
  expected_rate_10y = 0.3 * yield_3m + 0.7 * yield_2y
  term_premium      = yield_10y - expected_rate_10y
  Rolling z-score (252d) → percentile rank → 0-100 score

FRED fallback series: DGS10, DGS2, DTB3

Public API
----------
  TERM_PREMIUM_REGIMES      — module constant
  compute_term_premium(df)  -> pd.DataFrame
  get_current_term_premium(df) -> dict
  run_term_premium_analysis(df) -> dict
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERM_PREMIUM_REGIMES: dict[str, tuple[float, float]] = {
    "Suppressed":  (-99.0,  0.0),
    "Normal":      (  0.0,  1.5),
    "Elevated":    (  1.5,  2.5),
    "Very High":   (  2.5, 99.0),
}

_ZSCORE_WINDOW = 252
_MIN_PERIODS   = 63
_HIST_ROWS     = 504

# Weights for expected short rate blend
_W_3M = 0.30
_W_2Y = 0.70

# FRED series IDs
_FRED_10Y = "DGS10"
_FRED_2Y  = "DGS2"
_FRED_3M  = "DTB3"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(s: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    mu  = s.rolling(window, min_periods=_MIN_PERIODS).mean()
    sig = s.rolling(window, min_periods=_MIN_PERIODS).std(ddof=1)
    return (s - mu) / sig.replace(0.0, np.nan)


def _percentile_rank(s: pd.Series) -> pd.Series:
    out   = pd.Series(np.nan, index=s.index, dtype=float)
    s_arr = s.values
    for i in range(len(s_arr)):
        if np.isnan(s_arr[i]):
            continue
        window_vals = s_arr[: i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) < 2:
            continue
        rank = float(np.sum(valid <= s_arr[i]) - 1) / float(len(valid) - 1) * 100.0
        out.iloc[i] = float(np.clip(rank, 0.0, 100.0))
    return out


def _fetch_fred_series(series_id: str, start: str = "2000-01-01") -> pd.Series | None:
    try:
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            return None
        from fredapi import Fred  # type: ignore
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id, observation_start=start)
        if data is None or len(data) == 0:
            return None
        s = pd.Series(data, dtype=float)
        s.index = pd.to_datetime(s.index)
        return s.dropna()
    except Exception:
        return None


def _resolve_yield(df: pd.DataFrame, col_name: str, fred_id: str) -> pd.Series | None:
    if col_name in df.columns:
        s = df[col_name].dropna()
        if not s.empty:
            return df[col_name]
    fred_s = _fetch_fred_series(fred_id)
    if fred_s is not None and not fred_s.empty:
        return fred_s.reindex(df.index).ffill()
    return None


def _tp_regime(tp: float) -> str:
    if np.isnan(tp):
        return "Unknown"
    if tp < 0.0:
        return "Suppressed"
    if tp < 1.5:
        return "Normal"
    if tp < 2.5:
        return "Elevated"
    return "Very High"


def _credit_implication_text(regime: str, direction: str, score: float) -> str:
    base: dict[str, str] = {
        "Suppressed": (
            "Term premium is negative — rates are Fed-suppressed. Duration risk appears "
            "artificially low, which rewards carry but embeds mean-reversion risk. Credit "
            "spreads are likely compressed; any TP normalization would widen IG spreads "
            "faster than HY."
        ),
        "Normal": (
            "Term premium is in the normal range. Duration risk is fairly priced; credit "
            "markets should trade on fundamentals rather than rate distortion. Neutral for "
            "spread direction."
        ),
        "Elevated": (
            "Term premium is elevated — investors are demanding extra compensation for "
            "duration uncertainty. Typically negative for IG (duration drag) but can be "
            "neutral-to-positive for HY if elevation reflects growth expectations rather "
            "than inflation fear."
        ),
        "Very High": (
            "Term premium is at historically high levels (1990s / early 2000s analog). "
            "This signals a rate volatility regime that historically precedes IG spread "
            "widening and reduces risk appetite across credit. HY is vulnerable if equity "
            "volatility rises concurrently."
        ),
    }
    txt = base.get(regime, "Term premium regime unknown.")
    if direction == "Rising":
        txt += " Rising term premium is the more credit-negative scenario — it reflects "
        txt += "uncertainty rather than growth, compressing risk appetite."
    elif direction == "Falling":
        txt += " Falling term premium is a near-term tailwind for IG duration."
    return txt


def _rolling_r2(x: pd.Series, y: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """Rolling R² proxy: (rolling corr(x,y))²."""
    corr = x.rolling(window, min_periods=_MIN_PERIODS).corr(y)
    return corr ** 2


# ---------------------------------------------------------------------------
# Public: compute_term_premium
# ---------------------------------------------------------------------------

def compute_term_premium(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a copy of df with term_premium_* columns.

    New columns
    -----------
    tp_yield_10y           : 10y yield (sourced from df or FRED)
    tp_yield_2y            : 2y yield (sourced from df or FRED)
    tp_yield_3m            : 3m yield (sourced from df or FRED)
    tp_expected_rate       : blended expected short rate (0.3*3m + 0.7*2y)
    tp_term_premium        : yield_10y - expected_rate (%)
    tp_zscore              : rolling 252d z-score of term_premium
    tp_score               : 0-100 percentile rank (higher = more elevated)
    tp_regime              : str regime label
    tp_direction           : "Rising" / "Falling" / "Stable"
    tp_duration_risk_score : 0-100 duration risk
    tp_r2_vs_yield         : rolling R² of TP changes vs 10y yield changes
    """
    out = df.copy()

    try:
        y10 = _resolve_yield(out, "yield_10y", _FRED_10Y)
        y2  = _resolve_yield(out, "yield_2y",  _FRED_2Y)
        y3m = _resolve_yield(out, "yield_3m",  _FRED_3M)

        if y10 is None:
            for col in (
                "tp_yield_10y", "tp_yield_2y", "tp_yield_3m",
                "tp_expected_rate", "tp_term_premium", "tp_zscore",
                "tp_score", "tp_regime", "tp_direction",
                "tp_duration_risk_score", "tp_r2_vs_yield",
            ):
                out[col] = np.nan
            out["tp_regime"]    = "Unknown"
            out["tp_direction"] = "Unknown"
            return out

        out["tp_yield_10y"] = y10

        # Build expected rate from best available inputs
        if y2 is not None and y3m is not None:
            out["tp_yield_2y"]      = y2
            out["tp_yield_3m"]      = y3m
            expected = _W_3M * y3m + _W_2Y * y2
        elif y2 is not None:
            out["tp_yield_2y"] = y2
            out["tp_yield_3m"] = np.nan
            expected = y2
        elif y3m is not None:
            out["tp_yield_2y"] = np.nan
            out["tp_yield_3m"] = y3m
            # rolling mean as long-run expectation proxy
            expected = y3m.rolling(_ZSCORE_WINDOW, min_periods=_MIN_PERIODS).mean()
        else:
            # last-resort: rolling mean of the 10y itself
            expected = y10.rolling(_ZSCORE_WINDOW, min_periods=_MIN_PERIODS).mean()
            out["tp_yield_2y"] = np.nan
            out["tp_yield_3m"] = np.nan

        out["tp_expected_rate"] = expected
        out["tp_term_premium"]  = y10 - expected

        tp = out["tp_term_premium"]

        out["tp_zscore"] = _rolling_zscore(tp)
        out["tp_score"]  = _percentile_rank(tp.dropna()).reindex(out.index)

        out["tp_regime"] = tp.apply(
            lambda v: _tp_regime(float(v)) if pd.notna(v) else "Unknown"
        )

        # Direction: 21d change
        tp_21d_chg = tp - tp.shift(21)
        direction_s = pd.Series("Stable", index=out.index, dtype=object)
        direction_s[tp_21d_chg >  0.10] = "Rising"
        direction_s[tp_21d_chg < -0.10] = "Falling"
        out["tp_direction"] = direction_s

        # Duration risk score: full when TP rising, dampened when falling
        dur_raw = out["tp_score"].copy()
        falling_mask = out["tp_direction"] == "Falling"
        dur_raw[falling_mask] = dur_raw[falling_mask] * 0.5
        out["tp_duration_risk_score"] = dur_raw.clip(0.0, 100.0)

        # Rolling R² of TP changes vs 10y yield changes
        tp_chg    = tp.diff()
        y10_chg   = y10.diff()
        out["tp_r2_vs_yield"] = _rolling_r2(tp_chg, y10_chg)

    except Exception:
        for col in (
            "tp_yield_10y", "tp_yield_2y", "tp_yield_3m",
            "tp_expected_rate", "tp_term_premium", "tp_zscore",
            "tp_score", "tp_regime", "tp_direction",
            "tp_duration_risk_score", "tp_r2_vs_yield",
        ):
            if col not in out.columns:
                out[col] = np.nan
        for col in ("tp_regime", "tp_direction"):
            if col in out.columns:
                out[col] = out[col].fillna("Unknown")
            else:
                out[col] = "Unknown"

    return out


# ---------------------------------------------------------------------------
# Public: get_current_term_premium
# ---------------------------------------------------------------------------

def get_current_term_premium(df: pd.DataFrame) -> dict:
    """Return current term premium state.

    Returns
    -------
    dict with keys:
        available, term_premium, expected_short_rate, yield_10y,
        term_premium_score, term_premium_regime, term_premium_direction,
        credit_implication, duration_risk_score, warning
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        enriched = compute_term_premium(df)

        tp_valid = enriched["tp_term_premium"].dropna() if "tp_term_premium" in enriched.columns else pd.Series(dtype=float)
        if tp_valid.empty:
            return {"available": False}

        last = enriched.iloc[-1]

        def _fv(col: str) -> float | None:
            if col not in enriched.columns:
                return None
            v = last[col]
            try:
                fv = float(v)
                return None if np.isnan(fv) else fv
            except Exception:
                return None

        tp          = _fv("tp_term_premium")
        exp_rate    = _fv("tp_expected_rate")
        y10         = _fv("tp_yield_10y")
        tp_score    = _fv("tp_score")
        dur_score   = _fv("tp_duration_risk_score")

        if tp is None or tp_score is None:
            return {"available": False}

        regime    = str(last.get("tp_regime",    "Unknown"))
        direction = str(last.get("tp_direction", "Unknown"))

        credit_impl = _credit_implication_text(regime, direction, tp_score)

        # Warning: TP has spiked > 0.5% in 21 days
        warning: str | None = None
        tp_series = enriched["tp_term_premium"].dropna()
        if len(tp_series) >= 22:
            tp_21d_chg = float(tp_series.iloc[-1]) - float(tp_series.iloc[-22])
            if tp_21d_chg > 0.50:
                warning = (
                    f"Term premium has risen {tp_21d_chg * 100:.0f} bps over the past 21 days "
                    "— a rapid spike that historically precedes credit spread widening and "
                    "elevated rate volatility. Duration risk is elevated."
                )

        return {
            "available":               True,
            "term_premium":            round(tp, 4),
            "expected_short_rate":     round(exp_rate, 4) if exp_rate is not None else None,
            "yield_10y":               round(y10, 4) if y10 is not None else None,
            "term_premium_score":      round(tp_score, 1),
            "term_premium_regime":     regime,
            "term_premium_direction":  direction,
            "credit_implication":      credit_impl,
            "duration_risk_score":     round(dur_score, 1) if dur_score is not None else None,
            "warning":                 warning,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Public: run_term_premium_analysis
# ---------------------------------------------------------------------------

def run_term_premium_analysis(df: pd.DataFrame) -> dict:
    """Top-level term premium analysis entry point.

    Returns
    -------
    dict with keys:
        available, current, df, term_premium_history, expected_rate_history,
        term_premium_score_history, regime_history, tp_fraction_of_yield_move
    """
    _unavail: dict = {"available": False}

    try:
        if df is None or df.empty:
            return _unavail

        enriched = compute_term_premium(df)
        current  = get_current_term_premium(df)

        if not current.get("available"):
            return _unavail

        def _hist(col: str) -> pd.Series:
            if col not in enriched.columns:
                return pd.Series(dtype=float)
            return enriched[col].tail(_HIST_ROWS).copy()

        term_premium_history       = _hist("tp_term_premium")
        expected_rate_history      = _hist("tp_expected_rate")
        term_premium_score_history = _hist("tp_score")
        regime_history             = _hist("tp_regime")

        # tp_fraction_of_yield_move: last available rolling R² value
        tp_fraction = float("nan")
        if "tp_r2_vs_yield" in enriched.columns:
            r2_valid = enriched["tp_r2_vs_yield"].dropna()
            if not r2_valid.empty:
                tp_fraction = round(float(r2_valid.iloc[-1]), 4)

        return {
            "available":                   True,
            "current":                     current,
            "df":                          enriched,
            "term_premium_history":        term_premium_history,
            "expected_rate_history":       expected_rate_history,
            "term_premium_score_history":  term_premium_score_history,
            "regime_history":              regime_history,
            "tp_fraction_of_yield_move":   tp_fraction,
        }

    except Exception:
        return _unavail
