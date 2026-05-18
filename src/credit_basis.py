"""
CDX / Cash Credit Basis Monitor.

The credit basis = CDS_spread - Cash_spread (in bps equivalent).

  Positive basis : CDS more expensive than cash bonds — normal.
                   Institutional buyers of protection > cash sellers.
  Negative basis : Cash bonds more expensive to protect.
                   Signals forced cash-bond selling / real-money deleveraging.
                   This is the most reliable "institutional stress" signal.

Because live CDX data is not freely available, this module builds a proxy
from ETF prices and the existing spread columns in the master DataFrame.

Priority stack for basis construction:
  1. "cdx_hy" / "cdx_ig" columns  →  direct basis vs hy_spread / ig_spread
  2. "cdx_basis" column            →  used directly
  3. ETF NAV discount proxy        →  yfinance HYG / JNK / BKLN return differential

Basis signal labels (bps equivalent):
  < -20 bps  →  "Negative Basis (Forced Selling)"
  -20 to -5  →  "Compressed Basis"
   -5 to +15 →  "Normal"
      > +15   →  "Rich Basis"

Public API
----------
  get_basis_snapshot()            -> dict
  compute_historical_basis(df)    -> pd.DataFrame
  run_credit_basis(df)            -> dict
"""
from __future__ import annotations

import datetime
import functools

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIST_ROWS = 504          # ~2 trading years of history to surface
_ZSCORE_WINDOW = 252
_MIN_PERIODS = 63
_LIVE_PERIOD = "1y"       # yfinance download period

# Basis signal thresholds (bps equivalent)
_NEG_FORCED   = -20.0
_NEG_COMPRESS =  -5.0
_RICH         =  15.0

# HYG approximate spread duration (years) — used to convert price return → bps
_HYG_DURATION = 4.0
# LQD approximate spread duration
_LQD_DURATION = 7.0

# BKLN (loans) is floating-rate, near-zero spread duration — 21-day return
# differential vs HYG captures basis between secured loans and HY bonds.
_BKLN_WINDOW = 21


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _basis_signal(basis_bps: float) -> str:
    """Categorical label for a basis level in bps equivalent."""
    if basis_bps < _NEG_FORCED:
        return "Negative Basis (Forced Selling)"
    if basis_bps < _NEG_COMPRESS:
        return "Compressed Basis"
    if basis_bps <= _RICH:
        return "Normal"
    return "Rich Basis"


def _historical_percentile(series: pd.Series, current_val: float) -> float:
    """Fraction of history at or below current_val (0–100 scale)."""
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float((clean <= current_val).mean() * 100.0)


def _rolling_zscore(s: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    rm = s.rolling(window=window, min_periods=_MIN_PERIODS).mean()
    rs = s.rolling(window=window, min_periods=_MIN_PERIODS).std(ddof=1)
    return ((s - rm) / rs.replace(0, np.nan)).clip(-3.0, 3.0)


def _price_series_to_implied_spread_change(
    prices: pd.Series, duration: float
) -> pd.Series:
    """
    Convert ETF price daily log-returns into implied spread changes (bps).

    P_return ≈ -duration × Δspread  →  Δspread ≈ -P_return / duration
    Returns cumulative implied spread change from the first observation (bps).
    """
    log_ret = np.log(prices / prices.iloc[0])
    implied_spread_chg = -log_ret / duration * 10_000  # bps
    return implied_spread_chg


# ---------------------------------------------------------------------------
# Live yfinance fetch
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _cached_etf_history(date_key: str) -> dict:  # noqa: ARG001
    """Download 1-year daily closes for HYG, JNK, LQD, BKLN. Cached per day."""
    try:
        import yfinance as yf
        tickers = ["HYG", "JNK", "LQD", "BKLN"]
        raw = yf.download(tickers, period=_LIVE_PERIOD, auto_adjust=True, progress=False)
        if raw.empty:
            return {"available": False}
        closes = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
        closes = closes.dropna(how="all")
        result: dict = {"available": True, "closes": closes}
        return result
    except Exception:
        return {"available": False}


def _get_etf_closes() -> dict:
    date_key = datetime.date.today().isoformat()
    return _cached_etf_history(date_key)


# ---------------------------------------------------------------------------
# Public: get_basis_snapshot
# ---------------------------------------------------------------------------

def get_basis_snapshot() -> dict:
    """
    Fetch live ETF data and compute a credit basis proxy.

    Returns
    -------
    dict with keys:
        available         : bool
        hyg_price         : float
        jnk_price         : float
        lqd_price         : float
        hyg_jnk_basis     : float   — spread between HYG and JNK implied yield
                                       change over the last 21 days (bps proxy)
        bkln_hyg_basis    : float   — BKLN vs HYG cumulative return differential
                                       over 21 days, converted to bps
        basis_signal      : str
        stress_flag       : bool    — True if signal is "Negative Basis"
        as_of             : str
    """
    _unavail: dict = {
        "available": False,
        "hyg_price": float("nan"),
        "jnk_price": float("nan"),
        "lqd_price": float("nan"),
        "hyg_jnk_basis": float("nan"),
        "bkln_hyg_basis": float("nan"),
        "basis_signal": "Unknown",
        "stress_flag": False,
        "as_of": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    }

    try:
        data = _get_etf_closes()
        if not data.get("available"):
            return _unavail

        closes: pd.DataFrame = data["closes"]
        required = {"HYG", "JNK"}
        if not required.issubset(set(closes.columns)):
            return _unavail
        if len(closes) < _BKLN_WINDOW + 1:
            return _unavail

        # Latest prices
        last_row = closes.iloc[-1]
        hyg_price = float(last_row.get("HYG", float("nan")))
        jnk_price = float(last_row.get("JNK", float("nan")))
        lqd_price = float(last_row.get("LQD", float("nan"))) if "LQD" in closes.columns else float("nan")

        # HYG vs JNK implied spread basis (21d)
        # Both are HY vehicles but different compositions; price divergence ≈ basis
        window_data = closes.tail(_BKLN_WINDOW + 1)
        hyg_ret_21d = float(window_data["HYG"].iloc[-1] / window_data["HYG"].iloc[0] - 1)
        jnk_ret_21d = float(window_data["JNK"].iloc[-1] / window_data["JNK"].iloc[0] - 1)
        # Convert to implied spread change in bps: -return / duration × 10000
        hyg_impl_spread_chg = -hyg_ret_21d / _HYG_DURATION * 10_000
        jnk_impl_spread_chg = -jnk_ret_21d / _HYG_DURATION * 10_000
        hyg_jnk_basis = float(hyg_impl_spread_chg - jnk_impl_spread_chg)

        # BKLN vs HYG basis (21d cumulative return differential, bps)
        # BKLN is floating-rate loans; HYG is fixed-rate bonds.
        # When BKLN underperforms HYG by more than carry, cash bonds are being sold.
        bkln_hyg_basis = float("nan")
        if "BKLN" in closes.columns:
            bkln_ret_21d = float(window_data["BKLN"].iloc[-1] / window_data["BKLN"].iloc[0] - 1)
            # Positive = BKLN outperformed HYG → loans stable, bonds cheap → negative basis signal
            bkln_hyg_raw = (bkln_ret_21d - hyg_ret_21d)
            # Scale to bps: use HYG duration as reference
            bkln_hyg_basis = float(-bkln_hyg_raw / _HYG_DURATION * 10_000)

        # Primary basis signal from bkln_hyg_basis if available, else hyg_jnk_basis
        primary_basis = bkln_hyg_basis if not np.isnan(bkln_hyg_basis) else hyg_jnk_basis
        signal = _basis_signal(primary_basis)
        stress_flag = signal == "Negative Basis (Forced Selling)"

        return {
            "available": True,
            "hyg_price": round(hyg_price, 4) if not np.isnan(hyg_price) else float("nan"),
            "jnk_price": round(jnk_price, 4) if not np.isnan(jnk_price) else float("nan"),
            "lqd_price": round(lqd_price, 4) if not np.isnan(lqd_price) else float("nan"),
            "hyg_jnk_basis": round(hyg_jnk_basis, 2) if not np.isnan(hyg_jnk_basis) else float("nan"),
            "bkln_hyg_basis": round(bkln_hyg_basis, 2) if not np.isnan(bkln_hyg_basis) else float("nan"),
            "basis_signal": signal,
            "stress_flag": bool(stress_flag),
            "as_of": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }
    except Exception:
        return _unavail


# ---------------------------------------------------------------------------
# Public: compute_historical_basis
# ---------------------------------------------------------------------------

def compute_historical_basis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute credit basis from the master DataFrame columns.

    Priority:
      1. cdx_hy / hy_spread  →  direct basis_cdx_hy = cdx_hy - hy_spread
      2. cdx_ig / ig_spread  →  direct basis_cdx_ig = cdx_ig - ig_spread
      3. cdx_basis           →  used as-is
      4. Proxy: (hy_spread - hy_spread.rolling(252).mean()) - analogous IG term,
         re-centred so the long-run mean is zero (structural basis proxy).

    Appended columns:
      basis_cdx_hy      — HY CDX basis in bps (NaN if not computable)
      basis_cdx_ig      — IG CDX basis in bps (NaN if not computable)
      basis_primary     — best available basis estimate (bps equivalent)
      basis_signal      — categorical label
      basis_stress_flag — bool

    Returns a copy of df.
    """
    out = df.copy()

    # ---- 1. Direct CDX-based basis -----------------------------------------
    basis_hy = pd.Series(float("nan"), index=out.index, name="basis_cdx_hy")
    basis_ig = pd.Series(float("nan"), index=out.index, name="basis_cdx_ig")

    if "cdx_hy" in out.columns and "hy_spread" in out.columns:
        basis_hy = out["cdx_hy"] - out["hy_spread"]
    if "cdx_ig" in out.columns and "ig_spread" in out.columns:
        basis_ig = out["cdx_ig"] - out["ig_spread"]

    out["basis_cdx_hy"] = basis_hy
    out["basis_cdx_ig"] = basis_ig

    # ---- 2. Direct cdx_basis column ----------------------------------------
    if "cdx_basis" in out.columns:
        out["basis_primary"] = out["cdx_basis"].copy()

    # ---- 3. Prefer direct HY basis when available --------------------------
    elif not basis_hy.isna().all():
        out["basis_primary"] = basis_hy

    # ---- 4. Fall back to spread-ratio proxy --------------------------------
    else:
        if "hy_spread" in out.columns and "ig_spread" in out.columns:
            # Normalise each spread vs its own 252d rolling mean, then difference.
            # The idea: if HY richens relative to IG faster than normal, cash
            # bonds are being bid (positive basis). If HY cheapens relative to
            # IG, forced selling is present (negative basis).
            hy_z = _rolling_zscore(out["hy_spread"])
            ig_z = _rolling_zscore(out["ig_spread"])
            # Convert z-score differential to bps-equivalent by scaling to
            # the 252d rolling std of hy_spread (mean bps per z-score unit).
            hy_std_bps = (
                out["hy_spread"]
                .rolling(window=_ZSCORE_WINDOW, min_periods=_MIN_PERIODS)
                .std(ddof=1)
            )
            proxy_raw = (hy_z - ig_z) * hy_std_bps * 0.5
            # Re-centre so long-run mean is zero
            out["basis_primary"] = proxy_raw - proxy_raw.rolling(
                window=_ZSCORE_WINDOW, min_periods=_MIN_PERIODS
            ).mean()
        elif "hy_spread" in out.columns:
            # Single-asset proxy: deviation from long-run mean, sign-inverted
            hy_z = _rolling_zscore(out["hy_spread"])
            hy_std_bps = (
                out["hy_spread"]
                .rolling(window=_ZSCORE_WINDOW, min_periods=_MIN_PERIODS)
                .std(ddof=1)
            )
            out["basis_primary"] = -hy_z * hy_std_bps * 0.3
        else:
            out["basis_primary"] = float("nan")

    # ---- Signals -----------------------------------------------------------
    out["basis_signal"] = out["basis_primary"].apply(
        lambda v: _basis_signal(v) if not pd.isna(v) else "Unknown"
    )
    out["basis_stress_flag"] = out["basis_signal"] == "Negative Basis (Forced Selling)"

    return out


# ---------------------------------------------------------------------------
# Public: run_credit_basis
# ---------------------------------------------------------------------------

def run_credit_basis(df: pd.DataFrame) -> dict:
    """
    Top-level entry point for the credit basis module.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        available           : bool
        snapshot            : dict   from get_basis_snapshot()
        historical          : pd.DataFrame  last 504 rows of enriched df
        signal_series       : pd.Series     basis_primary last 504 rows
        current_basis       : float  live if snapshot available, else historical
        basis_signal        : str
        stress_flag         : bool
        percentile_current  : float  where current basis sits vs 504d history
        interpretation      : str
        warning             : str or None
    """
    _unavail: dict = {"available": False}

    try:
        # ---- availability check -------------------------------------------
        has_spread = "hy_spread" in df.columns or "ig_spread" in df.columns
        has_cdx = any(c in df.columns for c in ("cdx_hy", "cdx_ig", "cdx_basis"))
        if not has_spread and not has_cdx:
            return _unavail
        if len(df) < 21:
            return _unavail

        # ---- historical basis ---------------------------------------------
        enriched = compute_historical_basis(df)
        hist = enriched.tail(_HIST_ROWS).copy()
        signal_series = hist["basis_primary"].copy()

        # ---- live snapshot ------------------------------------------------
        snapshot = get_basis_snapshot()

        # ---- current basis: live first, then historical fallback -----------
        if snapshot.get("available"):
            live_basis = snapshot.get("bkln_hyg_basis", float("nan"))
            if np.isnan(live_basis):
                live_basis = snapshot.get("hyg_jnk_basis", float("nan"))
            if not np.isnan(live_basis):
                current_basis = live_basis
                source = "live"
            else:
                current_basis = float(enriched["basis_primary"].dropna().iloc[-1]) if not enriched["basis_primary"].dropna().empty else float("nan")
                source = "historical"
        else:
            current_basis = float(enriched["basis_primary"].dropna().iloc[-1]) if not enriched["basis_primary"].dropna().empty else float("nan")
            source = "historical"

        if np.isnan(current_basis):
            return _unavail

        # ---- signal and percentile ----------------------------------------
        basis_signal_str = _basis_signal(current_basis)
        stress_flag = basis_signal_str == "Negative Basis (Forced Selling)"
        percentile_current = _historical_percentile(signal_series.dropna(), current_basis)

        # ---- interpretation -----------------------------------------------
        basis_rounded = round(current_basis, 1)
        pct_rounded = round(percentile_current, 1) if not np.isnan(percentile_current) else None

        if stress_flag:
            interpretation = (
                f"Credit basis is deeply negative ({basis_rounded:+.1f} bps equivalent), "
                f"consistent with forced cash-bond selling. This sits at the "
                f"{pct_rounded}th percentile of the past {_HIST_ROWS}-day history, "
                f"indicating severe institutional deleveraging pressure."
            )
        elif basis_signal_str == "Compressed Basis":
            interpretation = (
                f"Basis is compressed ({basis_rounded:+.1f} bps equivalent), suggesting "
                f"elevated but not extreme cash-market stress. CDS protection is relatively "
                f"cheap vs the cost of holding cash bonds."
            )
        elif basis_signal_str == "Rich Basis":
            interpretation = (
                f"Basis is rich ({basis_rounded:+.1f} bps equivalent) — CDS protection "
                f"demand exceeds cash-bond selling pressure. Synthetic hedging is elevated."
            )
        else:
            interpretation = (
                f"Basis is in the normal range ({basis_rounded:+.1f} bps equivalent). "
                f"Cash and synthetic credit markets are broadly in equilibrium."
            )

        # ---- warning ------------------------------------------------------
        warning: str | None = None
        if not has_cdx:
            warning = (
                "No direct CDX column found — basis is a proxy constructed from "
                "spread z-scores and ETF return differentials, not true CDS-cash basis."
            )
        if not snapshot.get("available"):
            live_warn = "Live ETF fetch unavailable; current basis derived from historical data."
            warning = f"{warning}  {live_warn}" if warning else live_warn

        return {
            "available": True,
            "snapshot": snapshot,
            "historical": hist,
            "signal_series": signal_series,
            "current_basis": float(current_basis),
            "basis_signal": basis_signal_str,
            "stress_flag": bool(stress_flag),
            "percentile_current": float(percentile_current) if not np.isnan(percentile_current) else float("nan"),
            "interpretation": interpretation,
            "warning": warning,
        }

    except Exception:
        return _unavail
