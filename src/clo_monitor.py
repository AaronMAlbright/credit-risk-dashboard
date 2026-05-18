"""
CLO (Collateralized Loan Obligation) and structured credit stress monitor.

The $1T+ CLO market is the dominant buyer of leveraged loans. When CLO
mechanics are impaired — by widening BB spreads, failing OC tests, or
loan price deterioration — demand for leveraged loans collapses and HY/
loan spreads widen independently of fundamentals. This module tracks CLO
stress through ETF proxy construction.

Key mechanics:
  - BKLN/HYG relative return: loans underperforming bonds signals CLO
    buying is impaired (CLOs buy loans, not bonds).
  - OC (overcollateralization) test: if the loan portfolio falls below a
    threshold the CLO must stop paying junior tranches and deleverage —
    forced selling begets more selling.
  - CLO refi window: when AAA spreads are tight CLOs can refinance at lower
    cost, reducing overall CLO cost of capital and stimulating new issuance.

ETF proxies (via yfinance):
  BKLN — Invesco Senior Loan ETF       [primary loan proxy]
  SRLN — SPDR Blackstone Senior Loan   [secondary loan proxy]
  HYG  — iShares HY Bond ETF           [HY bond basis for relative return]
  CLO  — Janus Henderson CLO ETF       [direct CLO proxy]

Public API
----------
  CLO_STRESS_THRESHOLDS   — module constant: regime score ranges
  get_clo_snapshot()      -> dict  (live yfinance fetch)
  compute_historical_clo(df) -> pd.DataFrame  (adds clo_ columns)
  run_clo_monitor(df)     -> dict
"""
from __future__ import annotations

import datetime
import functools

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLO_STRESS_THRESHOLDS: dict[str, tuple[int, int]] = {
    "Benign":  (0,  25),
    "Caution": (25, 50),
    "Stress":  (50, 75),
    "Acute":   (75, 100),
}

_FETCH_PERIOD      = "1y"
_RATIO_WINDOW      = 21    # 21-trading-day relative return window
_52W_WINDOW        = 252   # approximate 1-year lookback for 52w high
_HIST_ROWS         = 504   # 2 years of daily rows for historical outputs

# OC cushion: loan prices below 90% of par → zero cushion; at par → full
_OC_FLOOR_PCT      = 0.90
_OC_RANGE          = 0.10  # 0.90 → 1.00 maps to 0 → 100

# CLO AAA spread proxy: ig_spread / 2, threshold ~130 bps → 1.30% / 2 = 0.65
_AAA_PROXY_REFI_THRESHOLD = 0.65   # in %-point units (ig_spread stored as %)

# yfinance tickers
_CLO_TICKERS = ["BKLN", "SRLN", "HYG", "CLO"]

_LOAN_TICKER_PREFERENCE = ["BKLN", "SRLN"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_clo_etf_history() -> dict[str, pd.Series]:
    """Download 1-year daily close history for CLO proxy tickers."""
    try:
        import yfinance as yf
        raw = yf.download(
            _CLO_TICKERS,
            period=_FETCH_PERIOD,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return {}
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw
        closes = closes.dropna(how="all")
        result: dict[str, pd.Series] = {}
        for ticker in _CLO_TICKERS:
            if ticker in closes.columns:
                s = closes[ticker].dropna()
                if not s.empty:
                    result[ticker] = s
        return result
    except Exception:
        return {}


def _drawdown_from_52w(series: pd.Series) -> float:
    """Return current price as % of the 52-week high minus 1 (negative = below high)."""
    if series is None or len(series) < 5:
        return float("nan")
    lookback = min(_52W_WINDOW, len(series))
    high = float(series.iloc[-lookback:].max())
    current = float(series.iloc[-1])
    if high <= 0:
        return float("nan")
    return (current / high - 1.0) * 100.0


def _rolling_rel_return(loan: pd.Series, hy: pd.Series, window: int = _RATIO_WINDOW) -> pd.Series:
    """Rolling `window`-day relative return of loan vs HY (percentage points)."""
    aligned = pd.DataFrame({"loan": loan, "hy": hy}).dropna()
    if len(aligned) < window + 2:
        return pd.Series(dtype=float)
    loan_ret = aligned["loan"].pct_change(window) * 100.0
    hy_ret   = aligned["hy"].pct_change(window) * 100.0
    return (loan_ret - hy_ret).dropna()


def _rel_return_latest(loan: pd.Series, hy: pd.Series, window: int = _RATIO_WINDOW) -> float:
    """Scalar 21-day relative return (loan minus HY, pp), latest observation."""
    rel = _rolling_rel_return(loan, hy, window)
    if rel.empty:
        return float("nan")
    return float(rel.iloc[-1])


def _zscore_series(s: pd.Series, window: int = _52W_WINDOW) -> pd.Series:
    """Rolling z-score of s using `window`-period expanding min_periods=63."""
    mu  = s.rolling(window, min_periods=63).mean()
    sig = s.rolling(window, min_periods=63).std(ddof=1)
    return (s - mu) / sig.replace(0, np.nan)


def _zscore_to_stress(z: float) -> float:
    """Map z-score to a 0–100 stress score: z ≥ +2 → 100, z ≤ -2 → 0."""
    return float(np.clip((z + 2.0) / 4.0 * 100.0, 0.0, 100.0))


def _drawdown_to_stress(drawdown_pct: float) -> float:
    """Map drawdown from 52w high (negative %) to 0–100 stress: -10% = 100."""
    return float(np.clip(-drawdown_pct / 10.0 * 100.0, 0.0, 100.0))


def _regime_from_score(score: float) -> str:
    """Map clo_stress_score to a regime label."""
    if np.isnan(score):
        return "Unknown"
    if score < 25:
        return "Benign"
    if score < 50:
        return "Caution"
    if score < 75:
        return "Stress"
    return "Acute"


def _oc_cushion(bkln_price: float, bkln_52w_high: float) -> float:
    """
    OC cushion proxy: (price / high - 0.90) / 0.10 × 100, clamped [0, 100].

    Higher score = more cushion. Falls to 0 when BKLN is 10%+ below its high.
    """
    if bkln_52w_high <= 0 or np.isnan(bkln_price) or np.isnan(bkln_52w_high):
        return float("nan")
    par_proxy = bkln_52w_high * 1.01
    ratio = bkln_price / par_proxy
    cushion = (ratio - _OC_FLOOR_PCT) / _OC_RANGE * 100.0
    return float(np.clip(cushion, 0.0, 100.0))


def _refi_window_from_ig(ig_spread: float | None) -> str:
    """Classify CLO refi window: 'Open' if AAA proxy < 130 bps equivalent."""
    if ig_spread is None or np.isnan(ig_spread):
        return "Unknown"
    aaa_proxy = ig_spread / 2.0
    return "Open" if aaa_proxy < _AAA_PROXY_REFI_THRESHOLD else "Closed"


# ---------------------------------------------------------------------------
# Public: get_clo_snapshot  (cached by date to avoid repeated yfinance calls)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=2)
def _cached_clo_fetch(date_key: str) -> dict:
    """Inner fetch cached by date string so repeated same-day calls are free."""
    return _fetch_clo_etf_history()


def get_clo_snapshot() -> dict:
    """
    Fetch live CLO/loan ETF data via yfinance and compute CLO stress metrics.

    Returns
    -------
    dict with keys:
        available          : bool
        loan_etf           : str or None   — which loan ETF found
        loan_price         : float or None
        hyg_price          : float or None
        clo_etf_price      : float or None
        bkln_vs_hyg_21d    : float or None — BKLN 21d return minus HYG 21d return (pp)
        bkln_drawdown      : float         — % below 52-week high (negative)
        clo_etf_drawdown   : float or None
        clo_stress_score   : float         — 0-100 composite
        regime             : str
        oc_cushion_proxy   : float         — 0-100 (100 = full cushion)
        refi_window        : str           — "Open" / "Closed" / "Unknown"
        as_of              : str           — ISO timestamp
    """
    _unavail: dict = {
        "available": False,
        "loan_etf": None,
        "loan_price": None,
        "hyg_price": None,
        "clo_etf_price": None,
        "bkln_vs_hyg_21d": None,
        "bkln_drawdown": float("nan"),
        "clo_etf_drawdown": None,
        "clo_stress_score": 50.0,
        "regime": "Unknown",
        "oc_cushion_proxy": float("nan"),
        "refi_window": "Unknown",
        "as_of": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    }

    try:
        date_key = datetime.date.today().isoformat()
        etf_data = _cached_clo_fetch(date_key)
        if not etf_data:
            return _unavail

        as_of = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        # ---- Identify loan ETF --------------------------------------------------
        loan_series: pd.Series | None = None
        loan_etf: str | None = None
        for ticker in _LOAN_TICKER_PREFERENCE:
            if ticker in etf_data and len(etf_data[ticker]) >= _RATIO_WINDOW + 2:
                loan_etf   = ticker
                loan_series = etf_data[ticker]
                break

        hyg_series = etf_data.get("HYG")
        clo_series = etf_data.get("CLO")

        def _last(s: pd.Series | None) -> float:
            if s is None or s.empty:
                return float("nan")
            return float(s.iloc[-1])

        loan_price    = _last(loan_series)
        hyg_price     = _last(hyg_series)
        clo_etf_price = _last(clo_series)

        # ---- Component A: BKLN/HYG 21d relative return z-score ----------------
        component_a: float | None = None
        bkln_vs_hyg_21d: float | None = None

        if loan_series is not None and hyg_series is not None:
            bkln_vs_hyg_21d = _rel_return_latest(loan_series, hyg_series)
            if not np.isnan(bkln_vs_hyg_21d):
                rel_hist = _rolling_rel_return(loan_series, hyg_series)
                if len(rel_hist) >= 5:
                    mu  = float(rel_hist.mean())
                    sig = float(rel_hist.std(ddof=1))
                    if sig > 0:
                        z = (bkln_vs_hyg_21d - mu) / sig
                        component_a = _zscore_to_stress(-z)  # invert: underperf = high stress

        # ---- Component B: BKLN drawdown from 52w high --------------------------
        bkln_drawdown = float("nan")
        component_b: float | None = None

        if loan_series is not None and len(loan_series) >= 10:
            bkln_drawdown = _drawdown_from_52w(loan_series)
            if not np.isnan(bkln_drawdown):
                component_b = _drawdown_to_stress(bkln_drawdown)

        # ---- Component C: CLO ETF drawdown from 52w high -----------------------
        clo_etf_drawdown: float | None = None
        component_c: float | None = None

        if clo_series is not None and len(clo_series) >= 10:
            clo_etf_drawdown = _drawdown_from_52w(clo_series)
            if not np.isnan(clo_etf_drawdown):
                component_c = _drawdown_to_stress(clo_etf_drawdown)

        # ---- Equal-weight available components --------------------------------
        components = [c for c in (component_a, component_b, component_c) if c is not None]
        clo_stress_score = float(np.mean(components)) if components else 50.0
        clo_stress_score = float(np.clip(clo_stress_score, 0.0, 100.0))

        # ---- OC cushion proxy -------------------------------------------------
        oc_cushion = float("nan")
        if loan_series is not None and len(loan_series) >= 10:
            lookback   = min(_52W_WINDOW, len(loan_series))
            high_52w   = float(loan_series.iloc[-lookback:].max())
            oc_cushion = _oc_cushion(loan_price, high_52w)

        # ---- Refi window: needs ig_spread, not available in live snapshot -----
        # Return "Unknown" here; run_clo_monitor resolves it from df
        refi_window = "Unknown"

        regime = _regime_from_score(clo_stress_score)

        return {
            "available":        True,
            "loan_etf":         loan_etf,
            "loan_price":       loan_price if not np.isnan(loan_price) else None,
            "hyg_price":        hyg_price  if not np.isnan(hyg_price)  else None,
            "clo_etf_price":    clo_etf_price if not np.isnan(clo_etf_price) else None,
            "bkln_vs_hyg_21d":  round(bkln_vs_hyg_21d, 2) if bkln_vs_hyg_21d is not None and not np.isnan(bkln_vs_hyg_21d) else None,
            "bkln_drawdown":    round(bkln_drawdown, 2) if not np.isnan(bkln_drawdown) else float("nan"),
            "clo_etf_drawdown": round(clo_etf_drawdown, 2) if clo_etf_drawdown is not None and not np.isnan(clo_etf_drawdown) else None,
            "clo_stress_score": round(clo_stress_score, 1),
            "regime":           regime,
            "oc_cushion_proxy": round(oc_cushion, 1) if not np.isnan(oc_cushion) else float("nan"),
            "refi_window":      refi_window,
            "as_of":            as_of,
        }

    except Exception:
        return _unavail


# ---------------------------------------------------------------------------
# Public: compute_historical_clo
# ---------------------------------------------------------------------------

def compute_historical_clo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a copy of df with CLO-related columns derived from df columns.

    Columns added:
        clo_stress_score      : float 0-100 composite
        clo_regime            : str
        clo_oc_cushion_proxy  : float 0-100 (from hy_spread-derived loan price proxy)
        clo_refi_window       : str  "Open" / "Closed" / "Unknown"
        clo_loan_vs_hy_zscore : float  (z-score of loan-vs-HY proxy from hy/ig spread divergence)

    When direct CLO spread columns (clo_aaa_spread, clo_bb_spread, clo_spread)
    are present they are used; otherwise the computation falls back to
    hy_spread and ig_spread proxies.
    """
    out = df.copy()

    try:
        cols = set(out.columns)

        # ---- Determine stress driver ------------------------------------------
        # If we have explicit CLO spread columns, use BB-AAA spread as the stress input
        has_clo_aaa = "clo_aaa_spread" in cols
        has_clo_bb  = "clo_bb_spread"  in cols
        has_hy      = "hy_spread"      in cols
        has_ig      = "ig_spread"      in cols

        stress_series: pd.Series | None = None

        if has_clo_aaa and has_clo_bb:
            # BB - AAA spread as stress signal (wide = stressed)
            spread_diff = out["clo_bb_spread"] - out["clo_aaa_spread"]
            z = _zscore_series(spread_diff)
            stress_series = z.apply(lambda v: _zscore_to_stress(float(v)) if pd.notna(v) else float("nan"))

        elif "clo_spread" in cols:
            z = _zscore_series(out["clo_spread"])
            stress_series = z.apply(lambda v: _zscore_to_stress(float(v)) if pd.notna(v) else float("nan"))

        elif has_hy:
            # Proxy: hy_spread z-score as CLO stress analog
            z = _zscore_series(out["hy_spread"])
            stress_series = z.apply(lambda v: _zscore_to_stress(float(v)) if pd.notna(v) else float("nan"))

        if stress_series is not None:
            out["clo_stress_score"] = stress_series.clip(0.0, 100.0)
        else:
            out["clo_stress_score"] = np.nan

        # ---- Regime -----------------------------------------------------------
        out["clo_regime"] = out["clo_stress_score"].apply(
            lambda v: _regime_from_score(float(v)) if pd.notna(v) else "Unknown"
        )

        # ---- OC cushion proxy via hy_spread (wider spread → lower loan price) -
        if has_hy:
            # Normalise hy_spread to a loan-price-like index: tight = near par
            hy = out["hy_spread"]
            hy_min = hy.rolling(252, min_periods=63).min()
            hy_max = hy.rolling(252, min_periods=63).max()
            range_ = (hy_max - hy_min).replace(0, np.nan)
            loan_price_proxy = 100.0 * (1.0 - (hy - hy_min) / range_ * 0.15)
            par_proxy = loan_price_proxy.rolling(252, min_periods=63).max() * 1.01
            cushion_raw = (loan_price_proxy / par_proxy - _OC_FLOOR_PCT) / _OC_RANGE * 100.0
            out["clo_oc_cushion_proxy"] = cushion_raw.clip(0.0, 100.0)
        else:
            out["clo_oc_cushion_proxy"] = np.nan

        # ---- Refi window via ig_spread proxy ----------------------------------
        if has_ig:
            out["clo_refi_window"] = out["ig_spread"].apply(
                lambda v: _refi_window_from_ig(float(v)) if pd.notna(v) else "Unknown"
            )
        else:
            out["clo_refi_window"] = "Unknown"

        # ---- Loan vs HY z-score proxy ----------------------------------------
        if has_hy and has_ig:
            hy_ig_ratio = out["hy_spread"] / out["ig_spread"].replace(0, np.nan)
            z_ratio = _zscore_series(hy_ig_ratio)
            out["clo_loan_vs_hy_zscore"] = z_ratio
        else:
            out["clo_loan_vs_hy_zscore"] = np.nan

    except Exception:
        for col in ("clo_stress_score", "clo_regime", "clo_oc_cushion_proxy",
                    "clo_refi_window", "clo_loan_vs_hy_zscore"):
            if col not in out.columns:
                out[col] = np.nan

    return out


# ---------------------------------------------------------------------------
# Public: run_clo_monitor
# ---------------------------------------------------------------------------

def run_clo_monitor(df: pd.DataFrame) -> dict:
    """
    Top-level CLO/structured credit analysis entry point.

    Combines a live yfinance snapshot with historical CLO metrics derived
    from the master DataFrame. Requires DatetimeIndex; no "date" column.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        available          : bool
        current            : dict (clo_stress_score, regime, oc_cushion_proxy,
                                   refi_window, bkln_vs_hyg_21d,
                                   interpretation, warning)
        df                 : pd.DataFrame (enriched with clo_ columns)
        historical_score   : pd.Series (DatetimeIndex, last _HIST_ROWS rows)
        regime_history     : pd.Series (str, DatetimeIndex)
        oc_cushion_history : pd.Series
        loan_vs_hy_history : pd.Series
    """
    _unavail: dict = {"available": False}

    try:
        if df is None or df.empty:
            return _unavail

        has_hy_col = "hy_spread" in df.columns

        # ---- Live snapshot ----------------------------------------------------
        snapshot = get_clo_snapshot()

        # ---- Enrich df -------------------------------------------------------
        enriched = compute_historical_clo(df)

        # ---- Refi window from df ig_spread (supplements snapshot) -----------
        refi_window = "Unknown"
        if "ig_spread" in df.columns:
            last_ig = df["ig_spread"].dropna()
            if not last_ig.empty:
                refi_window = _refi_window_from_ig(float(last_ig.iloc[-1]))
        if snapshot.get("available") and refi_window == "Unknown":
            refi_window = snapshot.get("refi_window", "Unknown")

        # ---- Determine current stress score ----------------------------------
        # Prefer live snapshot; fall back to last row of enriched df
        clo_stress_score: float = 50.0
        oc_cushion: float = float("nan")
        bkln_vs_hyg_21d: float | None = None

        if snapshot.get("available"):
            snap_score = snapshot.get("clo_stress_score")
            if snap_score is not None and not np.isnan(snap_score):
                clo_stress_score = float(snap_score)
            snap_oc = snapshot.get("oc_cushion_proxy")
            if snap_oc is not None and not np.isnan(snap_oc):
                oc_cushion = float(snap_oc)
            bkln_vs_hyg_21d = snapshot.get("bkln_vs_hyg_21d")

        elif has_hy_col and "clo_stress_score" in enriched.columns:
            last_score = enriched["clo_stress_score"].dropna()
            if not last_score.empty:
                clo_stress_score = float(last_score.iloc[-1])
            if "clo_oc_cushion_proxy" in enriched.columns:
                last_oc = enriched["clo_oc_cushion_proxy"].dropna()
                if not last_oc.empty:
                    oc_cushion = float(last_oc.iloc[-1])

        elif not snapshot.get("available") and not has_hy_col:
            return _unavail

        regime = _regime_from_score(clo_stress_score)

        # ---- Historical series slices ----------------------------------------
        historical_score = (
            enriched["clo_stress_score"].tail(_HIST_ROWS).copy()
            if "clo_stress_score" in enriched.columns
            else pd.Series(dtype=float)
        )
        regime_history = (
            enriched["clo_regime"].tail(_HIST_ROWS).copy()
            if "clo_regime" in enriched.columns
            else pd.Series(dtype=str)
        )
        oc_cushion_history = (
            enriched["clo_oc_cushion_proxy"].tail(_HIST_ROWS).copy()
            if "clo_oc_cushion_proxy" in enriched.columns
            else pd.Series(dtype=float)
        )
        loan_vs_hy_history = (
            enriched["clo_loan_vs_hy_zscore"].tail(_HIST_ROWS).copy()
            if "clo_loan_vs_hy_zscore" in enriched.columns
            else pd.Series(dtype=float)
        )

        # ---- Interpretation --------------------------------------------------
        bkln_detail = ""
        if bkln_vs_hyg_21d is not None and not np.isnan(bkln_vs_hyg_21d):
            direction = "underperforming" if bkln_vs_hyg_21d < 0 else "outperforming"
            bkln_detail = (
                f"Senior loans are {direction} HY bonds by "
                f"{abs(bkln_vs_hyg_21d):.2f}pp over 21 days — "
                f"{'CLO buying constrained' if bkln_vs_hyg_21d < 0 else 'CLO demand intact'}. "
            )

        oc_detail = ""
        if not np.isnan(oc_cushion):
            if oc_cushion < 20:
                oc_detail = "OC cushion proxy is critically low — forced deleveraging risk elevated. "
            elif oc_cushion < 50:
                oc_detail = "OC cushion proxy is below normal — monitor for OC test breach signals. "

        refi_detail = f"CLO refi window: {refi_window}. "

        interpretation = (
            f"CLO stress score: {clo_stress_score:.0f}/100 ({regime}). "
            f"{bkln_detail}"
            f"{oc_detail}"
            f"{refi_detail}"
        )

        # ---- Warning ---------------------------------------------------------
        warning: str | None = None
        if regime in ("Stress", "Acute"):
            warning = (
                f"CLO regime is '{regime}': loan market demand is structurally impaired. "
                "Spread widening may accelerate as CLO forced sellers overwhelm primary "
                "issuance capacity. Watch BKLN price action and CLO manager deal flow."
            )
        if not np.isnan(oc_cushion) and oc_cushion < 20:
            oc_warn = (
                "OC test breach risk is elevated. If CLOs begin redirecting cash flows "
                "from junior tranches to deleverage, the loan selling pressure can be "
                "self-reinforcing (falling prices → lower OC ratios → more selling)."
            )
            warning = f"{warning}  {oc_warn}" if warning else oc_warn

        current = {
            "clo_stress_score": round(clo_stress_score, 1),
            "regime":           regime,
            "oc_cushion_proxy": round(oc_cushion, 1) if not np.isnan(oc_cushion) else None,
            "refi_window":      refi_window,
            "bkln_vs_hyg_21d":  bkln_vs_hyg_21d,
            "interpretation":   interpretation,
            "warning":          warning,
        }

        return {
            "available":          True,
            "current":            current,
            "df":                 enriched,
            "historical_score":   historical_score,
            "regime_history":     regime_history,
            "oc_cushion_history": oc_cushion_history,
            "loan_vs_hy_history": loan_vs_hy_history,
        }

    except Exception:
        return _unavail
