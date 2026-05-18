"""
Commercial Real Estate (CRE) Stress Monitor.

CRE is a concentrated tail risk in current credit markets, transmitted via:
  - Office and multi-family vacancy → CMBS impairment
  - Regional bank CRE loan concentrations → capital shortfalls
  - Forced selling of CMBS at discounts → spread widening contagion to HY/IG

CRE stress leads broad credit spread widening because:
  1. Regional banks (KRE) are the primary lenders to CRE; equity distress signals
     loan-book problems before they show up in reported NPA data.
  2. CMBS spread widening bleeds into IG as insurance companies and pension funds
     are forced to sell to maintain credit quality limits.

Live data via yfinance (period="2y"):
  VNQ  — Vanguard Real Estate ETF (broad CRE proxy)
  IYR  — iShares US Real Estate ETF (alternative; higher office weight)
  MBB  — Mortgage-Backed Securities ETF (rate-adjusted MBS proxy)
  KRE  — SPDR S&P Regional Banking ETF (CRE-exposed lenders)
  KBE  — SPDR S&P Bank ETF (broader banking; KRE/KBE ratio = regional stress)
  SPY  — S&P 500 (benchmark for relative performance)
  XLRE — Real Estate Select Sector SPDR (large-cap REIT focus)

CRE stress score (0–100):
  Component 1 — VNQ/SPY 21d relative performance (underperf → high score)
  Component 2 — KRE/KBE 21d relative performance (underperf → high score)
  Component 3 — VNQ from 52-week high (deep drawdown → high score)
  Equal weight across available components; score clipped to [0, 100].

CRE regime thresholds:
   0–25  → "Stable"
  25–50  → "Mild Stress"
  50–75  → "Elevated Stress"
  ≥ 75   → "Acute CRE Stress"

Office stress flag: True if VNQ is > 20% below its 52-week high.

Public API
----------
  get_cre_snapshot()         -> dict
  compute_historical_cre(df) -> pd.DataFrame
  run_cre_stress(df)         -> dict
"""
from __future__ import annotations

import datetime
import functools

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIST_ROWS = 504          # ~2 trading years of history
_ZSCORE_WINDOW = 252      # 1 trading year
_MIN_PERIODS = 63         # minimum ~3 months

_LIVE_PERIOD = "2y"
_REL_PERF_WINDOW = 21     # trading days for relative performance calc
_52W_DAYS = 252           # approximate trading days in 52 weeks

# CRE regime thresholds
_STABLE_MAX = 25.0
_MILD_MAX = 50.0
_ELEVATED_MAX = 75.0

# Office stress flag: VNQ > 20% below 52w high
_OFFICE_STRESS_THRESHOLD = -0.20

# Score normalization bounds for each component
# VNQ/SPY 21d rel perf: -10% underperform = max stress (score 100)
_REL_PERF_FLOOR = -0.10   # worst expected relative perf (→ score 100)
_REL_PERF_CAP = 0.05      # best expected relative perf (→ score 0)

# VNQ from 52w high: -40% = max stress
_DD_FLOOR = -0.40
_DD_CAP = -0.02           # within 2% of 52w high = effectively no stress

# Column candidates in master DataFrame
_VNQ_CANDIDATES = ("vnq",)
_CRE_SPREAD_CANDIDATES = ("cre_spread", "cmbs_spread", "real_estate_stress")
_OFFICE_CANDIDATES = ("office_vacancy",)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _rolling_zscore(s: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    rm = s.rolling(window=window, min_periods=_MIN_PERIODS).mean()
    rs = s.rolling(window=window, min_periods=_MIN_PERIODS).std(ddof=1)
    return ((s - rm) / rs.replace(0, np.nan)).clip(-3.0, 3.0)


def _normalize_component(
    value: float, floor: float, cap: float, invert: bool = False
) -> float:
    """
    Map value onto [0, 100] between floor (score=100) and cap (score=0).

    If invert=True, floor maps to 0 and cap maps to 100.
    """
    if np.isnan(value):
        return float("nan")
    span = cap - floor
    if span == 0:
        return 50.0
    score = (value - floor) / span * 100.0
    if not invert:
        score = 100.0 - score
    return float(np.clip(score, 0.0, 100.0))


def _cre_regime(score: float) -> str:
    if np.isnan(score):
        return "Unknown"
    if score < _STABLE_MAX:
        return "Stable"
    if score < _MILD_MAX:
        return "Mild Stress"
    if score < _ELEVATED_MAX:
        return "Elevated Stress"
    return "Acute CRE Stress"


def _compute_stress_score(
    vnq_spy_rel: float,
    kre_kbe_rel: float,
    vnq_dd: float,
) -> float:
    """
    Compute composite CRE stress score from three components.

    Missing components are excluded from the average.
    Returns NaN if no components are available.
    """
    components = []

    if not np.isnan(vnq_spy_rel):
        c1 = _normalize_component(vnq_spy_rel, floor=_REL_PERF_FLOOR, cap=_REL_PERF_CAP)
        if not np.isnan(c1):
            components.append(c1)

    if not np.isnan(kre_kbe_rel):
        c2 = _normalize_component(kre_kbe_rel, floor=_REL_PERF_FLOOR, cap=_REL_PERF_CAP)
        if not np.isnan(c2):
            components.append(c2)

    if not np.isnan(vnq_dd):
        c3 = _normalize_component(vnq_dd, floor=_DD_FLOOR, cap=_DD_CAP)
        if not np.isnan(c3):
            components.append(c3)

    if not components:
        return float("nan")
    return float(np.mean(components))


# ---------------------------------------------------------------------------
# Live yfinance fetch (cached per calendar day)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _cached_cre_history(date_key: str) -> dict:  # noqa: ARG001
    """Download 2-year daily closes for CRE-related tickers. Cached per day."""
    try:
        import yfinance as yf
        tickers = ["VNQ", "IYR", "MBB", "KRE", "KBE", "SPY", "XLRE"]
        raw = yf.download(
            tickers, period=_LIVE_PERIOD, auto_adjust=True, progress=False
        )
        if raw.empty:
            return {"available": False}

        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].copy()
        else:
            closes = raw.copy()

        closes = closes.dropna(how="all")
        if closes.empty:
            return {"available": False}

        return {"available": True, "closes": closes}
    except Exception:
        return {"available": False}


def _get_cre_closes() -> dict:
    date_key = datetime.date.today().isoformat()
    return _cached_cre_history(date_key)


# ---------------------------------------------------------------------------
# Public: get_cre_snapshot
# ---------------------------------------------------------------------------

def get_cre_snapshot() -> dict:
    """
    Fetch live CRE ETF data via yfinance and compute stress snapshot.

    Returns
    -------
    dict with keys:
        available           : bool
        vnq_price           : float
        vnq_spy_21d_rel     : float   (VNQ 21d return − SPY 21d return)
        vnq_from_52w_high   : float   (% below 52w high; negative = below)
        kre_kbe_21d_rel     : float or None
        mbb_21d_return      : float or None
        cre_stress_score    : float   (0–100)
        cre_regime          : str
        office_stress_flag  : bool
        as_of               : str
    """
    _now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    _unavail: dict = {
        "available": False,
        "vnq_price": float("nan"),
        "vnq_spy_21d_rel": float("nan"),
        "vnq_from_52w_high": float("nan"),
        "kre_kbe_21d_rel": None,
        "mbb_21d_return": None,
        "cre_stress_score": float("nan"),
        "cre_regime": "Unknown",
        "office_stress_flag": False,
        "as_of": _now,
    }

    try:
        data = _get_cre_closes()
        if not data.get("available"):
            return _unavail

        closes: pd.DataFrame = data["closes"]

        # VNQ and SPY are the minimum required tickers
        if "VNQ" not in closes.columns or "SPY" not in closes.columns:
            return _unavail

        vnq = closes["VNQ"].dropna()
        spy = closes["SPY"].dropna()

        # Need at least 22 days for 21-day window + 1
        if len(vnq) < _REL_PERF_WINDOW + 1 or len(spy) < _REL_PERF_WINDOW + 1:
            return _unavail

        # ---- Latest prices ---------------------------------------------------
        vnq_price = float(vnq.iloc[-1])

        # ---- 21-day returns --------------------------------------------------
        def _ret21(s: pd.Series) -> float:
            if len(s) < _REL_PERF_WINDOW + 1:
                return float("nan")
            window = s.tail(_REL_PERF_WINDOW + 1)
            return float(window.iloc[-1] / window.iloc[0] - 1)

        vnq_ret21 = _ret21(vnq)
        spy_ret21 = _ret21(spy)
        vnq_spy_rel = vnq_ret21 - spy_ret21 if not (np.isnan(vnq_ret21) or np.isnan(spy_ret21)) else float("nan")

        # ---- 52-week high drawdown -------------------------------------------
        lookback = min(len(vnq), _52W_DAYS)
        vnq_52w_high = float(vnq.tail(lookback).max())
        vnq_dd = float(vnq_price / vnq_52w_high - 1)  # negative if below high

        # ---- KRE/KBE relative perf -------------------------------------------
        kre_kbe_rel: float | None = None
        if "KRE" in closes.columns and "KBE" in closes.columns:
            kre = closes["KRE"].dropna()
            kbe = closes["KBE"].dropna()
            if len(kre) >= _REL_PERF_WINDOW + 1 and len(kbe) >= _REL_PERF_WINDOW + 1:
                kre_ret21 = _ret21(kre)
                kbe_ret21 = _ret21(kbe)
                if not (np.isnan(kre_ret21) or np.isnan(kbe_ret21)):
                    kre_kbe_rel = float(kre_ret21 - kbe_ret21)

        # ---- MBB 21d return --------------------------------------------------
        mbb_21d: float | None = None
        if "MBB" in closes.columns:
            mbb = closes["MBB"].dropna()
            if len(mbb) >= _REL_PERF_WINDOW + 1:
                mbb_ret = _ret21(mbb)
                if not np.isnan(mbb_ret):
                    mbb_21d = float(mbb_ret)

        # ---- Stress score ----------------------------------------------------
        score = _compute_stress_score(
            vnq_spy_rel=vnq_spy_rel if not np.isnan(vnq_spy_rel) else float("nan"),
            kre_kbe_rel=kre_kbe_rel if kre_kbe_rel is not None else float("nan"),
            vnq_dd=vnq_dd,
        )
        regime = _cre_regime(score)
        office_flag = bool(vnq_dd < _OFFICE_STRESS_THRESHOLD)

        def _r(v: float | None, n: int = 4) -> float:
            if v is None or np.isnan(v):
                return float("nan")
            return round(v, n)

        return {
            "available": True,
            "vnq_price": _r(vnq_price, 2),
            "vnq_spy_21d_rel": _r(vnq_spy_rel, 4),
            "vnq_from_52w_high": _r(vnq_dd, 4),
            "kre_kbe_21d_rel": _r(kre_kbe_rel, 4) if kre_kbe_rel is not None else None,
            "mbb_21d_return": _r(mbb_21d, 4) if mbb_21d is not None else None,
            "cre_stress_score": _r(score, 1),
            "cre_regime": regime,
            "office_stress_flag": office_flag,
            "as_of": _now,
        }
    except Exception:
        return _unavail


# ---------------------------------------------------------------------------
# Public: compute_historical_cre
# ---------------------------------------------------------------------------

def compute_historical_cre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check master DataFrame for CRE-related columns and build historical series.

    Columns checked (in priority order):
      vnq          — direct CRE price proxy
      cre_spread   — CRE credit spread
      cmbs_spread  — CMBS spread
      real_estate_stress — pre-computed stress index

    Appended columns:
      cre_proxy          — best available CRE measure (raw level or z-score)
      cre_stress_regime  — categorical regime string

    The cre_proxy is normalized to a [0, 100] score where possible.
    Returns a copy of df (never raises).
    """
    try:
        out = df.copy()

        # ---- Identify best available CRE column ------------------------------
        vnq_col = _resolve_column(out, _VNQ_CANDIDATES)
        spread_col = _resolve_column(out, _CRE_SPREAD_CANDIDATES)

        if spread_col is not None:
            # Spread-based: higher spread = more stress; normalize via z-score
            z = _rolling_zscore(out[spread_col].copy())
            # Map z-score range [-3, 3] → [0, 100] where high z = high stress
            out["cre_proxy"] = ((z + 3.0) / 6.0 * 100.0).clip(0, 100)
        elif vnq_col is not None:
            # Price-based: drawdown from rolling 252d high as stress proxy
            prices = out[vnq_col].copy()
            rolling_max = prices.rolling(window=_52W_DAYS, min_periods=_MIN_PERIODS).max()
            drawdown = (prices / rolling_max - 1).clip(_DD_FLOOR, 0.0)
            # Invert so deeper drawdown = higher score
            out["cre_proxy"] = ((drawdown - 0.0) / _DD_FLOOR * 100.0).clip(0, 100)
        else:
            out["cre_proxy"] = pd.Series(float("nan"), index=out.index)

        # ---- Regime classification -------------------------------------------
        out["cre_stress_regime"] = out["cre_proxy"].apply(
            lambda v: _cre_regime(float(v)) if not np.isnan(float(v)) else "Unknown"
        )

        return out
    except Exception:
        return df.copy()


# ---------------------------------------------------------------------------
# Public: run_cre_stress
# ---------------------------------------------------------------------------

def run_cre_stress(df: pd.DataFrame) -> dict:
    """
    Top-level entry point for the CRE stress module.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        available               : bool
        snapshot                : dict  (from get_cre_snapshot)
        historical_banking_link : pd.Series or None
            banking_stress_score_smooth from df if available (for context)
        credit_implication      : str
        systemic_flag           : bool  (True if cre_stress_score > 75)
        interpretation          : str
        warning                 : str or None
    """
    _unavail: dict = {
        "available": False,
        "snapshot": {"available": False},
        "historical_banking_link": None,
        "credit_implication": "Insufficient data to assess CRE stress.",
        "systemic_flag": False,
        "interpretation": "CRE stress data unavailable.",
        "warning": None,
    }

    try:
        # ---- Live snapshot ---------------------------------------------------
        snapshot = get_cre_snapshot()

        # ---- Historical CRE data from df ------------------------------------
        historical_df = compute_historical_cre(df)

        # Check CRE columns in df
        vnq_col = _resolve_column(df, _VNQ_CANDIDATES)
        spread_col = _resolve_column(df, _CRE_SPREAD_CANDIDATES)
        has_df_cre = vnq_col is not None or spread_col is not None

        snap_available = snapshot.get("available", False)
        data_available = snap_available or has_df_cre

        if not data_available:
            return _unavail

        # ---- Banking linkage series ------------------------------------------
        banking_link: pd.Series | None = None
        if "banking_stress_score_smooth" in df.columns:
            banking_link = df["banking_stress_score_smooth"].tail(_HIST_ROWS).copy()
        elif "banking_stress_score" in df.columns:
            banking_link = df["banking_stress_score"].tail(_HIST_ROWS).copy()

        # ---- Determine stress score and regime for output --------------------
        cre_score: float = float("nan")
        cre_regime: str = "Unknown"

        if snap_available:
            cre_score = float(snapshot.get("cre_stress_score", float("nan")))
            cre_regime = snapshot.get("cre_regime", "Unknown")
        elif "cre_proxy" in historical_df.columns:
            proxy = historical_df["cre_proxy"].dropna()
            if not proxy.empty:
                cre_score = float(proxy.iloc[-1])
                cre_regime = _cre_regime(cre_score)

        systemic_flag = bool(not np.isnan(cre_score) and cre_score >= _ELEVATED_MAX)

        # ---- Credit implication ----------------------------------------------
        implication_map = {
            "Stable": (
                "CRE markets are not signaling material credit stress. Regional banks "
                "with CRE concentration are not underperforming, and REIT prices suggest "
                "stable NOI expectations. HY and IG credit are unlikely to face CRE-driven "
                "spread widening in the near term."
            ),
            "Mild Stress": (
                "Emerging CRE softness is visible in REIT underperformance relative to "
                "the broader market. Regional bank equity is beginning to price in CRE "
                "loan concerns. CMBS spreads may widen modestly. HY credit faces limited "
                "but non-trivial spillover risk via regional bank funding costs."
            ),
            "Elevated Stress": (
                "CRE stress is elevated. Regional banks (KRE) are materially underperforming "
                "larger banks (KBE), signaling rising CRE loan loss concerns. Office CMBS "
                "spreads are likely widening, with spillover to IG credit via insurance "
                "company and pension fund deleveraging. HY names with CRE exposure "
                "face idiosyncratic widening; watch for rating agency actions."
            ),
            "Acute CRE Stress": (
                "CRE stress is at systemic levels. The combination of deep REIT drawdowns, "
                "regional bank underperformance, and potential forced CMBS selling creates a "
                "negative feedback loop with broader credit markets. Historical analogs "
                "(2007-08, 2023 SVB episode) suggest HY spreads could widen 150-300 bps if "
                "CRE stress transmits fully. Immediate risk-off positioning is warranted."
            ),
            "Unknown": (
                "CRE stress regime cannot be determined from available data."
            ),
        }
        credit_implication = implication_map.get(cre_regime, implication_map["Unknown"])

        # ---- Interpretation --------------------------------------------------
        score_str = f"{cre_score:.0f}/100" if not np.isnan(cre_score) else "N/A"
        vnq_dd = snapshot.get("vnq_from_52w_high", float("nan")) if snap_available else float("nan")
        dd_str = f"{vnq_dd * 100:+.1f}% from 52w high" if not np.isnan(vnq_dd) else "N/A"
        kre_rel = snapshot.get("kre_kbe_21d_rel") if snap_available else None
        kre_str = (
            f"KRE/KBE 21d: {kre_rel * 100:+.1f}%"
            if kre_rel is not None and not np.isnan(kre_rel)
            else "KRE/KBE: N/A"
        )

        interpretation = (
            f"CRE stress score: {score_str} → Regime: {cre_regime}. "
            f"VNQ: {dd_str}. {kre_str}. "
            f"Systemic flag: {'YES' if systemic_flag else 'No'}."
        )

        # ---- Warning ---------------------------------------------------------
        office_flag = snapshot.get("office_stress_flag", False) if snap_available else False
        warning: str | None = None
        if systemic_flag:
            warning = (
                "ACUTE CRE STRESS DETECTED: Score >= 75. CRE stress is at systemic "
                "levels with material risk of contagion to HY and IG credit via regional "
                "bank funding and CMBS market dislocations."
            )
        elif office_flag:
            warning = (
                "Office stress flag active: VNQ is more than 20% below its 52-week high, "
                "indicating significant REIT distress. Monitor CMBS spreads for contagion."
            )

        return {
            "available": True,
            "snapshot": snapshot,
            "historical_banking_link": banking_link,
            "credit_implication": credit_implication,
            "systemic_flag": systemic_flag,
            "interpretation": interpretation,
            "warning": warning,
        }

    except Exception:
        return _unavail
