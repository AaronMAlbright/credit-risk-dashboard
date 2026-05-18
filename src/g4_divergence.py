"""
G4 Central Bank Divergence Monitor.

Fed vs ECB vs BOJ vs BOE policy rates and 10-year yields side by side.
Credit spread premiums and FX hedging costs vary systematically with G4
central bank divergence.

Public API
----------
  POLICY_RATE_COLS     — column candidates to look up in the master DataFrame
  POLICY_RATE_FALLBACK — hardcoded 2024-2025 approximations
  get_live_rates()               -> dict
  get_policy_rates(df)           -> dict
  compute_divergence_metrics(policy_rates, live_rates) -> dict
  run_g4_divergence(df)          -> dict
"""

from __future__ import annotations

import datetime

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

# Column names to check in df for policy rates
POLICY_RATE_COLS: dict[str, list[str]] = {
    "fed": ["fed_funds_rate", "dff", "fed_funds"],
    "ecb": ["ecb_rate", "ecb_deposit_rate"],
    "boe": ["boe_rate", "boe_bank_rate"],
    "boj": ["boj_rate", "boj_policy_rate"],
}

# Fallback recent values (2024-2025 approximate)
POLICY_RATE_FALLBACK: dict[str, float] = {
    "Fed": 4.33,
    "ECB": 2.50,
    "BOE": 4.50,
    "BOJ": 0.50,
}

# Approximate G4 10-year yield peer average (ex-US) used for yield premium calc
_PEER_10Y_APPROX: float = 2.60  # rough blended average of Bund/Gilt/JGB

# yfinance tickers
_TICKERS: dict[str, str] = {
    "us_10y":  "^TNX",
    "us_30y":  "^TYX",
    "eurusd":  "EURUSD=X",
    "gbpusd":  "GBPUSD=X",
    "usdjpy":  "JPY=X",
}

# FX hedging cost rough estimate per cross (100bps = 1%)
_HEDGE_COST_APPROX: dict[str, float] = {
    "EUR/USD": 1.80,   # annualised cross-currency basis cost, approx
    "GBP/USD": 1.20,
    "USD/JPY": 4.00,   # historically elevated due to large Fed-BOJ gap
}

# Divergence regime thresholds (Fed-ECB spread, %)
_DIVERGENCE_WIDE    = 2.0
_DIVERGENCE_MODERATE_LOW = 1.0

# Carry pressure thresholds (Fed-BOJ spread)
_CARRY_HIGH_THRESH     = 3.5
_CARRY_MODERATE_THRESH = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_last_close(ticker_obj) -> float | None:
    """Return the latest close price from a yfinance Ticker, or None."""
    try:
        hist = ticker_obj.history(period="5d", interval="1d")
        if hist.empty:
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            return None
        return float(closes.iloc[-1])
    except Exception:
        return None


def _safe_float(val: object) -> float | None:
    """Convert a value to float, returning None for NaN/None."""
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _last_valid(series: pd.Series) -> float | None:
    """Return the last non-NaN float in a Series, or None."""
    clean = series.dropna()
    if clean.empty:
        return None
    return _safe_float(clean.iloc[-1])


# ---------------------------------------------------------------------------
# Public: get_live_rates
# ---------------------------------------------------------------------------

def get_live_rates() -> dict:
    """Fetch live yield and FX data via yfinance.

    Returns
    -------
    dict:
        available : bool
        us_10y    : float or None — US 10-year yield (%)
        us_30y    : float or None — US 30-year yield (%)
        eurusd    : float or None — EUR/USD spot
        gbpusd    : float or None — GBP/USD spot
        usdjpy    : float or None — USD/JPY spot
        as_of     : str — ISO timestamp of fetch
    """
    try:
        if not _YF_AVAILABLE:
            return {
                "available": False,
                "us_10y":  None,
                "us_30y":  None,
                "eurusd":  None,
                "gbpusd":  None,
                "usdjpy":  None,
                "as_of":   datetime.datetime.utcnow().isoformat(),
            }

        results: dict[str, float | None] = {}
        for key, sym in _TICKERS.items():
            try:
                ticker = yf.Ticker(sym)
                results[key] = _safe_last_close(ticker)
            except Exception:
                results[key] = None

        # TNX and TYX are quoted in tenths of a percent (e.g., 45.2 = 4.52%)
        # yfinance returns them already divided, but check sanity
        us_10y = results.get("us_10y")
        us_30y = results.get("us_30y")
        # If the value looks like it's > 20 it's likely raw bps; divide by 10
        if us_10y is not None and us_10y > 20:
            us_10y = us_10y / 10.0
        if us_30y is not None and us_30y > 20:
            us_30y = us_30y / 10.0

        any_available = any(v is not None for v in results.values())

        return {
            "available":  any_available,
            "us_10y":     round(us_10y, 3) if us_10y is not None else None,
            "us_30y":     round(us_30y, 3) if us_30y is not None else None,
            "eurusd":     round(results["eurusd"], 4) if results.get("eurusd") is not None else None,
            "gbpusd":     round(results["gbpusd"], 4) if results.get("gbpusd") is not None else None,
            "usdjpy":     round(results["usdjpy"], 2) if results.get("usdjpy") is not None else None,
            "as_of":      datetime.datetime.utcnow().isoformat(),
        }
    except Exception:
        return {
            "available": False,
            "us_10y":  None,
            "us_30y":  None,
            "eurusd":  None,
            "gbpusd":  None,
            "usdjpy":  None,
            "as_of":   datetime.datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# Public: get_policy_rates
# ---------------------------------------------------------------------------

def get_policy_rates(df: pd.DataFrame) -> dict:
    """Extract policy rates from df columns or fall back to hardcoded values.

    Parameters
    ----------
    df : Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict:
        fed    : float
        ecb    : float
        boe    : float
        boj    : float
        source : str — "DataFrame" or "Fallback (approximate)"
    """
    try:
        rates: dict[str, float | None] = {
            "fed": None,
            "ecb": None,
            "boe": None,
            "boj": None,
        }

        if df is not None and not df.empty:
            for cb_key, candidates in POLICY_RATE_COLS.items():
                for col in candidates:
                    if col in df.columns:
                        val = _last_valid(df[col])
                        if val is not None:
                            rates[cb_key] = val
                            break

        # Fill any missing rates with fallback values
        fallback_map = {
            "fed": POLICY_RATE_FALLBACK["Fed"],
            "ecb": POLICY_RATE_FALLBACK["ECB"],
            "boe": POLICY_RATE_FALLBACK["BOE"],
            "boj": POLICY_RATE_FALLBACK["BOJ"],
        }
        all_from_df = True
        for cb_key in rates:
            if rates[cb_key] is None:
                rates[cb_key] = fallback_map[cb_key]
                all_from_df = False

        source = "DataFrame" if all_from_df else "Fallback (approximate)"

        return {
            "fed":    round(float(rates["fed"]), 4),
            "ecb":    round(float(rates["ecb"]), 4),
            "boe":    round(float(rates["boe"]), 4),
            "boj":    round(float(rates["boj"]), 4),
            "source": source,
        }
    except Exception:
        return {
            "fed":    POLICY_RATE_FALLBACK["Fed"],
            "ecb":    POLICY_RATE_FALLBACK["ECB"],
            "boe":    POLICY_RATE_FALLBACK["BOE"],
            "boj":    POLICY_RATE_FALLBACK["BOJ"],
            "source": "Fallback (approximate)",
        }


# ---------------------------------------------------------------------------
# Public: compute_divergence_metrics
# ---------------------------------------------------------------------------

def compute_divergence_metrics(policy_rates: dict, live_rates: dict) -> dict:
    """Compute all G4 divergence metrics from policy and live rate dicts.

    Parameters
    ----------
    policy_rates : output of get_policy_rates()
    live_rates   : output of get_live_rates()

    Returns
    -------
    dict:
        fed_ecb_spread    : float — Fed - ECB (pp)
        fed_boj_spread    : float — Fed - BOJ (pp)
        fed_boe_spread    : float — Fed - BOE (pp)
        g4_dispersion     : float — std dev of 4 policy rates
        divergence_regime : str
        us_yield_premium  : float or None — US 10y minus peer avg estimate
        carry_pressure    : str — "High" / "Moderate" / "Low"
        em_credit_signal  : str — "Headwind" / "Neutral" / "Tailwind"
        interpretation    : str
    """
    try:
        fed = float(policy_rates["fed"])
        ecb = float(policy_rates["ecb"])
        boe = float(policy_rates["boe"])
        boj = float(policy_rates["boj"])

        fed_ecb_spread = round(fed - ecb, 4)
        fed_boj_spread = round(fed - boj, 4)
        fed_boe_spread = round(fed - boe, 4)

        all_rates = np.array([fed, ecb, boe, boj], dtype=float)
        g4_dispersion = round(float(np.std(all_rates, ddof=1)), 4)

        # Divergence regime
        if fed_ecb_spread > _DIVERGENCE_WIDE:
            divergence_regime = "Wide US-EU Divergence"
        elif fed_ecb_spread > _DIVERGENCE_MODERATE_LOW:
            divergence_regime = "Moderate Divergence"
        else:
            divergence_regime = "Converging"

        # US 10y yield premium vs peer average estimate
        us_10y = live_rates.get("us_10y")
        if us_10y is not None:
            us_yield_premium = round(float(us_10y) - _PEER_10Y_APPROX, 4)
        else:
            us_yield_premium = None

        # Carry pressure (Fed-BOJ is the dominant carry driver)
        if fed_boj_spread >= _CARRY_HIGH_THRESH:
            carry_pressure = "High"
        elif fed_boj_spread >= _CARRY_MODERATE_THRESH:
            carry_pressure = "Moderate"
        else:
            carry_pressure = "Low"

        # EM credit signal
        # Wide US-EU divergence → USD bullish → EM headwind
        # Converging → USD weakening → EM tailwind
        if divergence_regime == "Wide US-EU Divergence":
            em_credit_signal = "Headwind"
        elif divergence_regime == "Converging":
            em_credit_signal = "Tailwind"
        else:
            em_credit_signal = "Neutral"

        # Interpretation narrative
        yield_note = (
            f"US 10y ({us_10y:.2f}%) trades "
            f"{'above' if (us_yield_premium or 0) >= 0 else 'below'} "
            f"estimated G4 peer average by {abs(us_yield_premium or 0):.1f}pp. "
        ) if us_10y is not None else ""

        carry_note = (
            f"USD/JPY carry pressure is {carry_pressure.lower()} "
            f"(Fed-BOJ spread: {fed_boj_spread:+.2f}pp). "
        )

        em_note = f"EM credit faces a {em_credit_signal.lower()} given current divergence dynamics."

        interpretation = (
            f"G4 divergence regime: '{divergence_regime}' "
            f"(Fed-ECB: {fed_ecb_spread:+.2f}pp, G4 dispersion: {g4_dispersion:.2f}pp). "
            f"{yield_note}"
            f"{carry_note}"
            f"{em_note}"
        )

        return {
            "fed_ecb_spread":   fed_ecb_spread,
            "fed_boj_spread":   fed_boj_spread,
            "fed_boe_spread":   fed_boe_spread,
            "g4_dispersion":    g4_dispersion,
            "divergence_regime": divergence_regime,
            "us_yield_premium": us_yield_premium,
            "carry_pressure":   carry_pressure,
            "em_credit_signal": em_credit_signal,
            "interpretation":   interpretation,
        }
    except Exception:
        return {
            "fed_ecb_spread":   0.0,
            "fed_boj_spread":   0.0,
            "fed_boe_spread":   0.0,
            "g4_dispersion":    0.0,
            "divergence_regime": "Unknown",
            "us_yield_premium": None,
            "carry_pressure":   "Unknown",
            "em_credit_signal": "Unknown",
            "interpretation":   "Insufficient data for divergence analysis.",
        }


# ---------------------------------------------------------------------------
# Public: run_g4_divergence
# ---------------------------------------------------------------------------

def run_g4_divergence(df: pd.DataFrame) -> dict:
    """Top-level G4 divergence analysis entry point.

    Always returns available=True (uses fallback policy rates if df lacks them).

    Parameters
    ----------
    df : Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict:
        available          : bool — always True
        policy_rates       : dict from get_policy_rates()
        live_rates         : dict from get_live_rates()
        metrics            : dict from compute_divergence_metrics()
        historical_fed_rate: pd.Series or None — last 252 rows from df if available
        rate_table         : pd.DataFrame — 4-row summary table
        interpretation     : str
    """
    try:
        policy_rates = get_policy_rates(df)
        live_rates   = get_live_rates()
        metrics      = compute_divergence_metrics(policy_rates, live_rates)

        # Historical fed rate series (last 252 rows)
        historical_fed_rate: pd.Series | None = None
        if df is not None and not df.empty:
            for col in POLICY_RATE_COLS["fed"]:
                if col in df.columns:
                    series = df[col].dropna()
                    if not series.empty:
                        historical_fed_rate = series.iloc[-252:]
                    break

        # Build the 4-row rate table
        # Columns: Central Bank, Policy Rate (%), 10y Yield (est.), vs Fed Spread
        us_10y = live_rates.get("us_10y")

        # Rough 10y yield estimates by CB (used only when live data absent)
        _10y_estimates: dict[str, float | None] = {
            "Fed":  us_10y,                          # live if available
            "ECB":  1.85,                            # Bund approx
            "BOE":  4.20,                            # Gilt approx
            "BOJ":  1.05,                            # JGB approx
        }

        rows = []
        for cb, rate_key, spread_key in [
            ("Fed",  "fed",  None),
            ("ECB",  "ecb",  "fed_ecb_spread"),
            ("BOE",  "boe",  "fed_boe_spread"),
            ("BOJ",  "boj",  "fed_boj_spread"),
        ]:
            policy_rate = policy_rates[rate_key]
            ten_y = _10y_estimates.get(cb)
            vs_fed = (
                None if spread_key is None
                else -metrics[spread_key]  # invert: "ECB vs Fed" = ECB - Fed
            )
            rows.append({
                "Central Bank":     cb,
                "Policy Rate (%)":  round(policy_rate, 2),
                "10y Yield (est.)": round(ten_y, 2) if ten_y is not None else "N/A",
                "vs Fed Spread (pp)": (
                    f"{vs_fed:+.2f}" if vs_fed is not None else "—"
                ),
            })

        rate_table = pd.DataFrame(rows)

        # Top-level interpretation combines metrics + live data availability
        live_note = (
            f" Live US 10y at {us_10y:.3f}%." if us_10y is not None
            else " Live yield data unavailable — using estimates."
        )
        interpretation = metrics["interpretation"] + live_note

        return {
            "available":           True,
            "policy_rates":        policy_rates,
            "live_rates":          live_rates,
            "metrics":             metrics,
            "historical_fed_rate": historical_fed_rate,
            "rate_table":          rate_table,
            "interpretation":      interpretation,
        }
    except Exception:
        # Absolute last-resort fallback — always return a usable dict
        try:
            fallback_policy = {
                "fed":    POLICY_RATE_FALLBACK["Fed"],
                "ecb":    POLICY_RATE_FALLBACK["ECB"],
                "boe":    POLICY_RATE_FALLBACK["BOE"],
                "boj":    POLICY_RATE_FALLBACK["BOJ"],
                "source": "Fallback (approximate)",
            }
            fallback_live = {
                "available": False,
                "us_10y":  None,
                "us_30y":  None,
                "eurusd":  None,
                "gbpusd":  None,
                "usdjpy":  None,
                "as_of":   datetime.datetime.utcnow().isoformat(),
            }
            fallback_metrics = compute_divergence_metrics(fallback_policy, fallback_live)
            rows = [
                {"Central Bank": "Fed", "Policy Rate (%)": POLICY_RATE_FALLBACK["Fed"],
                 "10y Yield (est.)": "N/A", "vs Fed Spread (pp)": "—"},
                {"Central Bank": "ECB", "Policy Rate (%)": POLICY_RATE_FALLBACK["ECB"],
                 "10y Yield (est.)": "N/A", "vs Fed Spread (pp)": f"{POLICY_RATE_FALLBACK['ECB'] - POLICY_RATE_FALLBACK['Fed']:+.2f}"},
                {"Central Bank": "BOE", "Policy Rate (%)": POLICY_RATE_FALLBACK["BOE"],
                 "10y Yield (est.)": "N/A", "vs Fed Spread (pp)": f"{POLICY_RATE_FALLBACK['BOE'] - POLICY_RATE_FALLBACK['Fed']:+.2f}"},
                {"Central Bank": "BOJ", "Policy Rate (%)": POLICY_RATE_FALLBACK["BOJ"],
                 "10y Yield (est.)": "N/A", "vs Fed Spread (pp)": f"{POLICY_RATE_FALLBACK['BOJ'] - POLICY_RATE_FALLBACK['Fed']:+.2f}"},
            ]
            return {
                "available":           True,
                "policy_rates":        fallback_policy,
                "live_rates":          fallback_live,
                "metrics":             fallback_metrics,
                "historical_fed_rate": None,
                "rate_table":          pd.DataFrame(rows),
                "interpretation":      fallback_metrics["interpretation"],
            }
        except Exception:
            return {"available": False}
