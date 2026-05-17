"""
Merton (1974) Distance-to-Default model.

The Merton structural credit model treats equity as a call option on firm assets.
Key insight: equity holders are long a call option on firm value V with strike D (debt).
When V falls below D, the firm defaults (equity = 0).

Distance-to-Default (DD):
  DD = (ln(V/D) + (μ - σ_V²/2) × T) / (σ_V × √T)

where:
  V  = asset value (estimated from equity + debt)
  D  = face value of debt (default barrier)
  σ_V = asset volatility (estimated from equity vol via Ito's lemma)
  T  = 1 year horizon
  μ  = expected asset return (use risk-free as proxy)

Simplified aggregate implementation for a market index:
  - Use SP500 as proxy for aggregate equity value E
  - Use VIX/100 as proxy for σ_E (implied equity volatility)
  - Estimate aggregate leverage L = D/(D+E) from HY spread level:
      L ≈ 0.35 + (hy_spread - 3.5) × 0.02  (spread encodes market's view of leverage)
      Clip L to [0.20, 0.80]
  - Asset value: V = E / (1 - L)  (E = V - D, so D = V × L)
  - Asset vol from Ito: σ_V = σ_E × E/V  (equity vol × equity/asset ratio)
  - DD = (ln(V/D) - 0.5 × σ_V²) / σ_V  (simplified, T=1 year)
       = (ln(1/L) - 0.5 × σ_V²) / σ_V
  - Implied default probability: P(default) = Φ(-DD)
    using math.erf for normal CDF

This is a stylised aggregate model for educational purposes. Real implementations
use firm-level balance sheet data (KMV model).

Background
----------
Robert Merton (1974) formalised the idea that default is an option event.
The firm's assets V follow a geometric Brownian motion. Debt D matures at
time T. Equity holders receive max(V_T - D, 0) at maturity — a call option.
Creditors receive min(V_T, D) — equivalently, the face value minus a put option.

Distance-to-Default measures how many standard deviations of asset return
separate the current asset value from the default barrier. A DD of 3 means
the firm is 3σ from insolvency on a one-year horizon. DD < 1 is typically
associated with distress; DD < 0 means the market-implied asset value is
already below the debt face value (technically insolvent).

In the KMV implementation (Moody's Analytics), DD maps to an Empirical
Default Frequency (EDF) using a historical database of defaults at each DD
level. Here we use the theoretical Φ(-DD) which assumes asset returns are
normally distributed — a simplification that overstates tail probabilities
at high DD but provides directionally correct signals.

Aggregate index application
---------------------------
Applying Merton to an index (SP500) rather than a single firm requires
several proxies:
- σ_E comes from the VIX (forward-looking, risk-neutral) rather than
  historical equity vol. This makes the measure more responsive to
  near-term stress.
- Aggregate leverage L is inferred from HY credit spreads rather than
  balance sheet data. This is defensible because credit markets price
  leverage risk in real time; rising spreads signal rising perceived
  leverage/default risk, which our linear mapping captures.
- The model therefore synthesises equity vol (VIX) and credit risk (HY spread)
  into a single Distance-to-Default figure, making it a natural cross-asset
  stress indicator.

Public API
----------
  compute_distance_to_default(df) → pd.DataFrame
  run_merton_analysis(df) → dict
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Normal CDF via math.erf (no scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """
    Standard normal CDF using the relationship Φ(x) = (1 + erf(x/√2)) / 2.

    math.erf is accurate to machine precision. This avoids any dependency
    on scipy.stats.norm while providing the same result.
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_cdf_series(s: pd.Series) -> pd.Series:
    """Vectorised normal CDF applied element-wise to a Series."""
    return s.apply(lambda x: _norm_cdf(x) if pd.notna(x) else float("nan"))


# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

_VIX_ALIASES = ["vix", "vixcls"]
_SP500_ALIASES = ["sp500"]
_HY_ALIASES = ["hy_spread"]


def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Return the first alias found in df.columns, or None."""
    for name in aliases:
        if name in df.columns:
            return name
    return None


# ---------------------------------------------------------------------------
# Merton regime classification
# ---------------------------------------------------------------------------

_DD_THRESHOLDS = [
    (3.0, "Very Low Risk (DD>3)"),
    (2.0, "Low Risk (2-3)"),
    (1.0, "Moderate (1-2)"),
    (0.0, "Elevated (0-1)"),
]


def _classify_dd(dd: float) -> str:
    """
    Map a Distance-to-Default value to a qualitative stress regime.

    Thresholds are based on empirical default research (Crosbie & Bohn 2003):
    - DD > 3: Remote default probability (< 0.1%)
    - DD 2-3: Low risk, consistent with investment-grade
    - DD 1-2: Moderate risk, consistent with BB/B territory
    - DD 0-1: Elevated, consistent with CCC or near-distress
    - DD < 0: Market-implied asset value below debt (technical insolvency)
    """
    if math.isnan(dd):
        return "Unknown"
    for threshold, label in _DD_THRESHOLDS:
        if dd >= threshold:
            return label
    return "Distressed (<0)"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_distance_to_default(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Merton Distance-to-Default for each row of the input DataFrame.

    Uses the aggregate index proxy method described in the module docstring.
    Returns a copy of the input DataFrame with five new columns appended.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain (or have aliases for) sp500, vix, and hy_spread columns.

    Returns
    -------
    pd.DataFrame
        Copy of input with additional columns:
        - sigma_E          : Implied equity volatility (VIX / 100)
        - leverage         : Estimated aggregate debt / asset ratio [0.20, 0.80]
        - sigma_V          : Implied asset volatility (Ito approximation)
        - dd               : Distance-to-Default in standard deviations [-5, 10]
        - default_prob_1y  : 1-year implied default probability [0, 1]
        - merton_regime    : Qualitative stress classification string

    Notes
    -----
    All computations are row-wise (no look-ahead). Missing values in any input
    column propagate as NaN in all output columns for that row.

    The Ito approximation used here (σ_V ≈ σ_E × E/V = σ_E × (1 - L)) is the
    first-order term of the Black-Scholes relationship between equity vol and
    asset vol. It slightly under-estimates σ_V for deep-in-the-money equity
    (low leverage) but is accurate enough for the leverage ratios typical of
    aggregate corporate sectors (L ≈ 0.30–0.60).
    """
    result = df.copy()

    sp500_col = _resolve_col(df, _SP500_ALIASES)
    vix_col = _resolve_col(df, _VIX_ALIASES)
    hy_col = _resolve_col(df, _HY_ALIASES)

    # If any required input is missing, add NaN columns and return early
    if sp500_col is None or vix_col is None or hy_col is None:
        for col in ["sigma_E", "leverage", "sigma_V", "dd", "default_prob_1y", "merton_regime"]:
            result[col] = float("nan")
        result["merton_regime"] = "Unknown"
        return result

    # --- σ_E: annualised implied equity volatility from VIX ---
    # VIX is expressed as a percentage (e.g. 20 means 20% vol), divide by 100.
    sigma_E = result[vix_col] / 100.0

    # --- Leverage L: implied aggregate debt/asset ratio from HY spread ---
    # Intuition: wider HY spreads reflect higher perceived default risk, which
    # is driven in part by higher leverage. The linear mapping is calibrated so
    # that a "normal" HY spread of ~3.5% maps to ~35% leverage (consistent with
    # long-run US corporate sector balance sheets). Each 50bp widening adds ~1%.
    leverage = (0.35 + (result[hy_col] - 3.5) * 0.02).clip(lower=0.20, upper=0.80)

    # --- σ_V: asset volatility via Ito's lemma ---
    # From dE = (∂E/∂V) × dV, where ∂E/∂V = N(d1) ≈ E/V = (1-L) for at-the-money.
    # This is the equity-value-weighted mapping from equity vol to asset vol.
    sigma_V = sigma_E * (1.0 - leverage)

    # --- Distance-to-Default ---
    # Full Merton formula: DD = (ln(V/D) + (μ - σ_V²/2) × T) / (σ_V × √T)
    # Simplifications: T=1, ln(V/D) = ln(1/(V×L/V)) = -ln(L) = ln(1/L), μ=0.
    # This gives: DD = (ln(1/L) - 0.5 × σ_V²) / σ_V
    # Setting μ=0 (risk-neutral drift) is conservative — it understates DD
    # slightly relative to a positive drift assumption, providing a prudent bound.
    dd_raw = (np.log(1.0 / leverage) - 0.5 * sigma_V ** 2) / sigma_V

    # Clip to [-5, 10]: beyond these extremes the CDF is at machine limits
    dd = dd_raw.clip(lower=-5.0, upper=10.0)

    # --- 1-year default probability ---
    # Under risk-neutral measure: P(default) = Φ(-DD)
    # Φ(-DD) increases as DD decreases — lower DD means higher default probability.
    default_prob_1y = _norm_cdf_series(-dd)

    # --- Merton regime classification ---
    merton_regime = dd.apply(lambda x: _classify_dd(x) if pd.notna(x) else "Unknown")

    result["sigma_E"] = sigma_E
    result["leverage"] = leverage
    result["sigma_V"] = sigma_V
    result["dd"] = dd
    result["default_prob_1y"] = default_prob_1y
    result["merton_regime"] = merton_regime

    return result


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def run_merton_analysis(df: pd.DataFrame) -> dict:
    """
    Run the complete Merton analysis pipeline and return a results bundle.

    Computes Distance-to-Default for all rows, extracts the current (most
    recent) reading, identifies historical stress periods, and measures
    correlation with the composite risk score.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset. Required columns: sp500 (or alias), vix (or alias),
        hy_spread. Optional: composite_risk_score_smooth for correlation.

    Returns
    -------
    dict with keys:
        df : pd.DataFrame
            Input DataFrame with Merton columns appended by
            compute_distance_to_default().
        current : dict
            Snapshot of Merton metrics at the most recent valid row:
            dd, default_prob_1y, default_prob_pct, sigma_E, sigma_V,
            leverage, merton_regime, sp500, vix, hy_spread.
        historical_low_dd : pd.DataFrame
            Subset of rows where dd < 1, representing historical stress
            periods where default risk was elevated. Useful for overlaying
            on time series charts to highlight crises.
        correlation_with_composite : float
            Pearson correlation between default_prob_1y and
            composite_risk_score_smooth. Expected to be strongly positive
            (both should rise during stress). NaN if composite column absent.
        available : bool
            False if any required input column (sp500, vix, hy_spread) is
            missing from the DataFrame.

    Notes
    -----
    The `current` dict uses the last row that has non-NaN values for all
    three input variables. This handles trailing NaN values that can arise
    from data pipeline lags on the most recent date.
    """
    sp500_col = _resolve_col(df, _SP500_ALIASES)
    vix_col = _resolve_col(df, _VIX_ALIASES)
    hy_col = _resolve_col(df, _HY_ALIASES)

    if sp500_col is None or vix_col is None or hy_col is None:
        return {
            "df": df.copy(),
            "current": {
                "dd": float("nan"),
                "default_prob_1y": float("nan"),
                "default_prob_pct": float("nan"),
                "sigma_E": float("nan"),
                "sigma_V": float("nan"),
                "leverage": float("nan"),
                "merton_regime": "Unknown",
                "sp500": float("nan"),
                "vix": float("nan"),
                "hy_spread": float("nan"),
            },
            "historical_low_dd": pd.DataFrame(),
            "correlation_with_composite": float("nan"),
            "available": False,
        }

    # --- Compute Merton columns ---
    mdf = compute_distance_to_default(df)

    # --- Current reading (last row with complete inputs) ---
    required_input_cols = [sp500_col, vix_col, hy_col]
    valid_mask = mdf[required_input_cols].notna().all(axis=1)
    valid_rows = mdf[valid_mask]

    if len(valid_rows) == 0:
        current_row = {col: float("nan") for col in required_input_cols}
        current_dd = float("nan")
        current_dp = float("nan")
        current_sigma_E = float("nan")
        current_sigma_V = float("nan")
        current_leverage = float("nan")
        current_regime = "Unknown"
    else:
        last = valid_rows.iloc[-1]
        current_dd = float(last["dd"])
        current_dp = float(last["default_prob_1y"])
        current_sigma_E = float(last["sigma_E"])
        current_sigma_V = float(last["sigma_V"])
        current_leverage = float(last["leverage"])
        current_regime = str(last["merton_regime"])
        current_sp500 = float(last[sp500_col])
        current_vix = float(last[vix_col])
        current_hy = float(last[hy_col])

    current = {
        "dd": current_dd,
        "default_prob_1y": current_dp,
        "default_prob_pct": current_dp * 100.0 if not math.isnan(current_dp) else float("nan"),
        "sigma_E": current_sigma_E,
        "sigma_V": current_sigma_V,
        "leverage": current_leverage,
        "merton_regime": current_regime,
        "sp500": current_sp500 if len(valid_rows) > 0 else float("nan"),
        "vix": current_vix if len(valid_rows) > 0 else float("nan"),
        "hy_spread": current_hy if len(valid_rows) > 0 else float("nan"),
    }

    # --- Historical stress periods (DD < 1) ---
    # DD < 1 historically corresponds to periods of genuine credit stress:
    # the 2001-02 downturn, the 2008-09 financial crisis, the 2020 COVID
    # shock, and the 2022 rate shock. These rows are returned for overlay
    # purposes on time-series visualisations.
    low_dd_mask = mdf["dd"].lt(1.0) & valid_mask
    historical_low_dd = mdf[low_dd_mask].copy()

    # --- Correlation with composite risk score ---
    correlation = float("nan")
    if "composite_risk_score_smooth" in mdf.columns:
        joint = mdf[["default_prob_1y", "composite_risk_score_smooth"]].dropna()
        if len(joint) >= 10:
            # Pearson correlation; expected to be positive because both measures
            # should rise together during risk-off environments.
            corr_matrix = joint.corr()
            correlation = float(corr_matrix.loc["default_prob_1y", "composite_risk_score_smooth"])

    return {
        "df": mdf,
        "current": current,
        "historical_low_dd": historical_low_dd,
        "correlation_with_composite": correlation,
        "available": True,
    }
