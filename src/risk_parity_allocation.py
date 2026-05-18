"""
Risk parity (Equal Risk Contribution) allocation engine.

Computes how much of each credit/rate/equity asset to hold so that each
contributes equally to portfolio volatility, using rolling realized
covariances and the iterative naive-ERC algorithm (pure numpy/pandas only).

Assets tracked: IG Credit, HY Credit, Equities (SP500), 10y Rates, USD (DXY),
plus optional EM, Loan, and BBB proxies when present in df.

Regime-adjusted weights scale ERC outputs for Risk-Off/Caution/Risk-On
environments, and a side-by-side weights table compares ERC vs equal-weight
vs regime-adjusted.

Public API
----------
  ASSET_PROXIES           — asset spec list (col, label, key)
  RISK_PARITY_WINDOW      — rolling cov/vol window (126 days)
  compute_asset_returns   — daily returns DataFrame for available assets
  compute_erc_weights     — ERC weights + risk contributions for last window
  compute_regime_adjusted_weights — regime-scaled ERC weights
  run_risk_parity_allocation — top-level orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSET_PROXIES: list[tuple[str, str, str]] = [
    ("ig_spread",   "IG Credit",        "ig"),
    ("hy_spread",   "HY Credit",        "hy"),
    ("sp500",       "Equities (SP500)", "eq"),
    ("yield_10y",   "10y Rates",        "rates"),
    ("dxy",         "USD (DXY)",        "usd"),
]

# Optional extra assets — added when present
_OPTIONAL_PROXIES: list[tuple[str, str, str]] = [
    ("em_spread",   "EM Credit",        "em"),
    ("loan_spread", "Loans",            "loan"),
    ("bbb_spread",  "BBB Credit",       "bbb"),
]

RISK_PARITY_WINDOW: int   = 126   # rolling window for vol/corr (~6 months)
CASH_VOL:           float = 0.001  # near-zero volatility for cash

_ANNUALIZE:  int   = 252
_ERC_ITERS:  int   = 100
_MIN_ROWS:   int   = 252
_MIN_ASSETS: int   = 2

# Regime adjustments: {key: additive fraction (before renorm)}
_REGIME_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "Risk-Off": {"hy": -0.30, "eq": -0.30, "rates": +0.20, "cash": +0.20},
    "Caution":  {"hy": -0.15, "eq": -0.15, "rates": +0.10, "cash": +0.10},
    "Neutral":  {},
    "Risk-On":  {"hy": +0.15, "eq": +0.15, "rates": -0.10, "cash": -0.10},
}

# Spread assets return = -pct_change (falling spread = positive return)
_SPREAD_KEYS = {"ig", "hy", "em", "loan", "bbb"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _all_asset_specs(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return base + optional proxy specs whose columns are present in df."""
    all_specs = ASSET_PROXIES + _OPTIONAL_PROXIES
    return [(col, label, key) for col, label, key in all_specs if col in df.columns]


# ---------------------------------------------------------------------------
# Asset returns
# ---------------------------------------------------------------------------

def compute_asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns for each available asset proxy.

    - Spread assets (IG, HY, EM, Loan, BBB): return = -1 * pct_change
      Rationale: falling spreads = positive economic return for credit holders.
    - Price/rate assets (SP500, yield_10y, DXY): return = pct_change

    Returns DataFrame of returns with one column per available asset, named
    by the asset key (e.g. 'hy', 'ig', 'eq').  NaN rows are NOT dropped here
    so that the index aligns with df; callers should dropna as needed.
    """
    out: dict[str, pd.Series] = {}
    specs = _all_asset_specs(df)

    for col, label, key in specs:
        series = df[col].copy().astype(float)
        # Replace 0s in price series to avoid division errors
        if key not in _SPREAD_KEYS:
            series = series.replace(0, np.nan)

        pct = series.pct_change()

        if key in _SPREAD_KEYS:
            out[key] = -1.0 * pct
        else:
            out[key] = pct

    if not out:
        return pd.DataFrame(index=df.index)

    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# ERC algorithm
# ---------------------------------------------------------------------------

def compute_erc_weights(returns_df: pd.DataFrame, window: int = RISK_PARITY_WINDOW) -> dict:
    """
    Compute Equal Risk Contribution (ERC) weights using the rolling covariance
    matrix from the most recent `window` observations.

    Algorithm (iterative naive risk parity, 100 iterations):
      1. Cov = rolling window covariance matrix (annualized)
      2. Start with equal weights: w = [1/N, ...]
      3. Iterate:
           sigma_p = sqrt(w @ Cov @ w)
           marginal_contrib = Cov @ w / sigma_p
           risk_contrib = w * marginal_contrib
           target = sigma_p / N
           w = w * (target / risk_contrib)
           w = w / sum(w)

    Returns
    -------
    {
        weights: {asset_key: weight},
        risk_contributions: {asset_key: pct_contribution},
        portfolio_vol: float,
        cov_matrix: np.ndarray,
        n_assets: int,
        available: bool,
    }
    """
    _fail = {
        "available": False,
        "weights": {},
        "risk_contributions": {},
        "portfolio_vol": None,
        "cov_matrix": None,
        "n_assets": 0,
    }

    try:
        if returns_df is None or returns_df.empty:
            return _fail

        # Use the last `window` rows after dropping any rows with all-NaN
        recent = returns_df.dropna(how="all").tail(window)
        # Drop columns that are entirely NaN in this window
        recent = recent.dropna(axis=1, how="all")

        cols = list(recent.columns)
        N    = len(cols)
        if N < _MIN_ASSETS:
            return _fail

        # Fill any remaining NaNs with 0 (missing observations within the window)
        mat = recent[cols].fillna(0.0).values  # shape (T, N)

        if mat.shape[0] < 20:
            return _fail

        # Annualized covariance
        cov = np.cov(mat, rowvar=False) * _ANNUALIZE  # (N, N)
        # Regularize: add tiny diagonal to ensure positive definiteness
        cov = cov + np.eye(N) * 1e-10

        # --- Iterative ERC ---
        w = np.ones(N) / N

        for _ in range(_ERC_ITERS):
            var_p = float(w @ cov @ w)
            if var_p < 1e-14:
                break
            sigma_p       = np.sqrt(var_p)
            marginal      = cov @ w / sigma_p
            risk_contrib  = w * marginal
            total_rc      = float(risk_contrib.sum())
            if abs(total_rc) < 1e-14:
                break
            target        = sigma_p / N
            # Avoid divide-by-zero in risk_contrib
            safe_rc       = np.where(np.abs(risk_contrib) < 1e-14, 1e-14, risk_contrib)
            w             = w * (target / safe_rc)
            w_sum         = w.sum()
            if w_sum < 1e-14:
                w = np.ones(N) / N
            else:
                w = w / w_sum

        # Clip negatives that can arise from numerical noise, renorm
        w = np.clip(w, 0.0, None)
        w_sum = w.sum()
        if w_sum < 1e-9:
            return _fail
        w = w / w_sum

        # Final portfolio vol
        portfolio_vol = float(np.sqrt(w @ cov @ w))

        # Risk contributions as percentages
        sigma_p      = portfolio_vol if portfolio_vol > 1e-9 else 1e-9
        marginal     = cov @ w / sigma_p
        risk_contrib = w * marginal
        rc_pct       = risk_contrib / risk_contrib.sum() * 100.0

        weights_dict       = {cols[i]: round(float(w[i]),      6) for i in range(N)}
        risk_contrib_dict  = {cols[i]: round(float(rc_pct[i]), 2) for i in range(N)}

        return {
            "available":          True,
            "weights":            weights_dict,
            "risk_contributions": risk_contrib_dict,
            "portfolio_vol":      round(portfolio_vol, 6),
            "cov_matrix":         cov,
            "n_assets":           N,
        }

    except Exception:
        return _fail


# ---------------------------------------------------------------------------
# Regime-adjusted weights
# ---------------------------------------------------------------------------

def compute_regime_adjusted_weights(
    erc_weights: dict,
    current_regime: str,
) -> dict:
    """
    Adjust ERC weights for the current regime.

    Adjustments (additive, then renormalized to sum to 1):
      Risk-Off: HY -30%, EQ -30%, Rates +20%, Cash +20%
      Caution:  HY -15%, EQ -15%, Rates +10%, Cash +10%
      Neutral:  no adjustment
      Risk-On:  HY +15%, EQ +15%, Rates -10%, Cash -10%

    "Cash" is a synthetic slot — added to the weight dict if not already
    present with weight 0, then adjusted and renormalized.

    Returns dict of adjusted weights (including 'cash' key if regime adds to it),
    all values summing to 1.0.
    """
    try:
        raw: dict[str, float] = {k: float(v) for k, v in erc_weights.items()}
        adjustments = _REGIME_ADJUSTMENTS.get(current_regime, {})

        if not adjustments:
            # Neutral: no change
            total = sum(raw.values())
            if total < 1e-9:
                return raw
            return {k: v / total for k, v in raw.items()}

        # Ensure cash slot exists
        if "cash" not in raw:
            raw["cash"] = 0.0

        # Apply additive adjustments
        for key, adj in adjustments.items():
            raw[key] = raw.get(key, 0.0) + adj

        # Clip negatives
        raw = {k: max(v, 0.0) for k, v in raw.items()}

        # Remove cash if it rounded to zero (cleaner output)
        if raw.get("cash", 0.0) < 1e-9:
            raw.pop("cash", None)

        total = sum(raw.values())
        if total < 1e-9:
            n = len(raw)
            return {k: 1.0 / n for k in raw} if n > 0 else {}

        return {k: round(v / total, 6) for k, v in raw.items()}

    except Exception:
        return erc_weights


# ---------------------------------------------------------------------------
# Rolling ERC weights (for history chart)
# ---------------------------------------------------------------------------

def _compute_rolling_erc_weights(
    returns_df: pd.DataFrame,
    window: int = RISK_PARITY_WINDOW,
    step: int = 5,
) -> pd.DataFrame:
    """
    Compute ERC weights on a rolling basis, sampled every `step` rows.
    Returns a DataFrame indexed by date with asset keys as columns.
    Last 252 rows sampled.
    """
    try:
        if returns_df is None or returns_df.empty:
            return pd.DataFrame()

        clean = returns_df.dropna(how="all")
        tail  = clean.tail(252)
        cols  = list(returns_df.columns)
        rows: list[dict] = []

        indices = list(range(window, len(tail) + 1, step))
        if not indices or indices[-1] < len(tail):
            indices.append(len(tail))

        for end in indices:
            if end < window:
                continue
            chunk = tail.iloc[max(0, end - window): end]
            result = compute_erc_weights(chunk, window=window)
            if not result.get("available"):
                continue
            row = {"date": tail.index[end - 1]}
            row.update(result["weights"])
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows).set_index("date")
        # Fill asset columns that were absent in some windows
        for col in cols:
            if col not in out.columns:
                out[col] = np.nan
        return out

    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_risk_parity_allocation(df: pd.DataFrame) -> dict:
    """
    Top-level entry point.  Requires >= 2 asset proxies present in df and
    at least 252 rows of history.

    Returns
    -------
    {
        available: bool,
        erc_weights: dict,
        regime_weights: dict,
        equal_weights: dict,
        risk_contributions: dict,
        portfolio_vol_erc: float,
        portfolio_vol_equal: float,
        current_regime: str,
        assets_used: list[str],
        weights_table: pd.DataFrame,
        rolling_weights: pd.DataFrame,
        interpretation: str,
    }
    """
    _fail = {
        "available":           False,
        "erc_weights":         {},
        "regime_weights":      {},
        "equal_weights":       {},
        "risk_contributions":  {},
        "portfolio_vol_erc":   None,
        "portfolio_vol_equal": None,
        "current_regime":      None,
        "assets_used":         [],
        "weights_table":       pd.DataFrame(),
        "rolling_weights":     pd.DataFrame(),
        "interpretation":      "Insufficient data for risk parity allocation.",
    }

    try:
        if df is None or df.empty:
            return _fail

        if len(df) < _MIN_ROWS:
            return _fail

        specs = _all_asset_specs(df)
        if len(specs) < _MIN_ASSETS:
            return _fail

        # ---- Detect current regime ----
        regime_col = None
        for cand in ("final_decision", "composite_risk_label"):
            if cand in df.columns:
                regime_col = cand
                break

        current_regime = "Neutral"
        if regime_col is not None:
            last_regime = df[regime_col].dropna()
            if not last_regime.empty:
                current_regime = str(last_regime.iloc[-1])
        else:
            # Fall back to composite score threshold
            if "composite_risk_score_smooth" in df.columns:
                last_score = _safe_float(df["composite_risk_score_smooth"].iloc[-1])
                if last_score is not None:
                    if last_score >= 70:
                        current_regime = "Risk-Off"
                    elif last_score >= 50:
                        current_regime = "Caution"
                    elif last_score >= 25:
                        current_regime = "Neutral"
                    else:
                        current_regime = "Risk-On"

        # ---- Compute asset returns ----
        returns_df = compute_asset_returns(df)
        if returns_df.empty:
            return _fail

        assets_used = list(returns_df.columns)
        if len(assets_used) < _MIN_ASSETS:
            return _fail

        N = len(assets_used)

        # ---- ERC weights ----
        erc_result = compute_erc_weights(returns_df, window=RISK_PARITY_WINDOW)
        if not erc_result.get("available"):
            return _fail

        erc_weights       = erc_result["weights"]
        risk_contributions = erc_result["risk_contributions"]
        portfolio_vol_erc  = erc_result["portfolio_vol"]

        # ---- Equal weights ----
        equal_weights = {k: round(1.0 / N, 6) for k in assets_used}

        # Portfolio vol under equal weights
        cov = erc_result["cov_matrix"]
        eq_w_arr = np.array([equal_weights.get(k, 0.0) for k in assets_used])
        portfolio_vol_equal = float(np.sqrt(eq_w_arr @ cov @ eq_w_arr))

        # ---- Regime-adjusted weights ----
        regime_weights = compute_regime_adjusted_weights(erc_weights, current_regime)

        # ---- Weights comparison table ----
        label_map = {col: lbl for col, lbl, key in _all_asset_specs(df)
                     for col2, lbl2, key2 in [(col, lbl, key)] if key2 == key}
        key_to_label = {key: lbl for col, lbl, key in _all_asset_specs(df)}

        table_rows: list[dict] = []
        all_keys = set(erc_weights) | set(regime_weights) | set(equal_weights)
        for key in assets_used:
            row = {
                "Asset":            key_to_label.get(key, key),
                "ERC Weight":       round(erc_weights.get(key, 0.0) * 100, 1),
                "Equal Weight":     round(equal_weights.get(key, 0.0) * 100, 1),
                "Regime-Adj Weight": round(regime_weights.get(key, 0.0) * 100, 1),
                "Risk Contribution %": round(risk_contributions.get(key, 0.0), 1),
            }
            table_rows.append(row)

        # Add cash row if regime adjustment added it
        if "cash" in regime_weights and "cash" not in assets_used:
            table_rows.append({
                "Asset":             "Cash",
                "ERC Weight":        0.0,
                "Equal Weight":      0.0,
                "Regime-Adj Weight": round(regime_weights["cash"] * 100, 1),
                "Risk Contribution %": 0.0,
            })

        weights_table = (
            pd.DataFrame(table_rows).set_index("Asset")
            if table_rows else pd.DataFrame()
        )

        # ---- Rolling ERC weights ----
        rolling_weights = _compute_rolling_erc_weights(
            returns_df, window=RISK_PARITY_WINDOW, step=5
        )

        # ---- Interpretation ----
        interpretation = _build_interpretation(
            erc_weights=erc_weights,
            regime_weights=regime_weights,
            equal_weights=equal_weights,
            risk_contributions=risk_contributions,
            portfolio_vol_erc=portfolio_vol_erc,
            portfolio_vol_equal=portfolio_vol_equal,
            current_regime=current_regime,
            key_to_label=key_to_label,
            N=N,
        )

        return {
            "available":           True,
            "erc_weights":         erc_weights,
            "regime_weights":      regime_weights,
            "equal_weights":       equal_weights,
            "risk_contributions":  risk_contributions,
            "portfolio_vol_erc":   round(portfolio_vol_erc, 6),
            "portfolio_vol_equal": round(portfolio_vol_equal, 6),
            "current_regime":      current_regime,
            "assets_used":         assets_used,
            "weights_table":       weights_table,
            "rolling_weights":     rolling_weights,
            "interpretation":      interpretation,
        }

    except Exception:
        return _fail


def _build_interpretation(
    erc_weights: dict,
    regime_weights: dict,
    equal_weights: dict,
    risk_contributions: dict,
    portfolio_vol_erc: float,
    portfolio_vol_equal: float,
    current_regime: str,
    key_to_label: dict,
    N: int,
) -> str:
    """Generate a plain-English narrative for the risk parity allocation."""
    try:
        parts: list[str] = []

        vol_erc_pct   = portfolio_vol_erc   * 100
        vol_equal_pct = portfolio_vol_equal * 100
        parts.append(
            f"Under Equal Risk Contribution, portfolio annualized volatility is "
            f"{vol_erc_pct:.1f}% vs {vol_equal_pct:.1f}% for equal weighting "
            f"across {N} assets."
        )

        # Largest ERC weight
        if erc_weights:
            top_key = max(erc_weights, key=lambda k: erc_weights[k])
            top_lbl = key_to_label.get(top_key, top_key)
            parts.append(
                f"The largest ERC weight goes to {top_lbl} "
                f"({erc_weights[top_key]*100:.1f}%), reflecting its lower volatility."
            )

        # Largest risk contributor
        if risk_contributions:
            top_rc_key = max(risk_contributions, key=lambda k: risk_contributions[k])
            top_rc_lbl = key_to_label.get(top_rc_key, top_rc_key)
            parts.append(
                f"{top_rc_lbl} contributes the most risk "
                f"({risk_contributions[top_rc_key]:.1f}%)."
            )

        # Regime adjustment summary
        parts.append(
            f"In {current_regime} regime, regime-adjusted weights "
            + (
                "are unchanged from ERC."
                if current_regime == "Neutral"
                else "reduce high-risk asset exposures and increase defensive allocations."
                if current_regime in ("Caution", "Risk-Off")
                else "tilt toward higher-risk assets relative to ERC."
            )
        )

        return " ".join(parts)

    except Exception:
        return "Risk parity interpretation unavailable."
