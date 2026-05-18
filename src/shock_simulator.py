"""
Shock simulator.

Estimates OLS sensitivities of each composite sub-signal to a set of
observable market inputs (hy_spread, vix, yield_10y, yield_3m, sp500),
then applies user-specified shocks to produce a stressed composite
risk score and regime estimate.

Public API
----------
  SHOCK_INPUTS        — list of market-input column names
  SUB_SIGNALS         — list of smoothed sub-signal column names
  COMPOSITE           — composite risk score column name
  estimate_sensitivities — OLS betas for each sub-signal vs SHOCK_INPUTS
  apply_shock         — delta shocks → stressed sub-signals + composite
  get_default_shocks  — five named pre-built shock scenarios
  run_shock_analysis  — top-level orchestrator returning full results dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

SHOCK_INPUTS: list[str] = [
    "hy_spread",
    "vix",
    "yield_10y",
    "yield_3m",
    "sp500",
]

SUB_SIGNALS: list[str] = [
    "treasury_stress_score_smooth",
    "credit_market_risk_score_smooth",
    "macro_risk_score_smooth",
    "enhanced_funding_stress_score_smooth",
    "rates_stress_score_smooth",
    "fx_commodity_score_smooth",
    "banking_stress_score_smooth",
]

COMPOSITE: str = "composite_risk_score_smooth"

# Regime thresholds (composite score → label)
_REGIME_THRESHOLDS = [
    (30.0, "Risk-On"),
    (50.0, "Neutral"),
    (65.0, "Caution"),
    (float("inf"), "Risk-Off"),
]

_OLS_LOOKBACK: int = 252
_OLS_MIN_ROWS: int = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_regime(score: float) -> str:
    """Map a composite score to a regime label."""
    for threshold, label in _REGIME_THRESHOLDS:
        if score < threshold:
            return label
    return "Risk-Off"


def _ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Fit OLS via the normal equations.

    Parameters
    ----------
    X : (n, p) design matrix (already includes constant column).
    y : (n,) target vector.

    Returns
    -------
    beta : (p,) coefficient vector.
    r2   : coefficient of determination (0–1).
    """
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Singular matrix — fall back to least squares
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return beta, float(np.clip(r2, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public: estimate_sensitivities
# ---------------------------------------------------------------------------

def estimate_sensitivities(df: pd.DataFrame) -> dict:
    """Estimate OLS beta for each sub-signal vs SHOCK_INPUTS using last 252 rows.

    Uses the last ``_OLS_LOOKBACK`` rows of ``df`` after dropping any rows that
    have NaN values in the sub-signal or any SHOCK_INPUTS column.

    Returns
    -------
    dict keyed by sub-signal name.  Each value is a dict mapping each
    SHOCK_INPUTS column name to its beta coefficient, plus ``"r2"``.
    If a sub-signal has fewer than ``_OLS_MIN_ROWS`` clean rows the entry
    is an empty dict.
    """
    try:
        tail = df.iloc[-_OLS_LOOKBACK:].copy()
        available_inputs = [c for c in SHOCK_INPUTS if c in tail.columns]
        result: dict[str, dict] = {}

        for signal in SUB_SIGNALS:
            if signal not in tail.columns:
                result[signal] = {}
                continue

            cols_needed = [signal] + available_inputs
            clean = tail[cols_needed].dropna()

            if len(clean) < _OLS_MIN_ROWS:
                result[signal] = {}
                continue

            y = clean[signal].values.astype(float)
            X_raw = clean[available_inputs].values.astype(float)
            # Add constant (intercept)
            ones = np.ones((len(y), 1))
            X = np.hstack([ones, X_raw])

            beta, r2 = _ols_fit(X, y)
            # beta[0] is the intercept; beta[1:] correspond to available_inputs
            entry: dict[str, float] = {}
            for i, col in enumerate(available_inputs):
                entry[col] = float(beta[i + 1])
            # Fill missing inputs with zero (no sensitivity data)
            for col in SHOCK_INPUTS:
                if col not in entry:
                    entry[col] = 0.0
            entry["r2"] = r2
            result[signal] = entry

        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public: apply_shock
# ---------------------------------------------------------------------------

def apply_shock(df: pd.DataFrame, shocks: dict) -> dict:
    """Apply a shock scenario and return updated signal estimates.

    Parameters
    ----------
    df     : Master DataFrame (DatetimeIndex, all signal columns present).
    shocks : Mapping of SHOCK_INPUTS column name → delta value.
             e.g. ``{"hy_spread": 50, "vix": 5, "sp500": -0.10}``.
             sp500 deltas are treated as fractions (e.g. -0.10 = -10 pct).
             Only keys present in ``SHOCK_INPUTS`` are used; others ignored.

    Returns
    -------
    dict with keys:
      available        : bool
      baseline         : {composite: float, sub_signals: {name: float}}
      shocked          : {composite: float, sub_signals: {name: float}}
      deltas           : {composite: float, sub_signals: {name: float}}
      regime_baseline  : str
      regime_shocked   : str
      biggest_movers   : list of {name, delta}, top 3 by abs(delta)
      interpretation   : str
    """
    try:
        if df.empty:
            return {"available": False}

        sensitivities = estimate_sensitivities(df)

        # ---- Baseline values (last row) ----
        last = df.iloc[-1]

        def _safe_float(val: object) -> float:
            try:
                v = float(val)
                return v if np.isfinite(v) else np.nan
            except (TypeError, ValueError):
                return np.nan

        baseline_subs: dict[str, float] = {}
        for sig in SUB_SIGNALS:
            baseline_subs[sig] = _safe_float(last.get(sig, np.nan))

        baseline_composite = _safe_float(last.get(COMPOSITE, np.nan))
        if np.isnan(baseline_composite):
            # Fall back to mean of available sub-signals
            vals = [v for v in baseline_subs.values() if not np.isnan(v)]
            baseline_composite = float(np.mean(vals)) if vals else np.nan

        # ---- Apply shocks to sub-signals ----
        # Validate shock keys
        valid_shocks: dict[str, float] = {}
        for k, v in shocks.items():
            if k in SHOCK_INPUTS:
                try:
                    valid_shocks[k] = float(v)
                except (TypeError, ValueError):
                    pass

        shocked_subs: dict[str, float] = {}
        for sig in SUB_SIGNALS:
            bsig = baseline_subs[sig]
            if np.isnan(bsig):
                shocked_subs[sig] = np.nan
                continue

            betas = sensitivities.get(sig, {})
            delta = 0.0
            for inp, shock_val in valid_shocks.items():
                b = betas.get(inp, 0.0)
                delta += b * shock_val
            shocked_subs[sig] = float(np.clip(bsig + delta, 0.0, 100.0))

        # ---- Shocked composite ----
        # Use equal weights across sub-signals that have valid values
        valid_shocked = [(k, v) for k, v in shocked_subs.items() if not np.isnan(v)]
        if valid_shocked:
            shocked_composite = float(
                np.clip(np.mean([v for _, v in valid_shocked]), 0.0, 100.0)
            )
        else:
            shocked_composite = np.nan

        # ---- Deltas ----
        sub_deltas: dict[str, float] = {}
        for sig in SUB_SIGNALS:
            b = baseline_subs[sig]
            s = shocked_subs[sig]
            if np.isnan(b) or np.isnan(s):
                sub_deltas[sig] = np.nan
            else:
                sub_deltas[sig] = round(s - b, 4)

        composite_delta = (
            round(shocked_composite - baseline_composite, 4)
            if not (np.isnan(shocked_composite) or np.isnan(baseline_composite))
            else np.nan
        )

        # ---- Regimes ----
        regime_baseline = (
            _classify_regime(baseline_composite)
            if not np.isnan(baseline_composite)
            else "Unknown"
        )
        regime_shocked = (
            _classify_regime(shocked_composite)
            if not np.isnan(shocked_composite)
            else "Unknown"
        )

        # ---- Biggest movers ----
        mover_list = [
            {"name": sig, "delta": sub_deltas[sig]}
            for sig in SUB_SIGNALS
            if not np.isnan(sub_deltas.get(sig, np.nan))
        ]
        mover_list.sort(key=lambda x: abs(x["delta"]), reverse=True)
        biggest_movers = mover_list[:3]

        # ---- Interpretation ----
        shock_desc = ", ".join(
            f"{k} {'+' if v >= 0 else ''}{v}" for k, v in valid_shocks.items()
        ) if valid_shocks else "no shocks"

        if np.isnan(composite_delta):
            interpretation = "Insufficient data to compute shock impact."
        else:
            direction = "rises" if composite_delta > 0 else "falls"
            regime_note = (
                f"Regime unchanged ({regime_baseline})."
                if regime_baseline == regime_shocked
                else f"Regime shifts from {regime_baseline} to {regime_shocked}."
            )
            mover_names = ", ".join(m["name"].replace("_smooth", "").replace("_score", "") for m in biggest_movers[:2])
            interpretation = (
                f"Under shock ({shock_desc}), composite risk score {direction} by "
                f"{abs(composite_delta):.1f} pts "
                f"(baseline {baseline_composite:.1f} → {shocked_composite:.1f}). "
                f"{regime_note} "
                f"Largest movers: {mover_names}."
            )

        return {
            "available": True,
            "baseline": {
                "composite": round(float(baseline_composite), 4) if not np.isnan(baseline_composite) else None,
                "sub_signals": {k: (round(float(v), 4) if not np.isnan(v) else None) for k, v in baseline_subs.items()},
            },
            "shocked": {
                "composite": round(float(shocked_composite), 4) if not np.isnan(shocked_composite) else None,
                "sub_signals": {k: (round(float(v), 4) if not np.isnan(v) else None) for k, v in shocked_subs.items()},
            },
            "deltas": {
                "composite": composite_delta,
                "sub_signals": {k: (round(float(v), 4) if not np.isnan(v) else None) for k, v in sub_deltas.items()},
            },
            "regime_baseline": regime_baseline,
            "regime_shocked": regime_shocked,
            "biggest_movers": biggest_movers,
            "interpretation": interpretation,
        }
    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Public: get_default_shocks
# ---------------------------------------------------------------------------

def get_default_shocks() -> dict:
    """Return five pre-built shock scenario dicts.

    Returns
    -------
    dict mapping scenario name → shocks dict (SHOCK_INPUTS keys → delta values).
    """
    return {
        "Mild Stress":    {"hy_spread": 50,  "vix": 5,  "sp500": -0.05},
        "Credit Crunch":  {"hy_spread": 200, "vix": 25, "sp500": -0.15},
        "Rate Shock":     {"yield_10y": 1.0, "yield_3m": 0.5},
        "Risk-Off":       {"hy_spread": 300, "vix": 40, "sp500": -0.20, "yield_10y": -0.5},
        "Stagflation":    {"yield_10y": 1.5, "vix": 20, "hy_spread": 150, "sp500": -0.10},
    }


# ---------------------------------------------------------------------------
# Public: run_shock_analysis
# ---------------------------------------------------------------------------

def run_shock_analysis(df: pd.DataFrame) -> dict:
    """Top-level shock analysis orchestrator.

    Estimates OLS sensitivities, gathers current market levels, and runs
    all five default shock scenarios.

    Parameters
    ----------
    df : Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict:
      available          : bool
      sensitivities      : dict from estimate_sensitivities()
      baseline_composite : float — latest composite risk score
      baseline_regime    : str  — current regime label
      default_scenarios  : dict mapping scenario name → apply_shock() result
      current_values     : dict mapping SHOCK_INPUTS col → latest observed value
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        # Check that we have at least the composite score
        if COMPOSITE not in df.columns:
            return {"available": False}

        sensitivities = estimate_sensitivities(df)

        last = df.iloc[-1]

        def _safe(col: str) -> object:
            v = last.get(col, np.nan)
            try:
                fv = float(v)
                return round(fv, 6) if np.isfinite(fv) else None
            except (TypeError, ValueError):
                return None

        current_values: dict[str, object] = {col: _safe(col) for col in SHOCK_INPUTS}

        baseline_composite_raw = _safe(COMPOSITE)
        baseline_composite = float(baseline_composite_raw) if baseline_composite_raw is not None else np.nan
        baseline_regime = _classify_regime(baseline_composite) if not np.isnan(baseline_composite) else "Unknown"

        default_shocks = get_default_shocks()
        default_scenarios: dict[str, dict] = {}
        for name, shocks in default_shocks.items():
            default_scenarios[name] = apply_shock(df, shocks)

        return {
            "available": True,
            "sensitivities": sensitivities,
            "baseline_composite": round(baseline_composite, 4) if not np.isnan(baseline_composite) else None,
            "baseline_regime": baseline_regime,
            "default_scenarios": default_scenarios,
            "current_values": current_values,
        }
    except Exception:
        return {"available": False}
