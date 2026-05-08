"""
Cross-asset confirmation engine.

Prevents single-score dominance by requiring multi-domain confirmation
before flagging strong regime warnings.

Five domains:
  credit    — credit spreads, credit regime score
  rates     — treasury stress, real yield elevation, MOVE
  liquidity — liquidity score, NFCI tightness
  fx_comm   — FX/commodity score, USD, oil
  equity    — market internals, drawdown, VIX

Each domain is "confirming" when its key signal exceeds its 75th-pct threshold
(or falls below 25th pct for inversely-oriented signals like sp500_drawdown).

Confirmation levels:
  "Confirmed"          — 4–5 domains confirming
  "Partially Confirmed" — 2–3 domains confirming
  "Unconfirmed"        — 0–1 domains confirming

Public API
----------
  compute_domain_thresholds   — per-column 75th/25th pct thresholds
  evaluate_domains            — per-domain confirmation status for one row
  compute_confirmation        — full confirmation result for one row
  run_confirmation_series     — time series of confirmation status
"""

from __future__ import annotations
import numpy as np
import pandas as pd

_PCT_HIGH = 0.75   # confirming when signal > this percentile
_PCT_LOW  = 0.25   # confirming for inverse signals (e.g. deep negative drawdown)

# Domain spec: {domain_name: [(column, direction), ...]}
# direction "high" = confirming when val > 75th pct
# direction "low"  = confirming when val < 25th pct (i.e. severely negative)
_DOMAIN_SPECS: dict[str, list[tuple[str, str]]] = {
    "credit": [
        ("credit_market_risk_score_smooth", "high"),
        ("hy_change_30d",                  "high"),
        ("hy_spread",                      "high"),
    ],
    "rates": [
        ("treasury_stress_score_smooth",   "high"),
        ("real_yield_z",                   "high"),
        ("rates_stress_score_smooth",      "high"),
    ],
    "liquidity": [
        ("liquidity_regime_score_smooth",  "high"),
        ("nfci",                           "high"),
        ("stlfsi",                         "high"),
    ],
    "fx_comm": [
        ("fx_commodity_score_smooth",      "high"),
        ("oil_change_30d",                 "low"),    # oil decline = stress
    ],
    "equity": [
        ("market_internals_score_smooth",  "high"),
        ("sp500_drawdown",                 "low"),    # deep drawdown = stress
        ("vix_change_30d",                 "high"),
    ],
}

DOMAIN_LABELS = {
    "credit":   "Credit",
    "rates":    "Rates",
    "liquidity": "Liquidity",
    "fx_comm":  "FX / Commodity",
    "equity":   "Equity",
}


def compute_domain_thresholds(df: pd.DataFrame) -> dict[str, tuple[float, str]]:
    """
    Compute per-column thresholds over the full df.

    Returns {col: (threshold_value, direction)}.
    """
    thresholds = {}
    for specs in _DOMAIN_SPECS.values():
        for col, direction in specs:
            if col not in df.columns or col in thresholds:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            pct = _PCT_LOW if direction == "low" else _PCT_HIGH
            thresholds[col] = (float(series.quantile(pct)), direction)
    return thresholds


def evaluate_domains(row: pd.Series, thresholds: dict) -> dict[str, dict]:
    """
    Per-domain confirmation status for a single row.

    Returns {domain: {"confirming": bool, "n_signals": int, "n_confirming": int}}.
    """
    result = {}
    for domain, specs in _DOMAIN_SPECS.items():
        n_avail = 0
        n_conf = 0
        for col, direction in specs:
            if col not in row.index or col not in thresholds:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            threshold, thresh_dir = thresholds[col]
            n_avail += 1
            if direction == "high" and thresh_dir == "high" and val > threshold:
                n_conf += 1
            elif direction == "low" and thresh_dir == "low" and val < threshold:
                n_conf += 1
        result[domain] = {
            "confirming":   n_avail > 0 and n_conf > 0,
            "n_signals":    n_avail,
            "n_confirming": n_conf,
        }
    return result


def compute_confirmation(row: pd.Series, thresholds: dict) -> dict:
    """
    Full confirmation result for a single row.

    Returns dict with:
      domain_results       — per-domain confirmation
      n_confirming         — count of confirming domains
      n_domains            — domains with available data
      status               — "Confirmed" / "Partially Confirmed" / "Unconfirmed"
      confirming_domains   — list of domain names that are confirming
    """
    domain_results = evaluate_domains(row, thresholds)
    confirming = [d for d, r in domain_results.items() if r["confirming"]]
    n_conf = len(confirming)
    n_domains = len([d for d, r in domain_results.items() if r["n_signals"] > 0])

    if n_domains == 0:
        status = "Insufficient Data"
    elif n_conf >= 4:
        status = "Confirmed"
    elif n_conf >= 2:
        status = "Partially Confirmed"
    else:
        status = "Unconfirmed"

    return {
        "domain_results":     domain_results,
        "n_confirming":       n_conf,
        "n_domains":          n_domains,
        "status":             status,
        "confirming_domains": confirming,
    }


def run_confirmation_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised confirmation status for every row in df.

    Returns DataFrame with columns:
      confirmation_status, n_confirming_domains,
      {domain}_confirming  (bool per domain)
    Same index as df.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    thresholds = compute_domain_thresholds(df)

    # Build per-domain confirming booleans vectorised
    domain_frames: dict[str, pd.Series] = {}
    for domain, specs in _DOMAIN_SPECS.items():
        any_confirming = pd.Series(False, index=df.index)
        for col, direction in specs:
            if col not in df.columns or col not in thresholds:
                continue
            threshold, _ = thresholds[col]
            if direction == "high":
                any_confirming = any_confirming | (df[col].fillna(np.nan) > threshold)
            else:
                any_confirming = any_confirming | (df[col].fillna(np.nan) < threshold)
        domain_frames[domain] = any_confirming.fillna(False)

    out = pd.DataFrame({f"{d}_confirming": s for d, s in domain_frames.items()})
    out["n_confirming_domains"] = out.sum(axis=1)

    def _status(n: int) -> str:
        if n >= 4:
            return "Confirmed"
        if n >= 2:
            return "Partially Confirmed"
        return "Unconfirmed"

    out["confirmation_status"] = out["n_confirming_domains"].map(_status)
    return out


def get_current_confirmation(df: pd.DataFrame) -> dict:
    """
    Confirmation result for the latest row of df.

    Returns the full confirmation dict from compute_confirmation().
    """
    if df.empty:
        return {}
    thresholds = compute_domain_thresholds(df)
    return compute_confirmation(df.iloc[-1], thresholds)
