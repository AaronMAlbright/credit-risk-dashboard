"""
Regime-conditional forward return table.

Given today's regime, what has historically happened to key assets over the
next 1/3/6/12 months? This translates the signal model into a concrete
probability-weighted return expectation.

Horizons: 30, 60, 126, 252 trading days (approx 1.5m, 3m, 6m, 12m)
Assets:   SP500, HY spread (level change), IG spread (level change),
          HY total return (if available), strategy return

Statistics per (regime, horizon, asset):
  mean_return   — average outcome
  hit_rate      — % of observations with positive return (or spread tightening)
  pct_10        — 10th percentile (downside scenario)
  pct_90        — 90th percentile (upside scenario)
  n_obs         — sample size (flag if < 20)

Key insight: Risk-On periods have historically produced dramatically better
forward returns across ALL horizons. Risk-Off entry points — while scary —
have produced some of the best 6–12 month forward returns when regimes
eventually recover. This teaches "the entry point matters most."

Public API
----------
  HORIZONS = [30, 60, 126, 252]
  ASSETS   = ["sp500", "hy_spread", "ig_spread", "hy_total_return", "strategy"]
  compute_regime_return_table(df) -> dict
  get_current_regime_outlook(df) -> dict
  run_regime_return_analysis(df) -> dict
"""

import numpy as np
import pandas as pd

HORIZONS = [30, 60, 126, 252]
ASSETS = ["sp500", "hy_spread", "ig_spread", "hy_total_return", "strategy"]

HORIZON_LABELS = {
    30: "1M",
    60: "3M",
    126: "6M",
    252: "12M",
}

REGIME_ORDER = ["Risk-On", "Neutral", "Caution", "Risk-Off"]

MIN_OBS = 20


def _compute_forward_series(df: pd.DataFrame) -> dict:
    """
    Compute forward return / spread-change series for every horizon and asset.

    Returns a dict: {(asset, horizon): pd.Series (index aligned with df)}
    """
    result = {}

    # ---- SP500 cumulative forward return ----
    if "sp500" in df.columns:
        for h in HORIZONS:
            fwd = df["sp500"].shift(-h) / df["sp500"] - 1
            result[("sp500", h)] = fwd

    # ---- HY spread level change (negative = tightening = good) ----
    if "hy_spread" in df.columns:
        for h in HORIZONS:
            fwd = df["hy_spread"].shift(-h) - df["hy_spread"]
            result[("hy_spread", h)] = fwd

    # ---- IG spread level change ----
    if "ig_spread" in df.columns:
        for h in HORIZONS:
            fwd = df["ig_spread"].shift(-h) - df["ig_spread"]
            result[("ig_spread", h)] = fwd

    # ---- HY total return (compound over h days, then shift back) ----
    if "hy_total_return_daily" in df.columns:
        ret = df["hy_total_return_daily"].fillna(0.0)
        for h in HORIZONS:
            rolling_prod = (1 + ret).rolling(h).apply(np.prod, raw=True)
            fwd = rolling_prod.shift(-h) - 1
            result[("hy_total_return", h)] = fwd

    # ---- Strategy return ----
    if "strategy_daily_return" in df.columns:
        ret = df["strategy_daily_return"].fillna(0.0)
        for h in HORIZONS:
            rolling_prod = (1 + ret).rolling(h).apply(np.prod, raw=True)
            fwd = rolling_prod.shift(-h) - 1
            result[("strategy", h)] = fwd

    return result


def _is_spread_asset(asset: str) -> bool:
    """Spread assets: lower (tightening) is good, so hit_rate = fraction < 0."""
    return asset in ("hy_spread", "ig_spread")


def _stats_for_series(series: pd.Series, is_spread: bool) -> dict:
    """Compute summary stats for a single (asset, horizon, regime) slice."""
    valid = series.dropna()
    n = len(valid)
    if n == 0:
        return {
            "n_obs": 0,
            "mean": np.nan,
            "pct_10": np.nan,
            "pct_90": np.nan,
            "hit_rate": np.nan,
        }

    hit_rate = (valid < 0).mean() if is_spread else (valid > 0).mean()
    return {
        "n_obs": n,
        "mean": valid.mean(),
        "pct_10": valid.quantile(0.10),
        "pct_90": valid.quantile(0.90),
        "hit_rate": hit_rate,
    }


def compute_regime_return_table(df: pd.DataFrame) -> dict:
    """
    Build the full regime-conditional forward return table.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with at minimum 'final_decision' and 'sp500'.

    Returns
    -------
    dict with keys:
        "table"            : pd.DataFrame, MultiIndex (regime, horizon)
        "regimes"          : list of regime strings found in data
        "horizons"         : HORIZONS
        "assets_available" : list of asset keys that had data
        "available"        : bool
    """
    if "final_decision" not in df.columns:
        return {"available": False, "table": pd.DataFrame(), "regimes": [],
                "horizons": HORIZONS, "assets_available": []}

    fwd_series = _compute_forward_series(df)
    assets_available = sorted({a for (a, _) in fwd_series.keys()})

    regimes_in_data = [r for r in REGIME_ORDER if r in df["final_decision"].values]

    rows = []
    index_tuples = []

    for regime in regimes_in_data:
        mask = df["final_decision"] == regime
        for h in HORIZONS:
            row = {}
            n_obs_min = None
            for asset in assets_available:
                key = (asset, h)
                if key not in fwd_series:
                    continue
                regime_series = fwd_series[key][mask]
                stats = _stats_for_series(regime_series, _is_spread_asset(asset))
                row[f"mean_{asset}"] = stats["mean"]
                row[f"hit_rate_{asset}"] = stats["hit_rate"]
                row[f"pct_10_{asset}"] = stats["pct_10"]
                row[f"pct_90_{asset}"] = stats["pct_90"]
                n_obs = stats["n_obs"]
                if n_obs_min is None or n_obs < n_obs_min:
                    n_obs_min = nobs = n_obs
            row["n_obs"] = n_obs_min if n_obs_min is not None else 0
            row["horizon_label"] = HORIZON_LABELS[h]
            row["low_sample"] = (row["n_obs"] < MIN_OBS)
            rows.append(row)
            index_tuples.append((regime, h))

    if not rows:
        return {"available": False, "table": pd.DataFrame(), "regimes": regimes_in_data,
                "horizons": HORIZONS, "assets_available": assets_available}

    table = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index_tuples, names=["regime", "horizon"]))

    return {
        "table": table,
        "regimes": regimes_in_data,
        "horizons": HORIZONS,
        "assets_available": assets_available,
        "available": True,
    }


def get_current_regime_outlook(df: pd.DataFrame) -> dict:
    """
    Return the forward-return outlook for the CURRENT regime.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys:
        "regime"          : str
        "outlook_table"   : pd.DataFrame (slice of full table for this regime)
        "recommendation"  : str (plain English narrative)
        "n_obs_min"       : int (lowest observation count across horizons)
        "available"       : bool
    """
    if "final_decision" not in df.columns or df.empty:
        return {"available": False, "regime": None, "outlook_table": pd.DataFrame(),
                "recommendation": "Data not available.", "n_obs_min": 0}

    current_regime = df.iloc[-1]["final_decision"]
    result = compute_regime_return_table(df)

    if not result["available"]:
        return {"available": False, "regime": current_regime, "outlook_table": pd.DataFrame(),
                "recommendation": "Insufficient data for regime analysis.", "n_obs_min": 0}

    table = result["table"]

    if current_regime not in table.index.get_level_values("regime"):
        return {"available": False, "regime": current_regime, "outlook_table": pd.DataFrame(),
                "recommendation": f"No historical data for regime '{current_regime}'.", "n_obs_min": 0}

    outlook = table.xs(current_regime, level="regime")
    n_obs_min = int(outlook["n_obs"].min()) if "n_obs" in outlook.columns else 0

    # Build narrative using 6-month (126d) and 12-month (252d) data where available
    recommendation_parts = [f"In **{current_regime}** regimes:"]

    for h, label in [(126, "6-month"), (252, "12-month")]:
        if h not in outlook.index:
            continue
        row = outlook.loc[h]
        sp500_mean = row.get("mean_sp500", np.nan)
        sp500_hr = row.get("hit_rate_sp500", np.nan)
        hy_mean = row.get("mean_hy_spread", np.nan)
        hy_hr = row.get("hit_rate_hy_spread", np.nan)
        n = int(row.get("n_obs", 0))

        parts = []
        if not np.isnan(sp500_mean):
            sign = "+" if sp500_mean >= 0 else ""
            parts.append(
                f"SP500 has returned {sign}{sp500_mean:.1%} on average "
                f"over {label} (hit rate: {sp500_hr:.0%}, n={n})"
            )
        if not np.isnan(hy_mean):
            direction = "tightened" if hy_mean < 0 else "widened"
            parts.append(
                f"HY spreads {direction} {abs(hy_mean):.0f}bps on average "
                f"({hy_hr:.0%} tightening rate)"
            )
        if parts:
            recommendation_parts.append(f"  [{label}] " + "; ".join(parts) + ".")

    # Risk-Off specific note about recovery opportunity
    if current_regime == "Risk-Off":
        h252 = outlook.loc[252] if 252 in outlook.index else None
        if h252 is not None:
            sp_12m = h252.get("mean_sp500", np.nan)
            if not np.isnan(sp_12m) and sp_12m > 0:
                recommendation_parts.append(
                    f"  Note: Risk-Off entry points have historically produced "
                    f"strong 12-month recoveries (+{sp_12m:.1%} average), "
                    "as regimes eventually normalize — entry point matters most."
                )

    recommendation = " ".join(recommendation_parts) if len(recommendation_parts) > 1 else recommendation_parts[0]

    return {
        "regime": current_regime,
        "outlook_table": outlook,
        "recommendation": recommendation,
        "n_obs_min": n_obs_min,
        "available": True,
    }


def _best_entry_analysis(df: pd.DataFrame, table: pd.DataFrame) -> dict:
    """
    For each regime, compute mean 12-month SP500 return when entering at the
    START of a new regime spell (i.e., first day the regime appears after a
    different regime). Compare to unconditional 12-month mean.

    Returns dict: {regime: {"mean_12m_entry": float, "n_entries": int}}
    """
    if "final_decision" not in df.columns or "sp500" not in df.columns:
        return {}

    # Compute unconditional 12-month forward return
    fwd_12m = df["sp500"].shift(-252) / df["sp500"] - 1

    unconditional_mean = fwd_12m.dropna().mean()

    # Identify regime-change entry points (day 1 of each new regime spell)
    regime_series = df["final_decision"]
    entry_mask = regime_series != regime_series.shift(1)
    entry_mask.iloc[0] = True  # first row is always an entry

    result = {"unconditional_12m_mean": unconditional_mean}

    for regime in REGIME_ORDER:
        spell_starts = entry_mask & (regime_series == regime)
        fwd_at_entry = fwd_12m[spell_starts].dropna()
        n = len(fwd_at_entry)
        mean_entry = fwd_at_entry.mean() if n > 0 else np.nan
        result[regime] = {
            "mean_12m_entry": mean_entry,
            "n_entries": n,
            "vs_unconditional": (mean_entry - unconditional_mean) if not np.isnan(mean_entry) else np.nan,
        }

    return result


def run_regime_return_analysis(df: pd.DataFrame) -> dict:
    """
    Full regime return analysis combining all sub-analyses.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys:
        "table"                 : dict from compute_regime_return_table
        "current_regime_outlook": dict from get_current_regime_outlook
        "best_entry_analysis"   : dict with entry-point return statistics
        "available"             : bool
    """
    table_result = compute_regime_return_table(df)
    if not table_result["available"]:
        return {
            "table": table_result,
            "current_regime_outlook": get_current_regime_outlook(df),
            "best_entry_analysis": {},
            "available": False,
        }

    # Check if at least 4 regimes have MIN_OBS observations at the 30d horizon
    tbl = table_result["table"]
    regimes_with_data = 0
    for regime in REGIME_ORDER:
        if (regime, 30) in tbl.index:
            if tbl.loc[(regime, 30), "n_obs"] >= MIN_OBS:
                regimes_with_data += 1

    sufficient_data = regimes_with_data >= 4

    entry_analysis = _best_entry_analysis(df, tbl)

    return {
        "table": table_result,
        "current_regime_outlook": get_current_regime_outlook(df),
        "best_entry_analysis": entry_analysis,
        "available": sufficient_data,
    }
