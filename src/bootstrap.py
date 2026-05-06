import numpy as np
import pandas as pd

DEFAULT_N_BOOT = 1000
DEFAULT_CI = 0.95
MIN_OBS_REGIME = 30
MIN_OBS_TRANSITION = 5

_WINDOW_METRICS = [
    "strategy_sharpe",
    "sp500_sharpe",
    "strategy_total_return",
    "sp500_total_return",
    "strategy_max_drawdown",
    "sp500_max_drawdown",
    "strategy_hit_rate",
]


def _ci_bounds(ci):
    alpha = (1 - ci) / 2
    return alpha * 100, (1 - alpha) * 100


def _boot_ci(vals, n_boot, lo_pct, hi_pct):
    """Return (mean, ci_lower, ci_upper) via percentile bootstrap."""
    n = len(vals)
    boot_means = np.array([
        vals[np.random.choice(n, size=n, replace=True)].mean()
        for _ in range(n_boot)
    ])
    return vals.mean(), np.percentile(boot_means, lo_pct), np.percentile(boot_means, hi_pct)


def bootstrap_window_metrics(windows_df, n_boot=DEFAULT_N_BOOT, ci=DEFAULT_CI):
    """
    Bootstrap CIs for walk-forward window-level metrics.
    Resamples windows (rows) with replacement.

    Returns DataFrame indexed by metric with columns:
    mean, ci_lower, ci_upper, n_windows, flagged
    """
    if windows_df is None or windows_df.empty:
        return pd.DataFrame(
            columns=["mean", "ci_lower", "ci_upper", "n_windows", "flagged"]
        )

    available = [c for c in _WINDOW_METRICS if c in windows_df.columns]
    lo_pct, hi_pct = _ci_bounds(ci)
    records = []

    for col in available:
        vals = windows_df[col].dropna().values
        n = len(vals)
        if n < 2:
            records.append({
                "metric": col, "mean": float(vals[0]) if n == 1 else np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "n_windows": n, "flagged": True,
            })
            continue
        mean, lo, hi = _boot_ci(vals, n_boot, lo_pct, hi_pct)
        records.append({
            "metric": col, "mean": mean,
            "ci_lower": lo, "ci_upper": hi,
            "n_windows": n, "flagged": n < 10,
        })

    return pd.DataFrame(records).set_index("metric")


def bootstrap_regime_forward_returns(
    df, regime_col, return_col="sp500_forward_30d_return",
    n_boot=DEFAULT_N_BOOT, ci=DEFAULT_CI, min_obs=MIN_OBS_REGIME,
):
    """
    Bootstrap CIs for mean forward return per regime.

    Returns DataFrame indexed by regime with columns:
    mean, ci_lower, ci_upper, n_obs, flagged
    """
    if regime_col not in df.columns or return_col not in df.columns:
        return pd.DataFrame(columns=["mean", "ci_lower", "ci_upper", "n_obs", "flagged"])

    lo_pct, hi_pct = _ci_bounds(ci)
    records = []

    for regime, grp in df.groupby(regime_col):
        vals = grp[return_col].dropna().values
        n = len(vals)
        if n < 2:
            records.append({
                "regime": regime, "mean": float(vals[0]) if n == 1 else np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "n_obs": n, "flagged": True,
            })
            continue
        mean, lo, hi = _boot_ci(vals, n_boot, lo_pct, hi_pct)
        records.append({
            "regime": regime, "mean": mean,
            "ci_lower": lo, "ci_upper": hi,
            "n_obs": n, "flagged": n < min_obs,
        })

    return pd.DataFrame(records).set_index("regime")


def bootstrap_hit_rates(
    df, regime_col, return_col="sp500_forward_30d_return",
    n_boot=DEFAULT_N_BOOT, ci=DEFAULT_CI, min_obs=MIN_OBS_REGIME,
):
    """
    Bootstrap CIs for hit rate (fraction of positive returns) per regime.

    Returns DataFrame indexed by regime with columns:
    hit_rate, ci_lower, ci_upper, n_obs, flagged
    """
    if regime_col not in df.columns or return_col not in df.columns:
        return pd.DataFrame(columns=["hit_rate", "ci_lower", "ci_upper", "n_obs", "flagged"])

    lo_pct, hi_pct = _ci_bounds(ci)
    records = []

    for regime, grp in df.groupby(regime_col):
        vals = (grp[return_col].dropna() > 0).astype(float).values
        n = len(vals)
        if n < 2:
            records.append({
                "regime": regime, "hit_rate": float(vals[0]) if n == 1 else np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "n_obs": n, "flagged": True,
            })
            continue
        mean, lo, hi = _boot_ci(vals, n_boot, lo_pct, hi_pct)
        records.append({
            "regime": regime, "hit_rate": mean,
            "ci_lower": lo, "ci_upper": hi,
            "n_obs": n, "flagged": n < min_obs,
        })

    return pd.DataFrame(records).set_index("regime")


def bootstrap_transition_probs(
    series, n_boot=DEFAULT_N_BOOT, ci=DEFAULT_CI,
    min_obs=MIN_OBS_TRANSITION,
):
    """
    Bootstrap CIs for each transition probability cell.

    For each from-regime, resamples its outgoing transitions with replacement
    and recomputes empirical probabilities.

    Returns DataFrame with MultiIndex (from_regime, to_regime) and columns:
    mean_prob, ci_lower, ci_upper, n_obs, flagged
    """
    lo_pct, hi_pct = _ci_bounds(ci)

    clean = series.dropna()
    from_vals = clean.iloc[:-1].values
    to_vals = clean.iloc[1:].values

    from_groups: dict[str, list] = {}
    for f, t in zip(from_vals, to_vals):
        from_groups.setdefault(f, []).append(t)

    records = []
    for from_regime, outcomes in from_groups.items():
        n = len(outcomes)
        arr = np.array(outcomes)
        to_set = sorted(set(outcomes))
        flagged = n < min_obs

        for to_regime in to_set:
            obs_prob = (arr == to_regime).mean()
            if n < 2:
                records.append({
                    "from_regime": from_regime, "to_regime": to_regime,
                    "mean_prob": obs_prob,
                    "ci_lower": np.nan, "ci_upper": np.nan,
                    "n_obs": n, "flagged": True,
                })
                continue
            boot_probs = np.array([
                (arr[np.random.choice(n, size=n, replace=True)] == to_regime).mean()
                for _ in range(n_boot)
            ])
            records.append({
                "from_regime": from_regime, "to_regime": to_regime,
                "mean_prob": obs_prob,
                "ci_lower": np.percentile(boot_probs, lo_pct),
                "ci_upper": np.percentile(boot_probs, hi_pct),
                "n_obs": n, "flagged": flagged,
            })

    if not records:
        return pd.DataFrame(
            columns=["mean_prob", "ci_lower", "ci_upper", "n_obs", "flagged"]
        )

    result = pd.DataFrame(records).set_index(["from_regime", "to_regime"])
    return result.sort_index()


def run_bootstrap_analysis(df, windows_df=None, n_boot=DEFAULT_N_BOOT, ci=DEFAULT_CI):
    """
    Full bootstrap analysis across all estimators.

    Returns dict with keys:
      window_metrics            — CI on walk-forward window-level metrics
      regime_returns            — CI on mean 30d forward return by final_decision
      regime_hit_rates          — CI on hit rate by final_decision
      transition_probs_final_decision   — CI on each transition cell (decisions)
      transition_probs_transition_regime — CI on each transition cell (regimes)
    """
    rng_state = np.random.get_state()
    np.random.seed(42)

    try:
        results = {}

        results["window_metrics"] = bootstrap_window_metrics(
            windows_df, n_boot=n_boot, ci=ci
        )

        results["regime_returns"] = bootstrap_regime_forward_returns(
            df, "final_decision", n_boot=n_boot, ci=ci
        )

        results["regime_hit_rates"] = bootstrap_hit_rates(
            df, "final_decision", n_boot=n_boot, ci=ci
        )

        for regime_col in ["final_decision", "transition_regime"]:
            key = f"transition_probs_{regime_col}"
            if regime_col in df.columns:
                results[key] = bootstrap_transition_probs(
                    df[regime_col], n_boot=n_boot, ci=ci
                )
            else:
                results[key] = pd.DataFrame()

    finally:
        np.random.set_state(rng_state)

    return results
