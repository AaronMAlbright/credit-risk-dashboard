"""
Regime validity testing.

Tests whether regimes are genuinely distinct economic states using:
  - Per-regime forward return / volatility / drawdown / spread / rates statistics
  - Kruskal-Wallis H-test (rank-based, scipy-free)
  - One-way ANOVA (F-statistic)
  - Bootstrap pairwise difference-of-means (empirical p-values)
  - Separability score (0-100, eta-squared analog)
  - Regime consolidation recommendations where separability is weak

Public API
----------
  compute_regime_stats          — economic summary per regime
  kruskal_wallis_test           — KW H-statistic + significance
  anova_test                    — F-statistic + significance
  bootstrap_pairwise            — pairwise bootstrap CIs and p-values
  compute_separability_score    — scalar 0-100 separability
  flag_consolidations           — regimes that should be merged
  run_regime_validity           — orchestrator returning full results dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

_MIN_OBS = 30           # warn when regime has fewer observations
_BOOTSTRAP_N = 2000     # bootstrap iterations for pairwise tests
_ANNUALIZE = 252

# Chi-squared 0.05 critical values for df 1..15 (no scipy needed)
_CHI2_CV_05 = {
    1: 3.841, 2: 5.991,  3: 7.815,  4: 9.488,  5: 11.070,
    6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
    11: 19.675, 12: 21.026, 13: 22.362, 14: 23.685, 15: 24.996,
}

# F-distribution 0.05 critical values for (df_between=k-1, df_within varies)
# Rows = df1 (between), columns indexed loosely; we just flag if F > 2.5 heuristic
_F_CV_HEURISTIC = 2.5


# ---------------------------------------------------------------------------
# Forward stat helpers (scipy-free, no look-ahead)
# ---------------------------------------------------------------------------

def _add_forward_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward-horizon columns used for regime comparison.
    Uses only backward-looking primitives shifted forward — no look-ahead.
    """
    df = df.copy()
    if "sp500_daily_return" in df.columns:
        r = df["sp500_daily_return"].values.astype(float)
        n = len(r)

        for h in [21, 63, 126]:
            col = f"_fwd_ret_{h}d"
            out = np.full(n, np.nan)
            for i in range(n - h):
                seg = r[i + 1: i + 1 + h]
                out[i] = float(np.nansum(seg))  # log-approx: sum of daily returns
            df[col] = out

        # Forward realised vol (21d)
        vol_out = np.full(n, np.nan)
        for i in range(n - 21):
            seg = r[i + 1: i + 22]
            if np.isfinite(seg).sum() >= 10:
                vol_out[i] = float(np.nanstd(seg, ddof=1) * np.sqrt(_ANNUALIZE))
        df["_fwd_vol_21d"] = vol_out

        # Forward max drawdown (21d)
        dd_out = np.full(n, np.nan)
        for i in range(n - 21):
            seg = r[i + 1: i + 22]
            if np.isfinite(seg).sum() >= 3:
                eq = np.cumprod(1 + np.nan_to_num(seg))
                dd_out[i] = float((eq / np.maximum.accumulate(eq) - 1).min())
        df["_fwd_dd_21d"] = dd_out

    # Forward 21d changes in spread indicators
    for src, col in [
        ("hy_spread",   "_fwd_hy_chg_21d"),
        ("vix",         "_fwd_vix_chg_21d"),
        ("move_index",  "_fwd_move_chg_21d"),
        ("yield_10y",   "_fwd_rates_chg_21d"),
        ("spread",      "_fwd_curve_chg_21d"),
    ]:
        if src in df.columns:
            s = df[src].values.astype(float)
            out = np.full(len(s), np.nan)
            for i in range(len(s) - 21):
                out[i] = float(s[i + 21] - s[i])
            df[col] = out

    return df


# ---------------------------------------------------------------------------
# 1. Per-regime economic summary
# ---------------------------------------------------------------------------

def compute_regime_stats(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
) -> pd.DataFrame:
    """
    Per-regime economic summary.

    Returns DataFrame indexed by regime with columns:
      n_obs, sample_warning,
      fwd_ret_1m, fwd_ret_3m, fwd_ret_6m,
      fwd_vol_1m, fwd_maxdd_1m,
      fwd_hy_chg, fwd_vix_chg, fwd_move_chg, fwd_rates_chg,
      mean_persistence_days
    """
    if regime_col not in df.columns or df.empty:
        return pd.DataFrame()

    df = _add_forward_cols(df)

    # Persistence per regime (average run length)
    from src.regime_transition import _regime_runs, compute_regime_durations
    dur = compute_regime_durations(df[regime_col].dropna())

    rows = []
    for regime, grp in df.groupby(regime_col):
        n = len(grp)

        def _mean(col):
            return round(float(grp[col].mean()), 4) if col in grp.columns else np.nan

        row = {
            "regime":           regime,
            "n_obs":            n,
            "sample_warning":   n < _MIN_OBS,
            "fwd_ret_1m":       _mean("_fwd_ret_21d"),
            "fwd_ret_3m":       _mean("_fwd_ret_63d"),
            "fwd_ret_6m":       _mean("_fwd_ret_126d"),
            "fwd_vol_1m":       _mean("_fwd_vol_21d"),
            "fwd_maxdd_1m":     _mean("_fwd_dd_21d"),
            "fwd_hy_chg":       _mean("_fwd_hy_chg_21d"),
            "fwd_vix_chg":      _mean("_fwd_vix_chg_21d"),
            "fwd_move_chg":     _mean("_fwd_move_chg_21d"),
            "fwd_rates_chg":    _mean("_fwd_rates_chg_21d"),
        }
        if not dur.empty and regime in dur.index:
            row["mean_persistence_days"] = float(dur.at[regime, "mean_days"])
        else:
            row["mean_persistence_days"] = np.nan

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .set_index("regime")
        .sort_values("fwd_ret_1m", ascending=True)
    )


# ---------------------------------------------------------------------------
# 2. Kruskal-Wallis (scipy-free)
# ---------------------------------------------------------------------------

def kruskal_wallis_test(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    outcome_col: str = "_fwd_ret_21d",
) -> dict:
    """
    Kruskal-Wallis H-test: are regime forward returns from the same distribution?

    Returns dict with:
      H_stat, df, critical_value_05, significant_05,
      n_groups, n_total, groups (list of (regime, n_obs, mean_rank))
      conclusion
    """
    if regime_col not in df.columns or outcome_col not in df.columns:
        return {"error": f"Missing column: {regime_col} or {outcome_col}"}

    sub = df[[regime_col, outcome_col]].dropna()
    if sub.empty:
        return {"error": "No valid observations after dropna"}

    groups = {r: grp[outcome_col].values for r, grp in sub.groupby(regime_col)}
    if len(groups) < 2:
        return {"error": "Need at least 2 regimes"}

    all_vals = sub[outcome_col].values
    N = len(all_vals)

    # Assign ranks (average for ties)
    sorted_idx = np.argsort(all_vals, kind="stable")
    ranks = np.empty(N)
    ranks[sorted_idx] = np.arange(1, N + 1, dtype=float)
    # Handle ties: assign average rank to tied values
    sorted_vals = all_vals[sorted_idx]
    i = 0
    while i < N:
        j = i
        while j < N and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[sorted_idx[i:j]] = avg_rank
        i = j

    # Build regime → ranks map
    regime_arr = sub[regime_col].values
    H = 0.0
    group_stats = []
    for regime, grp_vals in groups.items():
        mask = regime_arr == regime
        grp_ranks = ranks[mask]
        n_i = len(grp_ranks)
        R_i = grp_ranks.mean()
        H += n_i * (R_i - (N + 1) / 2.0) ** 2
        group_stats.append((regime, n_i, round(R_i, 2)))

    H = (12.0 / (N * (N + 1))) * H

    # Tie correction
    from collections import Counter
    counts_per_val = Counter(all_vals)
    tie_sum = sum(c ** 3 - c for c in counts_per_val.values() if c > 1)
    correction = 1.0 - tie_sum / (N ** 3 - N) if N > 1 else 1.0
    H_corrected = H / correction if correction > 1e-12 else H

    k = len(groups)
    df_kw = k - 1
    cv = _CHI2_CV_05.get(df_kw, _CHI2_CV_05.get(15, 24.996))
    sig = bool(H_corrected > cv)

    return {
        "H_stat":              round(H_corrected, 4),
        "df":                  df_kw,
        "critical_value_05":   cv,
        "significant_05":      sig,
        "n_groups":            k,
        "n_total":             N,
        "groups":              sorted(group_stats, key=lambda x: x[1], reverse=True),
        "conclusion": (
            "Regimes have significantly different forward return distributions (p < 0.05)"
            if sig else
            "INCONCLUSIVE: cannot reject null that regimes share the same distribution (p ≥ 0.05)"
        ),
    }


# ---------------------------------------------------------------------------
# 3. One-way ANOVA (F-statistic)
# ---------------------------------------------------------------------------

def anova_test(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    outcome_col: str = "_fwd_ret_21d",
) -> dict:
    """
    One-way ANOVA on forward returns across regimes.

    Returns dict with:
      F_stat, df_between, df_within,
      ss_between, ss_within, ss_total,
      significant_heuristic  (F > 2.5),
      eta_squared  — fraction of total variance explained by regime
    """
    if regime_col not in df.columns or outcome_col not in df.columns:
        return {"error": "Missing column"}

    sub = df[[regime_col, outcome_col]].dropna()
    if sub.empty or sub[regime_col].nunique() < 2:
        return {"error": "Insufficient data or groups"}

    overall_mean = sub[outcome_col].mean()
    ss_total = float(((sub[outcome_col] - overall_mean) ** 2).sum())

    ss_between = 0.0
    k = sub[regime_col].nunique()
    N = len(sub)

    group_stats = []
    for regime, grp in sub.groupby(regime_col):
        n_i = len(grp)
        mean_i = float(grp[outcome_col].mean())
        ss_between += n_i * (mean_i - overall_mean) ** 2
        group_stats.append((regime, n_i, round(mean_i, 4)))

    ss_within = ss_total - ss_between
    df_between = k - 1
    df_within = N - k

    if df_within <= 0 or ss_within < 1e-14:
        return {"error": "Insufficient degrees of freedom"}

    F = (ss_between / df_between) / (ss_within / df_within)
    eta2 = ss_between / ss_total if ss_total > 1e-14 else 0.0

    return {
        "F_stat":               round(F, 4),
        "df_between":           df_between,
        "df_within":            df_within,
        "ss_between":           round(ss_between, 6),
        "ss_within":            round(ss_within, 6),
        "ss_total":             round(ss_total, 6),
        "significant_heuristic": bool(F > _F_CV_HEURISTIC),
        "eta_squared":          round(eta2, 4),
        "groups":               sorted(group_stats, key=lambda x: x[1], reverse=True),
        "conclusion": (
            f"F={F:.2f} — regimes explain {eta2:.1%} of total return variance. "
            + ("Statistically separable by heuristic threshold." if F > _F_CV_HEURISTIC
               else "INCONCLUSIVE: F below heuristic threshold (2.5).")
        ),
    }


# ---------------------------------------------------------------------------
# 4. Bootstrap pairwise difference-of-means
# ---------------------------------------------------------------------------

def bootstrap_pairwise(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    outcome_col: str = "_fwd_ret_21d",
    n_boot: int = _BOOTSTRAP_N,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Pairwise bootstrap test: for each regime pair, is the difference in mean
    forward returns statistically significant?

    Returns DataFrame with columns:
      regime_a, regime_b, n_a, n_b,
      mean_a, mean_b, observed_diff,
      ci_low_95, ci_high_95,
      p_value_two_sided,
      significant_05, separable
    """
    if regime_col not in df.columns or outcome_col not in df.columns:
        return pd.DataFrame()

    sub = df[[regime_col, outcome_col]].dropna()
    rng = np.random.default_rng(seed)
    groups = {r: grp[outcome_col].values for r, grp in sub.groupby(regime_col)}
    regime_names = sorted(groups.keys())

    rows = []
    for a, b in combinations(regime_names, 2):
        x, y = groups[a], groups[b]
        n_a, n_b = len(x), len(y)

        if n_a < 5 or n_b < 5:
            rows.append({
                "regime_a": a, "regime_b": b, "n_a": n_a, "n_b": n_b,
                "mean_a": round(x.mean(), 4), "mean_b": round(y.mean(), 4),
                "observed_diff": round(x.mean() - y.mean(), 4),
                "ci_low_95": np.nan, "ci_high_95": np.nan,
                "p_value_two_sided": np.nan, "significant_05": False,
                "separable": False, "note": "insufficient observations",
            })
            continue

        obs_diff = float(x.mean() - y.mean())

        # Bootstrap CI: resample each group independently
        boot_diffs = np.empty(n_boot)
        for i in range(n_boot):
            d = rng.choice(x, n_a, replace=True).mean() - rng.choice(y, n_b, replace=True).mean()
            boot_diffs[i] = d

        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))

        # Permutation p-value: pool and resample under H0 (equal means)
        pooled = np.concatenate([x, y])
        perm_diffs = np.empty(n_boot)
        for i in range(n_boot):
            perm = rng.permutation(pooled)
            perm_diffs[i] = perm[:n_a].mean() - perm[n_a:].mean()

        p_val = float((np.abs(perm_diffs) >= np.abs(obs_diff)).mean())

        rows.append({
            "regime_a":          a,
            "regime_b":          b,
            "n_a":               n_a,
            "n_b":               n_b,
            "mean_a":            round(float(x.mean()), 4),
            "mean_b":            round(float(y.mean()), 4),
            "observed_diff":     round(obs_diff, 4),
            "ci_low_95":         round(ci_lo, 4),
            "ci_high_95":        round(ci_hi, 4),
            "p_value_two_sided": round(p_val, 4),
            "significant_05":    bool(p_val < 0.05),
            "separable":         bool(p_val < 0.05 and (ci_lo > 0 or ci_hi < 0)),
            "note":              "",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Separability score
# ---------------------------------------------------------------------------

def compute_separability_score(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    outcome_col: str = "_fwd_ret_21d",
) -> dict:
    """
    Scalar 0-100 regime separability score.

    Based on eta-squared (between-group / total variance) × 100.
    Higher = regimes explain more of the forward return distribution.

    Returns dict: {score, interpretation, eta_squared, n_obs, n_regimes}
    """
    anova = anova_test(df, regime_col, outcome_col)
    if "error" in anova:
        return {"score": np.nan, "interpretation": "Insufficient data", "eta_squared": np.nan}

    eta2 = anova["eta_squared"]
    score = min(100.0, max(0.0, eta2 * 100))

    if score >= 15:
        interpretation = "Strong separability — regimes are economically distinct"
    elif score >= 7:
        interpretation = "Moderate separability — partial economic distinction"
    elif score >= 3:
        interpretation = "Weak separability — limited economic distinction"
    else:
        interpretation = "POOR separability — regimes are not meaningfully distinct; consider consolidation"

    return {
        "score":           round(score, 1),
        "interpretation":  interpretation,
        "eta_squared":     eta2,
        "n_obs":           anova["df_between"] + anova["df_within"] + 1,
        "n_regimes":       anova["df_between"] + 1,
    }


# ---------------------------------------------------------------------------
# 6. Consolidation flagging
# ---------------------------------------------------------------------------

def flag_consolidations(
    pairwise: pd.DataFrame,
    stats: pd.DataFrame,
    min_obs: int = _MIN_OBS,
) -> list[dict]:
    """
    Identify regimes that should be merged based on:
      - Pairwise bootstrap: not significantly different (p >= 0.05)
      - OR regime has fewer than min_obs observations

    Returns list of consolidation recommendations.
    """
    recs = []

    # Small-sample flags
    if not stats.empty and "sample_warning" in stats.columns:
        for regime in stats.index[stats["sample_warning"]]:
            n = int(stats.at[regime, "n_obs"])
            recs.append({
                "type":    "small_sample",
                "regime":  regime,
                "partner": None,
                "reason":  f"Only {n} observations — results unreliable (min {min_obs})",
                "action":  "Consider merging into adjacent regime or dropping",
            })

    # Statistically indistinct pairs
    if not pairwise.empty and "significant_05" in pairwise.columns:
        not_sig = pairwise[~pairwise["significant_05"]]
        for _, row in not_sig.iterrows():
            # Only flag if both have sufficient data
            a_ok = stats.at[row["regime_a"], "n_obs"] >= min_obs if row["regime_a"] in stats.index else False
            b_ok = stats.at[row["regime_b"], "n_obs"] >= min_obs if row["regime_b"] in stats.index else False
            if a_ok and b_ok:
                recs.append({
                    "type":    "not_distinct",
                    "regime":  row["regime_a"],
                    "partner": row["regime_b"],
                    "reason":  (
                        f"Bootstrap p={row['p_value_two_sided']:.3f}: forward return distributions "
                        f"are not significantly different (Δmean={row['observed_diff']:.3%})"
                    ),
                    "action":  "Consider consolidating these regimes",
                })

    return recs


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------

def run_regime_validity(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
) -> dict:
    """
    Full regime validity analysis.

    Returns dict with:
      regime_stats        — per-regime economic summary DataFrame
      kw_test             — Kruskal-Wallis result dict (1m forward returns)
      anova               — ANOVA result dict
      pairwise            — bootstrap pairwise DataFrame
      separability        — separability score dict
      consolidations      — list of consolidation recommendations
      outcome_col         — column used as primary test outcome
    """
    df_aug = _add_forward_cols(df)
    outcome_col = "_fwd_ret_21d"

    stats = compute_regime_stats(df_aug, regime_col)
    kw = kruskal_wallis_test(df_aug, regime_col, outcome_col)
    anova = anova_test(df_aug, regime_col, outcome_col)
    pairwise = bootstrap_pairwise(df_aug, regime_col, outcome_col)
    sep = compute_separability_score(df_aug, regime_col, outcome_col)
    cons = flag_consolidations(pairwise, stats)

    return {
        "regime_stats":   stats,
        "kw_test":        kw,
        "anova":          anova,
        "pairwise":       pairwise,
        "separability":   sep,
        "consolidations": cons,
        "outcome_col":    outcome_col,
    }
