"""
Score orthogonality analysis.

Measures how independent the six composite sub-scores are from each other.
High multicollinearity reduces the regime engine's discriminating power and
makes composite weights unstable.

Metrics produced
----------------
  correlation_matrix   — pairwise Pearson correlations
  vif                  — Variance Inflation Factor per score (via OLS, no sklearn)
  pca                  — explained variance + effective rank from SVD
  rolling_correlations — time-varying correlations for the highest-correlated pairs
  summary              — headline diversity stats
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The six scores that feed the composite engine (in composite_engine weight order)
COMPOSITE_SCORE_COLS: dict[str, str] = {
    "Macro Risk":     "macro_risk_score_smooth",
    "Credit Risk":    "credit_market_risk_score_smooth",
    "Complacency":    "complacency_score_smooth",
    "Liquidity":      "liquidity_regime_score_smooth",
    "Treasury":       "treasury_stress_score_smooth",
    "Mean Reversion": "mean_reversion_score_smooth",
}

# VIF thresholds
VIF_MODERATE = 5.0
VIF_HIGH = 10.0

# Rolling window in trading days (≈ 3 months)
DEFAULT_ROLLING_WINDOW = 60

# Number of top correlated pairs to track in rolling view
TOP_N_PAIRS = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _available_scores(df: pd.DataFrame) -> dict[str, str]:
    """Return subset of COMPOSITE_SCORE_COLS whose columns exist in df."""
    return {name: col for name, col in COMPOSITE_SCORE_COLS.items() if col in df.columns}


def _score_matrix(df: pd.DataFrame, available: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """
    Return (X, names) where X is a float array of standardised scores (row = obs)
    and names matches the column order.
    """
    cols = list(available.values())
    names = list(available.keys())
    X = df[cols].dropna().astype(float).values
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1.0  # avoid divide-by-zero for constant columns
    return (X - mean) / std, names


def _vif_single(X_std: np.ndarray, i: int) -> float:
    """VIF for column i: 1 / (1 - R²) from OLS of col_i ~ all other cols."""
    y = X_std[:, i]
    rest = np.delete(X_std, i, axis=1)
    A = np.column_stack([np.ones(len(y)), rest])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return np.nan
    r2 = 1.0 - ss_res / ss_tot
    if r2 >= 1.0 - 1e-9:
        return np.inf
    return 1.0 / (1.0 - r2)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Pearson correlation matrix for the six composite sub-scores.

    Returns square DataFrame indexed and columned by display name.
    Empty if fewer than 2 scores are present.
    """
    available = _available_scores(df)
    if len(available) < 2:
        return pd.DataFrame()

    cols = list(available.values())
    names = list(available.keys())
    clean = df[cols].dropna()
    if len(clean) < 10:
        return pd.DataFrame()

    corr = clean.corr()
    corr.index = names
    corr.columns = names
    return corr.round(3)


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variance Inflation Factor for each composite sub-score.

    VIF = 1 / (1 - R²) where R² is from regressing each score on all others.
    Requires at least 3 scores; returns empty DataFrame otherwise.

    Returns DataFrame indexed by score name with columns:
      vif, flag ("Low" / "Moderate" / "High")
    """
    available = _available_scores(df)
    if len(available) < 3:
        return pd.DataFrame()

    X_std, names = _score_matrix(df, available)
    if len(X_std) < len(names) + 2:
        return pd.DataFrame()

    rows = []
    for i, name in enumerate(names):
        vif = _vif_single(X_std, i)
        if np.isinf(vif):
            flag = "High"
        elif np.isnan(vif):
            flag = "N/A"
        elif vif >= VIF_HIGH:
            flag = "High"
        elif vif >= VIF_MODERATE:
            flag = "Moderate"
        else:
            flag = "Low"
        rows.append({"score": name, "vif": round(vif, 2) if np.isfinite(vif) else np.nan, "flag": flag})

    return pd.DataFrame(rows).set_index("score").sort_values("vif", ascending=False)


def compute_pca_decomposition(df: pd.DataFrame) -> dict:
    """
    PCA via SVD on the standardised score matrix.

    Returns dict with:
      explained_variance_ratio  — list of floats, length = n_scores
      cumulative_variance       — cumulative sum of the above
      n_components_90pct        — fewest components reaching ≥ 90% explained variance
      effective_rank            — Shannon-entropy effective dimensionality (1 = one factor, n = fully diverse)
      component_names           — score names in column order
    """
    available = _available_scores(df)
    if len(available) < 2:
        return {}

    X_std, names = _score_matrix(df, available)
    if len(X_std) < 4:
        return {}

    _, s, _ = np.linalg.svd(X_std, full_matrices=False)
    var = s ** 2
    total = var.sum()
    evr = (var / total).tolist()
    cumvar = np.cumsum(evr).tolist()

    n90 = next((i + 1 for i, cv in enumerate(cumvar) if cv >= 0.90), len(evr))

    # Shannon entropy effective rank
    p = np.array(evr)
    p = p[p > 0]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    return {
        "explained_variance_ratio": [round(v, 4) for v in evr],
        "cumulative_variance":      [round(v, 4) for v in cumvar],
        "n_components_90pct":       n90,
        "effective_rank":           round(eff_rank, 2),
        "component_names":          names,
    }


def compute_rolling_correlations(
    df: pd.DataFrame,
    window: int = DEFAULT_ROLLING_WINDOW,
    top_n: int = TOP_N_PAIRS,
) -> dict[tuple[str, str], pd.Series]:
    """
    Rolling Pearson correlations for the top_n most correlated pairs.

    Pairs are ranked by mean absolute correlation over the full sample.
    Returns dict mapping (name_a, name_b) → pd.Series of rolling correlations
    indexed by date.
    """
    available = _available_scores(df)
    if len(available) < 2:
        return {}

    cols = list(available.values())
    names = list(available.keys())
    date_col = "date" if "date" in df.columns else None

    clean = df[[date_col] + cols].dropna() if date_col else df[cols].dropna()
    if len(clean) < window + 1:
        return {}

    score_df = clean[cols].copy()
    score_df.columns = names

    # Full-sample mean abs correlation to rank pairs
    full_corr = score_df.corr().abs()
    pairs = [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    pairs_sorted = sorted(pairs, key=lambda p: full_corr.loc[p[0], p[1]], reverse=True)
    top_pairs = pairs_sorted[:top_n]

    result: dict[tuple[str, str], pd.Series] = {}
    for a, b in top_pairs:
        rolling = score_df[a].rolling(window).corr(score_df[b])
        if date_col:
            rolling.index = pd.to_datetime(clean[date_col].values)
        result[(a, b)] = rolling.dropna().round(3)

    return result


def run_orthogonality_analysis(
    df: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> dict:
    """
    Full orthogonality audit across all four analytical layers.

    Returns dict with:
      correlation_matrix    — pairwise Pearson correlations
      vif                   — VIF per score with flag
      pca                   — explained variance and effective rank
      rolling_correlations  — {(a, b): pd.Series} for top pairs
      summary               — headline stats dict
    """
    corr_matrix = compute_correlation_matrix(df)
    vif_df      = compute_vif(df)
    pca         = compute_pca_decomposition(df)
    rolling     = compute_rolling_correlations(df, window=rolling_window)

    # Summary
    n_high = int((vif_df["flag"] == "High").sum()) if not vif_df.empty else 0
    n_moderate = int((vif_df["flag"] == "Moderate").sum()) if not vif_df.empty else 0

    max_pairwise = np.nan
    if not corr_matrix.empty:
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        off_diag = corr_matrix.values[mask]
        max_pairwise = float(np.abs(off_diag).max())

    summary = {
        "effective_rank":     pca.get("effective_rank", np.nan),
        "n_components_90pct": pca.get("n_components_90pct", np.nan),
        "max_pairwise_corr":  round(max_pairwise, 3) if not np.isnan(max_pairwise) else np.nan,
        "n_high_vif":         n_high,
        "n_moderate_vif":     n_moderate,
    }

    return {
        "correlation_matrix":   corr_matrix,
        "vif":                  vif_df,
        "pca":                  pca,
        "rolling_correlations": rolling,
        "summary":              summary,
    }
