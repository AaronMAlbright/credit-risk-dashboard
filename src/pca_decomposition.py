"""
PCA Decomposition — factor analysis of the 7 composite sub-scores.

Decomposes the current composite risk score into named latent factors
(Rates, Credit, Macro) using pure numpy PCA (no sklearn).

Public API
----------
  SUB_SCORES                  — ordered list of (col, label)
  compute_pca(df)             -> dict
  get_current_decomposition(df) -> dict
  run_pca_analysis(df)        -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SUB_SCORES: list[tuple[str, str]] = [
    ("treasury_stress_score_smooth",          "Treasury Stress"),
    ("credit_market_risk_score_smooth",       "Credit Risk"),
    ("macro_risk_score_smooth",               "Macro Risk"),
    ("enhanced_funding_stress_score_smooth",  "Funding Stress"),
    ("rates_stress_score_smooth",             "Rates Stress"),
    ("fx_commodity_score_smooth",             "FX/Commodity"),
    ("banking_stress_score_smooth",           "Banking Stress"),
]

# Minimum requirements
_MIN_SCORES    = 3    # at least this many sub-score columns must be present
_MIN_ROWS      = 252  # at least this many non-NaN rows needed
_N_COMPONENTS  = 3    # target number of principal components

# Factor naming rules: maps to canonical group name
_RATES_NAMES   = {"treasury stress", "rates stress"}
_CREDIT_NAMES  = {"credit risk", "funding stress", "banking stress"}
_MACRO_NAMES   = {"macro risk", "fx/commodity"}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _available_sub_scores(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the subset of SUB_SCORES whose columns exist in df."""
    return [(col, label) for col, label in SUB_SCORES if col in df.columns]


def _name_component(loadings: np.ndarray, labels: list[str]) -> str:
    """
    Name a PCA component based on its dominant (highest absolute) loading.

    Parameters
    ----------
    loadings : 1-D array of length n_sub_scores — loadings for one component
    labels   : list of human-readable sub-score labels (same order as loadings)

    Returns
    -------
    str: "Rates Factor", "Credit Factor", "Macro Factor", or "Factor N"
    """
    if len(loadings) == 0:
        return "Factor"

    top_idx   = int(np.argmax(np.abs(loadings)))
    top_label = labels[top_idx].lower()

    for key in _RATES_NAMES:
        if key in top_label:
            return "Rates Factor"
    for key in _CREDIT_NAMES:
        if key in top_label:
            return "Credit Factor"
    for key in _MACRO_NAMES:
        if key in top_label:
            return "Macro Factor"

    return "Factor"


def _deduplicate_names(names: list[str]) -> list[str]:
    """Append numeric suffixes to duplicate component names (e.g., two 'Rates Factor')."""
    seen: dict[str, int] = {}
    result = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            result.append(name)
        else:
            seen[name] += 1
            result.append(f"{name} {seen[name] + 1}")
    return result


# ---------------------------------------------------------------------------
# compute_pca
# ---------------------------------------------------------------------------

def compute_pca(df: pd.DataFrame) -> dict:
    """
    Run PCA on available sub-scores using numpy eigendecomposition.

    Steps:
      1. Extract sub-score columns that exist in df, dropna rows
      2. Standardize (zero mean, unit variance)
      3. Covariance matrix → eigendecomposition (np.linalg.eigh)
      4. Sort eigenvectors descending by eigenvalue
      5. Keep top min(_N_COMPONENTS, n_available) components

    Returns
    -------
    {
        available               : bool,
        n_components            : int,
        explained_variance_ratio: list[float],
        cumulative_variance     : float,
        loadings                : pd.DataFrame  (n_sub_scores × n_components),
        scores                  : pd.DataFrame  (n_rows × n_components, df.index aligned),
        component_names         : list[str],
    }
    """
    try:
        avail = _available_sub_scores(df)

        if len(avail) < _MIN_SCORES:
            return {"available": False}

        cols   = [col for col, _ in avail]
        labels = [label for _, label in avail]

        sub_df = df[cols].dropna()

        if len(sub_df) < _MIN_ROWS:
            return {"available": False}

        X = sub_df.values.astype(float)  # (n_rows, n_features)

        # Standardize
        mean = X.mean(axis=0)
        std  = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0  # guard constant columns
        X_std = (X - mean) / std

        # Covariance matrix (n_features × n_features)
        cov = np.cov(X_std, rowvar=False)

        # Eigendecomposition — eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        order       = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]  # each column is one eigenvector

        # Clamp negative eigenvalues to 0 (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        n_comp = min(_N_COMPONENTS, len(avail))

        total_var = float(eigenvalues.sum()) if eigenvalues.sum() > 0 else 1.0
        evr = [float(eigenvalues[k] / total_var) for k in range(n_comp)]
        cum_var = float(sum(evr))

        # Loadings: shape (n_features, n_components)
        loadings_arr = eigenvectors[:, :n_comp]  # (n_features, n_comp)

        # Component names from dominant loadings
        raw_names = [
            _name_component(loadings_arr[:, k], labels)
            for k in range(n_comp)
        ]
        component_names = _deduplicate_names(raw_names)

        loadings_df = pd.DataFrame(
            loadings_arr,
            index=cols,
            columns=component_names,
        )

        # Factor scores: project standardized data onto eigenvectors
        scores_arr = X_std @ loadings_arr  # (n_rows, n_comp)
        scores_df  = pd.DataFrame(
            scores_arr,
            index=sub_df.index,
            columns=component_names,
        )

        return {
            "available":                True,
            "n_components":             n_comp,
            "explained_variance_ratio": evr,
            "cumulative_variance":      round(cum_var, 4),
            "loadings":                 loadings_df,
            "scores":                   scores_df,
            "component_names":          component_names,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# get_current_decomposition
# ---------------------------------------------------------------------------

def get_current_decomposition(df: pd.DataFrame) -> dict:
    """
    For the most recent row, show factor contributions to the composite risk score.

    Contribution of factor k = current_factor_score[k] (the projection of today's
    standardized sub-scores onto eigenvector k).  pct_contribution expresses each
    factor's absolute score as a fraction of the sum of all absolute factor scores.

    Returns
    -------
    {
        available           : bool,
        date                : str,
        factor_contributions: list[{name, score, pct_contribution}],
        dominant_factor     : str,
        interpretation      : str,
    }
    """
    try:
        pca = compute_pca(df)
        if not pca.get("available", False):
            return {"available": False}

        scores_df       = pca["scores"]
        component_names = pca["component_names"]
        evr             = pca["explained_variance_ratio"]

        # Most recent date that has a score
        last_row = scores_df.iloc[-1]
        last_date = str(scores_df.index[-1].date()) if hasattr(scores_df.index[-1], "date") else str(scores_df.index[-1])

        current_scores = last_row.values.astype(float)  # length n_comp

        abs_sum = float(np.abs(current_scores).sum())
        if abs_sum < 1e-10:
            abs_sum = 1.0

        contributions = []
        for k, name in enumerate(component_names):
            score = float(current_scores[k])
            pct   = float(np.abs(score) / abs_sum * 100.0)
            contributions.append({
                "name":             name,
                "score":            round(score, 4),
                "pct_contribution": round(pct, 1),
            })

        # Dominant factor: highest absolute score
        dom_idx    = int(np.argmax(np.abs(current_scores)))
        dom_factor = component_names[dom_idx]
        dom_score  = current_scores[dom_idx]

        # Human-readable interpretation
        direction  = "elevated" if dom_score > 0 else "subdued"
        dom_pct    = contributions[dom_idx]["pct_contribution"]
        interp = (
            f"{dom_factor} is the primary driver ({dom_pct:.0f}% of factor activity), "
            f"currently {direction}. "
            f"Top 3 components explain {pca['cumulative_variance'] * 100:.0f}% of variance."
        )

        return {
            "available":            True,
            "date":                 last_date,
            "factor_contributions": contributions,
            "dominant_factor":      dom_factor,
            "interpretation":       interp,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# run_pca_analysis
# ---------------------------------------------------------------------------

def run_pca_analysis(df: pd.DataFrame) -> dict:
    """
    Top-level entry point for PCA factor decomposition.

    Returns
    -------
    {
        available            : bool,
        pca                  : dict  (from compute_pca),
        current              : dict  (from get_current_decomposition),
        rolling_factor_scores: pd.DataFrame  (last 504 rows of factor scores),
    }
    """
    try:
        pca     = compute_pca(df)
        current = get_current_decomposition(df)

        if not pca.get("available", False):
            return {"available": False}

        scores_df = pca["scores"]
        rolling   = scores_df.iloc[-504:].copy() if len(scores_df) > 504 else scores_df.copy()

        return {
            "available":             True,
            "pca":                   pca,
            "current":               current,
            "rolling_factor_scores": rolling,
        }

    except Exception:
        return {"available": False}
