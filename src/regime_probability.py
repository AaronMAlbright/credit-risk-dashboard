"""
Regime probability nowcast.

Converts the current component-score vector into a posterior probability
distribution across final-decision regimes using Gaussian Naive Bayes.

For each regime R and feature vector x:
  P(R | x) ∝ P(x | R) × P(R)
  P(x | R) = Π_i  N(x_i ; μ_Ri, σ_Ri)   [naive Bayes / feature independence]

All computation is done in log-space (log-sum-exp) for numerical stability.
Regimes with fewer than _MIN_OBS observations are excluded to avoid
degenerate Gaussian fits.

Public API
----------
  fit_naive_bayes        — fit per-regime Gaussian parameters from df
  score_to_regime_probs  — single feature vector → probability dict
  compute_prob_history   — full-history probability time series (vectorised)
  get_current_probs      — latest-row summary dict
  run_regime_probability — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FEATURE_COLS: list[str] = [
    "macro_risk_score_smooth",
    "credit_market_risk_score_smooth",
    "liquidity_regime_score_smooth",
    "complacency_score_smooth",
    "mean_reversion_score_smooth",
    "risk_appetite_score_smooth",
    "treasury_stress_score_smooth",
]
_LABEL_COL = "final_decision"
_DATE_COL  = "date"
_MIN_OBS   = 10    # regimes below this are excluded
_STD_FLOOR = 1.0   # minimum per-feature std (prevents degenerate Gaussians)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_naive_bayes(df: pd.DataFrame) -> dict:
    """
    Fit per-regime Gaussian parameters from df.

    Returns dict keyed by regime name:
      {regime: {prior, means, stds, n_obs, features}}
    Regimes with < _MIN_OBS clean observations are silently omitted.
    """
    feats = [c for c in _FEATURE_COLS if c in df.columns]
    if not feats or _LABEL_COL not in df.columns or df.empty:
        return {}

    regime_arrays: dict[str, np.ndarray] = {}
    for regime, grp in df.groupby(_LABEL_COL):
        sub = grp[feats].dropna()
        if len(sub) >= _MIN_OBS:
            regime_arrays[regime] = sub.values.astype(float)

    if not regime_arrays:
        return {}

    total = sum(len(v) for v in regime_arrays.values())
    model: dict[str, dict] = {}
    for regime, vals in regime_arrays.items():
        model[regime] = {
            "prior":    float(len(vals) / total),
            "means":    vals.mean(axis=0),
            "stds":     np.maximum(vals.std(axis=0, ddof=1), _STD_FLOOR),
            "n_obs":    len(vals),
            "features": feats,
        }
    return model


# ---------------------------------------------------------------------------
# Probability computation
# ---------------------------------------------------------------------------

def score_to_regime_probs(scores: np.ndarray, model: dict) -> dict[str, float]:
    """
    Convert a feature vector to posterior regime probabilities.

    scores  : 1-D float array aligned with model["features"].
    Returns {regime: probability}, values sum to 1.0.
    NaN scores return an empty dict.
    """
    if not model or np.isnan(scores).any():
        return {}

    regimes = list(model.keys())
    log_posts = np.empty(len(regimes))
    for j, (regime, params) in enumerate(model.items()):
        z   = (scores - params["means"]) / params["stds"]
        lll = -0.5 * (z ** 2 + np.log(2 * np.pi * params["stds"] ** 2)).sum()
        log_posts[j] = np.log(params["prior"]) + lll

    shift     = log_posts.max()
    log_norm  = shift + np.log(np.exp(log_posts - shift).sum())
    probs     = np.exp(log_posts - log_norm)
    return {r: float(probs[j]) for j, r in enumerate(regimes)}


def compute_prob_history(
    df: pd.DataFrame,
    model: dict | None = None,
) -> pd.DataFrame:
    """
    Compute regime probabilities for every row in df (vectorised).

    If model is None, fits from the full df (in-sample; fine for long series).

    Returns DataFrame indexed by date (if 'date' col present), columns are
    regime names + 'top_regime' (str) + 'entropy' (float, bits).
    Rows with any NaN feature values receive NaN probabilities.
    """
    if model is None:
        model = fit_naive_bayes(df)
    if not model:
        return pd.DataFrame()

    feats   = list(model.values())[0]["features"]
    X       = df[feats].values.astype(float)        # (n, p)
    valid   = ~np.isnan(X).any(axis=1)              # (n,) boolean mask

    regimes     = list(model.keys())
    n_rows      = len(X)
    n_reg       = len(regimes)
    log_posts   = np.full((n_rows, n_reg), np.nan)

    for j, params in enumerate(model.values()):
        z = (X[valid] - params["means"]) / params["stds"]      # (valid_n, p)
        ll = -0.5 * (z ** 2 + np.log(2 * np.pi * params["stds"] ** 2)).sum(axis=1)
        log_posts[valid, j] = np.log(params["prior"]) + ll

    # Normalise valid rows via log-sum-exp
    probs = np.full_like(log_posts, np.nan)
    lp_v  = log_posts[valid]
    shift = lp_v.max(axis=1, keepdims=True)
    log_norm = shift + np.log(np.exp(lp_v - shift).sum(axis=1, keepdims=True))
    probs[valid] = np.exp(lp_v - log_norm)

    out = pd.DataFrame(probs, columns=regimes)
    if _DATE_COL in df.columns:
        out.index = pd.to_datetime(df[_DATE_COL].values)

    # Shannon entropy per row (bits)
    p_clip = np.where(probs > 0, probs, np.nan)
    out["entropy"] = np.nansum(-p_clip * np.log2(p_clip), axis=1)

    # Top regime
    out["top_regime"] = out[regimes].idxmax(axis=1)
    return out


def get_current_probs(
    df: pd.DataFrame,
    model: dict | None = None,
) -> dict:
    """
    Regime probability distribution for the latest observation.

    Returns dict:
      probs          — {regime: float}, sum to 1.0
      top_regime     — highest-probability regime
      top_prob       — its probability (0–1)
      second         — second-highest regime name
      second_prob    — its probability
      entropy        — Shannon entropy in bits
      model_regimes  — ordered list of regimes in the model
    """
    if model is None:
        model = fit_naive_bayes(df)
    if not model:
        return {}

    feats  = list(model.values())[0]["features"]
    latest = df.iloc[-1]
    x      = np.array([float(latest.get(f, np.nan)) for f in feats])

    probs = score_to_regime_probs(x, model)
    if not probs:
        return {}

    sorted_p = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    p_vals   = np.array([v for v in probs.values() if v > 0])
    entropy  = float(-np.sum(p_vals * np.log2(p_vals))) if len(p_vals) > 1 else 0.0

    out = {
        "probs":         probs,
        "top_regime":    sorted_p[0][0],
        "top_prob":      round(sorted_p[0][1], 4),
        "model_regimes": list(model.keys()),
        "entropy":       round(entropy, 3),
    }
    if len(sorted_p) >= 2:
        out["second"]      = sorted_p[1][0]
        out["second_prob"] = round(sorted_p[1][1], 4)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_regime_probability(df: pd.DataFrame) -> dict:
    """
    Full regime probability analysis.

    Returns dict:
      model    — fitted Naive Bayes parameters per regime
      current  — get_current_probs() result dict
      history  — compute_prob_history() DataFrame (indexed by date)
    """
    model   = fit_naive_bayes(df)
    current = get_current_probs(df, model=model)
    history = compute_prob_history(df, model=model)
    return {"model": model, "current": current, "history": history}
