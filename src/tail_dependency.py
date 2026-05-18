"""
Empirical joint tail probability for major asset pairs.

How often do HY spreads and VIX both blow out simultaneously?
Pure empirical quantile approach — tells you which "diversifiers" actually
diversify under stress and which cluster against you.

Tail dependency coefficient = P(A in tail AND B in tail) / P(A in tail)
  > 1.0: assets cluster in stress
  < 1.0: asset B diversifies when A is stressed

Public API
----------
  TAIL_ASSETS       : list of (col, label, invert) tuples — module constant
  TAIL_QUANTILE     : float = 0.10
  _tail_mask                         — boolean tail indicator for one series
  compute_pair_tail_dependency       — all metrics for one asset pair
  compute_tail_dependency_matrix     — NxN matrix for all available assets
  run_tail_dependency                — top-level entry point
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAIL_ASSETS: list[tuple[str, str, bool]] = [
    ("hy_spread",  "HY Spread",  False),  # tail = high values (widening)
    ("ig_spread",  "IG Spread",  False),  # tail = high values
    ("vix",        "VIX",        False),  # tail = high VIX
    ("sp500",      "S&P 500",    True),   # tail = low values (falling)
    ("yield_10y",  "10y Yield",  True),   # tail = falling yields (flight to safety)
    ("dxy",        "DXY",        False),  # tail = strong dollar (risk-off)
]

TAIL_QUANTILE: float = 0.10   # 10th/90th percentile defines the tail

_MIN_ASSETS = 3     # minimum available assets for the analysis to run
_MIN_OBS    = 252   # minimum rows required


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _tail_mask(series: pd.Series, invert: bool, quantile: float = TAIL_QUANTILE) -> pd.Series:
    """Boolean mask: True when the series value is in its stress tail.

    invert=False: tail = values above the (1 - quantile) percentile (e.g. >90th)
    invert=True : tail = values below the quantile percentile          (e.g. <10th)

    NaN values are always False (not in tail).
    """
    clean = series.dropna()
    if len(clean) == 0:
        return pd.Series(False, index=series.index)

    if invert:
        threshold = float(np.percentile(clean.values, quantile * 100))
        return series <= threshold
    else:
        threshold = float(np.percentile(clean.values, (1.0 - quantile) * 100))
        return series >= threshold


def _safe_corr(s_a: pd.Series, s_b: pd.Series) -> float:
    """Pearson correlation on the aligned non-NaN intersection. Returns NaN if < 3 obs."""
    aligned = pd.concat([s_a, s_b], axis=1).dropna()
    if len(aligned) < 3:
        return float("nan")
    arr_a = aligned.iloc[:, 0].values.astype(float)
    arr_b = aligned.iloc[:, 1].values.astype(float)
    # numpy corrcoef on 2-row matrix
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.corrcoef(arr_a, arr_b)
    val = float(cc[0, 1])
    return val if np.isfinite(val) else float("nan")


# ---------------------------------------------------------------------------
# Pair-level computation
# ---------------------------------------------------------------------------

def compute_pair_tail_dependency(
    s_a: pd.Series,
    invert_a: bool,
    s_b: pd.Series,
    invert_b: bool,
) -> dict:
    """Compute empirical tail dependency metrics for one asset pair.

    Parameters
    ----------
    s_a, s_b   : aligned pd.Series (same index, or alignment done internally)
    invert_a/b : True = tail is low values; False = tail is high values

    Returns
    -------
    dict with keys:
        joint_tail_prob     : float — P(A in tail AND B in tail)
        tail_dep_coef       : float — P(joint) / P(A in tail)
        tail_dep_coef_rev   : float — P(joint) / P(B in tail)
        stress_corr         : float — correlation conditional on A OR B in tail
        unconditional_corr  : float — standard Pearson correlation
        n_joint_tail_obs    : int
        interpretation      : str
    """
    try:
        # align on shared non-NaN index
        combined = pd.concat([s_a.rename("a"), s_b.rename("b")], axis=1).dropna()
        n = len(combined)
        if n < 30:
            return {
                "joint_tail_prob": float("nan"),
                "tail_dep_coef": float("nan"),
                "tail_dep_coef_rev": float("nan"),
                "stress_corr": float("nan"),
                "unconditional_corr": float("nan"),
                "n_joint_tail_obs": 0,
                "interpretation": "Insufficient data",
            }

        ser_a = combined["a"]
        ser_b = combined["b"]

        mask_a = _tail_mask(ser_a, invert_a)
        mask_b = _tail_mask(ser_b, invert_b)

        p_a = float(mask_a.mean())
        p_b = float(mask_b.mean())
        p_joint = float((mask_a & mask_b).mean())
        n_joint = int((mask_a & mask_b).sum())

        # tail dependency coefficients
        tdc   = (p_joint / p_a)  if p_a  > 0 else float("nan")
        tdc_r = (p_joint / p_b)  if p_b  > 0 else float("nan")

        # conditional correlation: rows where A OR B is in its tail
        stress_mask = mask_a | mask_b
        stress_rows = combined[stress_mask]
        if len(stress_rows) >= 3:
            with np.errstate(invalid="ignore", divide="ignore"):
                cc = np.corrcoef(stress_rows["a"].values.astype(float),
                                 stress_rows["b"].values.astype(float))
            stress_corr = float(cc[0, 1]) if np.isfinite(cc[0, 1]) else float("nan")
        else:
            stress_corr = float("nan")

        unconditional_corr = _safe_corr(ser_a, ser_b)

        # interpretation buckets
        if not np.isfinite(tdc):
            interp = "Insufficient data"
        elif tdc >= 2.5:
            interp = "Clusters strongly in stress"
        elif tdc >= 1.2:
            interp = "Partial clustering in stress"
        elif tdc >= 0.8:
            interp = "Near-independent in tails"
        else:
            interp = "Diversifies in stress"

        return {
            "joint_tail_prob":    round(p_joint, 6),
            "tail_dep_coef":      round(tdc,   4) if np.isfinite(tdc)   else float("nan"),
            "tail_dep_coef_rev":  round(tdc_r, 4) if np.isfinite(tdc_r) else float("nan"),
            "stress_corr":        round(stress_corr, 4) if np.isfinite(stress_corr) else float("nan"),
            "unconditional_corr": round(unconditional_corr, 4) if np.isfinite(unconditional_corr) else float("nan"),
            "n_joint_tail_obs":   n_joint,
            "interpretation":     interp,
        }

    except Exception:
        return {
            "joint_tail_prob": float("nan"),
            "tail_dep_coef": float("nan"),
            "tail_dep_coef_rev": float("nan"),
            "stress_corr": float("nan"),
            "unconditional_corr": float("nan"),
            "n_joint_tail_obs": 0,
            "interpretation": "Computation error",
        }


# ---------------------------------------------------------------------------
# Full matrix
# ---------------------------------------------------------------------------

def compute_tail_dependency_matrix(df: pd.DataFrame) -> dict:
    """Compute the full NxN tail dependency matrix for available TAIL_ASSETS.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex; columns include asset columns listed in
         TAIL_ASSETS (not required to have all — missing columns are skipped).

    Returns
    -------
    dict with keys:
        available             : bool
        assets                : list[str] — labels of available assets
        tail_dep_matrix       : pd.DataFrame (NxN, tail_dep_coef, labeled)
        stress_corr_matrix    : pd.DataFrame (NxN stress correlations)
        unconditional_corr_matrix : pd.DataFrame
        pair_details          : dict — {(label_a, label_b): compute_pair_... result}
        highest_dependency    : tuple — (label_a, label_b, coef) most clustered
        lowest_dependency     : tuple — (label_a, label_b, coef) most diversifying
        n_assets              : int
        n_obs                 : int
    """
    try:
        # filter to columns present in df
        available_assets = [
            (col, label, invert)
            for col, label, invert in TAIL_ASSETS
            if col in df.columns and df[col].notna().sum() >= 30
        ]

        if len(available_assets) < _MIN_ASSETS:
            return {"available": False}

        n = len(available_assets)
        labels = [label for _, label, _ in available_assets]

        # initialise matrices with NaN
        tdc_mat   = pd.DataFrame(np.eye(n), index=labels, columns=labels)
        sc_mat    = pd.DataFrame(np.full((n, n), float("nan")), index=labels, columns=labels)
        uc_mat    = pd.DataFrame(np.full((n, n), float("nan")), index=labels, columns=labels)

        # diagonal: stress_corr and unconditional_corr of a series with itself = 1
        for lbl in labels:
            sc_mat.loc[lbl, lbl] = 1.0
            uc_mat.loc[lbl, lbl] = 1.0

        pair_details: dict[tuple[str, str], dict] = {}

        for i, (col_a, label_a, inv_a) in enumerate(available_assets):
            for j, (col_b, label_b, inv_b) in enumerate(available_assets):
                if j <= i:
                    continue

                result = compute_pair_tail_dependency(
                    df[col_a], inv_a,
                    df[col_b], inv_b,
                )

                pair_details[(label_a, label_b)] = result

                tdc   = result["tail_dep_coef"]
                tdc_r = result["tail_dep_coef_rev"]
                sc    = result["stress_corr"]
                uc    = result["unconditional_corr"]

                # fill both triangles: (A,B) uses tdc; (B,A) uses tdc_rev
                tdc_mat.loc[label_a, label_b] = tdc
                tdc_mat.loc[label_b, label_a] = tdc_r
                sc_mat.loc[label_a, label_b]  = sc
                sc_mat.loc[label_b, label_a]  = sc
                uc_mat.loc[label_a, label_b]  = uc
                uc_mat.loc[label_b, label_a]  = uc

        # find highest and lowest off-diagonal tdc
        off_diag_vals: list[tuple[str, str, float]] = []
        for (la, lb), res in pair_details.items():
            v = res["tail_dep_coef"]
            if np.isfinite(v):
                off_diag_vals.append((la, lb, v))

        if off_diag_vals:
            off_diag_vals.sort(key=lambda x: x[2], reverse=True)
            highest = off_diag_vals[0]
            lowest  = off_diag_vals[-1]
        else:
            highest = ("", "", float("nan"))
            lowest  = ("", "", float("nan"))

        n_obs = int(df[[col for col, _, _ in available_assets]].dropna().shape[0])

        return {
            "available":                  True,
            "assets":                     labels,
            "tail_dep_matrix":            tdc_mat,
            "stress_corr_matrix":         sc_mat,
            "unconditional_corr_matrix":  uc_mat,
            "pair_details":               pair_details,
            "highest_dependency":         highest,
            "lowest_dependency":          lowest,
            "n_assets":                   n,
            "n_obs":                      n_obs,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Pair-finding helpers for key_findings
# ---------------------------------------------------------------------------

def _format_pair_finding(label_a: str, label_b: str, res: dict) -> str:
    """Format one pair result into an actionable sentence."""
    tdc = res["tail_dep_coef"]
    sc  = res["stress_corr"]
    interp = res["interpretation"]

    sc_str = f", stress corr {sc:+.2f}" if np.isfinite(sc) else ""
    if not np.isfinite(tdc):
        return f"{label_a} / {label_b}: insufficient data."

    return (
        f"{label_a} and {label_b} {interp.lower()} "
        f"(tail dep = {tdc:.2f}x{sc_str})."
    )


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_tail_dependency(df: pd.DataFrame) -> dict:
    """Top-level entry point for tail dependency analysis.

    Requires at least _MIN_ASSETS (3) available TAIL_ASSETS columns in df
    and at least _MIN_OBS (252) rows.

    Parameters
    ----------
    df : master DataFrame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        available       : bool
        matrix          : dict from compute_tail_dependency_matrix
        interpretation  : str — high-level summary
        key_findings    : list[str] — top 3 most actionable pair findings
        warning         : str or None — set if any pair tdc > 3.0
    """
    try:
        if df is None or len(df) < _MIN_OBS:
            return {"available": False}

        # check that at least MIN_ASSETS columns exist and have data
        present = [
            col for col, _, _ in TAIL_ASSETS
            if col in df.columns and df[col].notna().sum() >= 30
        ]
        if len(present) < _MIN_ASSETS:
            return {"available": False}

        matrix = compute_tail_dependency_matrix(df)
        if not matrix.get("available", False):
            return {"available": False}

        pair_details = matrix["pair_details"]

        # --- build key_findings: top 3 pairs by most extreme tdc ---
        scored: list[tuple[float, str, str, dict]] = []
        for (la, lb), res in pair_details.items():
            tdc = res["tail_dep_coef"]
            if np.isfinite(tdc):
                # score by distance from 1.0 (most extreme in either direction)
                scored.append((abs(tdc - 1.0), la, lb, res))

        scored.sort(key=lambda x: x[0], reverse=True)
        key_findings = [
            _format_pair_finding(la, lb, res)
            for _, la, lb, res in scored[:3]
        ]

        # --- warning if any tdc > 3.0 ---
        extreme = [
            f"{la}/{lb} ({res['tail_dep_coef']:.1f}x)"
            for (la, lb), res in pair_details.items()
            if np.isfinite(res["tail_dep_coef"]) and res["tail_dep_coef"] > 3.0
        ]
        warning = (
            f"Extreme tail clustering detected: {', '.join(extreme)}. "
            "These assets move together almost deterministically in stress — "
            "treating them as separate risk positions overstates diversification."
        ) if extreme else None

        # --- high-level interpretation ---
        hi_la, hi_lb, hi_coef = matrix["highest_dependency"]
        lo_la, lo_lb, lo_coef = matrix["lowest_dependency"]

        hi_str = (
            f"{hi_la} and {hi_lb} cluster strongly in stress "
            f"(tail dep = {hi_coef:.1f}x) — when one blows out, so does the other."
            if np.isfinite(hi_coef) else "No high-dependency pair identified."
        )
        lo_str = (
            f"{lo_la} and {lo_lb} are near-independent in tails "
            f"({lo_coef:.2f}x) — {lo_lb} provides genuine stress diversification."
            if np.isfinite(lo_coef) else "No low-dependency pair identified."
        )

        interpretation = f"{hi_str} {lo_str}"

        return {
            "available":    True,
            "matrix":       matrix,
            "interpretation": interpretation,
            "key_findings": key_findings,
            "warning":      warning,
        }

    except Exception:
        return {"available": False}
