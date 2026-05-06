import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

OUTPUT_DIR = os.path.join("outputs", "regime_transition")

DECISION_ORDER = [
    "Buy Stress",
    "Watch Entry",
    "Hold / Do Not Chase",
    "Neutral",
    "Wait",
    "Avoid Chasing Risk",
    "Divergence Warning",
    "Credit Warning",
    "Stress / Stabilization Watch",
]

TRANSITION_ORDER = [
    "Stable / Neutral",
    "Complacency / Late Cycle",
    "Early Deterioration",
    "Deteriorating / Rising Stress",
    "Panic / Stabilizing",
]

_FORWARD_RETURN_COLS = [
    "sp500_forward_30d_return",
    "sp500_forward_60d_return",
    "strategy_forward_30d_return",
    "sp500_future_drawdown_30d",
]


def _regime_runs(series):
    """Return DataFrame of (regime, start, end, length) consecutive runs."""
    s = series.reset_index(drop=True)
    if s.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "length"])

    runs = []
    current = s.iloc[0]
    start = 0
    for i in range(1, len(s)):
        if s.iloc[i] != current:
            runs.append({"regime": current, "start": start, "end": i - 1, "length": i - start})
            current = s.iloc[i]
            start = i
    runs.append({"regime": current, "start": start, "end": len(s) - 1, "length": len(s) - start})
    return pd.DataFrame(runs)


def compute_transition_matrix(series, ordered=None):
    """
    Build a regime transition count and probability matrix.

    Returns (count_df, prob_df) where rows = from_regime, cols = to_regime.
    Only regimes that actually appear in the series are included.
    """
    present = [r for r in (ordered or []) if r in series.values]
    if not present:
        present = sorted(series.dropna().unique().tolist())

    counts = pd.DataFrame(0, index=present, columns=present, dtype=int)
    from_vals = series.iloc[:-1].values
    to_vals = series.iloc[1:].values
    for f, t in zip(from_vals, to_vals):
        if f in counts.index and t in counts.columns:
            counts.loc[f, t] += 1

    # Drop rows/cols with no transitions
    active_rows = counts.index[counts.sum(axis=1) > 0]
    active_cols = counts.columns[counts.sum(axis=0) > 0]
    all_active = active_rows.union(active_cols)
    counts = counts.loc[all_active, all_active]

    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0)

    return counts, probs


def compute_regime_durations(series):
    """
    Per-regime persistence statistics (in trading days).

    Returns DataFrame indexed by regime with columns:
    count, mean_days, median_days, std_days, min_days, max_days
    """
    runs = _regime_runs(series.dropna())
    if runs.empty:
        return pd.DataFrame()

    stats = (
        runs.groupby("regime")["length"]
        .agg(
            count="count",
            mean_days="mean",
            median_days="median",
            std_days="std",
            min_days="min",
            max_days="max",
        )
        .round(1)
    )
    stats["std_days"] = stats["std_days"].fillna(0.0)
    return stats


def compute_regime_forward_returns(df, regime_col):
    """Mean forward returns and max drawdown by regime."""
    available = [c for c in _FORWARD_RETURN_COLS if c in df.columns]
    result = df.groupby(regime_col)[available].mean().round(4)
    return result


def compute_transition_forward_returns(df, regime_col):
    """
    Forward returns measured at the first day of each new regime
    (i.e., entry-point returns when a regime transition occurs).

    Returns DataFrame with MultiIndex (from_regime, to_regime).
    """
    series = df[regime_col]
    is_transition = (series != series.shift(1)) & series.notna() & series.shift(1).notna()
    is_transition.iloc[0] = False

    sub = df[is_transition].copy()
    sub["from_regime"] = series.shift(1)[is_transition].values

    available = [c for c in _FORWARD_RETURN_COLS if c in sub.columns]
    result = (
        sub.groupby(["from_regime", regime_col])[available]
        .mean()
        .round(4)
    )
    result.index.names = ["from_regime", "to_regime"]
    return result


def _save_heatmap(matrix, title, path):
    """Save a transition probability heatmap as PNG."""
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.3), max(4, n_rows * 0.9)))

    annot = matrix.map(lambda v: f"{v:.0%}" if v > 0 else "")

    sns.heatmap(
        matrix.astype(float),
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Transition Probability"},
        vmin=0,
        vmax=1,
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("To Regime", fontsize=11)
    ax.set_ylabel("From Regime", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_regime_analysis(df):
    """
    Run full regime transition analysis on both final_decision and transition_regime.

    Saves CSVs and heatmap PNGs to outputs/regime_transition/.
    Returns nested dict keyed by regime column name.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    targets = [
        ("final_decision", "Decision", DECISION_ORDER),
        ("transition_regime", "Transition Regime", TRANSITION_ORDER),
    ]

    results = {}
    for regime_col, label, order in targets:
        if regime_col not in df.columns:
            continue

        clean = df[df[regime_col].notna()].copy()
        series = clean[regime_col]

        counts, probs = compute_transition_matrix(series, ordered=order)
        counts.to_csv(os.path.join(OUTPUT_DIR, f"{regime_col}_counts.csv"))
        probs.to_csv(os.path.join(OUTPUT_DIR, f"{regime_col}_probs.csv"))

        _save_heatmap(
            probs,
            f"{label} Transition Probabilities",
            os.path.join(OUTPUT_DIR, f"{regime_col}_heatmap.png"),
        )

        durations = compute_regime_durations(series)
        durations.to_csv(os.path.join(OUTPUT_DIR, f"{regime_col}_durations.csv"))

        fwd_returns = compute_regime_forward_returns(clean, regime_col)
        fwd_returns.to_csv(os.path.join(OUTPUT_DIR, f"{regime_col}_forward_returns.csv"))

        trans_returns = compute_transition_forward_returns(clean, regime_col)
        trans_returns.to_csv(os.path.join(OUTPUT_DIR, f"{regime_col}_transition_returns.csv"))

        results[regime_col] = {
            "transition_counts": counts,
            "transition_probs": probs,
            "durations": durations,
            "forward_returns": fwd_returns,
            "transition_returns": trans_returns,
        }

    return results
