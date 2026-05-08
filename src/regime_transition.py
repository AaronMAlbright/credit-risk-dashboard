import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

OUTPUT_DIR = os.path.join("outputs", "regime_transition")

DECISION_ORDER = [
    "Risk-On",
    "Neutral",
    "Caution",
    "Risk-Off",
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


def compute_hazard_rates(series: pd.Series) -> dict:
    """
    Discrete hazard rate h(d) = P(leaving regime at day d | survived to day d).

    Also returns:
      expected_remaining_duration(age)  — given current age, expected days left
      prob_exit_within_N(age, N=20)     — P(exit within N days given current age)

    Returns dict keyed by regime, each containing:
      hazard_df     — DataFrame(age_days, hazard_rate, n_at_risk, n_exits)
      median_age    — median run length
    """
    runs = _regime_runs(series.dropna())
    if runs.empty:
        return {}

    result = {}
    for regime, grp in runs.groupby("regime"):
        lengths = grp["length"].values.astype(int)
        if len(lengths) < 3:
            continue
        max_len = int(lengths.max())

        ages = np.arange(1, max_len + 1)
        n_at_risk = np.array([(lengths >= d).sum() for d in ages])
        n_exits   = np.array([(lengths == d).sum() for d in ages])

        with np.errstate(divide="ignore", invalid="ignore"):
            hazard = np.where(n_at_risk > 0, n_exits / n_at_risk, np.nan)

        result[regime] = {
            "hazard_df": pd.DataFrame({
                "age_days":   ages,
                "hazard_rate": hazard,
                "n_at_risk":  n_at_risk,
                "n_exits":    n_exits,
            }),
            "median_age": float(np.median(lengths)),
        }

    return result


def prob_exit_within(hazard_df: pd.DataFrame, current_age: int, n_days: int = 20) -> float:
    """
    Given current regime age (days) and hazard DataFrame,
    compute P(exit within n_days).

    P(survive n more days) = product(1 - h(current_age + d)) for d in 1..n_days
    P(exit within n days) = 1 - P(survive)
    """
    if hazard_df.empty:
        return np.nan

    max_age = int(hazard_df["age_days"].max())
    survival = 1.0
    for d in range(1, n_days + 1):
        age_d = current_age + d
        if age_d > max_age:
            h = float(hazard_df["hazard_rate"].iloc[-1])
        else:
            row = hazard_df[hazard_df["age_days"] == age_d]
            h = float(row["hazard_rate"].iloc[0]) if not row.empty else 0.0
        if not np.isnan(h):
            survival *= (1.0 - h)

    return round(float(1.0 - survival), 4)


def compute_current_regime_forecast(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    n_days: int = 20,
) -> dict:
    """
    Compute transition probability for the current (latest) regime.

    Returns dict with:
      current_regime    — name of the latest regime
      current_age_days  — how many days in the current streak
      prob_exit_20d     — probability of leaving this regime in next 20 days
      transition_probs  — most likely next regimes (from transition matrix)
    """
    if regime_col not in df.columns or df.empty:
        return {}

    series = df[regime_col].dropna()
    if series.empty:
        return {}

    # Current streak
    current = str(series.iloc[-1])
    age = 1
    for v in reversed(series.iloc[:-1].values):
        if str(v) == current:
            age += 1
        else:
            break

    # Hazard-based exit probability
    hazard_by_regime = compute_hazard_rates(series)
    p_exit = np.nan
    if current in hazard_by_regime:
        p_exit = prob_exit_within(hazard_by_regime[current]["hazard_df"], age, n_days)

    # Next regime probabilities from transition matrix
    _, probs = compute_transition_matrix(series)
    next_probs = {}
    if current in probs.index:
        row = probs.loc[current].sort_values(ascending=False)
        next_probs = {str(k): round(float(v), 3) for k, v in row.items() if v > 0.01}

    return {
        "current_regime":  current,
        "current_age_days": age,
        "prob_exit_20d":   p_exit,
        "transition_probs": next_probs,
    }


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

        hazard_by_regime = compute_hazard_rates(series)
        current_forecast = compute_current_regime_forecast(clean, regime_col)

        results[regime_col] = {
            "transition_counts": counts,
            "transition_probs":  probs,
            "durations":         durations,
            "forward_returns":   fwd_returns,
            "transition_returns": trans_returns,
            "hazard_by_regime":  hazard_by_regime,
            "current_forecast":  current_forecast,
        }

    return results
