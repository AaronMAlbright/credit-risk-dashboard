"""
Parameter sensitivity framework.

Tests strategy robustness by perturbing three independent parameter groups:

1. score_scale        — Each smoothed sub-score column × multiplier (0.7–1.3).
                        Flows into generate_final_decision → strategy_weight → backtest.
2. composite_weights  — The six composite-blend weights (sum = 1.0).
                        Affects composite_risk_score and equity_weight allocation.
3. backtester_weights — Decision-to-equity-weight mapping inside the backtester.
                        Directly scales strategy_daily_return.

No production scoring files are modified. All re-scoring is done via direct
calls to the same engine functions used by the pipeline.

Entry point:
    from src.sensitivity import run_sensitivity, save_sensitivity_results
    from src.pipeline import run_analytics, run_scoring_pipeline

    df, _, _ = run_analytics(pipeline_df)
    results_df, unstable = run_sensitivity(df)
    save_sensitivity_results(results_df, unstable)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.composite_engine import build_composite_risk
from src.portfolio_engine import generate_portfolio_weights
from src.backtester import build_strategy_backtest, assign_strategy_return
from src.risk_engine import generate_final_decision, classify_transition_regime


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------

COMPOSITE_WEIGHTS_DEFAULT = {
    "macro":          0.25,
    "credit":         0.25,
    "liquidity":      0.15,
    "treasury":       0.10,
    "complacency":    0.20,
    "mean_reversion": 0.05,
}

BACKTESTER_WEIGHTS_DEFAULT = {
    "buy_entry": 1.00,   # Buy Stress, Watch Entry
    "neutral":   0.70,   # Neutral
    "hold":      0.45,   # Hold / Do Not Chase, Avoid Chasing Risk
    "warning":   0.25,   # Credit Warning, Divergence Warning
    "stress":    0.15,   # Wait, Stress / Stabilization Watch
}

SCORE_SCALE_COLS = [
    "macro_risk_score_smooth",
    "credit_market_risk_score_smooth",
    "liquidity_regime_score_smooth",
    "treasury_stress_score_smooth",
    "complacency_score_smooth",
    "mean_reversion_score_smooth",
]

SCORE_COL_LABELS = {
    "macro_risk_score_smooth":             "macro",
    "credit_market_risk_score_smooth":     "credit",
    "liquidity_regime_score_smooth":       "liquidity",
    "treasury_stress_score_smooth":        "treasury",
    "complacency_score_smooth":            "complacency",
    "mean_reversion_score_smooth":         "mean_reversion",
}

_REQUIRED_COLS = [
    "sp500",
    "final_decision",
    "transition_regime",
    "shock_flag",
    "credit_equity_divergence",
    "vol_credit_mismatch",
    "strategy_daily_return",
    "sp500_daily_return",
    "strategy_forward_30d_return",
    "equity_weight",
    "cash_weight",
] + SCORE_SCALE_COLS


# ---------------------------------------------------------------------------
# Private: metric helpers
# ---------------------------------------------------------------------------

def _compute_daily_metrics(df: pd.DataFrame) -> dict:
    """Sharpe, total return, max drawdown, hit rate from strategy_daily_return."""
    r = df["strategy_daily_return"].dropna()
    if len(r) < 2:
        return {
            "sharpe": np.nan, "total_return": np.nan,
            "max_drawdown": np.nan, "hit_rate": np.nan,
        }
    std = r.std()
    equity = (1 + r).cumprod()
    return {
        "sharpe":       round(float(r.mean() / std * np.sqrt(252)) if std > 0 else np.nan, 4),
        "total_return": round(float(equity.iloc[-1] - 1), 6),
        "max_drawdown": round(float((equity / equity.cummax() - 1).min()), 6),
        "hit_rate":     round(float((r > 0).mean()), 4),
    }


def _compute_forward_metrics(df: pd.DataFrame) -> dict:
    """Mean return, hit rate, worst 5th pct from strategy_forward_30d_return."""
    r = df["strategy_forward_30d_return"].dropna()
    if len(r) < 2:
        return {"fwd_mean_return": np.nan, "fwd_hit_rate": np.nan, "fwd_worst_5pct": np.nan}
    return {
        "fwd_mean_return": round(float(r.mean()), 6),
        "fwd_hit_rate":    round(float((r > 0).mean()), 4),
        "fwd_worst_5pct":  round(float(r.quantile(0.05)), 6) if len(r) >= 10 else np.nan,
    }


# ---------------------------------------------------------------------------
# Private: composite weight helpers
# ---------------------------------------------------------------------------

def _classify_composite(score: float) -> str:
    """Local mirror of composite_engine.classify_composite_risk."""
    if score >= 70:
        return "Defensive / High Risk"
    if score >= 50:
        return "Caution"
    if score >= 25:
        return "Neutral"
    return "Risk-On"


def _perturb_composite_weights(base: dict, param: str, new_value: float) -> dict:
    """
    Set param to new_value, rescale remaining weights proportionally so sum = 1.0.
    """
    weights = base.copy()
    weights[param] = new_value
    others = [k for k in weights if k != param]
    other_sum = sum(weights[k] for k in others)
    if other_sum > 0:
        scale = (1.0 - new_value) / other_sum
        for k in others:
            weights[k] = round(weights[k] * scale, 8)
    return weights


# ---------------------------------------------------------------------------
# Private: re-builders
# ---------------------------------------------------------------------------

def _apply_score_scale(df: pd.DataFrame, col: str, scale: float) -> pd.DataFrame:
    """
    Scale one score column by a multiplier (clipped to [0, 100]).
    Re-derives transition_regime when macro or credit is scaled.
    Re-runs generate_final_decision and build_strategy_backtest.
    """
    df = df.copy()
    df[col] = (df[col] * scale).clip(0, 100)

    if col in ("macro_risk_score_smooth", "credit_market_risk_score_smooth"):
        df["macro_risk_momentum_10d"]  = df["macro_risk_score_smooth"].diff(10)
        df["credit_risk_momentum_10d"] = df["credit_market_risk_score_smooth"].diff(10)
        df["combined_stress_momentum_10d"] = (
            df[["macro_risk_momentum_10d", "credit_risk_momentum_10d"]].mean(axis=1)
        )
        df["transition_regime"] = df.apply(
            lambda row: classify_transition_regime(
                row["macro_risk_score_smooth"],
                row["credit_market_risk_score_smooth"],
                row["combined_stress_momentum_10d"],
            ),
            axis=1,
        )

    df["final_decision_obj"] = df.apply(
        lambda row: generate_final_decision(
            row["macro_risk_score_smooth"],
            row["credit_market_risk_score_smooth"],
            row["liquidity_regime_score_smooth"],
            row["cross_asset_divergence_score_smooth"],
            row["complacency_score_smooth"],
            row["mean_reversion_score_smooth"],
            row["transition_regime"],
            row["shock_flag"],
            row["credit_equity_divergence"],
            row["vol_credit_mismatch"],
        ),
        axis=1,
    )
    df["final_decision"] = df["final_decision_obj"].apply(lambda x: x["final_decision"])
    df = build_strategy_backtest(df)
    return df


def _apply_composite_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Re-blend composite score with perturbed weights.
    Re-runs portfolio weight allocation and forward-return computation.
    Daily backtest is also rebuilt (final_decision is unchanged).
    """
    df = df.copy()
    mr_comp = 100 - df["mean_reversion_score_smooth"]

    df["composite_risk_score"] = (
        weights["macro"]          * df["macro_risk_score_smooth"]
        + weights["credit"]       * df["credit_market_risk_score_smooth"]
        + weights["liquidity"]    * df["liquidity_regime_score_smooth"]
        + weights["treasury"]     * df["treasury_stress_score_smooth"]
        + weights["complacency"]  * df["complacency_score_smooth"]
        + weights["mean_reversion"] * mr_comp
    ).clip(0, 100)

    df["composite_risk_score_smooth"] = (
        df["composite_risk_score"].rolling(10, min_periods=1).mean()
    )
    df["composite_risk_label"] = df["composite_risk_score_smooth"].apply(_classify_composite)

    pw = df.apply(generate_portfolio_weights, axis=1)
    df["equity_weight"] = pw.apply(lambda x: x["equity_weight"])
    df["credit_weight"] = pw.apply(lambda x: x["credit_weight"])
    df["cash_weight"]   = pw.apply(lambda x: x["cash_weight"])

    df["strategy_forward_30d_return"] = df.apply(assign_strategy_return, axis=1)
    df = build_strategy_backtest(df)
    return df


def _apply_backtester_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Re-run backtest with a custom decision → strategy_weight mapping.
    Only the strategy return columns are updated; scores and decisions are unchanged.
    """
    df = df.copy()

    def _w(row):
        d = row.get("final_decision", "Neutral")
        if d in ("Buy Stress", "Watch Entry"):
            return weights["buy_entry"]
        if d == "Neutral":
            return weights["neutral"]
        if d in ("Hold / Do Not Chase", "Avoid Chasing Risk"):
            return weights["hold"]
        if d in ("Credit Warning", "Divergence Warning"):
            return weights["warning"]
        if d in ("Wait", "Stress / Stabilization Watch"):
            return weights["stress"]
        return weights["neutral"]

    df["strategy_weight"] = df.apply(_w, axis=1)
    df["strategy_weight_lagged"] = df["strategy_weight"].shift(1)
    df["strategy_daily_return"] = (
        df["strategy_weight_lagged"] * df["sp500_daily_return"]
    ).fillna(0)
    df["strategy_equity_curve"] = (1 + df["strategy_daily_return"]).cumprod()
    df["strategy_drawdown"] = (
        df["strategy_equity_curve"] / df["strategy_equity_curve"].cummax() - 1
    )
    return df


# ---------------------------------------------------------------------------
# Public sweeps
# ---------------------------------------------------------------------------

def sweep_score_scales(df: pd.DataFrame, n_steps: int = 7) -> pd.DataFrame:
    """
    Sweep each smoothed score column from 0.7× to 1.3× its baseline.

    Returns one row per (score_column × scale) with daily backtest metrics.
    """
    scales = np.linspace(0.7, 1.3, n_steps)
    rows = []
    for col in SCORE_SCALE_COLS:
        label = SCORE_COL_LABELS[col]
        for scale in scales:
            perturbed = _apply_score_scale(df, col, float(scale))
            rows.append({
                "group":       "score_scale",
                "param":       label,
                "param_value": round(float(scale), 4),
                **_compute_daily_metrics(perturbed),
            })
    return pd.DataFrame(rows)


def sweep_composite_weights(df: pd.DataFrame, n_steps: int = 7) -> pd.DataFrame:
    """
    Sweep each composite blend weight ±30% relative to its default, renormalizing others.

    Reports both forward-return metrics (primary — these are directly affected)
    and daily backtest metrics (unchanged when final_decision is not modified).
    """
    rows = []
    for param, base_value in COMPOSITE_WEIGHTS_DEFAULT.items():
        lo = max(0.01, base_value * 0.7)
        hi = min(0.99, base_value * 1.3)
        for v in np.linspace(lo, hi, n_steps):
            weights   = _perturb_composite_weights(COMPOSITE_WEIGHTS_DEFAULT, param, float(v))
            perturbed = _apply_composite_weights(df, weights)
            rows.append({
                "group":       "composite_weights",
                "param":       param,
                "param_value": round(float(v), 6),
                **_compute_forward_metrics(perturbed),
                **_compute_daily_metrics(perturbed),
            })
    return pd.DataFrame(rows)


def sweep_backtester_weights(df: pd.DataFrame, n_steps: int = 7) -> pd.DataFrame:
    """
    Sweep each backtester decision-weight ±30% relative to its default.

    Reports daily backtest metrics (directly driven by this mapping).
    """
    rows = []
    for param, base_value in BACKTESTER_WEIGHTS_DEFAULT.items():
        lo = max(0.01, base_value * 0.7)
        hi = min(1.00, base_value * 1.3)
        for v in np.linspace(lo, hi, n_steps):
            weights   = {**BACKTESTER_WEIGHTS_DEFAULT, param: float(v)}
            perturbed = _apply_backtester_weights(df, weights)
            rows.append({
                "group":       "backtester_weights",
                "param":       param,
                "param_value": round(float(v), 4),
                **_compute_daily_metrics(perturbed),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Instability detection
# ---------------------------------------------------------------------------

def identify_unstable_parameters(
    results_df: pd.DataFrame,
    cv_threshold: float = 0.25,
    metrics: list = None,
) -> list:
    """
    Flag parameters whose coefficient of variation (std / |mean|) exceeds cv_threshold
    for any tracked metric.

    Returns a list of dicts sorted descending by CV:
        {group, param, metric, cv, min_value, max_value, mean_value}
    """
    if metrics is None:
        metrics = ["sharpe", "total_return", "max_drawdown", "hit_rate",
                   "fwd_mean_return", "fwd_hit_rate"]

    available = [m for m in metrics if m in results_df.columns]
    unstable = []

    for (group, param), sub in results_df.groupby(["group", "param"]):
        for metric in available:
            vals = sub[metric].dropna()
            if len(vals) < 2:
                continue
            mean = vals.mean()
            if abs(mean) < 1e-9:
                continue
            cv = vals.std() / abs(mean)
            if cv > cv_threshold:
                unstable.append({
                    "group":      group,
                    "param":      param,
                    "metric":     metric,
                    "cv":         round(float(cv), 4),
                    "min_value":  round(float(vals.min()), 6),
                    "max_value":  round(float(vals.max()), 6),
                    "mean_value": round(float(mean), 6),
                })

    return sorted(unstable, key=lambda x: x["cv"], reverse=True)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

_HEATMAP_METRICS = {
    "score_scale":        ["sharpe", "total_return", "max_drawdown", "hit_rate"],
    "composite_weights":  ["fwd_mean_return", "fwd_hit_rate"],
    "backtester_weights": ["sharpe", "total_return", "max_drawdown"],
}


def plot_sensitivity_heatmaps(
    results_df: pd.DataFrame,
    output_dir: str = "outputs/sensitivity/heatmaps",
) -> None:
    """Generate one heatmap PNG per (group × metric) combination."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for group in results_df["group"].unique():
        sub = results_df[results_df["group"] == group]
        for metric in _HEATMAP_METRICS.get(group, ["sharpe"]):
            if metric not in sub.columns:
                continue

            pivot = sub.pivot_table(
                index="param_value", columns="param", values=metric, aggfunc="mean"
            )
            if pivot.empty:
                continue

            n_params = len(pivot.columns)
            fig, ax = plt.subplots(figsize=(max(6, n_params * 1.4), 5))
            im = ax.imshow(
                pivot.values, aspect="auto", cmap="RdYlGn",
                origin="lower", interpolation="nearest",
            )
            plt.colorbar(im, ax=ax, label=metric)

            ax.set_xticks(range(n_params))
            ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.3f}" for v in pivot.index], fontsize=8)
            ax.set_xlabel("Parameter")
            ax.set_ylabel("Perturbation Value")
            ax.set_title(f"{group} — {metric}", fontsize=11)

            plt.tight_layout()
            fig.savefig(Path(output_dir) / f"{group}_{metric}.png", dpi=120)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_sensitivity(
    df: pd.DataFrame,
    n_steps: int = 7,
) -> tuple:
    """
    Run all three parameter sweeps and identify unstable parameters.

    Parameters
    ----------
    df      : Scored + analytics DataFrame from pipeline.run_analytics().
              Must contain all columns in _REQUIRED_COLS.
    n_steps : Perturbation steps per parameter (default 7, use 3 for quick runs).

    Returns
    -------
    results_df : DataFrame with all sweep results (group, param, param_value, metrics).
    unstable   : List of dicts from identify_unstable_parameters().
    """
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"run_sensitivity: missing required columns: {missing}")

    score_df      = sweep_score_scales(df, n_steps)
    composite_df  = sweep_composite_weights(df, n_steps)
    backtester_df = sweep_backtester_weights(df, n_steps)

    results_df = pd.concat([score_df, composite_df, backtester_df], ignore_index=True)
    unstable   = identify_unstable_parameters(results_df)

    return results_df, unstable


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_sensitivity_results(
    results_df: pd.DataFrame,
    unstable: list,
    output_dir: str = "outputs/sensitivity",
) -> None:
    """
    Save sensitivity_results.csv, heatmap PNGs, and sensitivity_report.txt.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_path = Path(output_dir) / "sensitivity_results.csv"
    results_df.to_csv(results_path, index=False)

    plot_sensitivity_heatmaps(results_df, str(Path(output_dir) / "heatmaps"))

    lines = ["PARAMETER SENSITIVITY REPORT", "=" * 50, ""]

    if not unstable:
        lines.append("No unstable parameters detected (CV threshold: 0.25).")
    else:
        lines.append(f"{len(unstable)} unstable parameter(s) detected (CV > 0.25):\n")
        for item in unstable:
            lines.append(
                f"  [{item['group']}] {item['param']} / {item['metric']}"
                f"   CV={item['cv']:.3f}"
                f"   range=[{item['min_value']:.4f}, {item['max_value']:.4f}]"
                f"   mean={item['mean_value']:.4f}"
            )

    counts = results_df.groupby("group")["param"].nunique()
    lines += [
        "",
        "Parameter groups tested:",
        f"  score_scale        : {counts.get('score_scale', 0)} params",
        f"  composite_weights  : {counts.get('composite_weights', 0)} params",
        f"  backtester_weights : {counts.get('backtester_weights', 0)} params",
        f"  Total sweep rows   : {len(results_df)}",
    ]

    report_path = Path(output_dir) / "sensitivity_report.txt"
    report_path.write_text("\n".join(lines))

    print("\n=== SENSITIVITY RESULTS SAVED ===")
    print(f"Results  : {results_path}  ({len(results_df)} rows)")
    print(f"Heatmaps : {Path(output_dir) / 'heatmaps'}/")
    print(f"Report   : {report_path}")
    if unstable:
        print(f"\nTop unstable parameters (by CV):")
        for item in unstable[:5]:
            print(
                f"  [{item['group']}] {item['param']} / {item['metric']}"
                f"  CV={item['cv']:.3f}"
            )
