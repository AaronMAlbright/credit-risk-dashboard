"""
Walk-forward validation framework.

Takes the fully scored DataFrame produced by pipeline.run_analytics() and
evaluates strategy performance on rolling out-of-sample test windows.

No production scoring logic is modified or re-run. Each test window uses
signal columns (final_decision, score columns, equity_weight) that were
computed from data strictly prior to the test window start date, because
all rolling features in the pipeline use only trailing windows.

Usage:
    from src.pipeline import run_analytics, run_scoring_pipeline
    from src.walk_forward import run_walk_forward, save_results

    df, _, _ = run_analytics(pipeline_df)
    windows_df, regimes_df = run_walk_forward(df)
    save_results(windows_df, regimes_df)
"""
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Required columns — asserted at entry to run_walk_forward
# ---------------------------------------------------------------------------
_REQUIRED_COLS = [
    "strategy_daily_return",
    "sp500_daily_return",
    "final_decision",
    "equity_weight",
    "sp500_forward_30d_return",
]

# Forward-looking columns that must never appear as scoring inputs.
# Checked by assert_no_leakage().
_FORWARD_LABEL_COLS = [
    "sp500_forward_30d_return",
    "sp500_forward_60d_return",
    "hy_forward_30d_change",
    "hy_forward_60d_change",
    "sp500_future_min_30d",
    "sp500_future_min_60d",
    "sp500_future_drawdown_30d",
    "sp500_future_drawdown_60d",
]

# Columns that drive the final_decision — none should overlap with label cols.
_SIGNAL_INPUT_COLS = [
    "macro_risk_score_smooth",
    "credit_market_risk_score_smooth",
    "liquidity_regime_score_smooth",
    "cross_asset_divergence_score_smooth",
    "complacency_score_smooth",
    "mean_reversion_score_smooth",
    "transition_regime",
    "shock_flag",
    "credit_equity_divergence",
    "vol_credit_mismatch",
]


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------
def assert_no_leakage(df: pd.DataFrame) -> None:
    """
    Verify that forward-looking label columns are not used as signal inputs.
    Raises AssertionError if any overlap is detected.

    Two checks:
      1. Name overlap — a forward-label column name appears in the signal inputs list.
      2. Data contamination — a signal column's values are identical to a forward-label
         column's values (detects explicit overwrite of signal data with future returns).
    """
    signal_set = set(_SIGNAL_INPUT_COLS)
    label_set = set(_FORWARD_LABEL_COLS)

    name_overlap = signal_set & label_set & set(df.columns)
    assert not name_overlap, f"Leakage: forward-looking columns in signal inputs: {name_overlap}"

    for sig_col in _SIGNAL_INPUT_COLS:
        if sig_col not in df.columns:
            continue
        for label_col in _FORWARD_LABEL_COLS:
            if label_col not in df.columns:
                continue
            mask = df[sig_col].notna() & df[label_col].notna()
            if mask.sum() >= 2 and (df.loc[mask, sig_col] == df.loc[mask, label_col]).all():
                assert False, (
                    f"Leakage: signal column '{sig_col}' has identical values to "
                    f"forward label '{label_col}'"
                )


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------
def generate_windows(
    df: pd.DataFrame,
    train_days: int = 126,
    test_days: int = 42,
    step_days: int = 42,
) -> list:
    """
    Generate rolling train/test window definitions.

    Each window guarantees test_start > train_end so no test row can have
    contributed to any signal used to score that same row.

    Returns a list of dicts with keys:
        window_id, train_start, train_end, test_start, test_end
    """
    dates = df.index
    n = len(dates)
    windows = []
    window_id = 0
    cursor = train_days  # first test window starts after minimum training history

    while cursor + test_days <= n:
        train_end_idx = cursor - 1
        test_end_idx  = cursor + test_days - 1

        windows.append({
            "window_id":   window_id,
            "train_start": dates[0],
            "train_end":   dates[train_end_idx],
            "test_start":  dates[cursor],
            "test_end":    dates[test_end_idx],
            "_test_slice": slice(cursor, cursor + test_days),
        })
        window_id += 1
        cursor += step_days

    return windows


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def _window_metrics(daily_returns: pd.Series, prefix: str) -> dict:
    """
    Compute annualised performance metrics from a daily return series.

    Returns keys:
        {prefix}_total_return, {prefix}_sharpe, {prefix}_hit_rate,
        {prefix}_max_drawdown, {prefix}_volatility
    """
    r = daily_returns.dropna()

    if len(r) < 2:
        return {
            f"{prefix}_total_return":  np.nan,
            f"{prefix}_sharpe":        np.nan,
            f"{prefix}_hit_rate":      np.nan,
            f"{prefix}_max_drawdown":  np.nan,
            f"{prefix}_volatility":    np.nan,
        }

    equity   = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    std      = r.std()

    return {
        f"{prefix}_total_return": round(float(equity.iloc[-1] - 1), 6),
        f"{prefix}_sharpe":       round(float(r.mean() / std * np.sqrt(252)) if std > 0 else np.nan, 4),
        f"{prefix}_hit_rate":     round(float((r > 0).mean()), 4),
        f"{prefix}_max_drawdown": round(float(drawdown.min()), 6),
        f"{prefix}_volatility":   round(float(std * np.sqrt(252)), 6),
    }


# ---------------------------------------------------------------------------
# Single-window evaluation
# ---------------------------------------------------------------------------
def evaluate_window(df_test: pd.DataFrame, window_id: int, train_end: pd.Timestamp) -> dict:
    """
    Compute strategy and SP500 benchmark metrics for a single test window.

    Returns a single flat dict suitable for appending to the windows results CSV.
    """
    strategy_metrics = _window_metrics(df_test["strategy_daily_return"], "strategy")
    sp500_metrics    = _window_metrics(df_test["sp500_daily_return"],     "sp500")

    dominant_regime = (
        df_test["final_decision"].value_counts().index[0]
        if df_test["final_decision"].notna().any()
        else "Unknown"
    )

    row = {
        "window_id":        window_id,
        "train_end":        train_end,
        "test_start":       df_test.index[0],
        "test_end":         df_test.index[-1],
        "n_obs":            len(df_test),
        "dominant_regime":  dominant_regime,
    }
    row.update(strategy_metrics)
    row.update(sp500_metrics)
    return row


# ---------------------------------------------------------------------------
# Regime breakdown for a single window
# ---------------------------------------------------------------------------
def compute_regime_breakdown(
    df_test: pd.DataFrame,
    window_id: int,
    train_end: pd.Timestamp,
) -> list:
    """
    Break test-window performance down by final_decision regime.

    Uses sp500_forward_30d_return as the outcome measure — this is the label
    the signal is trying to predict. avg_equity_weight tracks how exposed the
    strategy was while in each regime.

    Returns a list of dicts, one per regime present in this window.
    """
    rows = []

    for regime, group in df_test.groupby("final_decision", observed=True):
        fwd = group["sp500_forward_30d_return"].dropna()
        n_fwd = len(fwd)

        rows.append({
            "window_id":              window_id,
            "train_end":              train_end,
            "test_start":             df_test.index[0],
            "test_end":               df_test.index[-1],
            "regime":                 regime,
            "n_obs":                  len(group),
            "n_obs_with_fwd_return":  n_fwd,
            "avg_forward_30d_return": round(float(fwd.mean()), 6)  if n_fwd > 0 else np.nan,
            "hit_rate":               round(float((fwd > 0).mean()), 4) if n_fwd > 0 else np.nan,
            "worst_5pct":             round(float(fwd.quantile(0.05)), 6) if n_fwd >= 10 else np.nan,
            "avg_equity_weight":      round(float(group["equity_weight"].mean()), 4),
        })

    return rows


# ---------------------------------------------------------------------------
# Full walk-forward run
# ---------------------------------------------------------------------------
def run_walk_forward(
    df: pd.DataFrame,
    train_days: int = 126,
    test_days: int = 42,
    step_days: int = 42,
) -> tuple:
    """
    Run the complete walk-forward evaluation.

    Parameters
    ----------
    df          : Scored DataFrame from pipeline.run_analytics(). Must contain
                  all columns in _REQUIRED_COLS.
    train_days  : Minimum trading days of history before the first test window.
    test_days   : Trading days per test window.
    step_days   : Trading days between consecutive test window starts.
                  step_days == test_days → non-overlapping windows.
                  step_days <  test_days → overlapping windows.

    Returns
    -------
    windows_df  : DataFrame with one row per test window (performance metrics).
    regimes_df  : DataFrame with one row per regime per test window.
    """
    # --- validate inputs ---
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"run_walk_forward: missing required columns: {missing}")

    assert_no_leakage(df)

    windows = generate_windows(df, train_days=train_days, test_days=test_days, step_days=step_days)

    if not windows:
        raise ValueError(
            f"No windows generated. Dataset has {len(df)} rows but "
            f"train_days={train_days} leaves only {len(df) - train_days} rows "
            f"for test windows of size {test_days}."
        )

    window_rows = []
    regime_rows = []

    for w in windows:
        df_test    = df.iloc[w["_test_slice"]]
        train_end  = w["train_end"]
        window_id  = w["window_id"]

        window_rows.append(evaluate_window(df_test, window_id, train_end))
        regime_rows.extend(compute_regime_breakdown(df_test, window_id, train_end))

    windows_df = pd.DataFrame(window_rows)
    regimes_df = pd.DataFrame(regime_rows)

    return windows_df, regimes_df


# ---------------------------------------------------------------------------
# Frozen OOS splits
# ---------------------------------------------------------------------------

FROZEN_SPLITS: list[dict] = [
    {
        "name":         "Pre-2016 train → 2016–2025 test",
        "train_end":    "2015-12-31",
        "test_start":   "2016-01-01",
        "description":  "Post-QE3 normalisation and political risk era",
    },
    {
        "name":         "Pre-2019 train → 2019–2025 test",
        "train_end":    "2018-12-31",
        "test_start":   "2019-01-01",
        "description":  "Includes COVID crash and 2022 rate shock as pure OOS",
    },
    {
        "name":         "Pre-COVID train → Post-COVID test",
        "train_end":    "2020-02-19",
        "test_start":   "2020-03-01",
        "description":  "Zero-rate era, inflation regime, regional bank stress OOS",
    },
    {
        "name":         "Pre-2022 train → 2022–2025 test",
        "train_end":    "2021-12-31",
        "test_start":   "2022-01-01",
        "description":  "Rate shock, SVB, and post-pandemic normalisation OOS",
    },
]

_CAVEAT = (
    "NOTE: These splits evaluate performance on held-out time periods, but the "
    "scoring rules themselves were designed with awareness of the full dataset. "
    "Results are pseudo-OOS: they show regime framework robustness across time, "
    "not true walk-forward with frozen parameters. Treat as directional only."
)


def run_frozen_splits(
    df: pd.DataFrame,
    splits: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Evaluate strategy metrics on pre-defined frozen train/test date splits.

    Parameters
    ----------
    df     : Scored DataFrame from pipeline.run_analytics(). Must have
             strategy_daily_return, sp500_daily_return, final_decision.
    splits : List of split dicts (default: FROZEN_SPLITS).

    Returns
    -------
    DataFrame with one row per split showing IS and OOS performance metrics.
    """
    if splits is None:
        splits = FROZEN_SPLITS

    missing = [c for c in ["strategy_daily_return", "sp500_daily_return"] if c not in df.columns]
    if missing:
        raise ValueError(f"run_frozen_splits: missing columns {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        else:
            raise ValueError("df must have DatetimeIndex or 'date' column")

    rows = []
    for sp in splits:
        train_end   = pd.Timestamp(sp["train_end"])
        test_start  = pd.Timestamp(sp["test_start"])

        df_train = df[df.index <= train_end]
        df_test  = df[df.index >= test_start]

        row = {
            "name":         sp["name"],
            "description":  sp["description"],
            "train_end":    str(train_end.date()),
            "test_start":   str(test_start.date()),
            "test_end":     str(df_test.index[-1].date()) if not df_test.empty else "—",
            "n_train_days": len(df_train),
            "n_test_days":  len(df_test),
        }

        for label, sub in [("is", df_train), ("oos", df_test)]:
            if len(sub) < 2:
                for k in ["strategy_sharpe", "sp500_sharpe", "strategy_maxdd",
                          "sp500_maxdd", "strategy_total_ret", "sp500_total_ret",
                          "excess_sharpe"]:
                    row[f"{label}_{k}"] = np.nan
                continue

            for prefix, col in [("strategy", "strategy_daily_return"),
                                 ("sp500",    "sp500_daily_return")]:
                m = _window_metrics(sub[col], prefix)
                for k, v in m.items():
                    row[f"{label}_{k}"] = v

            s_sh = row.get(f"{label}_strategy_sharpe", np.nan)
            b_sh = row.get(f"{label}_sp500_sharpe", np.nan)
            row[f"{label}_excess_sharpe"] = (
                round(float(s_sh - b_sh), 4)
                if not (np.isnan(s_sh) or np.isnan(b_sh)) else np.nan
            )

        # IS → OOS Sharpe degradation
        is_sh  = row.get("is_strategy_sharpe", np.nan)
        oos_sh = row.get("oos_strategy_sharpe", np.nan)
        row["sharpe_degradation"] = (
            round(float(oos_sh - is_sh), 4)
            if not (np.isnan(is_sh) or np.isnan(oos_sh)) else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results(
    windows_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    output_dir: str = "outputs/validation",
) -> None:
    """
    Save walk-forward results to CSV.

    Writes:
        {output_dir}/walk_forward_windows.csv
        {output_dir}/walk_forward_regimes.csv
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    windows_path = Path(output_dir) / "walk_forward_windows.csv"
    regimes_path = Path(output_dir) / "walk_forward_regimes.csv"

    windows_df.to_csv(windows_path, index=False)
    regimes_df.to_csv(regimes_path, index=False)

    print(f"\n=== WALK-FORWARD RESULTS SAVED ===")
    print(f"Windows : {windows_path}  ({len(windows_df)} windows)")
    print(f"Regimes : {regimes_path}  ({len(regimes_df)} rows)")
