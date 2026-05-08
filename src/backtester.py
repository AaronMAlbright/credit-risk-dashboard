import numpy as np
import pandas as pd

# Hard OOS cutoff — thresholds and signal weights were developed on data
# before this date. Everything from here onward is a genuine out-of-sample test.
OOS_CUTOFF = "2020-01-01"

# Minimum equity allocation regardless of how bearish the signal is.
# Prevents going to near-zero exposure and missing sharp V-shaped recoveries.
_EQUITY_FLOOR = 0.40

_DECISION_WEIGHTS = {
    "Buy Stress":         1.00,
    "Watch Entry":        0.85,
    "Risk On":            0.85,
    "Neutral":            0.70,
    "Hold / Do Not Chase": 0.55,
    "Avoid Chasing Risk": 0.55,
    "Credit Warning":     0.45,
    "Reduce Risk":        0.45,
    "Active Stress":      0.40,
    "Wait":               0.40,
}


def assign_strategy_return(row):
    if row["sp500_forward_30d_return"] != row["sp500_forward_30d_return"]:
        return None

    return (
        row["equity_weight"] * row["sp500_forward_30d_return"]
        + row["cash_weight"] * 0.002
    )


def build_strategy_backtest(df):
    df = df.copy()

    df["sp500_daily_return"] = df["sp500"].pct_change()

    # Use piecewise score-based sizing when the composite score is available.
    # Falls back to decision-bucket lookup for data that lacks the score column.
    if "composite_risk_score_smooth" in df.columns and df["composite_risk_score_smooth"].notna().any():
        from src.position_sizing import compute_score_sizing, SCORE_BREAKPOINTS
        score_weights = compute_score_sizing(df, SCORE_BREAKPOINTS)
        df["strategy_weight"] = score_weights.values
    else:
        df["strategy_weight"] = df["final_decision"].map(_DECISION_WEIGHTS).fillna(0.60)

    # Apply minimum equity floor — never go below 40% regardless of signal
    df["strategy_weight"] = df["strategy_weight"].clip(lower=_EQUITY_FLOOR)

    df["strategy_weight_lagged"] = df["strategy_weight"].shift(1)

    df["strategy_daily_return"] = (
        df["strategy_weight_lagged"] * df["sp500_daily_return"]
    )

    df["strategy_daily_return"] = df["strategy_daily_return"].fillna(0)
    df["sp500_daily_return"] = df["sp500_daily_return"].fillna(0)

    df["strategy_equity_curve"] = (1 + df["strategy_daily_return"]).cumprod()
    df["sp500_equity_curve"] = (1 + df["sp500_daily_return"]).cumprod()

    df["strategy_drawdown"] = (
        df["strategy_equity_curve"] / df["strategy_equity_curve"].cummax()
        - 1
    )

    df["sp500_backtest_drawdown"] = (
        df["sp500_equity_curve"] / df["sp500_equity_curve"].cummax()
        - 1
    )

    return df


def compute_backtest_summary(df, tc_bps=10):
    """
    Compute backtest summary stats with optional transaction costs.

    tc_bps : basis points charged per day of weight change (default 10 bps).
             Applied when strategy_weight_lagged differs from the prior day.
    """
    d = df.copy()

    # Transaction cost: deduct tc_bps on days where weight changes
    if "strategy_weight_lagged" in d.columns and tc_bps > 0:
        weight_change = d["strategy_weight_lagged"].diff().abs().fillna(0)
        tc = weight_change * (tc_bps / 10_000)
        d["strategy_daily_return"] = d["strategy_daily_return"] - tc

    valid = d.dropna(subset=["strategy_forward_30d_return"])

    summary = {
        "avg_strategy_30d_return": valid["strategy_forward_30d_return"].mean(),
        "avg_sp500_30d_return":    valid["sp500_forward_30d_return"].mean(),
        "strategy_hit_rate":  (valid["strategy_forward_30d_return"] > 0).mean(),
        "sp500_hit_rate":     (valid["sp500_forward_30d_return"] > 0).mean(),
        "strategy_worst_5pct": valid["strategy_forward_30d_return"].quantile(0.05),
        "sp500_worst_5pct":    valid["sp500_forward_30d_return"].quantile(0.05),
    }

    if "strategy_equity_curve" in d.columns:
        # Recompute equity curve with transaction costs applied
        eq = (1 + d["strategy_daily_return"].fillna(0)).cumprod()
        sp = (1 + d["sp500_daily_return"].fillna(0)).cumprod()
        dd_strat = (eq / eq.cummax() - 1).min()
        dd_sp    = (sp / sp.cummax() - 1).min()
        summary.update({
            "strategy_total_return": eq.iloc[-1] - 1,
            "sp500_total_return":    sp.iloc[-1] - 1,
            "strategy_volatility":   d["strategy_daily_return"].std() * np.sqrt(252),
            "sp500_volatility":      d["sp500_daily_return"].std() * np.sqrt(252),
            "strategy_max_drawdown": dd_strat,
            "sp500_max_drawdown":    dd_sp,
            "strategy_sharpe": (
                d["strategy_daily_return"].mean() /
                d["strategy_daily_return"].std() * np.sqrt(252)
                if d["strategy_daily_return"].std() > 0 else np.nan
            ),
            "sp500_sharpe": (
                d["sp500_daily_return"].mean() /
                d["sp500_daily_return"].std() * np.sqrt(252)
                if d["sp500_daily_return"].std() > 0 else np.nan
            ),
        })

    return summary


def compute_oos_split(df, cutoff=OOS_CUTOFF, tc_bps=10):
    """
    Split the backtest into in-sample and out-of-sample periods and compute
    separate performance stats for each.

    Parameters
    ----------
    df      : output of build_strategy_backtest()
    cutoff  : ISO date string — first date of the OOS period
    tc_bps  : transaction cost in basis points per day of weight change

    Returns dict:
        cutoff          — the cutoff date used
        in_sample       — summary dict for IS period
        out_of_sample   — summary dict for OOS period
        full_period     — summary dict for full history
        is_start        — first IS date (str)
        is_end          — last IS date (str)
        oos_start       — first OOS date (str)
        oos_end         — last OOS date (str)
        is_n_days       — number of IS trading days
        oos_n_days      — number of OOS trading days
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"] if "date" in d.columns else d.index)

    cutoff_ts = pd.Timestamp(cutoff)
    is_mask   = d["date"] < cutoff_ts
    oos_mask  = d["date"] >= cutoff_ts

    is_df  = d[is_mask].copy()
    oos_df = d[oos_mask].copy()

    def _safe_summary(sub):
        if sub.empty:
            return {}
        # Rebase equity curves to 1.0 for the sub-period
        sub = sub.copy()
        if "strategy_daily_return" in sub.columns:
            sub["strategy_equity_curve"] = (
                1 + sub["strategy_daily_return"].fillna(0)
            ).cumprod()
            sub["sp500_equity_curve"] = (
                1 + sub["sp500_daily_return"].fillna(0)
            ).cumprod()
            sub["strategy_drawdown"] = (
                sub["strategy_equity_curve"] /
                sub["strategy_equity_curve"].cummax() - 1
            )
            sub["sp500_backtest_drawdown"] = (
                sub["sp500_equity_curve"] /
                sub["sp500_equity_curve"].cummax() - 1
            )
        return compute_backtest_summary(sub, tc_bps=tc_bps)

    def _date_str(sub, which="first"):
        if sub.empty:
            return "—"
        return str(sub["date"].iloc[0 if which == "first" else -1].date())

    return {
        "cutoff":        cutoff,
        "in_sample":     _safe_summary(is_df),
        "out_of_sample": _safe_summary(oos_df),
        "full_period":   _safe_summary(d),
        "is_start":      _date_str(is_df, "first"),
        "is_end":        _date_str(is_df, "last"),
        "oos_start":     _date_str(oos_df, "first"),
        "oos_end":       _date_str(oos_df, "last"),
        "is_n_days":     int(is_mask.sum()),
        "oos_n_days":    int(oos_mask.sum()),
    }