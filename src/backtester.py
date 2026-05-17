import numpy as np
import pandas as pd

# Hard OOS cutoff — thresholds and signal weights were developed on data
# before this date. Everything from here onward is a genuine out-of-sample test.
OOS_CUTOFF = "2016-01-01"

# Default sizing parameters — all overridable via build_strategy_backtest kwargs
_EQUITY_FLOOR     = 0.40   # minimum equity allocation
_EQUITY_CAP       = 1.00   # maximum equity allocation
_TREND_MA_WINDOW  = 200    # days for momentum trend filter
_TREND_BOOST      = 1.15   # weight scalar when SP500 > MA
_TREND_DAMP       = 0.85   # weight scalar when SP500 < MA
_VOL_TARGET       = 0.10   # annualised vol target for vol-targeting component
_VOL_WINDOW       = 21     # rolling window for realised vol estimate
_MIN_PERSISTENCE  = 3      # EWM span to smooth rapid weight flips (days)

# Regime → target weight map (used as the regime-probability component)
_DECISION_WEIGHTS = {
    "Buy Stress":                   1.00,
    "Watch Entry":                  0.85,
    "Risk On":                      0.85,
    "Neutral":                      0.70,
    "Stress / Stabilization Watch": 0.60,
    "Hold / Do Not Chase":          0.55,
    "Divergence Warning":           0.50,
    "Avoid Chasing Risk":           0.45,
    "Credit Warning":               0.40,
    "Reduce Risk":                  0.40,
    "Active Stress":                0.40,
    "Wait":                         0.40,
}


def _trend_scalar(sp500: pd.Series, window: int = _TREND_MA_WINDOW) -> pd.Series:
    """
    Per-row multiplier: _TREND_BOOST when SP500 > rolling MA, _TREND_DAMP when below.
    Returns 1.0 during MA warm-up (no look-ahead).
    """
    ma = sp500.rolling(window, min_periods=window).mean()
    scalar = np.where(sp500 > ma, _TREND_BOOST, _TREND_DAMP)
    scalar = pd.Series(scalar, index=sp500.index, dtype=float)
    scalar[ma.isna()] = 1.0
    return scalar


def assign_strategy_return(row):
    if row["sp500_forward_30d_return"] != row["sp500_forward_30d_return"]:
        return None

    return (
        row["equity_weight"] * row["sp500_forward_30d_return"]
        + row["cash_weight"] * 0.002
    )


def build_strategy_backtest(
    df,
    equity_floor: float    = _EQUITY_FLOOR,
    equity_cap: float      = _EQUITY_CAP,
    target_vol: float      = _VOL_TARGET,
    vol_window: int        = _VOL_WINDOW,
    ma_window: int         = _TREND_MA_WINDOW,
    min_persistence: int   = _MIN_PERSISTENCE,
):
    """
    Build a full strategy backtest from a scored DataFrame.

    Four complementary sizing methods are blended into a single equity weight:

      1. score_weight     — piecewise-linear map from composite_risk_score_smooth
      2. regime_weight    — decision label → target weight lookup
      3. vol_target_weight — target_vol / rolling_SP500_vol (avoids circular dependency)
      4. momentum_weight  — score_weight scaled by the SP500 vs MA trend filter

    The blend is the row-wise mean of all four methods.  An EWM with span
    min_persistence then smooths rapid flip-flopping before the floor/cap clip.
    """
    df = df.copy()
    df["sp500_daily_return"] = df["sp500"].pct_change()

    # ── 1. Score-based weight ─────────────────────────────────────────────────
    if "composite_risk_score_smooth" in df.columns and df["composite_risk_score_smooth"].notna().any():
        from src.position_sizing import compute_score_sizing, SCORE_BREAKPOINTS
        score_s = compute_score_sizing(df, SCORE_BREAKPOINTS)
        df["score_weight"] = score_s.values
    else:
        df["score_weight"] = 0.60

    # ── 2. Regime-label weight ────────────────────────────────────────────────
    if "final_decision" in df.columns:
        df["regime_weight"] = df["final_decision"].map(_DECISION_WEIGHTS).fillna(0.60)
    else:
        df["regime_weight"] = 0.60

    # ── 3. Volatility-target weight ───────────────────────────────────────────
    # Uses SP500 realised vol — no circular dependency on strategy returns.
    _sp_ret = df["sp500_daily_return"].fillna(0)
    _realised_vol = _sp_ret.rolling(vol_window, min_periods=5).std() * np.sqrt(252)
    _realised_vol = _realised_vol.replace(0, np.nan)
    df["vol_target_weight"] = (target_vol / _realised_vol).clip(
        lower=equity_floor, upper=equity_cap
    )
    # Fill warm-up NaNs with score weight so blend stays meaningful
    df["vol_target_weight"] = df["vol_target_weight"].fillna(df["score_weight"])

    # ── 4. Momentum-adjusted weight ───────────────────────────────────────────
    df["trend_scalar"] = _trend_scalar(df["sp500"], window=ma_window).values
    df["momentum_weight"] = (df["score_weight"] * df["trend_scalar"]).clip(
        lower=equity_floor, upper=equity_cap
    )

    # ── Blend: row-wise mean of all four components ───────────────────────────
    _sizing_cols = ["score_weight", "regime_weight", "vol_target_weight", "momentum_weight"]
    df["strategy_weight_raw"] = df[_sizing_cols].mean(axis=1)

    # ── Persistence smoothing: EWM reduces rapid day-to-day flipping ──────────
    if min_persistence > 1:
        df["strategy_weight_raw"] = (
            df["strategy_weight_raw"]
            .ewm(span=min_persistence, adjust=False)
            .mean()
        )

    # ── Apply floor and cap ───────────────────────────────────────────────────
    df["strategy_weight"] = df["strategy_weight_raw"].clip(
        lower=equity_floor, upper=equity_cap
    )

    df["strategy_weight_lagged"] = df["strategy_weight"].shift(1)
    df["strategy_turnover"] = df["strategy_weight"].diff().abs().fillna(0)

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


def compute_benchmark_returns(
    df: "pd.DataFrame",
    bond_annual_return: float = 0.03,
    bond_vol: float = 0.04,
) -> "pd.DataFrame":
    """
    Add 60/40 and simplified risk-parity benchmark columns to the backtest DataFrame.

    Bond proxy is a constant annual return (approximates UST aggregate index).
    Risk parity uses 21-day rolling realised SP500 vol vs a fixed bond vol estimate,
    lagged 1 day to avoid look-ahead.

    Adds columns:
      sixty_forty_daily, sixty_forty_curve, sixty_forty_drawdown
      risk_parity_daily, risk_parity_curve, risk_parity_drawdown
    """
    d = df.copy()
    sp_ret = d["sp500"].pct_change().fillna(0)
    bond_ret = pd.Series(bond_annual_return / 252, index=d.index)

    # 60/40 — static, no rebalancing cost
    d["sixty_forty_daily"]    = 0.60 * sp_ret + 0.40 * bond_ret
    d["sixty_forty_curve"]    = (1 + d["sixty_forty_daily"]).cumprod()
    d["sixty_forty_drawdown"] = d["sixty_forty_curve"] / d["sixty_forty_curve"].cummax() - 1

    # Risk parity — inverse-vol weights, 21-day rolling, lagged 1 day
    sp_vol_raw = sp_ret.rolling(21, min_periods=5).std() * np.sqrt(252)
    sp_vol     = sp_vol_raw.replace(0, np.nan).fillna(0.15).clip(lower=0.05)
    bond_vol_s = pd.Series(bond_vol, index=d.index)
    total_inv  = (1 / sp_vol) + (1 / bond_vol_s)
    sp_rp_w    = ((1 / sp_vol) / total_inv).shift(1).fillna(0.5)
    bond_rp_w  = ((1 / bond_vol_s) / total_inv).shift(1).fillna(0.5)

    d["risk_parity_daily"]    = sp_rp_w * sp_ret + bond_rp_w * bond_ret
    d["risk_parity_curve"]    = (1 + d["risk_parity_daily"]).cumprod()
    d["risk_parity_drawdown"] = d["risk_parity_curve"] / d["risk_parity_curve"].cummax() - 1

    return d


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