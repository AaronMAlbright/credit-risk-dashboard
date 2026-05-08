"""
Model failure analysis.

Identifies and characterises periods where the regime framework failed:
  - False positives   — model defensive while market rallied strongly
  - False negatives   — model bullish while market crashed
  - Missed crashes    — severe drawdowns the model failed to anticipate
  - Missed rallies    — strong recoveries the model stayed defensive through
  - Unstable windows  — rolling windows where strategy badly lagged SP500
  - Regime confusion  — rapid regime oscillation (flip-flopping)
  - Persistent underperformance — stretches of continuous strategy lag

Public API
----------
  classify_model_stance          — per-row defensive/neutral/bullish label
  compute_false_positives        — defensive while market rallied
  compute_false_negatives        — bullish while market crashed
  compute_missed_episodes        — named stress episodes the model missed
  compute_unstable_windows       — rolling windows of persistent lag
  compute_regime_confusion       — periods of frequent regime switching
  run_failure_analysis           — orchestrator returning full dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUALIZE = 252

# Thresholds for classifying failures
_FP_RALLY_THRESHOLD      = 0.04    # missed a 4%+ 30d rally while defensive
_FN_CRASH_THRESHOLD      = -0.05   # exposed to 5%+ drawdown while bullish
_DEFENSIVE_WEIGHT        = 0.45    # equity_weight ≤ this → model is defensive
_BULLISH_WEIGHT          = 0.65    # equity_weight ≥ this → model is bullish
_UNSTABLE_WINDOW_DAYS    = 63      # rolling window for underperformance detection
_UNDERPERFORM_THRESHOLD  = -0.08   # 63d strategy−SP500 cumulative lag triggers flag
_CONFUSION_WINDOW        = 21      # days in which to count regime transitions
_CONFUSION_THRESHOLD     = 4       # transitions in window triggers confusion flag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_model_stance(df: pd.DataFrame) -> pd.Series:
    """
    Per-row model stance: "defensive" / "neutral" / "bullish".
    Uses equity_weight if present, else composite_risk_score_smooth.
    """
    if "equity_weight" in df.columns:
        w = df["equity_weight"]
        stance = pd.Series("neutral", index=df.index)
        stance[w <= _DEFENSIVE_WEIGHT] = "defensive"
        stance[w >= _BULLISH_WEIGHT]   = "bullish"
        return stance

    if "composite_risk_score_smooth" in df.columns:
        s = df["composite_risk_score_smooth"]
        stance = pd.Series("neutral", index=df.index)
        stance[s >= 65] = "defensive"
        stance[s <= 35] = "bullish"
        return stance

    return pd.Series("unknown", index=df.index)


def _date_str(idx) -> str:
    try:
        return str(idx.date())
    except Exception:
        return str(idx)


# ---------------------------------------------------------------------------
# 1. False positives (defensive while market rallied)
# ---------------------------------------------------------------------------

def compute_false_positives(
    df: pd.DataFrame,
    rally_threshold: float = _FP_RALLY_THRESHOLD,
) -> pd.DataFrame:
    """
    Rows where the model was defensive and the market rallied > threshold.

    Returns DataFrame with columns:
      date, equity_weight, composite_risk_score_smooth,
      fwd_return_30d, magnitude, regime, likely_reason
    """
    stance = classify_model_stance(df)
    defensive = stance == "defensive"

    fwd_col = "sp500_forward_30d_return"
    if fwd_col not in df.columns:
        return pd.DataFrame()

    sub = df[defensive & (df[fwd_col] > rally_threshold)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for idx, row in sub.iterrows():
        score = float(row.get("composite_risk_score_smooth", np.nan))
        ew    = float(row.get("equity_weight", np.nan))
        fwd   = float(row[fwd_col])
        regime = str(row.get("final_decision", "Unknown"))

        # Diagnose likely reason
        reasons = []
        if not np.isnan(score) and score > 70:
            reasons.append("high composite score (justified caution that didn't materialise)")
        if "credit_market_risk_score_smooth" in row.index:
            if float(row.get("credit_market_risk_score_smooth", 0)) > 65:
                reasons.append("elevated credit risk signal")
        if "treasury_stress_score_smooth" in row.index:
            if float(row.get("treasury_stress_score_smooth", 0)) > 65:
                reasons.append("elevated treasury stress signal")
        if "vix_change_30d" in row.index:
            if float(row.get("vix_change_30d", 0)) > 5:
                reasons.append("recent VIX spike creating false alarm")
        if not reasons:
            reasons = ["composite score above defensive threshold"]

        rows.append({
            "date":                       _date_str(idx),
            "equity_weight":              round(ew, 3),
            "composite_score":            round(score, 1),
            "fwd_return_30d":             round(fwd, 4),
            "magnitude":                  round(fwd, 4),
            "regime":                     regime,
            "likely_reason":              "; ".join(reasons),
            "failure_type":               "false_positive",
        })

    return pd.DataFrame(rows).sort_values("magnitude", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. False negatives (bullish while market crashed)
# ---------------------------------------------------------------------------

def compute_false_negatives(
    df: pd.DataFrame,
    crash_threshold: float = _FN_CRASH_THRESHOLD,
) -> pd.DataFrame:
    """
    Rows where the model was bullish and the market dropped below threshold.

    Returns DataFrame with same schema as compute_false_positives.
    """
    stance = classify_model_stance(df)
    bullish = stance == "bullish"

    dd_col = "sp500_future_drawdown_30d"
    fwd_col = "sp500_forward_30d_return"
    out_col = dd_col if dd_col in df.columns else (fwd_col if fwd_col in df.columns else None)
    if out_col is None:
        return pd.DataFrame()

    sub = df[bullish & (df[out_col] < crash_threshold)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for idx, row in sub.iterrows():
        score  = float(row.get("composite_risk_score_smooth", np.nan))
        ew     = float(row.get("equity_weight", np.nan))
        outcome = float(row[out_col])
        regime = str(row.get("final_decision", "Unknown"))

        reasons = []
        if not np.isnan(score) and score < 35:
            reasons.append("composite score was in benign zone — signal missed the turn")
        for sig, label in [
            ("vix_change_30d", "VIX"),
            ("hy_change_30d", "HY spread"),
            ("real_yield_change_90d", "real yields"),
        ]:
            if sig in row.index and not np.isnan(float(row.get(sig, np.nan))):
                val = float(row[sig])
                if abs(val) > 0:
                    pass  # sign check
        if "shock_flag" in row.index and bool(row.get("shock_flag", False)):
            reasons.append("shock_flag was raised but composite did not re-enter defensive")
        if not reasons:
            reasons = ["market moved faster than signal could react"]

        rows.append({
            "date":              _date_str(idx),
            "equity_weight":     round(ew, 3),
            "composite_score":   round(score, 1),
            "outcome_30d":       round(outcome, 4),
            "magnitude":         round(abs(outcome), 4),
            "regime":            regime,
            "likely_reason":     "; ".join(reasons) or "rapid market decline, signal lag",
            "failure_type":      "false_negative",
        })

    return (
        pd.DataFrame(rows)
        .sort_values("magnitude", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 3. Named episode miss detection
# ---------------------------------------------------------------------------

_NAMED_EPISODES = [
    {"name": "GFC",             "start": "2008-09-01", "end": "2009-03-31",  "type": "crash",  "sp500_actual": -0.47},
    {"name": "EU Sovereign",    "start": "2011-07-01", "end": "2012-01-31",  "type": "crash",  "sp500_actual": -0.19},
    {"name": "China/EM 2015",   "start": "2015-08-01", "end": "2016-02-29",  "type": "crash",  "sp500_actual": -0.13},
    {"name": "Q4 2018",         "start": "2018-10-01", "end": "2018-12-31",  "type": "crash",  "sp500_actual": -0.20},
    {"name": "COVID Crash",     "start": "2020-02-19", "end": "2020-04-30",  "type": "crash",  "sp500_actual": -0.34},
    {"name": "2022 Rate Shock", "start": "2022-01-03", "end": "2022-10-14",  "type": "crash",  "sp500_actual": -0.25},
    {"name": "SVB 2023",        "start": "2023-03-08", "end": "2023-05-31",  "type": "stress", "sp500_actual": -0.08},
    {"name": "2020 Recovery",   "start": "2020-04-01", "end": "2020-12-31",  "type": "rally",  "sp500_actual":  0.65},
    {"name": "2021 Bull Run",   "start": "2021-01-01", "end": "2021-12-31",  "type": "rally",  "sp500_actual":  0.29},
    {"name": "2023 Recovery",   "start": "2023-01-01", "end": "2023-12-31",  "type": "rally",  "sp500_actual":  0.24},
]


def compute_missed_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each named episode, assess model stance and outcome.

    Returns DataFrame with columns:
      name, type, start, end, sp500_actual,
      mean_equity_weight, dominant_regime, composite_mean,
      model_assessment (correctly_positioned / missed / partial),
      miss_magnitude
    """
    stance = classify_model_stance(df)
    df_aug = df.copy()
    df_aug["_stance"] = stance

    idx = df_aug.index
    if not isinstance(idx, pd.DatetimeIndex):
        if "date" in df_aug.columns:
            df_aug = df_aug.set_index("date")
        else:
            return pd.DataFrame()

    rows = []
    for ep in _NAMED_EPISODES:
        start = pd.Timestamp(ep["start"])
        end   = pd.Timestamp(ep["end"])
        sub   = df_aug.loc[(df_aug.index >= start) & (df_aug.index <= end)]

        if sub.empty:
            rows.append({
                "name":               ep["name"],
                "type":               ep["type"],
                "start":              ep["start"],
                "end":                ep["end"],
                "sp500_actual":       ep["sp500_actual"],
                "mean_equity_weight": np.nan,
                "dominant_regime":    "No data in range",
                "composite_mean":     np.nan,
                "model_assessment":   "no_data",
                "miss_magnitude":     np.nan,
            })
            continue

        mean_ew = float(sub.get("equity_weight", pd.Series([np.nan])).mean()) \
            if "equity_weight" in sub else np.nan
        comp_mean = float(sub["composite_risk_score_smooth"].mean()) \
            if "composite_risk_score_smooth" in sub else np.nan

        dominant = (
            sub["final_decision"].value_counts().index[0]
            if "final_decision" in sub and not sub["final_decision"].isna().all()
            else "Unknown"
        )

        # Assessment
        if ep["type"] in ("crash", "stress"):
            # Good: model was defensive (low equity weight) during the crash
            if not np.isnan(mean_ew):
                if mean_ew <= _DEFENSIVE_WEIGHT:
                    assessment = "correctly_positioned"
                    miss = 0.0
                elif mean_ew <= 0.60:
                    assessment = "partial"
                    miss = round(abs(ep["sp500_actual"]) * (mean_ew - _DEFENSIVE_WEIGHT), 4)
                else:
                    assessment = "missed_crash"
                    miss = round(abs(ep["sp500_actual"]) * (mean_ew - _DEFENSIVE_WEIGHT), 4)
            else:
                assessment = "unknown"
                miss = np.nan
        else:  # rally
            if not np.isnan(mean_ew):
                if mean_ew >= _BULLISH_WEIGHT:
                    assessment = "correctly_positioned"
                    miss = 0.0
                elif mean_ew >= 0.55:
                    assessment = "partial"
                    miss = round(ep["sp500_actual"] * (_BULLISH_WEIGHT - mean_ew), 4)
                else:
                    assessment = "missed_rally"
                    miss = round(ep["sp500_actual"] * (_BULLISH_WEIGHT - mean_ew), 4)
            else:
                assessment = "unknown"
                miss = np.nan

        rows.append({
            "name":               ep["name"],
            "type":               ep["type"],
            "start":              ep["start"],
            "end":                ep["end"],
            "sp500_actual":       ep["sp500_actual"],
            "mean_equity_weight": round(mean_ew, 3) if not np.isnan(mean_ew) else np.nan,
            "dominant_regime":    str(dominant),
            "composite_mean":     round(comp_mean, 1) if not np.isnan(comp_mean) else np.nan,
            "model_assessment":   assessment,
            "miss_magnitude":     miss,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Unstable windows (rolling strategy − SP500 lag)
# ---------------------------------------------------------------------------

def compute_unstable_windows(
    df: pd.DataFrame,
    window: int = _UNSTABLE_WINDOW_DAYS,
    lag_threshold: float = _UNDERPERFORM_THRESHOLD,
) -> pd.DataFrame:
    """
    Rolling windows where strategy cumulative lag vs SP500 exceeded threshold.

    Returns DataFrame with start, end, strategy_cum, sp500_cum, lag, regime.
    """
    if "strategy_daily_return" not in df.columns or "sp500_daily_return" not in df.columns:
        return pd.DataFrame()

    strat = df["strategy_daily_return"].fillna(0).values
    bench = df["sp500_daily_return"].fillna(0).values
    n = len(strat)

    if n < window:
        return pd.DataFrame()

    idx = df.index
    rows = []
    for i in range(window, n):
        s_seg = strat[i - window:i]
        b_seg = bench[i - window:i]
        s_cum = float(np.prod(1 + s_seg) - 1)
        b_cum = float(np.prod(1 + b_seg) - 1)
        lag   = s_cum - b_cum
        if lag < lag_threshold:
            regime = str(df.iloc[i].get("final_decision", "Unknown")) \
                if "final_decision" in df.columns else "Unknown"
            rows.append({
                "start":        _date_str(idx[i - window]),
                "end":          _date_str(idx[i]),
                "strategy_cum": round(s_cum, 4),
                "sp500_cum":    round(b_cum, 4),
                "lag":          round(lag, 4),
                "regime":       regime,
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Collapse overlapping windows: keep worst lag per calendar month
    try:
        out["end_dt"] = pd.to_datetime(out["end"])
        out = (
            out.assign(month=out["end_dt"].dt.to_period("M"))
            .sort_values("lag")
            .drop_duplicates(subset="month", keep="first")
            .drop(columns=["end_dt", "month"])
            .sort_values("lag")
            .reset_index(drop=True)
        )
    except Exception:
        out = out.sort_values("lag").reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# 5. Regime confusion (rapid oscillation)
# ---------------------------------------------------------------------------

def compute_regime_confusion(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    window: int = _CONFUSION_WINDOW,
    threshold: int = _CONFUSION_THRESHOLD,
) -> pd.DataFrame:
    """
    Identifies windows with rapid regime switching (oscillation / instability).

    Returns DataFrame with: date, n_transitions, regime_sequence, composite_score.
    """
    if regime_col not in df.columns:
        return pd.DataFrame()

    regimes = df[regime_col].values
    n = len(regimes)
    transitions = (regimes[1:] != regimes[:-1]).astype(int)

    rows = []
    idx = df.index
    for i in range(window, n):
        n_trans = int(transitions[i - window:i].sum())
        if n_trans >= threshold:
            seq = regimes[i - window:i + 1]
            unique_seq = [str(seq[0])]
            for v in seq[1:]:
                if str(v) != unique_seq[-1]:
                    unique_seq.append(str(v))
            comp = float(df.iloc[i].get("composite_risk_score_smooth", np.nan)) \
                if "composite_risk_score_smooth" in df.columns else np.nan
            rows.append({
                "date":            _date_str(idx[i]),
                "n_transitions":   n_trans,
                "regime_sequence": " → ".join(unique_seq[-6:]),
                "composite_score": round(comp, 1) if not np.isnan(comp) else np.nan,
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Keep one row per month at peak confusion
    try:
        out["date_dt"] = pd.to_datetime(out["date"])
        out = (
            out.assign(month=out["date_dt"].dt.to_period("M"))
            .sort_values("n_transitions", ascending=False)
            .drop_duplicates(subset="month", keep="first")
            .drop(columns=["date_dt", "month"])
            .sort_values("n_transitions", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# 6. Orchestrator
# ---------------------------------------------------------------------------

def run_failure_analysis(df: pd.DataFrame) -> dict:
    """
    Full failure analysis.

    Returns dict with:
      false_positives      — DataFrame
      false_negatives      — DataFrame
      missed_episodes      — DataFrame
      unstable_windows     — DataFrame
      regime_confusion     — DataFrame
      summary              — dict of high-level counts
    """
    fp  = compute_false_positives(df)
    fn  = compute_false_negatives(df)
    ep  = compute_missed_episodes(df)
    uw  = compute_unstable_windows(df)
    rc  = compute_regime_confusion(df)

    n_missed = 0
    if not ep.empty and "model_assessment" in ep.columns:
        n_missed = int((ep["model_assessment"].isin(["missed_crash", "missed_rally"])).sum())

    summary = {
        "n_false_positives":    len(fp),
        "n_false_negatives":    len(fn),
        "n_missed_episodes":    n_missed,
        "n_unstable_windows":   len(uw),
        "n_confusion_periods":  len(rc),
        "total_failures":       len(fp) + len(fn),
    }

    return {
        "false_positives":  fp,
        "false_negatives":  fn,
        "missed_episodes":  ep,
        "unstable_windows": uw,
        "regime_confusion": rc,
        "summary":          summary,
    }
