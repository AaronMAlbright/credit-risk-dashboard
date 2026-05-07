"""
Stress episode analysis.

Maps known historical and recent market stress episodes onto the model's
composite risk signal. For episodes within the dataset's date range, computes
actual model behaviour (mean/peak composite, decision distribution, drawdown,
lead-time). Episodes outside the range are flagged as historical references.

Public API
----------
  STRESS_EPISODES            — canonical list of episode metadata dicts
  compute_episode_stats      — per-episode summary DataFrame
  compute_score_paths        — daily composite + decision series per episode
  build_stress_timeline      — Plotly figure: composite score + shaded bands
  run_stress_analysis        — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Episode registry
# ---------------------------------------------------------------------------

# severity: "Severe" | "Moderate" | "Mild"
STRESS_EPISODES: list[dict] = [
    {
        "name":     "GFC",
        "start":    "2008-09-01",
        "end":      "2009-03-31",
        "label":    "Global Financial Crisis",
        "severity": "Severe",
    },
    {
        "name":     "EU Sovereign",
        "start":    "2011-07-01",
        "end":      "2012-01-31",
        "label":    "EU Sovereign Debt Crisis",
        "severity": "Moderate",
    },
    {
        "name":     "China / EM",
        "start":    "2015-08-01",
        "end":      "2016-02-29",
        "label":    "China / EM Selloff",
        "severity": "Moderate",
    },
    {
        "name":     "Q4 2018",
        "start":    "2018-10-01",
        "end":      "2018-12-31",
        "label":    "Q4 2018 Credit Selloff",
        "severity": "Moderate",
    },
    {
        "name":     "COVID Crash",
        "start":    "2020-02-19",
        "end":      "2020-04-30",
        "label":    "COVID-19 Crash",
        "severity": "Severe",
    },
    {
        "name":     "Rate Shock",
        "start":    "2022-01-03",
        "end":      "2022-10-14",
        "label":    "Fed Rate Shock 2022",
        "severity": "Moderate",
    },
    {
        "name":     "Aug 2024 Flash",
        "start":    "2024-07-29",
        "end":      "2024-08-12",
        "label":    "Yen Carry Unwind / BOJ Flash Crash",
        "severity": "Mild",
    },
    {
        "name":     "Q4 2024",
        "start":    "2024-10-01",
        "end":      "2024-11-15",
        "label":    "Q4 2024 Macro Uncertainty",
        "severity": "Mild",
    },
    {
        "name":     "Jan–Feb 2025",
        "start":    "2025-01-20",
        "end":      "2025-02-28",
        "label":    "Jan–Feb 2025 Tariff Fears",
        "severity": "Mild",
    },
    {
        "name":     "Liberation Day",
        "start":    "2025-04-02",
        "end":      "2025-05-06",
        "label":    "Apr 2025 Liberation Day Tariff Shock",
        "severity": "Moderate",
    },
]

_SEVERITY_COLORS: dict[str, str] = {
    "Severe":   "rgba(231, 76,  60,  0.18)",
    "Moderate": "rgba(230, 126, 34,  0.18)",
    "Mild":     "rgba(241, 196, 15,  0.15)",
}

_COMPOSITE_COL  = "composite_risk_score_smooth"
_DECISION_COL   = "final_decision"
_RETURN_COL     = "sp500_daily_return"
_SP500_COL      = "sp500"
_HIGH_ALERT_THRESHOLD = 60.0   # composite ≥ this → model is in warning territory


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _episode_mask(df: pd.DataFrame, ep: dict) -> pd.Series:
    """Boolean mask for rows within episode date range."""
    return (df["date"] >= pd.Timestamp(ep["start"])) & (df["date"] <= pd.Timestamp(ep["end"]))


def _sp500_episode_drawdown(sub: pd.DataFrame) -> float:
    """Max drawdown of SP500 price level within episode sub-frame."""
    if _SP500_COL not in sub.columns:
        return np.nan
    prices = sub[_SP500_COL].dropna().values.astype(float)
    if len(prices) < 2:
        return np.nan
    peak = prices[0]
    worst = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (p - peak) / peak
        if dd < worst:
            worst = dd
    return float(worst)


def _lead_days(df: pd.DataFrame, ep: dict, threshold: float = _HIGH_ALERT_THRESHOLD,
               lookback: int = 90) -> int | None:
    """
    Days before episode start that composite first crossed `threshold`.
    Returns None when episode is not in dataset or threshold never crossed.
    """
    if _COMPOSITE_COL not in df.columns:
        return None
    ep_start = pd.Timestamp(ep["start"])
    lookback_start = ep_start - pd.Timedelta(days=lookback)
    pre = df[(df["date"] >= lookback_start) & (df["date"] < ep_start)].copy()
    if pre.empty:
        return None
    above = pre[pre[_COMPOSITE_COL] >= threshold]
    if above.empty:
        return None
    first_cross = above["date"].min()
    return max(0, (ep_start - first_cross).days)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_episode_stats(
    df: pd.DataFrame,
    episodes: list[dict] = STRESS_EPISODES,
) -> pd.DataFrame:
    """
    Per-episode summary table.

    Returns DataFrame indexed by episode name with columns:
      label, severity, start, end, covered,
      n_obs, mean_composite, peak_composite, peak_date,
      sp500_drawdown, primary_decision, high_alert_pct, lead_days
    For uncovered episodes: numeric columns are NaN, covered=False.
    """
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    rows = []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for ep in episodes:
        mask = _episode_mask(df, ep)
        sub = df[mask]
        covered = len(sub) > 0

        if not covered:
            rows.append({
                "name":             ep["name"],
                "label":            ep["label"],
                "severity":         ep["severity"],
                "start":            ep["start"],
                "end":              ep["end"],
                "covered":          False,
                "n_obs":            0,
                "mean_composite":   np.nan,
                "peak_composite":   np.nan,
                "peak_date":        None,
                "sp500_drawdown":   np.nan,
                "primary_decision": None,
                "high_alert_pct":   np.nan,
                "lead_days":        None,
            })
            continue

        comp = sub[_COMPOSITE_COL].dropna() if _COMPOSITE_COL in sub.columns else pd.Series(dtype=float)
        mean_comp = float(comp.mean()) if len(comp) > 0 else np.nan
        peak_comp = float(comp.max()) if len(comp) > 0 else np.nan
        peak_date = None
        if len(comp) > 0:
            idx_peak = comp.idxmax()
            peak_date = str(sub.loc[idx_peak, "date"].date()) if idx_peak in sub.index else None

        primary_decision = None
        if _DECISION_COL in sub.columns and not sub[_DECISION_COL].dropna().empty:
            primary_decision = sub[_DECISION_COL].mode().iloc[0]

        high_alert_pct = np.nan
        if len(comp) > 0:
            high_alert_pct = float((comp >= _HIGH_ALERT_THRESHOLD).mean())

        sp500_dd = _sp500_episode_drawdown(sub)
        lead = _lead_days(df, ep)

        rows.append({
            "name":             ep["name"],
            "label":            ep["label"],
            "severity":         ep["severity"],
            "start":            ep["start"],
            "end":              ep["end"],
            "covered":          True,
            "n_obs":            len(sub),
            "mean_composite":   round(mean_comp, 1) if not np.isnan(mean_comp) else np.nan,
            "peak_composite":   round(peak_comp, 1) if not np.isnan(peak_comp) else np.nan,
            "peak_date":        peak_date,
            "sp500_drawdown":   round(sp500_dd, 4) if not np.isnan(sp500_dd) else np.nan,
            "primary_decision": primary_decision,
            "high_alert_pct":   round(high_alert_pct, 4) if not np.isnan(high_alert_pct) else np.nan,
            "lead_days":        lead,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("name")


def compute_score_paths(
    df: pd.DataFrame,
    episodes: list[dict] = STRESS_EPISODES,
    pad_days: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    Daily composite score + decision for each covered episode, with `pad_days`
    of context on each side.

    Returns dict: {episode_name: DataFrame} with columns date, composite, decision.
    Only includes episodes with data in df.
    """
    if df.empty or "date" not in df.columns:
        return {}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    result = {}

    for ep in episodes:
        ep_start = pd.Timestamp(ep["start"])
        ep_end   = pd.Timestamp(ep["end"])
        win_start = ep_start - pd.Timedelta(days=pad_days)
        win_end   = ep_end   + pd.Timedelta(days=pad_days)

        sub = df[(df["date"] >= win_start) & (df["date"] <= win_end)].copy()
        if sub.empty:
            continue

        cols = {"date": sub["date"]}
        if _COMPOSITE_COL in sub.columns:
            cols["composite"] = sub[_COMPOSITE_COL].values
        if _DECISION_COL in sub.columns:
            cols["decision"] = sub[_DECISION_COL].values

        result[ep["name"]] = pd.DataFrame(cols).reset_index(drop=True)

    return result


def build_stress_timeline(
    df: pd.DataFrame,
    episodes: list[dict] = STRESS_EPISODES,
) -> go.Figure:
    """
    Composite risk score over full dataset with shaded stress episode bands.

    Only draws bands for covered episodes. Uncovered episodes shown in legend
    as greyed reference annotations.
    """
    fig = go.Figure()

    if df.empty or _COMPOSITE_COL not in df.columns:
        return fig

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Composite score trace
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df[_COMPOSITE_COL],
        mode="lines",
        name="Composite Risk Score",
        line=dict(color="#1a1a2e", width=1.5),
    ))

    # High-alert threshold line
    fig.add_hline(
        y=_HIGH_ALERT_THRESHOLD,
        line_dash="dot",
        line_color="#e74c3c",
        line_width=1,
        annotation_text="High Alert (60)",
        annotation_position="top right",
        annotation_font_size=10,
    )

    # Episode shaded regions (covered only)
    added_severities: set[str] = set()
    for ep in episodes:
        mask = _episode_mask(df, ep)
        if not mask.any():
            continue
        sev = ep["severity"]
        color = _SEVERITY_COLORS.get(sev, "rgba(150,150,150,0.15)")
        show_legend = sev not in added_severities
        added_severities.add(sev)
        fig.add_vrect(
            x0=ep["start"],
            x1=ep["end"],
            fillcolor=color,
            opacity=1,
            layer="below",
            line_width=0,
            annotation_text=ep["name"],
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="#555",
        )
        if show_legend:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=color.replace("0.18", "0.8").replace("0.15", "0.7"), symbol="square"),
                name=f"{sev} episode",
                showlegend=True,
            ))

    fig.update_layout(
        title="Composite Risk Score with Stress Episode Overlay",
        xaxis_title="Date",
        yaxis_title="Composite Risk Score",
        yaxis=dict(range=[0, 105]),
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def run_stress_analysis(
    df: pd.DataFrame,
    episodes: list[dict] = STRESS_EPISODES,
) -> dict:
    """
    Full stress episode analysis.

    Returns dict with:
      episode_stats   — per-episode summary DataFrame
      score_paths     — dict of daily composite DataFrames per covered episode
      timeline_chart  — Plotly figure
      n_covered       — int, episodes with data in dataset
      n_historical    — int, episodes outside dataset range
    """
    stats = compute_episode_stats(df, episodes)
    paths = compute_score_paths(df, episodes)
    chart = build_stress_timeline(df, episodes)

    n_covered   = int(stats["covered"].sum()) if not stats.empty else 0
    n_historical = len(episodes) - n_covered

    return {
        "episode_stats":  stats,
        "score_paths":    paths,
        "timeline_chart": chart,
        "n_covered":      n_covered,
        "n_historical":   n_historical,
    }
