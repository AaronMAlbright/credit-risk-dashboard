"""
Historical credit episode case studies.
"""

from __future__ import annotations

import pandas as pd

from src.channel_attribution import channel_contribution_table


EPISODES = {
    "COVID Stress": ("2020-02-15", "2020-04-30"),
    "2022 Rates Shock": ("2022-01-01", "2022-12-31"),
    "Banking Stress": ("2023-03-01", "2023-05-31"),
    "Tight-Spread Complacency": ("2024-01-01", "2024-12-31"),
}


def episode_case_study(df: pd.DataFrame, episode: str = "2022 Rates Shock") -> dict:
    if df.empty:
        return {"available": False, "reason": "No data available"}
    if episode not in EPISODES:
        return {"available": False, "reason": f"Unknown episode: {episode}"}

    work = df.copy()
    date = pd.to_datetime(work["date"]) if "date" in work.columns else pd.to_datetime(work.index)
    start, end = [pd.Timestamp(x) for x in EPISODES[episode]]
    sub = work[(date >= start) & (date <= end)]
    if sub.empty:
        return {"available": False, "reason": f"No rows in episode window {start.date()} to {end.date()}"}

    first = sub.iloc[0]
    last = sub.iloc[-1]
    peak_idx = sub["composite_risk_score_smooth"].idxmax() if "composite_risk_score_smooth" in sub.columns else sub.index[-1]
    peak = sub.loc[peak_idx]

    drivers = channel_contribution_table(sub)
    top_driver = drivers.iloc[0]["channel"] if not drivers.empty else "Unavailable"

    result = {
        "available": True,
        "episode": episode,
        "start": str(start.date()),
        "end": str(end.date()),
        "n_rows": len(sub),
        "start_regime": first.get("final_decision", "Unknown"),
        "end_regime": last.get("final_decision", "Unknown"),
        "peak_score": peak.get("composite_risk_score_smooth"),
        "peak_regime": peak.get("final_decision", "Unknown"),
        "top_driver": top_driver,
        "hy_spread_change": None,
        "sp500_return": None,
        "lesson": "",
    }

    if "hy_spread" in sub.columns:
        result["hy_spread_change"] = float(sub["hy_spread"].iloc[-1] - sub["hy_spread"].iloc[0])
    if "sp500" in sub.columns:
        result["sp500_return"] = float(sub["sp500"].iloc[-1] / sub["sp500"].iloc[0] - 1)

    result["lesson"] = (
        f"During {episode}, the framework moved from {result['start_regime']} to "
        f"{result['end_regime']}; peak regime was {result['peak_regime']} with "
        f"{top_driver} as the largest institutional channel contributor."
    )
    return result


def case_study_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode in EPISODES:
        result = episode_case_study(df, episode)
        if not result.get("available"):
            continue
        rows.append(
            {
                "episode": result["episode"],
                "window": f"{result['start']} to {result['end']}",
                "start_regime": result["start_regime"],
                "end_regime": result["end_regime"],
                "peak_score": result["peak_score"],
                "peak_regime": result["peak_regime"],
                "top_driver": result["top_driver"],
                "hy_spread_change": result["hy_spread_change"],
                "sp500_return": result["sp500_return"],
            }
        )
    return pd.DataFrame(rows)

