"""PM review packet for the credit compensation scorecard."""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from src.credit_compensation_scorecard import build_credit_compensation_scorecard
from src.credit_compensation_validation import (
    analyze_scorecard_prediction_errors,
    analyze_scorecard_transitions,
    replay_scorecard_stress_episodes,
    validate_scorecard_recommendations,
)


def _markdown_table(table: pd.DataFrame, max_rows: int = 25) -> str:
    if table is None or table.empty:
        return "_Unavailable_"
    view = table.head(max_rows).copy()
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in view.itertuples(index=False):
        values = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    if len(table) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(table)} rows._")
    return "\n".join(lines)


def _csv_bytes(tables: dict[str, pd.DataFrame], markdown: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pm_review_packet.md", markdown)
        for name, table in tables.items():
            if table is not None and not table.empty:
                zf.writestr(f"{name}.csv", table.to_csv(index=False))
    return buf.getvalue()


def _add_section(lines: list[str], title: str, text: str | None = None, table: pd.DataFrame | None = None) -> None:
    lines.extend(["", f"## {title}", ""])
    if text:
        lines.extend([text, ""])
    if table is not None:
        lines.extend([_markdown_table(table), ""])


def build_pm_review_packet(df: pd.DataFrame) -> dict:
    """Build downloadable PM packet markdown and CSV bundle."""
    scorecard = build_credit_compensation_scorecard(df)
    if not scorecard.get("available"):
        return scorecard

    validation = validate_scorecard_recommendations(df)
    transitions = analyze_scorecard_transitions(df)
    errors = analyze_scorecard_prediction_errors(df)
    stress_replay = replay_scorecard_stress_episodes(df)

    current = scorecard.get("current", {})
    as_of = scorecard.get("audit_summary", {}).get("as_of")
    recommendation = scorecard.get("recommendation", "Unavailable")
    lines = [
        "# Credit Compensation PM Review Packet",
        "",
        f"As of: **{as_of}**",
        "",
        f"Recommendation: **{recommendation}**",
        "",
        scorecard.get("pm_final_verdict", ""),
        "",
        "## Current Snapshot",
        "",
        f"- HY OAS: {current.get('hy_oas_bps'):.0f} bps" if current.get("hy_oas_bps") is not None else "- HY OAS: unavailable",
        f"- Excess spread: {current.get('excess_spread_bps'):.0f} bps" if current.get("excess_spread_bps") is not None else "- Excess spread: unavailable",
        f"- Compensation ratio: {current.get('spread_compensation_ratio'):.2f}x" if current.get("spread_compensation_ratio") is not None else "- Compensation ratio: unavailable",
        f"- Net spread beta: {scorecard['net_spread_beta_summary']['net_spread_beta']:.2f}x",
        f"- Incremental CDX HY protection: {scorecard['cdx_hedge_summary']['incremental_cdx_hy_protection_pct']:.1f}% NAV",
    ]

    _add_section(lines, "Trade Memo", table=scorecard.get("memo_table"))
    _add_section(lines, "Portfolio Expression", table=scorecard.get("action_table"))
    _add_section(lines, "Rating-Bucket Allocation", table=scorecard.get("rating_bucket_table"))
    _add_section(lines, "Net Spread Beta", scorecard.get("net_spread_beta_summary_text"), scorecard.get("net_spread_beta_table"))
    _add_section(lines, "CDX Hedge Sizing", scorecard.get("cdx_hedge_summary_text"), scorecard.get("cdx_hedge_table"))
    _add_section(lines, "Risk / Reward", table=scorecard.get("risk_reward_table"))
    _add_section(lines, "Validation", validation.get("summary") if validation.get("available") else validation.get("reason"), validation.get("table"))
    _add_section(lines, "Transition Stability", transitions.get("summary_text") if transitions.get("available") else transitions.get("reason"), transitions.get("transition_outcome_table"))
    _add_section(lines, "False Positives / False Negatives", errors.get("summary_text") if errors.get("available") else errors.get("reason"), errors.get("error_table"))
    _add_section(lines, "Stress Replay", stress_replay.get("summary_text") if stress_replay.get("available") else stress_replay.get("reason"), stress_replay.get("table"))
    _add_section(lines, "Audit Inputs", table=scorecard.get("audit_input_table"))
    _add_section(lines, "Audit Rules", scorecard.get("audit_summary_text"), scorecard.get("audit_table"))

    markdown = "\n".join(lines).strip() + "\n"
    tables = {
        "scorecard_metrics": scorecard.get("table"),
        "trade_memo": scorecard.get("memo_table"),
        "portfolio_expression": scorecard.get("action_table"),
        "rating_bucket_allocation": scorecard.get("rating_bucket_table"),
        "risk_reward": scorecard.get("risk_reward_table"),
        "marginal_allocation": scorecard.get("marginal_allocation_table"),
        "net_spread_beta": scorecard.get("net_spread_beta_table"),
        "cdx_hedge_sizing": scorecard.get("cdx_hedge_table"),
        "validation": validation.get("table") if validation.get("available") else pd.DataFrame(),
        "transition_outcomes": transitions.get("transition_outcome_table") if transitions.get("available") else pd.DataFrame(),
        "prediction_errors": errors.get("error_table") if errors.get("available") else pd.DataFrame(),
        "stress_replay": stress_replay.get("table") if stress_replay.get("available") else pd.DataFrame(),
        "audit_inputs": scorecard.get("audit_input_table"),
        "audit_rules": scorecard.get("audit_table"),
    }

    return {
        "available": True,
        "markdown": markdown,
        "zip": _csv_bytes(tables, markdown),
        "tables": tables,
        "scorecard": scorecard,
        "validation": validation,
        "transitions": transitions,
        "prediction_errors": errors,
        "stress_replay": stress_replay,
        "summary": f"PM packet built for {recommendation} recommendation.",
    }
