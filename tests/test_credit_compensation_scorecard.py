import pandas as pd

from src.credit_compensation_scorecard import build_credit_compensation_scorecard


def _df(**overrides):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    data = {
        "hy_spread": [3.0, 3.2, 3.4, 6.0],
        "ig_spread_bps": [120, 125, 130, 150],
        "hy_yield": [7.0, 7.1, 7.2, 8.0],
        "ig_yield": [5.0, 5.1, 5.2, 5.4],
        "hy_spread_percentile": [30, 35, 40, 80],
        "final_decision": ["Neutral", "Neutral", "Neutral", "Neutral"],
        "composite_risk_score_smooth": [30, 32, 34, 35],
        "default_cycle_pct": [4.5, 4.7, 4.9, 5.1],
        "actual_chargeoff_pct": [1.0, 1.0, 1.0, 1.0],
        "actual_delinq_pct": [1.2, 1.2, 1.2, 1.2],
        "implied_vs_actual_gap": [3.5, 3.7, 3.9, 4.1],
        "sloos_change_90d": [-5.0, -4.0, -3.0, -2.0],
        "chargeoff_change_90d": [0.0, 0.0, 0.0, 0.0],
        "delinquency_change_90d": [0.0, 0.0, 0.0, 0.0],
    }
    data.update(overrides)
    return pd.DataFrame(data, index=idx)


def _forward_df(**overrides):
    periods = 14
    idx = pd.date_range("2024-01-01", periods=periods, freq="B")
    data = {
        "hy_spread": [5.6, 5.8, 6.0, 6.2, 5.9, 6.1, 5.7, 6.3, 5.8, 6.0, 5.9, 6.1, 6.0, 6.0],
        "ig_spread_bps": [130, 132, 135, 136, 134, 137, 133, 138, 134, 136, 135, 137, 136, 136],
        "hy_yield": [8.0] * periods,
        "ig_yield": [5.2] * periods,
        "hy_spread_percentile": [70, 75, 80, 82, 78, 79, 74, 83, 76, 81, 77, 80, 79, 80],
        "ig_spread_percentile": [45, 48, 50, 52, 49, 51, 46, 53, 49, 52, 48, 51, 50, 50],
        "final_decision": ["Neutral"] * periods,
        "composite_risk_score_smooth": [35] * periods,
        "default_cycle_pct": [5.0] * periods,
        "actual_chargeoff_pct": [1.0] * periods,
        "actual_delinq_pct": [1.2] * periods,
        "implied_vs_actual_gap": [4.0] * periods,
        "sloos_change_90d": [-2.0] * periods,
        "chargeoff_change_90d": [0.0] * periods,
        "delinquency_change_90d": [0.0] * periods,
        "hy_spread_forward_21d_change": [-30, -25, -20, -15, -35, -10, -40, -5, -22, -18, -28, -16, -12, None],
        "hy_spread_forward_63d_change": [-45, -35, -25, -20, -50, -18, -55, -12, -30, -24, -38, -22, -16, None],
        "hy_spread_forward_126d_change": [-60, -45, -30, -25, -70, -20, -75, -15, -35, -28, -42, -26, -20, None],
        "ig_spread_forward_21d_change": [-5, -4, -3, -2, -6, -2, -7, -1, -4, -3, -5, -2, -2, None],
        "ig_spread_forward_63d_change": [-8, -6, -4, -3, -9, -3, -10, -2, -5, -4, -7, -3, -3, None],
        "ig_spread_forward_126d_change": [-10, -8, -5, -4, -12, -4, -14, -3, -6, -5, -9, -4, -4, None],
    }
    data.update(overrides)
    return pd.DataFrame(data, index=idx)


def test_scorecard_available_and_table_shaped():
    result = build_credit_compensation_scorecard(_df())
    assert result["available"] is True
    assert result["recommendation"] in {"Add", "Hold", "Upgrade Quality", "Hedge", "De-risk"}
    assert {"metric", "value", "interpretation"}.issubset(result["table"].columns)
    assert {"decision", "guidance", "why it matters"}.issubset(result["action_table"].columns)
    assert {"section", "view"}.issubset(result["memo_table"].columns)
    assert {"trigger", "condition", "portfolio action"}.issubset(result["trigger_table"].columns)
    assert {"bucket", "target_weight", "tilt", "rationale"}.issubset(result["rating_bucket_table"].columns)
    assert {
        "bucket",
        "target_weight",
        "spread_carry_bps",
        "expected_default_drag_bps",
        "expected_excess_return_bps",
        "recession_stress_loss_bps",
        "weighted_expected_return_bps",
        "weighted_stress_loss_bps",
    }.issubset(result["bucket_return_table"].columns)
    assert {"metric", "value", "interpretation"}.issubset(result["risk_reward_table"].columns)
    assert {
        "action",
        "bucket",
        "suggested_shift",
        "funding_source",
        "reason",
        "expected_return_impact_bps",
        "stress_impact_bps",
    }.issubset(result["marginal_allocation_table"].columns)
    assert {
        "bucket",
        "target_weight",
        "+25bps_loss_bps",
        "+50bps_loss_bps",
        "+100bps_loss_bps",
        "weighted_+25bps_bps",
        "weighted_+50bps_bps",
        "weighted_+100bps_bps",
    }.issubset(result["spread_shock_table"].columns)
    assert {
        "bucket",
        "target_weight",
        "spread_beta",
        "spread_duration",
        "weighted_spread_beta",
        "weighted_duration_beta",
    }.issubset(result["net_spread_beta_table"].columns)
    assert {"metric", "value", "unit"}.issubset(result["cdx_hedge_table"].columns)
    assert {
        "scenario",
        "hy_shock_bps",
        "default_loss_multiplier",
        "modeled_portfolio_loss_bps",
        "hedge_offset_bps",
        "incremental_default_drag_bps",
        "primary_driver",
        "portfolio_action",
    }.issubset(result["scenario_preset_table"].columns)
    assert {
        "constraint",
        "current_pct",
        "limit_pct",
        "status",
        "gap_pct",
        "portfolio_action",
    }.issubset(result["constraint_table"].columns)
    assert {"metric", "current", "prior", "change", "unit", "interpretation"}.issubset(
        result["pm_attribution_table"].columns
    )
    assert {
        "bucket",
        "spread_source",
        "spread_carry_factor",
        "spread_beta",
        "loss_factor",
        "spread_duration",
        "recession_widening_bps",
    }.issubset(result["bucket_assumptions_table"].columns)
    assert {"output", "confidence", "basis"}.issubset(result["confidence_table"].columns)
    assert {"as_of", "rule", "observed", "threshold", "fired", "decision_impact"}.issubset(
        result["audit_table"].columns
    )
    assert {"input", "value", "unit"}.issubset(result["audit_input_table"].columns)
    assert set(result["rating_weights"]) == {"IG", "BBB", "BB", "B", "CCC", "Cash", "Hedge"}
    assert round(sum(result["rating_weights"].values()), 1) == 100.0
    assert result["pm_final_verdict"]
    assert result["pm_attribution_summary_text"]


def test_scorecard_adds_when_compensation_is_high_and_conditions_stable():
    result = build_credit_compensation_scorecard(_df())
    assert result["recommendation"] == "Add"
    assert result["allocation"]["credit_beta"] == "Overweight credit beta selectively"
    assert result["memo"]["Trade Expression"].startswith("Favor BB/B carry")
    assert result["rating_weights"]["BB"] > result["rating_weights"]["IG"]


def test_scorecard_upgrades_quality_when_spreads_are_very_rich():
    result = build_credit_compensation_scorecard(_df(
        hy_spread=[2.5, 2.4, 2.3, 2.2],
        hy_spread_percentile=[10, 9, 8, 5],
    ))
    assert result["recommendation"] == "Upgrade Quality"


def test_scorecard_derisks_when_rich_and_fundamentals_worsen():
    result = build_credit_compensation_scorecard(_df(
        hy_spread=[2.5, 2.4, 2.3, 2.2],
        hy_spread_percentile=[10, 9, 8, 5],
        sloos_change_90d=[4, 5, 6, 7],
        chargeoff_change_90d=[0.0, 0.1, 0.1, 0.1],
    ))
    assert result["recommendation"] == "De-risk"


def test_scorecard_defensive_composite_derisks():
    result = build_credit_compensation_scorecard(_df(
        composite_risk_score_smooth=[60, 65, 72, 80],
    ))
    assert result["recommendation"] == "De-risk"


def test_scorecard_unavailable_without_spreads():
    result = build_credit_compensation_scorecard(pd.DataFrame(index=pd.date_range("2024-01-01", periods=3)))
    assert result["available"] is False


def test_scorecard_uses_raw_observed_default_columns():
    result = build_credit_compensation_scorecard(_df(
        actual_chargeoff_pct=[None, None, None, None],
        actual_delinq_pct=[None, None, None, None],
        business_chargeoff_rate=[1.0, 1.1, 1.2, 1.3],
        ci_loan_delinquency=[1.4, 1.5, 1.6, 1.7],
    ))
    current = result["current"]
    assert current["actual_chargeoff_pct"] == 1.3
    assert current["business_loan_delinquency_pct"] == 1.7
    assert current["implied_default_pct"] is not None


def test_scorecard_adds_portfolio_guidance_for_quality_pressure():
    result = build_credit_compensation_scorecard(_df(
        hy_spread_percentile=[60, 65, 70, 75],
        ig_spread_percentile=[30, 35, 40, 45],
        hy_ig_ratio=[3.5, 3.7, 3.9, 4.1],
        bbb_ig_ratio=[1.1, 1.2, 1.3, 1.5],
    ))
    assert result["current"]["hy_ig_spread_ratio"] is not None
    assert "BBB pressure" in result["allocation"]["quality_bias"]
    assert "HY screens cheaper" in result["allocation"]["hy_ig_tilt"]
    assert "Watch level" in set(result["action_table"]["decision"])


def test_scorecard_adds_trade_memo_and_mind_change_triggers():
    result = build_credit_compensation_scorecard(_df(
        hy_spread=[2.5, 2.4, 2.3, 2.2],
        hy_spread_percentile=[10, 9, 8, 5],
    ))
    assert set(result["memo"]) == {
        "Current View",
        "What Changed",
        "Why It Matters",
        "Trade Expression",
        "Invalidation Level",
    }
    assert "Wait for HY OAS" in result["triggers"]["add_risk"]
    assert "Add risk" in set(result["trigger_table"]["trigger"])
    assert result["memo"]["Invalidation Level"] == result["triggers"]["add_risk"]


def test_scorecard_rating_allocation_upgrades_quality_when_hy_is_rich():
    result = build_credit_compensation_scorecard(_df(
        hy_spread=[2.5, 2.4, 2.3, 2.2],
        hy_spread_percentile=[10, 9, 8, 5],
    ))
    weights = result["rating_weights"]
    assert weights["IG"] > weights["BB"]
    assert weights["CCC"] == 0.0
    assert weights["Cash"] >= 5.0
    assert "Rating allocation" in result["memo"]["Trade Expression"]


def test_scorecard_rating_allocation_uses_cash_and_hedge_when_derisking():
    result = build_credit_compensation_scorecard(_df(
        composite_risk_score_smooth=[60, 65, 72, 80],
        sloos_change_90d=[4, 5, 6, 7],
        chargeoff_change_90d=[0.0, 0.1, 0.1, 0.1],
    ))
    weights = result["rating_weights"]
    assert result["recommendation"] == "De-risk"
    assert weights["Cash"] >= weights["IG"]
    assert weights["Hedge"] >= 15.0
    assert weights["B"] == 0.0


def test_scorecard_adds_historical_forward_outcomes_when_forward_columns_exist():
    result = build_credit_compensation_scorecard(_forward_df())
    outcomes = result["forward_outcomes"]
    assert outcomes["available"] is True
    assert outcomes["sample_count"] >= 8
    assert {"horizon", "sample_count", "hy_median_change_bps", "ig_median_change_bps"}.issubset(
        result["forward_outcomes_table"].columns
    )
    assert result["forward_outcomes_table"]["hy_median_change_bps"].iloc[0] < 0
    assert "HY spreads typically tightened" in result["forward_outcomes_summary"]


def test_scorecard_adds_bucket_expected_return_and_stress_loss():
    result = build_credit_compensation_scorecard(_df())
    table = result["bucket_return_table"].set_index("bucket")
    summary = result["bucket_return_summary"]
    assert table.loc["BB", "expected_excess_return_bps"] > table.loc["IG", "expected_excess_return_bps"]
    assert table.loc["CCC", "recession_stress_loss_bps"] > table.loc["BB", "recession_stress_loss_bps"]
    assert table.loc["Hedge", "recession_stress_loss_bps"] < 0
    assert summary["portfolio_expected_excess_return_bps"] > 0
    assert summary["portfolio_recession_stress_loss_bps"] > 0
    assert summary["expected_hy_spread_change_source"] == "Rules"
    assert "expected excess return" in result["bucket_return_summary_text"]


def test_scorecard_bucket_returns_penalize_rich_tightening_credit():
    result = build_credit_compensation_scorecard(_df(
        hy_spread=[2.5, 2.4, 2.3, 2.2],
        ig_spread_bps=[80, 78, 76, 75],
        hy_spread_percentile=[10, 9, 8, 5],
        sloos_change_90d=[4, 5, 6, 7],
        chargeoff_change_90d=[0.0, 0.1, 0.1, 0.1],
    ))
    table = result["bucket_return_table"].set_index("bucket")
    assert result["recommendation"] == "De-risk"
    assert result["bucket_return_summary"]["expected_hy_spread_change_bps"] > 0
    assert table.loc["B", "expected_excess_return_bps"] < table.loc["IG", "expected_excess_return_bps"]


def test_scorecard_surfaces_risk_reward_and_assumptions():
    result = build_credit_compensation_scorecard(_df())
    metrics = result["risk_reward_summary"]
    assumptions = result["bucket_assumptions_table"].set_index("bucket")
    table_metrics = set(result["risk_reward_table"]["metric"])
    assert "Expected return / stress loss" in table_metrics
    assert "B/CCC tail weight" in table_metrics
    assert metrics["risk_reward_ratio"] > 0
    assert metrics["tail_weight_pct"] == result["rating_weights"]["B"] + result["rating_weights"]["CCC"]
    assert metrics["hedge_stress_offset_pct"] > 0
    assert assumptions.loc["CCC", "spread_beta"] > assumptions.loc["BB", "spread_beta"]
    assert assumptions.loc["Hedge", "recession_widening_bps"] < 0


def test_scorecard_blends_historical_analogs_into_bucket_return_model():
    result = build_credit_compensation_scorecard(_forward_df())
    summary = result["bucket_return_summary"]
    table = result["bucket_return_table"].set_index("bucket")
    assert summary["expected_hy_spread_change_source"] == "Blended historical analogs + rules"
    assert summary["rule_hy_spread_change_bps"] < 0
    assert summary["historical_hy_spread_change_bps"] < 0
    assert summary["expected_hy_spread_change_bps"] < 0
    assert "blended historical analogs" in result["bucket_return_summary_text"]
    assert table.loc["BB", "expected_spread_mtm_bps"] > 0
    conf = result["confidence_table"].set_index("output")
    assert conf.loc["Historical analogs", "confidence"] == "Thin sample"
    assert conf.loc["Expected return model", "confidence"] == "Thin sample"


def test_scorecard_confidence_flags_rules_only_without_analogs():
    result = build_credit_compensation_scorecard(_df())
    conf = result["confidence_table"].set_index("output")
    assert conf.loc["Historical analogs", "confidence"] == "Rules-only"
    assert conf.loc["Expected return model", "confidence"] == "Rules-only"
    assert result["confidence_summary"]["overall_confidence"] == "Rules-only"


def test_scorecard_adds_marginal_allocation_advice():
    result = build_credit_compensation_scorecard(_df())
    advice = result["marginal_allocation_table"]
    primary = advice.iloc[0]
    assert primary["action"] == "Add"
    assert primary["suggested_shift"] == "+5%"
    assert primary["bucket"] in {"IG", "BBB", "BB", "B", "CCC", "Cash", "Hedge"}
    assert primary["funding_source"] != primary["bucket"]
    assert primary["expected_return_impact_bps"] != 0
    assert "stress-adjusted compensation" in primary["reason"]


def test_scorecard_marginal_advice_flags_tail_beta_when_stress_share_high():
    result = build_credit_compensation_scorecard(_df())
    advice = result["marginal_allocation_table"]
    tail_trims = advice[
        (advice["action"] == "Trim")
        & (advice["bucket"].isin(["B", "CCC"]))
        & (advice["funding_source"] == "Hedge")
    ]
    assert result["risk_reward_summary"]["tail_stress_share_pct"] >= 35.0
    assert not tail_trims.empty
    assert tail_trims.iloc[0]["stress_impact_bps"] < 0


def test_scorecard_adds_spread_shock_sensitivity():
    result = build_credit_compensation_scorecard(_df())
    shocks = result["spread_shock_table"].set_index("bucket")
    summary = result["spread_shock_summary"]
    assert shocks.loc["BB", "+100bps_loss_bps"] > shocks.loc["IG", "+100bps_loss_bps"]
    assert shocks.loc["Hedge", "+100bps_loss_bps"] < 0
    assert shocks.loc["Portfolio", "weighted_+100bps_bps"] == summary["portfolio_+100bps_loss_bps"]
    assert summary["portfolio_+100bps_loss_bps"] > 0
    assert summary["hedge_+100bps_offset_bps"] > 0
    assert "+100 bps spread shock" in summary["interpretation"]


def test_scorecard_adds_net_spread_beta():
    result = build_credit_compensation_scorecard(_df())
    table = result["net_spread_beta_table"].set_index("bucket")
    summary = result["net_spread_beta_summary"]

    assert table.loc["Hedge", "weighted_spread_beta"] < 0
    assert table.loc["Portfolio", "weighted_spread_beta"] == summary["net_spread_beta"]
    assert summary["gross_long_spread_beta"] > summary["net_spread_beta"]
    assert summary["hedge_offset_pct"] > 0
    assert summary["parallel_100bps_loss_bps"] == result["spread_shock_summary"]["portfolio_+100bps_loss_bps"]
    assert "Net spread beta" in result["net_spread_beta_summary_text"]


def test_scorecard_adds_cdx_hedge_sizing():
    result = build_credit_compensation_scorecard(_df(
        composite_risk_score_smooth=[60, 65, 72, 80],
        sloos_change_90d=[4, 5, 6, 7],
        chargeoff_change_90d=[0.0, 0.1, 0.1, 0.1],
    ))
    summary = result["cdx_hedge_summary"]
    table = result["cdx_hedge_table"].set_index("metric")

    assert result["recommendation"] == "De-risk"
    assert summary["target_net_spread_beta"] == 0.05
    assert summary["incremental_cdx_hy_protection_pct"] >= 0
    assert summary["post_trade_net_spread_beta"] <= result["net_spread_beta_summary"]["net_spread_beta"]
    assert table.loc["Incremental CDX HY protection", "unit"] == "% NAV"
    assert "CDX HY protection" in result["cdx_hedge_summary_text"]


def test_scorecard_adds_audit_trail():
    result = build_credit_compensation_scorecard(_df())
    audit = result["audit_table"].set_index("rule")
    inputs = result["audit_input_table"].set_index("input")

    assert result["audit_summary"]["recommendation"] == result["recommendation"]
    assert result["audit_summary"]["fired_rule_count"] == int(audit["fired"].sum())
    assert bool(audit.loc["Cheap spread", "fired"]) is True
    assert bool(audit.loc["Add compensation gate", "fired"]) is True
    assert inputs.loc["Compensation ratio", "value"] == round(result["current"]["spread_compensation_ratio"], 2)
    assert "Audit trail" in result["audit_summary_text"]


def test_scorecard_adds_pm_final_verdict():
    result = build_credit_compensation_scorecard(_df())
    verdict = result["pm_final_verdict"]
    primary = result["marginal_allocation_table"].iloc[0]
    assert result["recommendation"] in verdict
    assert str(primary["bucket"]) in verdict
    assert "expected excess return" in verdict
    assert "+100 bps spread shock" in verdict
    assert "key risk trigger" in verdict


def test_scorecard_adds_pm_attribution_view():
    result = build_credit_compensation_scorecard(_df())
    table = result["pm_attribution_table"].set_index("metric")
    summary = result["pm_attribution_summary"]

    assert summary["available"] is True
    assert summary["current_recommendation"] == result["recommendation"]
    assert table.loc["Recommendation", "current"] == result["recommendation"]
    assert table.loc["HY OAS", "change"] == "+300.0"
    assert table.loc["Excess spread", "change"] == "+300.0"
    assert "recommendation" in summary["interpretation"]


def test_scorecard_pm_attribution_uses_date_column_for_as_of():
    df = _df()
    df = df.reset_index().rename(columns={"index": "date"})

    result = build_credit_compensation_scorecard(df)
    summary = result["pm_attribution_summary"]

    assert summary["current_as_of"] == "2024-01-04"
    assert summary["prior_as_of"] == "2024-01-01"


def test_scorecard_adds_scenario_preset_stress():
    result = build_credit_compensation_scorecard(_df())
    table = result["scenario_preset_table"].set_index("scenario")
    summary = result["scenario_preset_summary"]
    assert {"Soft landing", "Recession widening", "Liquidity shock", "Fallen-angel wave", "Default-cycle shock"}.issubset(table.index)
    assert table.loc["Soft landing", "modeled_portfolio_loss_bps"] < table.loc["Recession widening", "modeled_portfolio_loss_bps"]
    assert table.loc["Default-cycle shock", "incremental_default_drag_bps"] > table.loc["Liquidity shock", "incremental_default_drag_bps"]
    assert summary["worst_scenario"] in set(table.index)
    assert "Worst preset" in result["scenario_preset_summary_text"]


def test_scorecard_adds_portfolio_constraints():
    result = build_credit_compensation_scorecard(_df())
    table = result["constraint_table"].set_index("constraint")
    assert {"Max CCC", "Max B/CCC tail", "Min cash", "Min hedge", "Max HY beta"}.issubset(table.index)
    assert set(table["status"]).issubset({"OK", "Breach"})
    assert result["constraint_summary"]["breach_count"] >= 0
    assert result["constraint_summary_text"]


def test_scorecard_constraints_flag_cash_or_hedge_breach_when_tightening():
    result = build_credit_compensation_scorecard(_df(
        composite_risk_score_smooth=[60, 65, 72, 80],
        sloos_change_90d=[4, 5, 6, 7],
        chargeoff_change_90d=[0.0, 0.1, 0.1, 0.1],
    ))
    table = result["constraint_table"].set_index("constraint")
    assert result["recommendation"] == "De-risk"
    assert table.loc["Min cash", "current_pct"] >= table.loc["Min cash", "limit_pct"]
    assert table.loc["Min hedge", "current_pct"] >= table.loc["Min hedge", "limit_pct"]
