import pandas as pd

from src.credit_compensation_validation import (
    add_scorecard_recommendations,
    analyze_scorecard_prediction_errors,
    analyze_scorecard_transitions,
    build_scorecard_validation_report,
    validate_scorecard_recommendations,
)


def _validation_df() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=160, freq="B")
    hy = []
    pct = []
    score = []
    sloos = []
    charge = []
    for i in range(160):
        block = i // 40
        if block == 0:
            hy.append(6.0 - i * 0.01)
            pct.append(80)
            score.append(35)
            sloos.append(-3)
            charge.append(0.0)
        elif block == 1:
            hy.append(3.0 + (i - 40) * 0.005)
            pct.append(5)
            score.append(35)
            sloos.append(-1)
            charge.append(0.0)
        elif block == 2:
            hy.append(4.0 + (i - 80) * 0.02)
            pct.append(45)
            score.append(35)
            sloos.append(8)
            charge.append(0.0)
        else:
            hy.append(5.6 + (i - 120) * 0.03)
            pct.append(15)
            score.append(80)
            sloos.append(8)
            charge.append(0.1)

    return pd.DataFrame(
        {
            "hy_spread": hy,
            "ig_spread_bps": [120 + i * 0.1 for i in range(160)],
            "hy_spread_percentile": pct,
            "final_decision": ["Neutral"] * 160,
            "composite_risk_score_smooth": score,
            "sloos_change_90d": sloos,
            "chargeoff_change_90d": charge,
            "delinquency_change_90d": [0.0] * 160,
        },
        index=idx,
    )


def test_add_scorecard_recommendations_adds_expected_labels():
    result = add_scorecard_recommendations(_validation_df())
    labels = set(result["scorecard_recommendation"])
    assert {"Add", "Upgrade Quality", "Hedge", "De-risk"}.issubset(labels)


def test_validate_scorecard_recommendations_shapes_table():
    result = validate_scorecard_recommendations(_validation_df(), horizons=(21, 63))
    assert result["available"] is True
    table = result["table"]
    assert {
        "recommendation",
        "horizon_days",
        "n_obs",
        "confidence",
        "hy_median_change_bps",
        "ig_median_change_bps",
        "favorable_hit_rate_pct",
        "worst_hy_widening_bps",
        "avg_excess_return_proxy_bps",
    }.issubset(table.columns)
    assert set(table["horizon_days"]) == {21, 63}
    assert table["n_obs"].min() > 0
    assert result["summary"]


def test_validate_scorecard_recommendations_favorable_hit_rates():
    table = validate_scorecard_recommendations(_validation_df(), horizons=(21,))["table"]
    add = table[(table["recommendation"] == "Add") & (table["horizon_days"] == 21)].iloc[0]
    hedge = table[(table["recommendation"] == "Hedge") & (table["horizon_days"] == 21)].iloc[0]
    assert add["hy_median_change_bps"] < 0
    assert add["favorable_hit_rate_pct"] > 50
    assert hedge["hy_median_change_bps"] > 0
    assert hedge["favorable_hit_rate_pct"] > 50


def test_validate_scorecard_recommendations_unavailable_without_hy():
    result = validate_scorecard_recommendations(pd.DataFrame({"x": [1, 2, 3]}))
    assert result["available"] is False
    assert "hy_spread" in result["reason"]


def test_build_scorecard_validation_report_exports_markdown_and_csv():
    result = build_scorecard_validation_report(_validation_df(), horizons=(21,))
    assert result["available"] is True
    assert "# Credit Compensation Scorecard Validation" in result["markdown"]
    assert "Validation Table" in result["markdown"]
    assert "recommendation,horizon_days" in result["csv"]
    assert result["current_recommendation"] in result["markdown"]


def test_analyze_scorecard_transitions_shapes_outputs():
    result = analyze_scorecard_transitions(_validation_df(), horizon_days=21)
    assert result["available"] is True
    assert not result["matrix_table"].empty
    assert not result["duration_table"].empty
    assert not result["transition_outcome_table"].empty
    assert {"transition_count", "episode_count", "whipsaw_rate_pct", "most_common_transition"}.issubset(
        result["summary"]
    )
    assert result["summary"]["transition_count"] >= 3
    assert "transitions" in result["summary_text"]


def test_analyze_scorecard_transitions_duration_metrics():
    result = analyze_scorecard_transitions(_validation_df(), horizon_days=21)
    duration = result["duration_table"].set_index("recommendation")
    assert {"Add", "Upgrade Quality", "Hedge", "De-risk"}.issubset(duration.index)
    assert duration.loc["Add", "median_duration_days"] >= 20
    assert duration["episode_count"].sum() == result["summary"]["episode_count"]


def test_analyze_scorecard_transitions_unavailable_without_hy():
    result = analyze_scorecard_transitions(pd.DataFrame({"x": [1, 2, 3]}))
    assert result["available"] is False
    assert "hy_spread" in result["reason"]


def test_analyze_scorecard_prediction_errors_splits_false_positive_and_negative_tables():
    df = _validation_df()
    df.iloc[:50, df.columns.get_loc("hy_spread")] = [6.0 + i * 0.02 for i in range(50)]

    result = analyze_scorecard_prediction_errors(df, horizon_days=21, materiality_bps=25)
    assert result["available"] is True
    assert {"false_positive_count", "false_negative_count", "error_rate_pct", "worst_error"}.issubset(
        result["summary"]
    )
    assert not result["false_positive_table"].empty
    assert set(result["false_positive_table"]["classification"]) == {"false_positive"}
    assert not result["summary_by_recommendation"].empty
    assert "error_rate_pct" in result["summary_by_recommendation"].columns
    assert "false positives" in result["summary_text"]


def test_analyze_scorecard_prediction_errors_identifies_hold_false_negatives():
    df = _validation_df()
    df["hy_spread_percentile"] = 45
    df["composite_risk_score_smooth"] = 35
    df["sloos_change_90d"] = 0
    df["chargeoff_change_90d"] = 0
    df["delinquency_change_90d"] = 0

    result = analyze_scorecard_prediction_errors(df, horizon_days=21, materiality_bps=25)
    false_negatives = result["false_negative_table"]

    assert result["available"] is True
    assert not false_negatives.empty
    assert false_negatives["classification"].str.startswith("false_negative").all()
    assert "Hold" in set(false_negatives["recommendation"])


def test_analyze_scorecard_prediction_errors_unavailable_without_hy():
    result = analyze_scorecard_prediction_errors(pd.DataFrame({"x": [1, 2, 3]}))
    assert result["available"] is False
    assert "hy_spread" in result["reason"]
