"""
Tests for src/report_generator.py.

Covers HTML structure, section presence/absence, helper edge cases,
and graceful handling of None/empty inputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.report_generator import (
    _color_class,
    _fmt,
    _pct,
    _safe_str,
    generate_html_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_df():
    n = 60
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "date":                         dates,
        "final_decision":               ["Neutral"] * 30 + ["Avoid Chasing Risk"] * 30,
        "final_environment":            ["Risk-Off"] * n,
        "final_action":                 ["Reduce Equity"] * n,
        "composite_risk_score_smooth":  rng.uniform(40, 80, n),
        "sp500_daily_return":           rng.normal(0.0003, 0.01, n),
    })


@pytest.fixture(scope="module")
def audit_dict():
    return {
        "summary": {
            "overall_confidence": "Robust",
            "n_robust": 3,
            "n_indicative": 1,
            "n_exploratory": 0,
        },
        "warnings": [],
    }


@pytest.fixture(scope="module")
def audit_with_warnings():
    return {
        "summary": {
            "overall_confidence": "Indicative",
            "n_robust": 1,
            "n_indicative": 2,
            "n_exploratory": 1,
        },
        "warnings": ["Short regime history: Neutral (45 obs)", "Walk-forward window too small"],
    }


@pytest.fixture(scope="module")
def decay_results(base_df):
    idx = pd.MultiIndex.from_tuples(
        [("Avoid Chasing Risk", 7), ("Avoid Chasing Risk", 30), ("Neutral", 7), ("Neutral", 30)],
        names=["regime", "horizon"],
    )
    return {
        "decay_by_regime": pd.DataFrame(
            {
                "mean_return": [-0.005, -0.010, 0.002, 0.004],
                "hit_rate":    [0.40,   0.35,   0.55, 0.60],
                "n_obs":       [30,     30,     30,   30],
                "ci_lower":    [-0.015, -0.020, -0.003, 0.001],
                "ci_upper":    [0.005,  0.002,  0.007, 0.009],
            },
            index=idx,
        )
    }


@pytest.fixture(scope="module")
def tail_risk_dict():
    return {
        "vs_benchmark": {
            "strategy": {"var": -0.012, "cvar": -0.020, "max_loss": -0.045, "ann_vol": 0.12, "n_obs": 60},
            "sp500":    {"var": -0.015, "cvar": -0.025, "max_loss": -0.060, "ann_vol": 0.16, "n_obs": 60},
        },
        "regime_tail_stats": pd.DataFrame(
            {
                "n_obs": [30, 30],
                "var_95": [-0.014, -0.018],
                "cvar_95": [-0.022, -0.030],
                "max_loss": [-0.040, -0.060],
            },
            index=["Neutral", "Avoid Chasing Risk"],
        ),
    }


@pytest.fixture(scope="module")
def weight_opt_dict():
    return {
        "current_sharpe": 0.85,
        "optimal_sharpe": 1.10,
        "optimal_weights": {
            "macro_risk": 0.30,
            "credit_risk": 0.20,
            "complacency": 0.20,
            "liquidity": 0.15,
            "treasury": 0.10,
            "mean_reversion": 0.05,
        },
        "grid_results": pd.DataFrame(),
        "tornado": pd.DataFrame(),
    }


@pytest.fixture(scope="module")
def attr_dict():
    return {
        "top_drivers_current": [
            {"name": "Credit Risk", "level": 72.3, "excess": 14.5, "elevated": True},
            {"name": "Complacency", "level": 65.1, "excess": 8.2, "elevated": True},
            {"name": "Macro Risk",  "level": 55.0, "excess": -2.1, "elevated": False},
        ]
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_pct_positive(self):
        assert _pct(0.05) == "5.00%"

    def test_pct_negative(self):
        assert _pct(-0.025) == "-2.50%"

    def test_pct_nan_returns_dash(self):
        assert _pct(float("nan")) == "—"

    def test_pct_none_returns_dash(self):
        assert _pct(None) == "—"

    def test_fmt_positive(self):
        assert _fmt(72.3) == "72.3"

    def test_fmt_none_returns_dash(self):
        assert _fmt(None) == "—"

    def test_fmt_nan_returns_dash(self):
        assert _fmt(float("nan")) == "—"

    def test_color_class_positive(self):
        assert _color_class(0.01) == "pos"

    def test_color_class_negative(self):
        assert _color_class(-0.01) == "neg"

    def test_color_class_zero(self):
        assert _color_class(0.0) == ""

    def test_color_class_none(self):
        assert _color_class(None) == ""

    def test_safe_str_normal(self):
        assert _safe_str("Neutral") == "Neutral"

    def test_safe_str_none_returns_fallback(self):
        assert _safe_str(None) == "N/A"

    def test_safe_str_custom_fallback(self):
        assert _safe_str(None, fallback="Unknown") == "Unknown"

    def test_safe_str_nan_returns_fallback(self):
        assert _safe_str(float("nan")) == "N/A"


# ---------------------------------------------------------------------------
# HTML structure
# ---------------------------------------------------------------------------


class TestHtmlStructure:
    def test_returns_string(self, base_df):
        html = generate_html_report(base_df)
        assert isinstance(html, str)

    def test_has_doctype(self, base_df):
        html = generate_html_report(base_df)
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_has_html_and_body_tags(self, base_df):
        html = generate_html_report(base_df)
        assert "<html" in html
        assert "<body" in html
        assert "</html>" in html

    def test_has_style_tag(self, base_df):
        html = generate_html_report(base_df)
        assert "<style>" in html

    def test_title_contains_date(self, base_df):
        html = generate_html_report(base_df)
        assert "Signal Report" in html

    def test_no_external_dependencies(self, base_df):
        html = generate_html_report(base_df)
        assert "https://" not in html
        assert "http://" not in html
        assert "cdn" not in html.lower()

    def test_footer_present(self, base_df):
        html = generate_html_report(base_df)
        assert "research purposes only" in html.lower()

    def test_empty_df_does_not_crash(self):
        html = generate_html_report(pd.DataFrame())
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_all_none_args_does_not_crash(self, base_df):
        html = generate_html_report(base_df, audit=None, decay_results=None,
                                    tail_risk=None, weight_opt=None, attr=None)
        assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# Signal section
# ---------------------------------------------------------------------------


class TestSignalSection:
    def test_decision_name_in_html(self, base_df):
        html = generate_html_report(base_df)
        assert "Avoid Chasing Risk" in html

    def test_known_decision_gets_badge_color(self, base_df):
        html = generate_html_report(base_df)
        assert "#e74c3c" in html  # Avoid Chasing Risk color

    def test_composite_score_present(self, base_df):
        html = generate_html_report(base_df)
        assert "/ 100" in html


# ---------------------------------------------------------------------------
# Confidence section
# ---------------------------------------------------------------------------


class TestConfidenceSection:
    def test_confidence_section_absent_when_no_audit(self, base_df):
        html = generate_html_report(base_df, audit=None)
        assert "Statistical Confidence" not in html

    def test_confidence_section_present_with_audit(self, base_df, audit_dict):
        html = generate_html_report(base_df, audit=audit_dict)
        assert "Statistical Confidence" in html
        assert "Robust" in html

    def test_warnings_shown(self, base_df, audit_with_warnings):
        html = generate_html_report(base_df, audit=audit_with_warnings)
        assert "Short regime history" in html

    def test_no_warnings_shows_ok_message(self, base_df, audit_dict):
        html = generate_html_report(base_df, audit=audit_dict)
        assert "No confidence warnings" in html

    def test_regime_counts_in_html(self, base_df, audit_dict):
        html = generate_html_report(base_df, audit=audit_dict)
        assert "Robust: <b>3</b>" in html


# ---------------------------------------------------------------------------
# Drivers section
# ---------------------------------------------------------------------------


class TestDriversSection:
    def test_drivers_absent_when_no_attr(self, base_df):
        html = generate_html_report(base_df, attr=None)
        assert "Top Drivers" not in html

    def test_drivers_present_with_attr(self, base_df, attr_dict):
        html = generate_html_report(base_df, attr=attr_dict)
        assert "Top Drivers" in html
        assert "Credit Risk" in html

    def test_elevated_label_present(self, base_df, attr_dict):
        html = generate_html_report(base_df, attr=attr_dict)
        assert "Elevated" in html

    def test_at_most_three_drivers(self, base_df):
        attr = {
            "top_drivers_current": [
                {"name": f"Driver {i}", "level": 50.0, "excess": 5.0, "elevated": True}
                for i in range(6)
            ]
        }
        html = generate_html_report(base_df, attr=attr)
        # Only first 3 should appear
        assert "Driver 0" in html
        assert "Driver 3" not in html


# ---------------------------------------------------------------------------
# Signal decay section
# ---------------------------------------------------------------------------


class TestDecaySection:
    def test_decay_absent_when_none(self, base_df):
        html = generate_html_report(base_df, decay_results=None)
        assert "Signal Decay" not in html

    def test_decay_present_for_current_regime(self, base_df, decay_results):
        html = generate_html_report(base_df, decay_results=decay_results)
        # Latest row is "Avoid Chasing Risk"
        assert "Signal Decay" in html
        assert "Avoid Chasing Risk" in html

    def test_decay_section_absent_when_regime_not_in_index(self, base_df):
        decay = {
            "decay_by_regime": pd.DataFrame(
                {"mean_return": [0.001], "hit_rate": [0.5], "n_obs": [10],
                 "ci_lower": [-0.005], "ci_upper": [0.007]},
                index=pd.MultiIndex.from_tuples([("Unknown Regime", 30)],
                                                names=["regime", "horizon"]),
            )
        }
        html = generate_html_report(base_df, decay_results=decay)
        assert "Signal Decay" not in html


# ---------------------------------------------------------------------------
# Tail risk section
# ---------------------------------------------------------------------------


class TestTailSection:
    def test_tail_section_absent_when_none(self, base_df):
        html = generate_html_report(base_df, tail_risk=None)
        assert "Tail Risk" not in html

    def test_tail_section_present(self, base_df, tail_risk_dict):
        html = generate_html_report(base_df, tail_risk=tail_risk_dict)
        assert "Tail Risk" in html
        assert "CVaR" in html

    def test_strategy_and_sp500_labels(self, base_df, tail_risk_dict):
        html = generate_html_report(base_df, tail_risk=tail_risk_dict)
        assert "Strategy" in html
        assert "SP500 Buy" in html


# ---------------------------------------------------------------------------
# Weights section
# ---------------------------------------------------------------------------


class TestWeightsSection:
    def test_weights_absent_when_none(self, base_df):
        html = generate_html_report(base_df, weight_opt=None)
        assert "Composite Weight" not in html

    def test_weights_present(self, base_df, weight_opt_dict):
        html = generate_html_report(base_df, weight_opt=weight_opt_dict)
        assert "Composite Weight" in html
        assert "Macro Risk" in html

    def test_sharpe_values_shown(self, base_df, weight_opt_dict):
        html = generate_html_report(base_df, weight_opt=weight_opt_dict)
        assert "0.85" in html
        assert "1.10" in html

    def test_optimal_column_present(self, base_df, weight_opt_dict):
        html = generate_html_report(base_df, weight_opt=weight_opt_dict)
        assert "Optimal" in html
