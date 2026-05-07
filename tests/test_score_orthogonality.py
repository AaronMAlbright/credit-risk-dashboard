"""
Tests for src/score_orthogonality.py.

Covers correlation matrix, VIF, PCA decomposition, rolling correlations,
and the run_orthogonality_analysis orchestrator.
"""

import numpy as np
import pandas as pd
import pytest

from src.score_orthogonality import (
    COMPOSITE_SCORE_COLS,
    VIF_HIGH,
    VIF_MODERATE,
    compute_correlation_matrix,
    compute_pca_decomposition,
    compute_rolling_correlations,
    compute_vif,
    run_orthogonality_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def independent_df():
    """Six fully independent (orthogonal) score columns — 200 rows."""
    n = 200
    rng = np.random.default_rng(1)
    data = {col: rng.uniform(10, 80, n) for col in COMPOSITE_SCORE_COLS.values()}
    data["date"] = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def correlated_df():
    """Two highly correlated score columns (macro ≈ credit), rest independent."""
    n = 200
    rng = np.random.default_rng(2)
    base = rng.uniform(10, 80, n)
    data = {col: rng.uniform(10, 80, n) for col in COMPOSITE_SCORE_COLS.values()}
    # Make credit_market_risk_score_smooth a near-copy of macro_risk_score_smooth
    data["macro_risk_score_smooth"] = base
    data["credit_market_risk_score_smooth"] = base + rng.normal(0, 1, n)
    data["date"] = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def one_score_df():
    """Only one score present — below threshold for most analyses."""
    n = 50
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "macro_risk_score_smooth": rng.uniform(10, 80, n),
    })


@pytest.fixture(scope="module")
def tiny_df():
    """Fewer rows than the rolling window threshold."""
    n = 10
    rng = np.random.default_rng(4)
    data = {col: rng.uniform(10, 80, n) for col in COMPOSITE_SCORE_COLS.values()}
    data["date"] = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# TestComputeCorrelationMatrix
# ---------------------------------------------------------------------------


class TestComputeCorrelationMatrix:
    def test_returns_dataframe(self, independent_df):
        result = compute_correlation_matrix(independent_df)
        assert isinstance(result, pd.DataFrame)

    def test_square_matrix(self, independent_df):
        result = compute_correlation_matrix(independent_df)
        assert result.shape[0] == result.shape[1]

    def test_diagonal_is_one(self, independent_df):
        result = compute_correlation_matrix(independent_df)
        assert (result.values.diagonal() == 1.0).all()

    def test_symmetric(self, independent_df):
        result = compute_correlation_matrix(independent_df)
        np.testing.assert_array_almost_equal(result.values, result.values.T)

    def test_values_in_valid_range(self, independent_df):
        result = compute_correlation_matrix(independent_df)
        assert (result.values >= -1.0).all()
        assert (result.values <= 1.0).all()

    def test_index_and_columns_are_display_names(self, independent_df):
        result = compute_correlation_matrix(independent_df)
        expected = list(COMPOSITE_SCORE_COLS.keys())
        assert list(result.index) == expected
        assert list(result.columns) == expected

    def test_high_correlation_detected(self, correlated_df):
        result = compute_correlation_matrix(correlated_df)
        corr = result.loc["Macro Risk", "Credit Risk"]
        assert corr > 0.95

    def test_independent_scores_low_correlation(self, independent_df):
        result = compute_correlation_matrix(independent_df)
        mask = np.triu(np.ones(result.shape, dtype=bool), k=1)
        off_diag = result.values[mask]
        assert np.abs(off_diag).max() < 0.30

    def test_one_score_returns_empty(self, one_score_df):
        result = compute_correlation_matrix(one_score_df)
        assert result.empty

    def test_empty_df_returns_empty(self):
        result = compute_correlation_matrix(pd.DataFrame())
        assert result.empty

    def test_rounded_to_3dp(self, independent_df):
        result = compute_correlation_matrix(independent_df)
        for val in result.values.flatten():
            assert round(val, 3) == val


# ---------------------------------------------------------------------------
# TestComputeVIF
# ---------------------------------------------------------------------------


class TestComputeVIF:
    def test_returns_dataframe(self, independent_df):
        result = compute_vif(independent_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, independent_df):
        result = compute_vif(independent_df)
        assert "vif" in result.columns
        assert "flag" in result.columns

    def test_index_is_score_names(self, independent_df):
        result = compute_vif(independent_df)
        for name in COMPOSITE_SCORE_COLS:
            assert name in result.index

    def test_independent_scores_low_vif(self, independent_df):
        result = compute_vif(independent_df)
        assert (result["vif"].dropna() < VIF_MODERATE).all()

    def test_independent_scores_flagged_low(self, independent_df):
        result = compute_vif(independent_df)
        assert (result["flag"] == "Low").all()

    def test_correlated_scores_high_vif(self, correlated_df):
        result = compute_vif(correlated_df)
        macro_vif = result.loc["Macro Risk", "vif"]
        credit_vif = result.loc["Credit Risk", "vif"]
        assert macro_vif > VIF_HIGH or credit_vif > VIF_HIGH

    def test_correlated_scores_flagged_high(self, correlated_df):
        result = compute_vif(correlated_df)
        flags = result["flag"].tolist()
        assert "High" in flags

    def test_vif_positive(self, independent_df):
        result = compute_vif(independent_df)
        assert (result["vif"].dropna() >= 1.0).all()

    def test_one_score_returns_empty(self, one_score_df):
        result = compute_vif(one_score_df)
        assert result.empty

    def test_sorted_descending_by_vif(self, independent_df):
        result = compute_vif(independent_df)
        vifs = result["vif"].dropna().tolist()
        assert vifs == sorted(vifs, reverse=True)


# ---------------------------------------------------------------------------
# TestComputePCADecomposition
# ---------------------------------------------------------------------------


class TestComputePCADecomposition:
    def test_returns_dict(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        for key in ["explained_variance_ratio", "cumulative_variance",
                    "n_components_90pct", "effective_rank", "component_names"]:
            assert key in result

    def test_evr_sums_to_one(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        total = sum(result["explained_variance_ratio"])
        assert abs(total - 1.0) < 1e-6

    def test_evr_values_positive(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        assert all(v >= 0 for v in result["explained_variance_ratio"])

    def test_cumulative_variance_monotone(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        cv = result["cumulative_variance"]
        assert all(cv[i] <= cv[i + 1] for i in range(len(cv) - 1))

    def test_cumulative_variance_ends_at_one(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        assert abs(result["cumulative_variance"][-1] - 1.0) < 1e-6

    def test_n_components_90pct_range(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        n = result["n_components_90pct"]
        assert 1 <= n <= len(COMPOSITE_SCORE_COLS)

    def test_effective_rank_range(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        er = result["effective_rank"]
        assert 1.0 <= er <= len(COMPOSITE_SCORE_COLS)

    def test_independent_scores_high_effective_rank(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        # Truly independent scores → effective rank close to n_scores
        assert result["effective_rank"] > 4.0

    def test_correlated_scores_low_effective_rank(self, correlated_df):
        result = compute_pca_decomposition(correlated_df)
        # One pair nearly identical → fewer effective dimensions
        assert result["effective_rank"] < 5.5

    def test_correlated_scores_fewer_components_for_90pct(self, independent_df, correlated_df):
        ind_result = compute_pca_decomposition(independent_df)
        corr_result = compute_pca_decomposition(correlated_df)
        # Correlated data → first PC captures more variance → fewer components to hit 90%
        assert corr_result["n_components_90pct"] <= ind_result["n_components_90pct"]

    def test_one_score_returns_empty(self, one_score_df):
        result = compute_pca_decomposition(one_score_df)
        assert result == {}

    def test_component_names_match_score_keys(self, independent_df):
        result = compute_pca_decomposition(independent_df)
        assert set(result["component_names"]) == set(COMPOSITE_SCORE_COLS.keys())


# ---------------------------------------------------------------------------
# TestComputeRollingCorrelations
# ---------------------------------------------------------------------------


class TestComputeRollingCorrelations:
    def test_returns_dict(self, independent_df):
        result = compute_rolling_correlations(independent_df, window=30)
        assert isinstance(result, dict)

    def test_keys_are_tuples_of_strings(self, independent_df):
        result = compute_rolling_correlations(independent_df, window=30)
        for key in result:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str) and isinstance(key[1], str)

    def test_returns_at_most_top_n_pairs(self, independent_df):
        result = compute_rolling_correlations(independent_df, window=30, top_n=2)
        assert len(result) <= 2

    def test_values_are_series(self, independent_df):
        result = compute_rolling_correlations(independent_df, window=30)
        for v in result.values():
            assert isinstance(v, pd.Series)

    def test_rolling_values_in_valid_range(self, independent_df):
        result = compute_rolling_correlations(independent_df, window=30)
        for series in result.values():
            assert (series.dropna().abs() <= 1.0).all()

    def test_most_correlated_pair_at_top(self, correlated_df):
        result = compute_rolling_correlations(correlated_df, window=30, top_n=1)
        assert len(result) == 1
        key = list(result.keys())[0]
        assert set(key) == {"Macro Risk", "Credit Risk"}

    def test_too_few_rows_returns_empty(self, tiny_df):
        result = compute_rolling_correlations(tiny_df, window=60)
        assert result == {}

    def test_index_is_datetime_when_date_col_present(self, independent_df):
        result = compute_rolling_correlations(independent_df, window=30)
        for series in result.values():
            assert pd.api.types.is_datetime64_any_dtype(series.index)

    def test_one_score_returns_empty(self, one_score_df):
        result = compute_rolling_correlations(one_score_df)
        assert result == {}


# ---------------------------------------------------------------------------
# TestRunOrthogonalityAnalysis
# ---------------------------------------------------------------------------


class TestRunOrthogonalityAnalysis:
    def test_returns_dict(self, independent_df):
        result = run_orthogonality_analysis(independent_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, independent_df):
        result = run_orthogonality_analysis(independent_df)
        for key in ["correlation_matrix", "vif", "pca",
                    "rolling_correlations", "summary"]:
            assert key in result

    def test_summary_has_all_fields(self, independent_df):
        result = run_orthogonality_analysis(independent_df)
        s = result["summary"]
        for field in ["effective_rank", "n_components_90pct",
                      "max_pairwise_corr", "n_high_vif", "n_moderate_vif"]:
            assert field in s

    def test_independent_df_zero_high_vif(self, independent_df):
        result = run_orthogonality_analysis(independent_df)
        assert result["summary"]["n_high_vif"] == 0

    def test_correlated_df_nonzero_high_vif(self, correlated_df):
        result = run_orthogonality_analysis(correlated_df)
        assert result["summary"]["n_high_vif"] > 0

    def test_max_pairwise_corr_is_float(self, independent_df):
        result = run_orthogonality_analysis(independent_df)
        assert isinstance(result["summary"]["max_pairwise_corr"], float)

    def test_correlated_df_high_max_pairwise(self, correlated_df):
        result = run_orthogonality_analysis(correlated_df)
        assert result["summary"]["max_pairwise_corr"] > 0.90

    def test_empty_df_graceful(self):
        result = run_orthogonality_analysis(pd.DataFrame())
        assert result["correlation_matrix"].empty
        assert result["vif"].empty
        assert result["pca"] == {}
