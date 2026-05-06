import numpy as np
import pandas as pd
import pytest

from src.regime_transition import (
    _regime_runs,
    compute_regime_durations,
    compute_regime_forward_returns,
    compute_transition_forward_returns,
    compute_transition_matrix,
    run_regime_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_series():
    return pd.Series(["A", "A", "B", "B", "B", "A", "C", "C"])


@pytest.fixture(scope="module")
def simple_df(simple_series):
    df = pd.DataFrame({"regime": simple_series})
    df["sp500_forward_30d_return"] = [0.01, 0.02, -0.03, -0.01, 0.00, 0.04, -0.05, 0.03]
    df["sp500_forward_60d_return"] = [0.02, 0.03, -0.04, -0.02, 0.01, 0.05, -0.06, 0.04]
    df["strategy_forward_30d_return"] = [0.01, 0.02, -0.02, -0.01, 0.00, 0.03, -0.04, 0.02]
    df["sp500_future_drawdown_30d"] = [-0.01, -0.01, -0.05, -0.04, -0.02, -0.01, -0.07, -0.02]
    return df


@pytest.fixture(scope="module")
def analytics_df(analytics_result):
    df, _, _ = analytics_result
    return df


# ---------------------------------------------------------------------------
# TestRegimeRuns
# ---------------------------------------------------------------------------


class TestRegimeRuns:
    def test_empty_series(self):
        result = _regime_runs(pd.Series([], dtype=str))
        assert result.empty
        assert list(result.columns) == ["regime", "start", "end", "length"]

    def test_single_element(self):
        result = _regime_runs(pd.Series(["A"]))
        assert len(result) == 1
        assert result.iloc[0]["regime"] == "A"
        assert result.iloc[0]["length"] == 1

    def test_all_same(self):
        result = _regime_runs(pd.Series(["X"] * 10))
        assert len(result) == 1
        assert result.iloc[0]["length"] == 10

    def test_alternating(self):
        result = _regime_runs(pd.Series(["A", "B"] * 4))
        assert len(result) == 8
        assert all(result["length"] == 1)

    def test_run_lengths(self, simple_series):
        result = _regime_runs(simple_series)
        assert list(result["length"]) == [2, 3, 1, 2]

    def test_run_regimes(self, simple_series):
        result = _regime_runs(simple_series)
        assert list(result["regime"]) == ["A", "B", "A", "C"]

    def test_start_end_indices(self, simple_series):
        result = _regime_runs(simple_series)
        assert result.iloc[0]["start"] == 0
        assert result.iloc[0]["end"] == 1
        assert result.iloc[1]["start"] == 2
        assert result.iloc[1]["end"] == 4


# ---------------------------------------------------------------------------
# TestComputeTransitionMatrix
# ---------------------------------------------------------------------------


class TestComputeTransitionMatrix:
    def test_returns_tuple(self, simple_series):
        result = compute_transition_matrix(simple_series)
        assert isinstance(result, tuple) and len(result) == 2

    def test_counts_dtype(self, simple_series):
        counts, _ = compute_transition_matrix(simple_series)
        assert counts.dtypes.unique()[0] == int

    def test_row_sums_to_one(self, simple_series):
        _, probs = compute_transition_matrix(simple_series)
        row_sums = probs.sum(axis=1)
        assert (abs(row_sums - 1.0) < 1e-9).all()

    def test_correct_count(self, simple_series):
        # A→B once, B→A once, A→C once, B→B twice
        counts, _ = compute_transition_matrix(simple_series)
        assert counts.loc["A", "B"] == 1
        assert counts.loc["B", "A"] == 1
        assert counts.loc["A", "C"] == 1
        assert counts.loc["B", "B"] == 2

    def test_self_persistence(self):
        # B appears three times in a row: B→B = 2
        s = pd.Series(["A", "B", "B", "B", "A"])
        counts, _ = compute_transition_matrix(s)
        assert counts.loc["B", "B"] == 2

    def test_ordered_filters_absent(self):
        s = pd.Series(["A", "B", "A"])
        counts, _ = compute_transition_matrix(s, ordered=["A", "B", "C", "D"])
        # C and D never appear, so they should be absent
        assert "C" not in counts.index
        assert "D" not in counts.columns

    def test_single_state(self):
        s = pd.Series(["A", "A", "A"])
        counts, probs = compute_transition_matrix(s)
        assert counts.loc["A", "A"] == 2
        assert probs.loc["A", "A"] == 1.0


# ---------------------------------------------------------------------------
# TestComputeRegimeDurations
# ---------------------------------------------------------------------------


class TestComputeRegimeDurations:
    def test_returns_dataframe(self, simple_series):
        result = compute_regime_durations(simple_series)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, simple_series):
        result = compute_regime_durations(simple_series)
        expected = {"count", "mean_days", "median_days", "std_days", "min_days", "max_days"}
        assert expected.issubset(set(result.columns))

    def test_count_per_regime(self, simple_series):
        result = compute_regime_durations(simple_series)
        assert result.loc["A", "count"] == 2   # two runs of A
        assert result.loc["B", "count"] == 1
        assert result.loc["C", "count"] == 1

    def test_mean_days(self, simple_series):
        result = compute_regime_durations(simple_series)
        # A runs: lengths 2, 1 → mean = 1.5
        assert result.loc["A", "mean_days"] == pytest.approx(1.5, abs=0.1)

    def test_max_days(self, simple_series):
        result = compute_regime_durations(simple_series)
        # B has one run of length 3
        assert result.loc["B", "max_days"] == 3

    def test_single_run_std_zero(self):
        result = compute_regime_durations(pd.Series(["X"] * 5))
        assert result.loc["X", "std_days"] == 0.0

    def test_empty_series(self):
        result = compute_regime_durations(pd.Series([], dtype=str))
        assert result.empty


# ---------------------------------------------------------------------------
# TestComputeRegimeForwardReturns
# ---------------------------------------------------------------------------


class TestComputeRegimeForwardReturns:
    def test_returns_dataframe(self, simple_df):
        result = compute_regime_forward_returns(simple_df, "regime")
        assert isinstance(result, pd.DataFrame)

    def test_indexed_by_regime(self, simple_df):
        result = compute_regime_forward_returns(simple_df, "regime")
        assert set(result.index) == {"A", "B", "C"}

    def test_mean_correct(self, simple_df):
        result = compute_regime_forward_returns(simple_df, "regime")
        # A rows: indices 0,1,5 → 30d returns: 0.01, 0.02, 0.04 → mean = 0.0233...
        a_mean = simple_df[simple_df["regime"] == "A"]["sp500_forward_30d_return"].mean()
        assert result.loc["A", "sp500_forward_30d_return"] == pytest.approx(a_mean, abs=0.0001)

    def test_missing_columns_ignored(self):
        df = pd.DataFrame({"regime": ["A", "B"], "some_other_col": [1, 2]})
        result = compute_regime_forward_returns(df, "regime")
        assert result.empty or result.shape[1] == 0


# ---------------------------------------------------------------------------
# TestComputeTransitionForwardReturns
# ---------------------------------------------------------------------------


class TestComputeTransitionForwardReturns:
    def test_returns_dataframe(self, simple_df):
        result = compute_transition_forward_returns(simple_df, "regime")
        assert isinstance(result, pd.DataFrame)

    def test_multiindex_names(self, simple_df):
        result = compute_transition_forward_returns(simple_df, "regime")
        assert result.index.names == ["from_regime", "to_regime"]

    def test_first_row_not_transition(self, simple_df):
        # No (NaN → A) transition should appear
        result = compute_transition_forward_returns(simple_df, "regime")
        from_regimes = [idx[0] for idx in result.index]
        # all from_regimes must be valid regime values
        assert all(r in {"A", "B", "C"} for r in from_regimes)

    def test_correct_transitions(self, simple_df):
        result = compute_transition_forward_returns(simple_df, "regime")
        transitions = set(result.index.tolist())
        # A→B at index 2, B→A at index 5, A→C at index 6
        assert ("A", "B") in transitions
        assert ("B", "A") in transitions
        assert ("A", "C") in transitions

    def test_entry_return_is_float(self, simple_df):
        result = compute_transition_forward_returns(simple_df, "regime")
        assert result["sp500_forward_30d_return"].dtype == float


# ---------------------------------------------------------------------------
# TestRunRegimeAnalysis (integration)
# ---------------------------------------------------------------------------


class TestRunRegimeAnalysis:
    def test_returns_dict(self, analytics_df):
        result = run_regime_analysis(analytics_df)
        assert isinstance(result, dict)

    def test_expected_keys(self, analytics_df):
        result = run_regime_analysis(analytics_df)
        assert "final_decision" in result
        assert "transition_regime" in result

    def test_subkeys(self, analytics_df):
        result = run_regime_analysis(analytics_df)
        for col in ["final_decision", "transition_regime"]:
            for key in ["transition_counts", "transition_probs", "durations",
                        "forward_returns", "transition_returns"]:
                assert key in result[col], f"Missing '{key}' in result['{col}']"

    def test_output_files_created(self, analytics_df, tmp_path, monkeypatch):
        monkeypatch.setattr("src.regime_transition.OUTPUT_DIR", str(tmp_path))
        run_regime_analysis(analytics_df)
        for col in ["final_decision", "transition_regime"]:
            assert (tmp_path / f"{col}_counts.csv").exists()
            assert (tmp_path / f"{col}_probs.csv").exists()
            assert (tmp_path / f"{col}_durations.csv").exists()
            assert (tmp_path / f"{col}_forward_returns.csv").exists()
            assert (tmp_path / f"{col}_transition_returns.csv").exists()

    def test_prob_rows_sum_to_one(self, analytics_df):
        result = run_regime_analysis(analytics_df)
        for col in ["final_decision", "transition_regime"]:
            probs = result[col]["transition_probs"]
            row_sums = probs.sum(axis=1)
            assert (row_sums.round(6) == 1.0).all(), f"Row sums not 1.0 for {col}"

    def test_durations_all_positive(self, analytics_df):
        result = run_regime_analysis(analytics_df)
        for col in ["final_decision", "transition_regime"]:
            durations = result[col]["durations"]
            assert (durations["mean_days"] > 0).all()
            assert (durations["min_days"] >= 1).all()

    def test_forward_returns_shape(self, analytics_df):
        result = run_regime_analysis(analytics_df)
        fwd = result["final_decision"]["forward_returns"]
        assert fwd.shape[0] > 0
        assert fwd.shape[1] >= 1
