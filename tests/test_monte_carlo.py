"""
Tests for src/monte_carlo.py.

Covers transition matrix construction (Laplace smoothing, row-stochastic),
score parameter extraction, path simulation (shape, bounds, seed
reproducibility, regime distribution convergence), and the orchestrator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import (
    _HORIZONS,
    _LAPLACE_EPS,
    _N_PATHS,
    _PERCENTILES,
    _SCORE_COL,
    _LABEL_COL,
    build_transition_matrix,
    run_monte_carlo,
    simulate_paths,
    _score_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Two-regime alternating sequence with synthetic composite scores."""
    rng    = np.random.default_rng(seed)
    labels = []
    regime = "A"
    for _ in range(n):
        labels.append(regime)
        if rng.random() < 0.15:
            regime = "B" if regime == "A" else "A"
    scores = np.where(
        np.array(labels) == "A",
        rng.normal(30.0, 3.0, n),
        rng.normal(60.0, 5.0, n),
    )
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "date":              dates,
        _LABEL_COL:          labels,
        _SCORE_COL:          np.clip(scores, 0, 100),
    })


def _make_three_regime_df(n_each: int = 50, seed: int = 1) -> pd.DataFrame:
    """Three regimes with known centroid scores."""
    rng   = np.random.default_rng(seed)
    rows  = []
    t     = 0
    for label, mean in [("Low", 20.0), ("Mid", 50.0), ("High", 80.0)]:
        for _ in range(n_each):
            rows.append({
                "date":     pd.Timestamp("2024-01-02") + pd.Timedelta(days=t),
                _LABEL_COL: label,
                _SCORE_COL: float(np.clip(rng.normal(mean, 4.0), 0, 100)),
            })
            t += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def two_regime_df():
    return _make_df()


@pytest.fixture(scope="module")
def three_regime_df():
    return _make_three_regime_df()


# ---------------------------------------------------------------------------
# build_transition_matrix
# ---------------------------------------------------------------------------

class TestBuildTransitionMatrix:
    def test_returns_tuple(self, two_regime_df):
        regimes, P = build_transition_matrix(two_regime_df)
        assert isinstance(regimes, list)
        assert isinstance(P, np.ndarray)

    def test_regimes_match_unique_labels(self, two_regime_df):
        regimes, P = build_transition_matrix(two_regime_df)
        assert set(regimes) == {"A", "B"}

    def test_matrix_shape(self, two_regime_df):
        regimes, P = build_transition_matrix(two_regime_df)
        n = len(regimes)
        assert P.shape == (n, n)

    def test_rows_sum_to_one(self, two_regime_df):
        _, P = build_transition_matrix(two_regime_df)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_all_entries_positive(self, two_regime_df):
        _, P = build_transition_matrix(two_regime_df)
        assert (P > 0).all()

    def test_laplace_smoothing_prevents_zeros(self):
        """A regime with no outgoing transitions still gets positive probs."""
        df = pd.DataFrame({
            _LABEL_COL: ["X"] * 10 + ["Y"] * 10,
            _SCORE_COL: [30.0] * 20,
            "date":     pd.date_range("2024-01-01", periods=20, freq="B"),
        })
        _, P = build_transition_matrix(df)
        assert (P > 0).all()

    def test_diagonal_dominant_for_sticky_regime(self, two_regime_df):
        """Stable two-regime df: self-transitions should dominate rows."""
        regimes, P = build_transition_matrix(two_regime_df)
        for i in range(len(regimes)):
            assert P[i, i] > 0.5

    def test_empty_df_returns_empty(self):
        regimes, P = build_transition_matrix(pd.DataFrame())
        assert regimes == []
        assert P.size == 0

    def test_missing_label_col_returns_empty(self, two_regime_df):
        df2 = two_regime_df.drop(columns=[_LABEL_COL])
        regimes, P = build_transition_matrix(df2)
        assert regimes == []

    def test_single_row_returns_empty(self):
        df = pd.DataFrame({_LABEL_COL: ["A"], _SCORE_COL: [30.0],
                           "date": [pd.Timestamp("2024-01-01")]})
        regimes, P = build_transition_matrix(df)
        assert regimes == []


# ---------------------------------------------------------------------------
# _score_params
# ---------------------------------------------------------------------------

class TestScoreParams:
    def test_returns_arrays(self, three_regime_df):
        regimes, _ = build_transition_matrix(three_regime_df)
        means, stds = _score_params(three_regime_df, regimes)
        assert isinstance(means, np.ndarray)
        assert isinstance(stds, np.ndarray)

    def test_length_matches_regimes(self, three_regime_df):
        regimes, _ = build_transition_matrix(three_regime_df)
        means, stds = _score_params(three_regime_df, regimes)
        assert len(means) == len(regimes)
        assert len(stds) == len(regimes)

    def test_means_near_known_centroids(self, three_regime_df):
        regimes, _ = build_transition_matrix(three_regime_df)
        means, stds = _score_params(three_regime_df, regimes)
        expected = {"Low": 20.0, "Mid": 50.0, "High": 80.0}
        for i, r in enumerate(regimes):
            assert abs(means[i] - expected[r]) < 8.0

    def test_stds_above_floor(self, three_regime_df):
        from src.monte_carlo import _SCORE_FLOOR
        regimes, _ = build_transition_matrix(three_regime_df)
        _, stds = _score_params(three_regime_df, regimes)
        assert (stds >= _SCORE_FLOOR).all()


# ---------------------------------------------------------------------------
# simulate_paths
# ---------------------------------------------------------------------------

class TestSimulatePaths:
    def _setup(self, df):
        regimes, P = build_transition_matrix(df)
        means, stds = _score_params(df, regimes)
        start = {r: 1.0 / len(regimes) for r in regimes}
        return regimes, P, means, stds, start

    def test_returns_dict(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start)
        assert isinstance(result, dict)

    def test_all_keys_present(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start)
        for k in ["paths", "percentiles", "regime_paths",
                  "regime_probs_at", "summary"]:
            assert k in result

    def test_paths_shape(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start,
                                horizons=[21], n_paths=200)
        assert result["paths"][21].shape == (200, 21)

    def test_scores_in_valid_range(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start,
                                horizons=[21], n_paths=500)
        scores = result["paths"][21]
        assert (scores >= 0).all()
        assert (scores <= 100).all()

    def test_regime_paths_shape(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start,
                                horizons=[21], n_paths=200)
        rp = result["regime_paths"][21]
        assert rp.shape == (200, 21)
        assert rp.min() >= 0
        assert rp.max() < len(regimes)

    def test_percentiles_shape(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start, horizons=[21])
        pdf = result["percentiles"][21]
        assert isinstance(pdf, pd.DataFrame)
        assert len(pdf) == 21
        for p in _PERCENTILES:
            assert f"p{p}" in pdf.columns

    def test_percentiles_monotone_across_cols(self, two_regime_df):
        """p5 <= p25 <= p50 <= p75 <= p95 at every time step."""
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start, horizons=[21])
        pdf = result["percentiles"][21]
        cols = [f"p{p}" for p in _PERCENTILES]
        for i in range(len(cols) - 1):
            assert (pdf[cols[i]] <= pdf[cols[i + 1]] + 1e-9).all()

    def test_reproducibility(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        r1 = simulate_paths(P, regimes, means, stds, start, horizons=[21], seed=0)
        r2 = simulate_paths(P, regimes, means, stds, start, horizons=[21], seed=0)
        np.testing.assert_array_equal(r1["paths"][21], r2["paths"][21])

    def test_different_seeds_differ(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        r1 = simulate_paths(P, regimes, means, stds, start, horizons=[21], seed=0)
        r2 = simulate_paths(P, regimes, means, stds, start, horizons=[21], seed=99)
        assert not np.array_equal(r1["paths"][21], r2["paths"][21])

    def test_regime_probs_at_sum_to_one(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start, horizons=_HORIZONS)
        for H in _HORIZONS:
            total = sum(result["regime_probs_at"][H].values())
            assert abs(total - 1.0) < 1e-6

    def test_summary_has_expected_keys(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start, horizons=[21])
        for k in ["median", "p5", "p95", "prob_above_50", "prob_above_70"]:
            assert k in result["summary"][21]

    def test_summary_p5_leq_median_leq_p95(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start, horizons=[21])
        s = result["summary"][21]
        assert s["p5"] <= s["median"] <= s["p95"]

    def test_probs_above_thresholds_between_0_and_1(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start, horizons=[21])
        s = result["summary"][21]
        assert 0.0 <= s["prob_above_50"] <= 1.0
        assert 0.0 <= s["prob_above_70"] <= 1.0
        assert s["prob_above_70"] <= s["prob_above_50"]

    def test_all_horizons_present(self, two_regime_df):
        regimes, P, means, stds, start = self._setup(two_regime_df)
        result = simulate_paths(P, regimes, means, stds, start, horizons=_HORIZONS)
        for H in _HORIZONS:
            assert H in result["paths"]
            assert H in result["percentiles"]
            assert H in result["summary"]

    def test_start_concentrated_regime_reflects_in_short_horizon(self):
        """If we start certain in regime A (score~30), short horizon median is near 30."""
        df = _make_df(seed=5)
        regimes, P = build_transition_matrix(df)
        means, stds = _score_params(df, regimes)
        start_a = {r: (1.0 if r == "A" else 0.0) for r in regimes}
        result  = simulate_paths(P, regimes, means, stds, start_a,
                                 horizons=[5], n_paths=2000, seed=0)
        # First step median should be near regime A's score (~30)
        first_step_median = float(np.median(result["paths"][5][:, 0]))
        assert 20.0 < first_step_median < 45.0

    def test_stationary_distribution_matches_at_long_horizon(self):
        """
        At a long horizon, the regime distribution should approximate the
        stationary distribution of the chain (independent of start).
        """
        df     = _make_df(seed=7)
        regimes, P = build_transition_matrix(df)
        means, stds = _score_params(df, regimes)

        # Two extreme starting distributions
        start_a = {r: (1.0 if r == "A" else 0.0) for r in regimes}
        start_b = {r: (0.0 if r == "A" else 1.0) for r in regimes}

        ra = simulate_paths(P, regimes, means, stds, start_a,
                            horizons=[126], n_paths=2000, seed=0)
        rb = simulate_paths(P, regimes, means, stds, start_b,
                            horizons=[126], n_paths=2000, seed=0)

        # Regime distributions at H=126 should be close regardless of start
        for r in regimes:
            pa = ra["regime_probs_at"][126].get(r, 0)
            pb = rb["regime_probs_at"][126].get(r, 0)
            assert abs(pa - pb) < 0.15, f"Regime {r}: start-dependence too large at H=126"


# ---------------------------------------------------------------------------
# run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def test_returns_dict(self, two_regime_df):
        result = run_monte_carlo(two_regime_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, two_regime_df):
        result = run_monte_carlo(two_regime_df)
        for k in ["regimes", "transition_matrix", "score_means",
                  "score_stds", "simulation", "horizons",
                  "last_actual", "last_date"]:
            assert k in result

    def test_horizons_default(self, two_regime_df):
        result = run_monte_carlo(two_regime_df)
        assert result["horizons"] == _HORIZONS

    def test_custom_horizons(self, two_regime_df):
        result = run_monte_carlo(two_regime_df, horizons=[10, 30])
        assert result["horizons"] == [10, 30]
        for H in [10, 30]:
            assert H in result["simulation"]["paths"]

    def test_last_actual_is_float(self, two_regime_df):
        result = run_monte_carlo(two_regime_df)
        assert isinstance(result["last_actual"], float)

    def test_transition_matrix_row_stochastic(self, two_regime_df):
        result = run_monte_carlo(two_regime_df)
        P = result["transition_matrix"]
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_empty_df_returns_empty(self):
        assert run_monte_carlo(pd.DataFrame()) == {}

    def test_missing_label_col_returns_empty(self, two_regime_df):
        df2 = two_regime_df.drop(columns=[_LABEL_COL])
        assert run_monte_carlo(df2) == {}

    def test_custom_start_probs_accepted(self, two_regime_df):
        result = run_monte_carlo(two_regime_df, current_probs={"A": 0.9, "B": 0.1})
        assert isinstance(result, dict)
        assert "simulation" in result

    def test_custom_n_paths(self, two_regime_df):
        result = run_monte_carlo(two_regime_df, n_paths=50, horizons=[21])
        assert result["simulation"]["paths"][21].shape[0] == 50
