"""
Tests for src/regime_probability.py.

Covers model fitting, single-row probability computation, full-history
vectorised pass, current-probs summary, and the orchestrator — including
edge cases and analytical invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime_probability import (
    _FEATURE_COLS,
    _LABEL_COL,
    _MIN_OBS,
    _STD_FLOOR,
    compute_prob_history,
    fit_naive_bayes,
    get_current_probs,
    run_regime_probability,
    score_to_regime_probs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n_per_regime: int = 30, seed: int = 0) -> pd.DataFrame:
    """
    Two well-separated regimes (RegimeA: all features ~ 20, RegimeB: ~ 80).
    Makes it easy to assert that the correct regime has highest probability.
    """
    rng = np.random.default_rng(seed)
    feats = _FEATURE_COLS[:3]  # use first 3 for simplicity
    rows = []
    for i, (label, mean) in enumerate([("RegimeA", 20.0), ("RegimeB", 80.0)]):
        for _ in range(n_per_regime):
            r = {f: rng.normal(mean, 3.0) for f in feats}
            r[_LABEL_COL] = label
            r["date"] = pd.Timestamp("2024-01-01") + pd.Timedelta(days=len(rows))
            rows.append(r)
    # Pad remaining _FEATURE_COLS with zeros so the function doesn't drop them
    df = pd.DataFrame(rows)
    for c in _FEATURE_COLS:
        if c not in df.columns:
            df[c] = 50.0
    return df.sort_values("date").reset_index(drop=True)


def _make_full_df(seed: int = 0) -> pd.DataFrame:
    """Five regimes, 25 obs each — uses all _FEATURE_COLS."""
    rng   = np.random.default_rng(seed)
    means = [10.0, 30.0, 50.0, 70.0, 90.0]
    labels = [f"Regime{i}" for i in range(5)]
    rows = []
    for label, mean in zip(labels, means):
        for _ in range(25):
            r = {f: rng.normal(mean, 4.0) for f in _FEATURE_COLS}
            r[_LABEL_COL] = label
            r["date"] = pd.Timestamp("2024-01-01") + pd.Timedelta(days=len(rows))
            rows.append(r)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@pytest.fixture(scope="module")
def two_regime_df():
    return _make_df()


@pytest.fixture(scope="module")
def five_regime_df():
    return _make_full_df()


# ---------------------------------------------------------------------------
# fit_naive_bayes
# ---------------------------------------------------------------------------

class TestFitNaiveBayes:
    def test_returns_dict(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        assert isinstance(model, dict)

    def test_correct_regimes_present(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        assert "RegimeA" in model
        assert "RegimeB" in model

    def test_prior_sums_to_one(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        total_prior = sum(v["prior"] for v in model.values())
        assert abs(total_prior - 1.0) < 1e-9

    def test_each_entry_has_expected_keys(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        for regime, params in model.items():
            for k in ["prior", "means", "stds", "n_obs", "features"]:
                assert k in params, f"Missing key {k} in regime {regime}"

    def test_stds_at_least_floor(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        for params in model.values():
            assert (params["stds"] >= _STD_FLOOR).all()

    def test_means_near_known_values(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        # _make_df randomises only the first 3 _FEATURE_COLS; the rest are
        # padded to 50.0, so we check only the active features.
        n_active = 3
        assert abs(model["RegimeA"]["means"][:n_active].mean() - 20.0) < 5.0
        assert abs(model["RegimeB"]["means"][:n_active].mean() - 80.0) < 5.0

    def test_n_obs_correct(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        assert model["RegimeA"]["n_obs"] == 30
        assert model["RegimeB"]["n_obs"] == 30

    def test_prior_balanced_for_equal_counts(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        assert abs(model["RegimeA"]["prior"] - 0.5) < 1e-6
        assert abs(model["RegimeB"]["prior"] - 0.5) < 1e-6

    def test_low_count_regime_excluded(self):
        rng = np.random.default_rng(1)
        rows = [{f: rng.normal(50, 5) for f in _FEATURE_COLS}
                | {_LABEL_COL: "Big", "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)}
                for i in range(30)]
        rows += [{f: rng.normal(80, 5) for f in _FEATURE_COLS}
                 | {_LABEL_COL: "Small", "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i + 30)}
                 for i in range(_MIN_OBS - 1)]  # just under threshold
        df = pd.DataFrame(rows)
        model = fit_naive_bayes(df)
        assert "Big"   in model
        assert "Small" not in model

    def test_empty_df_returns_empty(self):
        assert fit_naive_bayes(pd.DataFrame()) == {}

    def test_missing_label_col_returns_empty(self, two_regime_df):
        df2 = two_regime_df.drop(columns=[_LABEL_COL])
        assert fit_naive_bayes(df2) == {}

    def test_missing_feature_cols_returns_empty(self, two_regime_df):
        df2 = two_regime_df.drop(columns=_FEATURE_COLS[:3])
        # Only the remaining features should be used — model should still fit
        # if any _FEATURE_COLS remain
        model = fit_naive_bayes(df2)
        # We expect a model using the remaining features
        remaining = [c for c in _FEATURE_COLS if c in df2.columns]
        if remaining:
            assert len(model) > 0
        else:
            assert model == {}


# ---------------------------------------------------------------------------
# score_to_regime_probs
# ---------------------------------------------------------------------------

class TestScoreToRegimeProbs:
    def _model(self):
        return fit_naive_bayes(_make_df())

    def test_returns_dict(self):
        model = self._model()
        feats = list(model.values())[0]["features"]
        x = np.full(len(feats), 20.0)
        result = score_to_regime_probs(x, model)
        assert isinstance(result, dict)

    def test_probs_sum_to_one(self):
        model = self._model()
        feats = list(model.values())[0]["features"]
        x = np.full(len(feats), 50.0)
        result = score_to_regime_probs(x, model)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_probs_between_0_and_1(self):
        model = self._model()
        feats = list(model.values())[0]["features"]
        x = np.full(len(feats), 50.0)
        result = score_to_regime_probs(x, model)
        for p in result.values():
            assert 0.0 <= p <= 1.0

    def test_score_near_regime_a_favours_regime_a(self):
        model = self._model()
        feats = list(model.values())[0]["features"]
        x = np.full(len(feats), 20.0)  # RegimeA centroid
        result = score_to_regime_probs(x, model)
        assert result["RegimeA"] > result["RegimeB"]

    def test_score_near_regime_b_favours_regime_b(self):
        model = self._model()
        feats = list(model.values())[0]["features"]
        x = np.full(len(feats), 80.0)  # RegimeB centroid
        result = score_to_regime_probs(x, model)
        assert result["RegimeB"] > result["RegimeA"]

    def test_nan_input_returns_empty(self):
        model = self._model()
        feats = list(model.values())[0]["features"]
        x = np.full(len(feats), np.nan)
        result = score_to_regime_probs(x, model)
        assert result == {}

    def test_empty_model_returns_empty(self):
        x = np.array([50.0, 50.0, 50.0])
        assert score_to_regime_probs(x, {}) == {}

    def test_all_regimes_present_in_result(self):
        model = self._model()
        feats = list(model.values())[0]["features"]
        x = np.full(len(feats), 50.0)
        result = score_to_regime_probs(x, model)
        for r in model:
            assert r in result


# ---------------------------------------------------------------------------
# compute_prob_history
# ---------------------------------------------------------------------------

class TestComputeProbHistory:
    def test_returns_dataframe(self, two_regime_df):
        hist = compute_prob_history(two_regime_df)
        assert isinstance(hist, pd.DataFrame)

    def test_length_matches_input(self, two_regime_df):
        hist = compute_prob_history(two_regime_df)
        assert len(hist) == len(two_regime_df)

    def test_regime_columns_present(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        hist  = compute_prob_history(two_regime_df, model=model)
        for r in model:
            assert r in hist.columns

    def test_top_regime_column_present(self, two_regime_df):
        hist = compute_prob_history(two_regime_df)
        assert "top_regime" in hist.columns

    def test_entropy_column_present(self, two_regime_df):
        hist = compute_prob_history(two_regime_df)
        assert "entropy" in hist.columns

    def test_probs_sum_to_one_per_row(self, five_regime_df):
        model = fit_naive_bayes(five_regime_df)
        hist  = compute_prob_history(five_regime_df, model=model)
        regime_cols = list(model.keys())
        row_sums = hist[regime_cols].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-6

    def test_probs_nonneg(self, five_regime_df):
        model = fit_naive_bayes(five_regime_df)
        hist  = compute_prob_history(five_regime_df, model=model)
        regime_cols = list(model.keys())
        assert (hist[regime_cols] >= 0).all().all()

    def test_index_is_datetime(self, two_regime_df):
        hist = compute_prob_history(two_regime_df)
        assert pd.api.types.is_datetime64_any_dtype(hist.index)

    def test_top_regime_matches_argmax(self, five_regime_df):
        model = fit_naive_bayes(five_regime_df)
        hist  = compute_prob_history(five_regime_df, model=model)
        regime_cols = list(model.keys())
        expected = hist[regime_cols].idxmax(axis=1)
        pd.testing.assert_series_equal(hist["top_regime"], expected, check_names=False)

    def test_entropy_nonneg(self, five_regime_df):
        hist = compute_prob_history(five_regime_df)
        assert (hist["entropy"] >= 0).all()

    def test_entropy_near_zero_for_certain_regime(self, two_regime_df):
        # On the extreme end of RegimeA, entropy should be low
        model = fit_naive_bayes(two_regime_df)
        hist  = compute_prob_history(two_regime_df, model=model)
        # RegimeA rows (first 30) should have low entropy
        assert hist["entropy"].iloc[:30].mean() < 0.5

    def test_empty_df_returns_empty(self):
        assert compute_prob_history(pd.DataFrame()).empty

    def test_model_none_fits_internally(self, two_regime_df):
        hist = compute_prob_history(two_regime_df, model=None)
        assert not hist.empty

    def test_well_separated_regimes_correctly_classified(self, two_regime_df):
        model = fit_naive_bayes(two_regime_df)
        hist  = compute_prob_history(two_regime_df, model=model)
        # First 30 rows are RegimeA — top_regime should mostly be RegimeA
        a_correct = (hist["top_regime"].iloc[:30] == "RegimeA").mean()
        assert a_correct > 0.8


# ---------------------------------------------------------------------------
# get_current_probs
# ---------------------------------------------------------------------------

class TestGetCurrentProbs:
    def test_returns_dict(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        assert isinstance(result, dict)

    def test_all_required_keys_present(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        for k in ["probs", "top_regime", "top_prob", "entropy", "model_regimes"]:
            assert k in result

    def test_second_keys_present_for_multi_regime(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        assert "second" in result
        assert "second_prob" in result

    def test_probs_sum_to_one(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        assert abs(sum(result["probs"].values()) - 1.0) < 1e-6

    def test_top_prob_is_max(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        assert result["top_prob"] == max(result["probs"].values())

    def test_top_prob_between_0_and_1(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        assert 0.0 <= result["top_prob"] <= 1.0

    def test_entropy_nonneg(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        assert result["entropy"] >= 0.0

    def test_model_regimes_is_list(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        assert isinstance(result["model_regimes"], list)

    def test_empty_df_returns_empty(self):
        assert get_current_probs(pd.DataFrame()) == {}

    def test_top_regime_in_model_regimes(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        assert result["top_regime"] in result["model_regimes"]

    def test_second_prob_leq_top_prob(self, two_regime_df):
        result = get_current_probs(two_regime_df)
        assert result["second_prob"] <= result["top_prob"]


# ---------------------------------------------------------------------------
# run_regime_probability
# ---------------------------------------------------------------------------

class TestRunRegimeProbability:
    def test_returns_dict(self, two_regime_df):
        result = run_regime_probability(two_regime_df)
        assert isinstance(result, dict)

    def test_all_keys_present(self, two_regime_df):
        result = run_regime_probability(two_regime_df)
        for k in ["model", "current", "history"]:
            assert k in result

    def test_model_is_dict(self, two_regime_df):
        result = run_regime_probability(two_regime_df)
        assert isinstance(result["model"], dict)

    def test_current_is_dict(self, two_regime_df):
        result = run_regime_probability(two_regime_df)
        assert isinstance(result["current"], dict)

    def test_history_is_dataframe(self, two_regime_df):
        result = run_regime_probability(two_regime_df)
        assert isinstance(result["history"], pd.DataFrame)

    def test_history_length_matches_input(self, two_regime_df):
        result = run_regime_probability(two_regime_df)
        assert len(result["history"]) == len(two_regime_df)

    def test_empty_df_graceful(self):
        result = run_regime_probability(pd.DataFrame())
        assert result["model"] == {}
        assert result["current"] == {}
        assert result["history"].empty


# ---------------------------------------------------------------------------
# Analytical invariants
# ---------------------------------------------------------------------------

class TestAnalyticalInvariants:
    def test_uniform_prior_unaffected_by_label_imbalance(self):
        """
        If one regime is 10x larger but same centroid, the minority regime
        still wins probability at its own centroid.
        """
        rng = np.random.default_rng(99)
        f = _FEATURE_COLS[:2]
        big   = [{c: rng.normal(20, 3) for c in f} | {_LABEL_COL: "Big"}   for _ in range(100)]
        small = [{c: rng.normal(80, 3) for c in f} | {_LABEL_COL: "Small"} for _ in range(10)]
        for c in _FEATURE_COLS:
            if c not in f:
                for r in big + small:
                    r[c] = 50.0
        df    = pd.DataFrame(big + small)
        model = fit_naive_bayes(df)
        # At the Small centroid (all=80), Small should still win
        feats = model[list(model.keys())[0]]["features"]
        x     = np.full(len(feats), 80.0)
        probs = score_to_regime_probs(x, model)
        # Small should win even against much larger prior of Big
        assert probs["Small"] > probs["Big"]

    def test_midpoint_has_max_entropy(self, two_regime_df):
        """
        At the exact midpoint between the two model centroids, entropy
        should be higher than at either centroid (more uncertain).
        """
        model     = fit_naive_bayes(two_regime_df)
        c_a       = model["RegimeA"]["means"]
        c_b       = model["RegimeB"]["means"]
        midpoint  = (c_a + c_b) / 2.0

        p_mid = np.array(list(score_to_regime_probs(midpoint, model).values()))
        p_a   = np.array(list(score_to_regime_probs(c_a, model).values()))
        ent_mid = -np.sum(p_mid * np.log2(np.clip(p_mid, 1e-12, 1)))
        ent_a   = -np.sum(p_a   * np.log2(np.clip(p_a,   1e-12, 1)))
        assert ent_mid > ent_a  # midpoint is more uncertain than RegimeA centroid

    def test_five_regime_correct_assignment(self, five_regime_df):
        """
        At each regime's centroid, that regime should have highest probability.
        """
        model = fit_naive_bayes(five_regime_df)
        feats = list(model.values())[0]["features"]
        centroids = {r: params["means"] for r, params in model.items()}
        for regime, centroid in centroids.items():
            probs = score_to_regime_probs(centroid, model)
            assert probs[regime] == max(probs.values()), \
                f"Regime {regime} not top-ranked at its own centroid"
