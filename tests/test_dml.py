"""Tests for double ML (src/dml.py) on synthetic data. No data files needed.

Run with:  pytest
The 50-seed recovery check takes ~8 minutes, so it only runs with ICARUS_SLOW=1.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import dml, schema  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def fake():
    table, _ = make_synthetic(n_objects=200, seed=0)
    return table


@pytest.fixture(scope="module")
def result(fake):
    return dml.compare(fake, n_boot=499)


def test_prepare_uses_main_sample_and_mean_ap(fake):
    df = dml.prepare(fake)
    assert len(df) == len(schema.main_sample(fake))
    np.testing.assert_allclose(df["ap_mean"], df["T_ap_int"] / df["dt_days"])
    assert not any(f in c.lower() for c in dml.FEATURES for f in schema.FORBIDDEN)


def test_time_clusters(fake):
    df = dml.prepare(fake)
    linked = df["storm_id"].notna()
    assert (df.loc[linked, "cluster"] == "s" + df.loc[linked, "storm_id"].astype(str)).all()
    assert df.loc[~linked, "cluster"].str.startswith("w").all()


def test_folds_keep_each_cluster_in_one_fold(fake):
    clusters = dml.prepare(fake)["cluster"].to_numpy()
    folds = dml.time_folds(clusters, n_folds=5, seed=0)
    test_fold = {}
    for i, (_, test) in enumerate(folds):
        for c in np.unique(clusters[test]):
            assert c not in test_fold, f"cluster {c} split across folds"
            test_fold[c] = i
    assert len(test_fold) == len(np.unique(clusters))


def test_wls_cluster_matches_weighted_slope():
    rng = np.random.default_rng(0)
    x, w = rng.normal(size=500), rng.uniform(0.5, 1.5, 500)
    y = 2.0 * x + rng.normal(size=500)
    out = dml.wls_cluster(np.column_stack([x, np.ones(500)]), y, w, np.arange(500) % 25, n_boot=199)
    xc = x - np.average(x, weights=w)
    assert out["coef"] == pytest.approx((w * xc * y).sum() / (w * xc * xc).sum())
    assert out["se_cr1"] > 0 and out["ci_lo"] < out["coef"] < out["ci_hi"]
    assert out["n_clusters"] == 25


def test_select_model_prefers_linear_for_linear_target():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(0, 0.1, 2000)
    folds = dml.time_folds(np.arange(2000) % 20, n_folds=5)
    name, scores = dml.select_model(X, y, np.ones(2000), folds)
    assert name == "linear" and scores["linear"]["r2"] > 0.95


@pytest.fixture(scope="module")
def planted(fake):
    df = dml.prepare(fake)
    return dml.planted_theta(df, dml.true_leftover_ap(df, seed=0))


def test_true_leftover_ap_is_unrelated_to_sun(fake):
    df = dml.prepare(fake)
    v = dml.true_leftover_ap(df, seed=0, n_draws=100)
    assert abs(np.corrcoef(v, df["f107"])[0, 1]) < 0.05
    assert abs(np.average(v, weights=df["dt_days"])) < 0.1 * df["ap_mean"].mean()


def test_econml_estimate_matches_own_slope(result):
    assert result.attrs["theta_econml"] == pytest.approx(result.loc["dml", "coef"], rel=1e-6)


def test_reports_every_ap_model(result):
    for name in dml.candidate_models():
        assert f"dml_t_{name}" in result.index
    assert result.loc[f"dml_t_{result.attrs['model_t']}", "coef"] == result.loc["dml", "coef"]


def test_dml_recovers_planted_effect(result, planted):
    assert result.loc["dml", "ci_lo"] <= planted <= result.loc["dml", "ci_hi"]


def test_naive_is_fooled_by_sun_uv(result, planted):
    assert not result.loc["naive", "ci_lo"] <= planted <= result.loc["naive", "ci_hi"]
    assert abs(result.loc["naive", "coef"] - planted) > 3 * abs(result.loc["dml", "coef"] - planted)
    assert result.attrs["adjustment_share"] > 0.2


@pytest.mark.skipif(os.environ.get("ICARUS_SLOW") != "1", reason="slow: set ICARUS_SLOW=1")
def test_dml_covers_planted_in_90_percent_of_50_seeds():
    runs = dml.seed_study(n_seeds=50)
    assert runs["dml_covers"].sum() >= 45
