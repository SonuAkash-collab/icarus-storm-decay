"""Tests for the physics simulator (src/simulate.py). Synthetic space weather and
pymsis only: no data files, no downloads.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import event_study as es  # noqa: E402
from src import physics, schema  # noqa: E402
from src.event_checks import short_run_effect  # noqa: E402
from src.simulate import default_boost, planted_profile, simulate  # noqa: E402

SMALL = {"n_objects": 25, "n_high_alt": 2, "n_days": 300, "seed": 0}


@pytest.fixture(scope="module")
def null():
    return simulate(boost=0.0, **SMALL)


@pytest.fixture(scope="module")
def planted():
    return simulate(boost=0.5, return_hourly=True, **SMALL)


def weighted_mean(df, col):
    return (df[col] * df["dt_days"]).sum() / df["dt_days"].sum()


def test_matches_schema(planted):
    table, storms, _ = planted
    assert schema.problems(table) == []
    assert list(table.columns[:len(schema.COLUMNS)]) == list(schema.COLUMNS)
    for c in ["true_effect", "true_decay", "obs_decay", "true_drag_sensitivity", "drag_source", "role"]:
        assert c in table
    assert (table["is_active"] == 0).all() and (table["ecc"] < schema.MAX_ECC).all()
    assert not schema.bridges_onset(table, storms).any()
    assert table["alt_km"].between(250, 800).any() and (table["alt_km"] >= 1000).any()


def test_high_alt_uses_catalog_drag_sensitivity(planted):
    table, _, _ = planted
    high = table["alt_km"] >= schema.HIGH_ALT_MIN_KM
    assert (table.loc[high, "drag_source"] == "catalog").all()
    assert (table.loc[~high, "drag_source"] == "fit").all()
    assert table.loc[high, "Y_log_ratio"].isna().all()


def test_default_boost_shape():
    b = default_boost([-1, 0, 2, 3.5, 5, 8], peak=0.5)
    assert np.allclose(b, [0, 0.5, 0.5, 0.25, 0, 0])


def test_physics_matches_without_noise():
    # No TLE noise, no boost: the pipeline's prediction must follow the true decay
    table, _ = simulate(n_objects=6, n_high_alt=0, n_days=60, boost=0.0, sigma_a_m=0.0, seed=1)
    rel = table["Y_residual"] / table["true_decay"]
    assert abs(rel.median()) < 0.01
    assert rel.abs().mean() < 0.05
    assert (table["true_effect"] == 0).all()


def test_no_boost_residual_is_zero(null):
    table, _ = null
    main = schema.main_sample(table)
    assert (table["true_effect"] == 0).all()
    se = physics.block_se(main["Y_residual"], main["t_mid"])
    assert abs(weighted_mean(main, "Y_residual")) < 3 * se
    for role in ["fit", "holdout", "storm"]:
        part = main[main["role"] == role]
        assert abs(part["Y_residual"].median()) < 0.05 * part["true_decay"].abs().median() + 1.0


def test_holdout_quiet_residual_is_zero(planted):
    table, _, _ = planted
    hold = schema.main_sample(table).query("role == 'holdout'")
    assert (hold["true_effect"] == 0).all()
    m = weighted_mean(hold, "Y_residual")
    assert abs(m) < 3 * physics.block_se(hold["Y_residual"], hold["t_mid"])


def test_planted_boost_recovered_by_event_study(planted, null):
    table, storms, _ = planted
    res, report = es.event_study(table, storms, n_boot=199)
    assert report["used"].sum() >= 8
    short = short_run_effect(res)
    truth = planted_profile(table, storms).loc[0:2].mean()
    assert truth < -5
    assert short["ci_lo"] <= truth <= short["ci_hi"]
    assert short["ci_hi"] < 0
    # No boost: the interval covers zero
    short0 = short_run_effect(es.event_study(*null, n_boot=199)[0])
    assert short0["ci_lo"] <= 0 <= short0["ci_hi"]


def test_noise_larger_inside_storms(planted):
    table, _, _ = planted
    err = (table["obs_decay"] - table["true_decay"]) * table["dt_days"]   # m, noise on da
    inside = err[table["T_storm"] == 1].std()
    outside = err[table["T_storm"] == 0].std()
    assert inside > 2 * outside


def test_negative_boost_means_less_decay():
    table, storms = simulate(n_objects=5, n_high_alt=0, n_days=60, boost=-0.3, seed=2)
    early = table["event_day"].between(0, 2)
    assert (table["true_effect"] >= 0).all()
    assert (table.loc[early, "true_effect"] > 0).all()


def test_smoothing_blurs_the_storm():
    kw = {"n_objects": 5, "n_high_alt": 0, "n_days": 60, "boost": 1.0, "sigma_a_m": 0.0, "seed": 3}
    sharp, _ = simulate(**kw)
    blurred, _ = simulate(smooth_days=2.0, **kw)
    assert np.allclose(sharp["obs_decay"], sharp["true_decay"])
    day0 = blurred["event_day"].between(0, 1)
    after = blurred["event_day"].between(3, 5)
    # The trailing TLE fit sees the storm late: less extra decay on day 0, more after it
    assert (blurred.loc[day0, "obs_decay"] - blurred.loc[day0, "true_decay"]).mean() > 0
    assert (blurred.loc[after, "obs_decay"] - blurred.loc[after, "true_decay"]).mean() < 0


def test_accepts_given_space_weather(planted):
    table, storms, sw = planted
    again, _ = simulate(boost=0.5, spaceweather=sw, storms=storms, **SMALL)
    pd.testing.assert_frame_equal(again, table)
