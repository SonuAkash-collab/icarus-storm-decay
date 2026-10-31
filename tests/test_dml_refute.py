"""Tests for the DML refutations (src/dml_refute.py) on synthetic data.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import dml  # noqa: E402
from src import dml_refute as rf  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402

# Helper models that selection picks on synthetic data, fixed here to keep tests fast
FAST = {"model_y": "xgb", "model_t": "linear"}
Y = "Y_residual"


@pytest.fixture(scope="module")
def fake():
    return make_synthetic(n_objects=200, n_high_alt=15, return_hourly=True, seed=0)


@pytest.fixture(scope="module")
def df(fake):
    return dml.prepare(fake[0], Y)


@pytest.fixture(scope="module")
def base(df):
    return dml.dml(df, Y, n_boot=299, **FAST)


def test_high_altitude_control_is_equivalent_to_zero(fake, base):
    h = rf.high_alt(fake[0], Y, base["coef"], n_boot=299, **FAST)
    assert h["n_objects"] == 15
    assert h["ci_lo"] < 0 < h["ci_hi"]
    assert h["band"] == pytest.approx(rf.EQUIV_BAND * abs(base["coef"]))
    assert h["equivalent"]


def test_high_altitude_without_main_estimate_is_not_tested(fake):
    assert rf.high_alt(fake[0], Y, None, n_boot=19, **FAST)["equivalent"] is None


def test_random_common_cause_leaves_estimate_unchanged(df, base):
    r = rf.random_common_cause(df, Y, base, n_boot=299, **FAST)
    assert abs(r["shift_in_se"]) < 0.5


def test_circular_shift_moves_storms_but_keeps_every_row(fake, df):
    hourly = fake[2]
    np.testing.assert_allclose(rf.ap_mean_circular(df, hourly, 0), df["ap_mean"], atol=1e-6)
    moved = rf.ap_mean_circular(df, hourly, 100)
    assert not np.isnan(moved).any()
    assert np.corrcoef(moved, df["ap_mean"])[0, 1] < 0.2


def test_placebo_shifts_centre_on_zero(fake, df, base):
    p = rf.placebo_shifts(df, Y, fake[2], n_shifts=8, **FAST)
    assert abs(p["coef"].mean()) < 0.2 * abs(base["coef"])
    assert (p["coef"].abs() < abs(base["coef"])).all()


def test_drop_storm_rows(fake, df):
    storms = fake[1]
    drop = storms["storm_id"].iloc[:3].tolist()
    kept = rf.drop_storm_rows(df, storms, drop)
    assert 0 < len(kept) < len(df)
    assert not kept["storm_id"].isin(drop).any()


def test_without_storms_reports_both(fake, df):
    storms = fake[1]
    out = rf.without_storms(df, Y, storms, storms["storm_id"].iloc[:3].tolist(), n_boot=99, **FAST)
    assert len(out) == 2 and out.iloc[1]["n_rows"] < out.iloc[0]["n_rows"]
    assert (out["coef"].astype(float) < 0).all()


def test_lag_sweep_finds_a_planted_12h_lag(fake):
    table, _, hourly = fake
    planted = rf.plant_lag(table, hourly, rf.PLANTED_LAG_HOURS)
    sweep = dml.lag_sweep(dml.prepare(planted, Y), hourly, Y, n_boot=0, **FAST)
    assert sweep["coef"].astype(float).abs().idxmax() == rf.PLANTED_LAG_HOURS
