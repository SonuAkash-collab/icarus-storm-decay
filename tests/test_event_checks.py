"""Tests for the event-study checks (src/event_checks.py), on synthetic data.

Small numbers of seeds and placebo draws keep this fast; the real run uses
run_all(..., n_placebo=500). Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import event_checks as ec  # noqa: E402
from src import event_study as es  # noqa: E402
from src.schema import main_sample  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def fake():
    return make_synthetic(n_objects=120, seed=0)


@pytest.fixture(scope="module")
def null_fake():
    return make_synthetic(n_objects=120, seed=0, tau0=0.0)


def test_planted_recovery_over_seeds():
    rec = ec.planted_recovery(seeds=range(3), n_objects=120, n_boot=199)
    day0 = rec[rec["k"] == 0]
    assert (day0["coef"] < 0).all()
    # Mean error on day 0 within a quarter of the planted size
    assert abs(day0["error"].mean()) < 0.25 * abs(day0["truth"].mean())
    assert rec["covered"].mean() >= 2 / 3


def test_placebo_dates_centre_on_zero(fake):
    table, storms = fake
    placebo = ec.placebo_dates(table, storms, n_draws=25, seed=1)
    assert len(placebo) >= 20
    assert abs(placebo["est"].mean()) < 0.1
    real = ec.short_run_effect(es.event_study(table, storms, n_boot=0)[0])["est"]
    summary = ec.placebo_summary(placebo, real)
    assert summary["p_real"] < 0.05                  # the real storms stand out


def test_placebo_onsets_avoid_real_storms(fake):
    table, storms = fake
    days = ec.quiet_days(main_sample(table), storms)
    for _, start, end in es.contamination_windows(storms):
        assert not ((days >= start) & (days <= end)).any()


def test_pretrend_flat_on_null_data(null_fake):
    table, storms = null_fake
    res, _ = es.event_study(table, storms, n_boot=199)
    placebo = ec.placebo_dates(table, storms, n_draws=25, seed=2)
    test = ec.pretrend_test(res, placebo)
    assert test["p_placebo"] > 0.05
    assert test["n_days"] == 9


def test_pretrend_catches_a_fake_trend(fake):
    table, storms = fake
    res, _ = es.event_study(table, storms, n_boot=199)
    bent = res.copy()
    bent.loc[-10:-2, "coef"] += 0.5                  # pretend the pre-period drifts
    placebo = ec.placebo_dates(table, storms, n_draws=25, seed=3)
    assert ec.pretrend_test(bent, placebo)["p_placebo"] < 0.05


def test_key_storms_and_with_without(fake):
    table, storms = fake
    ids = ec.key_storm_ids(storms)
    onsets = storms.set_index("storm_id").loc[ids, "main_onset"]
    assert ((onsets.dt.year == 2024) & onsets.dt.month.isin([5, 10])).all()
    ww = ec.with_without(table, storms, n_boot=99)
    assert len(ww) == 2 and (ww["est"] < 0).all()
    assert ww["storms_used"].iloc[1] <= ww["storms_used"].iloc[0] - len(ids) + 1


def test_short_run_effect_without_bootstrap(fake):
    res, _ = es.event_study(*fake, n_boot=0)
    sr = ec.short_run_effect(res)
    assert sr["est"] < 0 and np.isnan(sr["se"])
