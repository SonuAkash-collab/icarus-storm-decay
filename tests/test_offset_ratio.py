"""Tests for the operator offset ratio (src/offset_ratio.py), on synthetic data.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import offset_ratio as orr  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def fake():
    # More objects, more actives and less noise than the default, so ~5 clean storms suffice
    return make_synthetic(n_objects=400, active_share=0.25, noise=1.0, seed=0)


@pytest.fixture(scope="module")
def noise_free(fake):
    table, storms = fake
    y = table["true_effect"] + table["true_manoeuvre"] + table["true_station_keeping"]
    return table.assign(Y_residual=y), storms


def test_noise_free_recovers_planted_shares(fake, noise_free):
    truth = fake[0].attrs["truth"]
    pairs, _ = orr.storm_pairs(*noise_free)
    got = orr.shares(pairs)
    assert got["ducking_share"] == pytest.approx(truth["duck_share"], abs=0.07)
    assert got["recovery_share"] == pytest.approx(truth["recovery_share"], abs=0.07)
    assert got["offset_ratio"] == pytest.approx(truth["offset_ratio"], abs=0.1)


def test_station_keeping_is_subtracted(noise_free):
    table, storms = noise_free
    without = table.assign(Y_residual=table["Y_residual"] - table["true_station_keeping"])
    a = orr.shares(orr.storm_pairs(table, storms)[0])
    b = orr.shares(orr.storm_pairs(without, storms)[0])
    pd.testing.assert_series_equal(a, b, atol=1e-9)


def test_noisy_estimate_and_interval(fake):
    truth = fake[0].attrs["truth"]
    summary, pairs, smd = orr.offset_ratio(*fake, n_boot=199)
    assert summary.loc["ducking_share", "est"] == pytest.approx(truth["duck_share"], abs=0.1)
    assert summary.loc["offset_ratio", "ci_lo"] < truth["offset_ratio"] < summary.loc["offset_ratio", "ci_hi"]
    assert summary.attrs["n_storms"] >= 3 and summary.attrs["n_actives"] >= 10
    # Sun UV part removed before the twins' loss is measured
    assert summary.attrs["controls"]["f107"] == pytest.approx(truth["gamma"], abs=0.02)


def test_matching_balance(fake):
    _, pairs, smd = orr.offset_ratio(*fake, n_boot=9)
    assert (smd.abs() < 0.1).all()
    assert pairs["n_twins"].between(1, orr.N_TWINS).all()


def test_match_respects_caliper():
    pool = pd.DataFrame({"alt_km": [500.0, 510, 900], "inc_deg": [53.0, 53, 53], "log_drag": [-4.6, -4.6, -4.6]},
                        index=[1, 2, 3])
    act = pd.DataFrame({"alt_km": [505.0], "inc_deg": [53.0], "log_drag": [-4.6]}, index=[99])
    pairs = orr.match(act, pool, n_twins=3, caliper=0.5)
    assert set(pairs["twin"]) == {1, 2}


def test_debris_placebo_noise_free_is_zero(noise_free):
    summary, _, _ = orr.debris_placebo(*noise_free, n_boot=0)
    assert abs(summary.loc["offset_ratio", "est"]) < 0.1


def test_debris_placebo_centres_on_zero(fake):
    dist = orr.placebo_distribution(*fake, n_reps=8)
    assert abs(dist["offset_ratio"].mean()) < 0.15
    assert abs(dist["ducking_share"].mean()) < 0.1
    # The real ducking share is far outside the placebo spread
    real = orr.offset_ratio(*fake, n_boot=9)[0].loc["ducking_share", "est"]
    assert real > dist["ducking_share"].mean() + 2 * dist["ducking_share"].std()


def test_outcome_argument(fake):
    table, storms = fake
    renamed = table.rename(columns={"Y_residual": "Y_total"}).assign(Y_residual=0.0)
    a = orr.offset_ratio(table, storms, n_boot=9)[0]
    b = orr.offset_ratio(renamed, storms, outcome="Y_total", n_boot=9)[0]
    pd.testing.assert_frame_equal(a, b)
    assert b.attrs["outcome"] == "Y_total"


def test_recovery_sensitivity(fake, noise_free):
    out = orr.recovery_sensitivity(*noise_free, n_boot=9)
    assert set(out.index.get_level_values("recovery_days")) == {10, 5}
    truth = fake[0].attrs["truth"]
    # Ducking share does not depend on the recovery window; a 5-day window counts
    # about half of the evenly spread 10-day climb
    assert out.loc[(5, "ducking_share"), "est"] == pytest.approx(truth["duck_share"], abs=0.07)
    assert out.loc[(5, "recovery_share"), "est"] == pytest.approx(truth["recovery_share"] / 2, abs=0.07)
    assert out.xs(5)["n_storms"].iloc[0] >= out.xs(10)["n_storms"].iloc[0]
