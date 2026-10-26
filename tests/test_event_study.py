"""Tests for the stacked event study (src/event_study.py), on synthetic data.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import event_study as es  # noqa: E402
from src.schema import main_sample  # noqa: E402
from src.synthetic import make_synthetic, planted_event_profile  # noqa: E402


@pytest.fixture(scope="module")
def fake():
    return make_synthetic(n_objects=150, seed=0)


@pytest.fixture(scope="module")
def result(fake):
    return es.event_study(*fake, n_boot=199)


def test_stacks_drop_rows_inside_other_storms(fake):
    table, storms = fake
    stacked, report = es.build_stacks(main_sample(table), storms)
    windows = es.contamination_windows(storms)
    for sid, g in stacked.groupby("stack"):
        for other, start, end in windows:
            if other != sid:
                assert not ((g["t_mid"] >= start) & (g["t_mid"] <= end)).any()
    assert report["rows_contaminated"].sum() > 0
    assert set(stacked["stack"]) == set(report.loc[report["used"], "storm_id"])


def test_strict_window_is_reported(fake):
    table, storms = fake
    _, active = es.build_stacks(main_sample(table), storms)
    _, strict = es.build_stacks(main_sample(table), storms, contam=(-10, 20))
    assert strict["rows_contaminated"].sum() > active["rows_contaminated"].sum()
    assert strict["used"].sum() <= active["used"].sum()
    assert "storms used" in es.summarise(strict)


def test_event_days_and_reference(result):
    res, _ = result
    assert list(res.index) == list(range(-10, 21))
    assert (res.loc[-1, ["coef", "se_cr1", "se_boot"]] == 0).all()
    assert (res["n_storms"] <= 20).all()


def test_recovers_planted_effect(fake, result):
    res, _ = result
    truth = planted_event_profile(fake[0])
    for k in (0, 1):
        assert res.loc[k, "ci_lo"] < truth[k] - truth[-1] < res.loc[k, "ci_hi"]
    assert res.loc[0, "ci_hi"] < 0                       # day 0 clearly negative
    # Pre-period coefficients are small compared with the day-0 effect
    assert res.loc[-10:-2, "coef"].abs().max() < 0.25 * abs(res.loc[0, "coef"])


def test_controls_recover_sun_uv_term(fake, result):
    controls = result[0].attrs["controls"]
    assert controls["f107"] == pytest.approx(fake[0].attrs["truth"]["gamma"], abs=0.03)


def test_bootstrap_and_cr1_agree(result):
    res = result[0].drop(index=-1)
    ratio = res["se_boot"] / res["se_cr1"]
    assert 0.6 < ratio.median() < 1.4


def test_exclude_storms(fake):
    table, storms = fake
    res, report = es.event_study(table, storms, exclude_storms=[1, 2], n_boot=19)
    assert not report["storm_id"].isin([1, 2]).any()
    assert res.attrs["n_clusters"] == report["used"].sum()


def test_too_few_storms_raises(fake):
    table, storms = fake
    with pytest.raises(ValueError, match="clean storms"):
        es.event_study(table, storms.iloc[:1], n_boot=9)


def test_null_effect_gives_zero():
    table, storms = make_synthetic(n_objects=100, tau0=0.0, seed=1)
    res, _ = es.event_study(table, storms, n_boot=199)
    inside = (res["ci_lo"] <= 0) & (res["ci_hi"] >= 0)
    assert inside.mean() > 0.8
    assert abs(res.loc[0, "coef"]) < 0.3
