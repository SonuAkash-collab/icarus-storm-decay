"""Tests for the fake-data generator (src/synthetic.py). No data files needed.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import schema  # noqa: E402
from src.synthetic import AP_QUIET, make_synthetic, planted_event_profile  # noqa: E402


@pytest.fixture(scope="module")
def fake():
    return make_synthetic(n_objects=150, seed=0)


def fe_slope(df, cols):
    """OLS slope of Y_residual on cols[0] with object fixed effects (within transform)."""
    g = df["norad_id"].to_numpy()
    X = df[cols].to_numpy(float)
    y = df["Y_residual"].to_numpy()
    X = X - pd.DataFrame(X).groupby(g).transform("mean").to_numpy()
    y = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
    return np.linalg.lstsq(X, y, rcond=None)[0][0]


def with_regressors(table):
    # x = planted effect size per row (true_tau * ap excess), so its true slope is -1
    return table.assign(x=table["true_tau"] * (table["T_ap_int"] / table["dt_days"] - AP_QUIET),
                        season=np.sin(2 * np.pi * table["day_of_year"] / 365.25))


def test_matches_schema(fake):
    table, storms = fake
    assert schema.problems(table) == []
    assert list(table.columns[:len(schema.COLUMNS)]) == list(schema.COLUMNS)
    assert {"true_tau", "true_effect", "true_manoeuvre", "true_artifact"} <= set(table.columns)
    assert len(storms) == 20


def test_reproducible():
    a, _ = make_synthetic(n_objects=10, seed=3)
    b, _ = make_synthetic(n_objects=10, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_intervals_never_bridge_onset(fake):
    table, storms = fake
    assert not schema.bridges_onset(table, storms).any()


def test_storms_come_with_high_f107(fake):
    table, _ = fake
    storm = table.groupby("T_storm")["f107"].mean()
    assert storm[1] > storm[0] + 3


def test_effect_fades_with_altitude(fake):
    table, _ = fake
    low = table[table["alt_km"] < 500]
    high = table[table["alt_km"] > 700]
    assert (low["true_tau"] / low["drag_sensitivity"]).mean() > 5 * (high["true_tau"] / high["drag_sensitivity"]).mean()


def test_event_profile_peaks_at_day_0(fake):
    profile = planted_event_profile(fake[0])
    assert profile.idxmin() == 0
    assert profile.loc[-10:-1].abs().max() < 0.1 * abs(profile[0])


def test_adjusted_regression_recovers_effect_and_naive_is_fooled(fake):
    table = with_regressors(schema.main_sample(fake[0]))
    adjusted = fe_slope(table, ["x", "f107", "f107_81d", "season"])
    naive = fe_slope(table, ["x"])
    assert adjusted == pytest.approx(-1.0, abs=0.25)
    assert naive < adjusted - 0.15            # Sun UV makes the naive effect too big


def test_tle_artifacts_bias_unfiltered_sample(fake):
    table = fake[0]
    report = schema.sample_report(table).set_index("filter")
    assert 0.02 < report.loc["ecc >= 0.01", "share"] < 0.2
    # Bad fits are more common inside storms
    assert (1 - table.groupby("T_storm")["tle_ok"].mean()).diff().iloc[-1] > 0.04
    cols = ["x", "f107", "f107_81d", "season"]
    clean = fe_slope(with_regressors(schema.main_sample(table)), cols)
    dirty = fe_slope(with_regressors(table[table["is_active"] == 0]), cols)
    assert dirty < clean - 0.2


def test_operator_offset(fake):
    table, storms = fake
    truth = table.attrs["truth"]
    assert truth["offset_ratio"] == pytest.approx(0.3 + 0.7 * 0.5)
    passive = table[table["is_active"] == 0]
    assert (passive["true_manoeuvre"] == 0).all()
    active = table[table["is_active"] == 1]
    during = active[active["T_storm"] == 1]
    # Ducking: operators cancel duck_share of the storm effect while it lasts
    assert (during["true_manoeuvre"] / during["true_effect"]).median() == pytest.approx(-0.3)
