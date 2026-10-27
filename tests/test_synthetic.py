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
from src.synthetic import AP_QUIET, HIGH_ALT_KM, make_synthetic, planted_event_profile  # noqa: E402


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
    assert passive["drag_sensitivity"].notna().all()
    assert table.loc[table["is_active"] == 1, "drag_sensitivity"].isna().any()
    active = table[table["is_active"] == 1]
    during = active[active["T_storm"] == 1]
    # Ducking: operators cancel duck_share of the storm effect while it lasts
    assert (during["true_manoeuvre"] / during["true_effect"]).median() == pytest.approx(-0.3)


@pytest.fixture(scope="module")
def with_high_alt():
    return make_synthetic(n_objects=60, n_high_alt=15, return_hourly=True, seed=4)


def test_high_alt_group(with_high_alt):
    table, _, _ = with_high_alt
    high = table[table["alt_km"] >= HIGH_ALT_KM[0]]
    objects = high.groupby("norad_id").first()
    assert len(objects) == 15
    assert objects["alt_km"].between(*HIGH_ALT_KM).all()
    assert (objects["ecc"] < schema.MAX_ECC).all() and (objects["is_active"] == 0).all()
    assert (high["true_manoeuvre"] == 0).all() and (high["true_station_keeping"] == 0).all()
    # Storm effect about zero up there, clearly negative lower down
    storm = table[table["T_storm"] == 1]
    assert abs(storm.loc[storm["alt_km"] >= 1000, "true_effect"].mean()) < 0.05
    assert storm.loc[storm["alt_km"] < 1000, "true_effect"].mean() < -0.3
    assert len(schema.main_sample(table)) > 0 and (schema.main_sample(table)["alt_km"] < 1000).all()
    assert len(schema.high_alt_sample(table)) > 0.9 * len(high)
    assert high["Y_log_ratio"].isna().all()
    assert schema.problems(table) == []


def test_high_alt_leaves_other_objects_unchanged(with_high_alt):
    table, storms, _ = with_high_alt
    base, base_storms = make_synthetic(n_objects=60, seed=4)
    pd.testing.assert_frame_equal(table[table["norad_id"] < 90060].reset_index(drop=True), base)
    pd.testing.assert_frame_equal(storms, base_storms)


def test_hourly_frame_like_load_spaceweather(with_high_alt):
    table, _, hourly = with_high_alt
    assert list(hourly.columns) == ["ap", "f107", "f107_81d"]
    assert hourly.index.name == "time"
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert hourly.notna().all().all()
    assert "hourly" not in table.attrs                 # kept out of attrs (save_table writes them)
    # Interval-mean ap of the table agrees with the hourly ap it was built from
    row = table[table["T_storm"] == 1].iloc[0]
    start = row["t_mid"] - pd.Timedelta(days=row["dt_days"] / 2)
    end = row["t_mid"] + pd.Timedelta(days=row["dt_days"] / 2)
    inside = hourly.loc[start.floor("h"):end.ceil("h") - pd.Timedelta(hours=1), "ap"]
    assert row["T_ap_int"] / row["dt_days"] == pytest.approx(inside.mean(), rel=0.3)
    day = row["t_mid"].floor("D")
    assert hourly.loc[day, "f107"] == pytest.approx(row["f107"])
