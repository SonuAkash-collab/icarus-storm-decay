"""Tests for the shared analysis-table schema (src/schema.py).

No data files needed. Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import schema  # noqa: E402

STORMS = pd.DataFrame({
    "storm_id": [1, 2],
    "main_onset": pd.to_datetime(["2024-05-10 17:00", "2024-05-25 00:00"]),
    "class": ["extreme", "moderate"],
    "recovery_end": pd.to_datetime(["2024-05-13 06:00", "2024-05-26 00:00"]),
})


def tiny_table():
    """Three valid intervals of one object around the May 2024 storm."""
    t_start = pd.to_datetime(["2024-05-01 00:00", "2024-05-10 17:00", "2024-05-11 17:00"])
    t_end = t_start + pd.Timedelta(days=1)
    df = pd.DataFrame({
        "norad_id": 10114,
        "t_mid": t_start + pd.Timedelta(hours=12),
        "dt_days": 1.0,
        "alt_km": 600.0,
        "inc_deg": 98.0,
        "local_solar_time": 10.5,
        "day_of_year": [122, 131, 132],
        "ecc": 0.002,
        "tle_ok": 1,
        "is_active": 0,
        "Y_residual": [0.5, -20.0, -15.0],
        "Y_residual_nrl": [0.8, -18.0, -14.0],
        "Y_log_ratio": [0.0, 0.3, 0.2],
        "T_ap_int": [5.0, 150.0, 80.0],
        "f107": 200.0,
        "f107_81d": 180.0,
        "drag_sensitivity": 0.01,
    })
    return pd.concat([df, schema.link_storms(t_start, t_end, STORMS)], axis=1)


def test_columns_match_plan():
    assert list(schema.empty_frame().columns) == list(schema.COLUMNS)
    assert set(schema.CONFOUNDERS) <= set(schema.COLUMNS)
    assert not any("bstar" in c.lower() for c in schema.COLUMNS)


def test_valid_table_passes():
    df = schema.validate(tiny_table())
    assert schema.problems(df) == []
    assert df["storm_id"].dtype == "Int64"


def test_link_storms():
    links = schema.link_storms(pd.to_datetime(["2024-04-01 00:00", "2024-05-10 12:00", "2024-05-11 12:00"]),
                               pd.to_datetime(["2024-04-02 00:00", "2024-05-11 12:00", "2024-05-12 12:00"]),
                               STORMS)
    assert links["storm_id"].isna().iloc[0] and links["T_storm"].iloc[0] == 0
    assert links["storm_id"].iloc[1] == 1 and links["T_storm"].iloc[1] == 1
    assert links["storm_class"].iloc[1] == "extreme"
    # Midpoint 2024-05-11 00:00 is 7 h after main_onset
    assert links["event_day"].iloc[1] == pytest.approx(7 / 24)


def test_nearest_storm_wins():
    # 2024-05-20: 9.3 days after storm 1, 5 days before storm 2
    links = schema.link_storms(pd.to_datetime(["2024-05-19 12:00"]),
                               pd.to_datetime(["2024-05-20 12:00"]), STORMS)
    assert links["storm_id"].iloc[0] == 2
    assert links["event_day"].iloc[0] == pytest.approx(-5.0)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.drop(columns="Y_residual"), "missing columns"),
    (lambda d: d.assign(BSTAR=1e-4), "forbidden"),
    (lambda d: d.assign(rho_msis21=1e-12), "forbidden"),
    (lambda d: d.assign(dt_days=0.0), "dt_days"),
    (lambda d: d.assign(is_active=2), "is_active"),
    (lambda d: d.assign(event_day=[np.nan, 25.0, 1.0]), "event_day"),
    (lambda d: d.assign(storm_class=pd.array([pd.NA, "huge", "extreme"], dtype="string")), "storm_class"),
    (lambda d: d.assign(f107=np.nan), "nulls"),
    (lambda d: d.assign(tle_ok=-1), "tle_ok"),
    (lambda d: d.assign(drag_sensitivity=np.nan), "drag_sensitivity"),
    (lambda d: d.assign(ecc=1.5), "ecc"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate"),
])
def test_bad_tables_fail(change, message):
    with pytest.raises(ValueError, match=message):
        schema.validate(change(tiny_table()))


def test_main_sample_filters():
    df = tiny_table().assign(ecc=[0.002, 0.02, 0.002], tle_ok=[1, 1, 0])
    assert len(schema.main_sample(df)) == 1
    report = schema.sample_report(df).set_index("filter")["removed"]
    assert report["ecc >= 0.01"] == 1 and report["TLE failed cleaning"] == 1 and report["kept"] == 1


def test_active_may_lack_drag_sensitivity():
    df = tiny_table().assign(is_active=1, drag_sensitivity=[np.nan, 0.01, 0.01])
    assert schema.problems(df) == []
    report = schema.sample_report(df.assign(is_active=[1, 0, 0])).set_index("filter")["removed"]
    assert report["active satellite"] == 1 and report["no drag_sensitivity"] == 0


def test_bridges_onset():
    df = tiny_table()
    assert not schema.bridges_onset(df, STORMS).any()
    # 2024-05-10 12:00 to 2024-05-11 12:00 contains the 17:00 onset of storm 1
    df = df.assign(t_mid=pd.to_datetime(["2024-05-01 12:00", "2024-05-11 00:00", "2024-05-12 00:00"]))
    assert schema.bridges_onset(df, STORMS).tolist() == [False, True, False]


def test_extra_columns_allowed():
    df = schema.validate(tiny_table().assign(true_effect=0.0))
    assert df.columns[-1] == "true_effect"


def test_save_load_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "table.parquet"
    saved = schema.save_table(tiny_table(), path)
    pd.testing.assert_frame_equal(schema.load_table(path), saved)
