"""Tests for src/orbit.py. Run with:  pytest"""

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.orbit import decay_rates, drop_duplicate_epochs, tle_to_row  # noqa: E402

# ISS TLE (the example used in the python-sgp4 documentation)
L1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
L2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"


def test_semi_major_axis_matches_hand_calculation():
    row = tle_to_row(L1, L2)
    # Hand calculation: Brouwer mean motion from sgp4 is n = 0.0676360233 rad/min.
    # n = 0.0676360233 / 60 = 1.127267e-3 rad/s;  mu (WGS-72) = 398600.8 km^3/s^2
    # a = (mu / n^2)^(1/3) = 6794.563 km
    n = 0.06763602333248933 / 60.0
    a_hand = (398600.8 / n**2) ** (1 / 3)
    assert math.isclose(a_hand, 6794.563, abs_tol=0.001)
    assert abs(row["a_km"] - a_hand) < 0.01
    assert math.isclose(row["ecc"], 0.0007417)


def test_epoch_is_parsed():
    row = tle_to_row(L1, L2)
    # Day 343.69339541 of 2019 = 9 Dec 2019, 16:38:29.363 UTC (0.69339541 * 86400 s = 59909.363 s)
    assert abs(row["epoch"] - pd.Timestamp("2019-12-09 16:38:29.363")) < pd.Timedelta("1ms")


def _toy_frame():
    """Three TLE rows for one object; the first epoch appears twice."""
    t = pd.Timestamp("2024-05-01")
    return pd.DataFrame({
        "norad_id": [1, 1, 1, 1],
        "epoch": [t, t, t + pd.Timedelta(hours=12), t + pd.Timedelta(days=1)],
        "created": [t, t + pd.Timedelta(hours=1), t + pd.Timedelta(hours=13), t + pd.Timedelta(days=1)],
        "a_km": [6900.0, 6899.0, 6898.5, 6898.0],
    })


def test_duplicate_removal_keeps_one_per_epoch_latest():
    clean = drop_duplicate_epochs(_toy_frame())
    assert not clean.duplicated(["norad_id", "epoch"]).any()
    assert len(clean) == 3
    # The later-published duplicate (a = 6899.0) must be the one kept
    assert clean.iloc[0]["a_km"] == 6899.0


def test_near_duplicate_epochs_are_merged():
    """Re-issued TLEs a few milliseconds apart count as the same epoch."""
    df = _toy_frame()
    df.loc[1, "epoch"] = df.loc[1, "epoch"] + pd.Timedelta(milliseconds=5)
    clean = drop_duplicate_epochs(df)
    assert len(clean) == 3
    assert clean.iloc[0]["a_km"] == 6899.0


def test_no_interval_has_nonpositive_dt():
    rates = decay_rates(drop_duplicate_epochs(_toy_frame()))
    assert (rates["dt_days"] > 0).all()
    # 6899.0 -> 6898.5 over 0.5 day = -1 km/day
    assert math.isclose(rates.iloc[0]["da_dt_km_per_day"], -1.0)
