"""Tests for the A7 precise-orbit parser and comparison."""

import gzip
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.precise_orbit import (  # noqa: E402
    compare_decay_rates,
    compare_tle_precise,
    parse_groops_orbit,
)


def test_parse_groops_orbit_reads_mjd_position_and_velocity(tmp_path):
    path = tmp_path / "orbit.txt.gz"
    row = " 60441.000000000000000000  7000000 0 0 0 7546.053290107542 0 0 0 0\n"
    with gzip.open(path, "wt") as stream:
        stream.write("# ORBIT\n")
        stream.write(row)
    parsed = parse_groops_orbit(path)
    assert len(parsed) == 1
    assert parsed.iloc[0]["time"] == pd.Timestamp("2024-05-11")
    assert parsed.iloc[0]["a_km_precise"] > 6900


def test_compare_tle_precise_returns_trend_metrics():
    times = pd.date_range("2024-05-01", periods=5, freq="D")
    tles = pd.DataFrame({"epoch": times, "a_km": [7000, 6999, 6998, 6997, 6996]})
    precise = pd.DataFrame({
        "time": times,
        "a_km_precise": [7003, 7002, 7001, 7000, 6999],
    })
    aligned, summary = compare_tle_precise(tles, precise)
    assert len(aligned) == 5
    assert summary.iloc[0]["centred_correlation"] > 0.99
    assert abs(summary.iloc[0]["slope_difference_km_per_day"]) < 1e-9


def test_compare_decay_rates_returns_lag_sweep():
    times = pd.date_range("2024-05-01", periods=96 * 5, freq="15min")
    precise = pd.DataFrame({
        "time": times,
        "a_km_precise": 7000 - 0.5 * (times - times[0]).total_seconds() / 86400,
    })
    tles = pd.DataFrame({
        "epoch": pd.date_range("2024-05-01", periods=5),
        "a_km": [7000, 6999.5, 6999, 6998.5, 6998],
    })
    rates, lags = compare_decay_rates(tles, precise, period_minutes=94)
    assert len(rates) >= 3
    assert set(lags["lag_days"]) == set(range(-3, 4))
