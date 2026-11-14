"""Tests for the sensitivity helpers (src/sensitivity.py). Run with:  pytest"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import physics  # noqa: E402
from src import sensitivity as sv  # noqa: E402


def test_event_bin_labels_whole_days():
    day = pd.Series([-10.0, -3.2, -2.5, -2.0, -0.1, 0.0, 0.9, 1.0, 2.9, 3.0, 20.0, np.nan])
    got = sv.event_bin(day).tolist()
    # -2.5 is whole day -3, -0.1 is whole day -1
    assert got[:9] == ["days -10..-3"] * 3 + ["days -2..-1"] * 2 + ["day 0"] * 2 + ["days 1..2"] * 2
    assert got[9:] == ["days 3..20", "days 3..20", "outside storm windows"]


def test_hampel_timing_counts():
    n = 40
    table = pd.DataFrame({"is_active": 0, "ecc": 0.001, "drag_sensitivity": 1.0, "alt_km": 600.0,
                          "event_day": [0.5] * 20 + [5.0] * 20, "storm_class": "strong",
                          "hampel_outlier": [1] * 10 + [0] * 10 + [1] * 2 + [0] * 18})
    table.loc[0, "alt_km"] = 1100.0           # high-altitude row: outside the main sample
    by_bin, by_class, _ = sv.hampel_timing(table)
    assert by_bin.loc["day 0", "rows"] == 19 and by_bin.loc["day 0", "hampel"] == 9
    assert by_bin.loc["days 3..20", "share_pct"] == 10.0
    assert by_class.loc["strong", "rows"] == n - 1


def test_hampel_setting_passes_k_and_can_switch_off():
    t = pd.date_range("2024-05-01", periods=60, freq="6h")
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 60)
    r[30] = 6.0                                 # 6 robust SDs: flagged at k = 3.5, not at k = 10
    rates = pd.DataFrame({"norad_id": 1, "t_mid": t, "resid_msis21": r})
    base = physics.flag_outliers(rates)["outlier"]
    with sv.hampel_setting(10.0):
        loose = physics.flag_outliers(rates)["outlier"]
    with sv.hampel_setting(None):
        off = physics.flag_outliers(rates)["outlier"]
    assert base.iloc[30] and not loose.iloc[30] and not off.any()
    assert physics.flag_outliers(rates)["outlier"].equals(base)     # patch is undone afterwards


def test_change_uses_parent_when_baseline_is_near_zero():
    assert sv._change(-16.0, -14.0, -14.0) == (100 * -2.0 / 14.0, "own baseline")
    pct, against = sv._change(-0.3, 0.2, -14.0)
    assert against == "parent headline" and abs(pct - (-0.5 / 14.0 * 100)) < 1e-9
