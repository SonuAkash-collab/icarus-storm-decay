"""Tests for the space-weather table (src/omni_io.py) and storm catalog (src/storms.py).

The first tests use the real OMNI2 files (downloaded into data/raw/ on first run).
Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.omni_io import PERIOD_END, PERIOD_START, build_spaceweather  # noqa: E402
from src.storms import find_dips, storm_catalog, storm_class  # noqa: E402

FILL_VALUES = [99, 999, 999.9, 99999]


@pytest.fixture(scope="module")
def spaceweather(tmp_path_factory):
    out = tmp_path_factory.mktemp("sw") / "spaceweather.parquet"
    return build_spaceweather(out_path=out).set_index("time")


@pytest.fixture(scope="module")
def storms(spaceweather):
    return storm_catalog(spaceweather["dst"])


# --- space-weather table ---------------------------------------------------

def test_one_row_per_hour_over_full_period(spaceweather):
    expected = pd.date_range(PERIOD_START, PERIOD_END, freq="h")
    assert spaceweather.index.equals(expected)


def test_no_fill_values_or_nan_remain(spaceweather):
    cols = ["ap", "dst", "kp", "f107", "f107_81d"]
    assert not spaceweather[cols].isna().any().any()
    for col in cols:
        assert not spaceweather[col].isin(FILL_VALUES).any(), col


def test_f107_in_physical_range(spaceweather):
    assert spaceweather["f107"].between(60, 350).all()
    assert spaceweather["f107_81d"].between(60, 350).all()


def test_kp_is_stored_on_its_own_scale(spaceweather):
    # Kp comes in thirds (0, 0.3, 0.7, 1.0, ...) up to 9; it is never averaged
    assert spaceweather["kp"].between(0, 9).all()
    assert np.allclose(spaceweather["kp"] * 3, np.round(spaceweather["kp"] * 3), atol=0.15)


# --- storm catalog on the real data -----------------------------------------

def test_may_2024_storm_is_extreme(storms):
    may = storms[(storms["peak_time"] >= "2024-05-10") & (storms["peak_time"] < "2024-05-12")]
    assert len(may) == 1
    assert may["min_dst"].iloc[0] < -350
    assert may["class"].iloc[0] == "extreme"


def test_oct_2024_storm_is_present(storms):
    oct_ = storms[(storms["peak_time"] >= "2024-10-10") & (storms["peak_time"] < "2024-10-12")]
    assert len(oct_) == 1
    assert oct_["min_dst"].iloc[0] < -200
    assert oct_["main_onset"].iloc[0] >= pd.Timestamp("2024-10-10")


def test_storms_at_least_3_days_apart(storms):
    gaps = storms["peak_time"].diff().dropna()
    assert (gaps >= pd.Timedelta(days=3)).all()
    # Storm windows do not overlap: each storm has recovered before the next one starts
    assert (storms["recovery_end"].iloc[:-1].to_numpy() <= storms["onset_time"].iloc[1:].to_numpy()).all()


def test_time_order_within_each_storm(storms):
    assert storms["recovery_end"].notna().all()
    assert (storms["onset_time"] < storms["peak_time"]).all()
    assert (storms["onset_time"] <= storms["main_onset"]).all()
    assert (storms["main_onset"] < storms["peak_time"]).all()
    assert (storms["peak_time"] < storms["recovery_end"]).all()


# --- storm logic on simulated Dst -------------------------------------------

def make_dst(dips, n_days=20):
    """Quiet Dst (-10 nT) with V-shaped dips: list of (start hour, depth nT, length h)."""
    t = pd.date_range("2024-01-01", periods=24 * n_days, freq="h")
    dst = np.full(len(t), -10.0)
    for start, depth, length in dips:
        half = length // 2
        shape = np.concatenate([np.linspace(-10, depth, half), np.linspace(depth, -10, length - half)])
        dst[start:start + length] = shape
    return pd.Series(dst, index=t)


def test_dips_closer_than_3_days_are_merged():
    dst = make_dst([(48, -80, 12), (48 + 48, -150, 12)])      # 2 days apart
    assert len(find_dips(dst)) == 1
    cat = storm_catalog(dst)
    assert len(cat) == 1
    assert cat["min_dst"].iloc[0] == -150
    assert cat["class"].iloc[0] == "strong"


def test_dips_further_than_3_days_stay_separate():
    dst = make_dst([(48, -80, 12), (48 + 24 * 5, -150, 12)])  # 5 days apart
    cat = storm_catalog(dst)
    assert len(cat) == 2
    assert list(cat["class"]) == ["moderate", "strong"]


def test_storm_class_boundaries():
    assert storm_class(-60) == "moderate"
    assert storm_class(-100) == "moderate"
    assert storm_class(-150) == "strong"
    assert storm_class(-250) == "severe"
    assert storm_class(-406) == "extreme"
