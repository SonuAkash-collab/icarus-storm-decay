"""Tests for the real-table builder (src/build_table.py) on a small fake cohort.

The fake cohort is made of real-format TLE lines (python-sgp4 sgp4init + export_tle)
for a few decaying objects, run through Piyush's cleaning and interval code, with
synthetic space weather and planted storms. No data files, no downloads.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sgp4.api import WGS72, Satrec
from sgp4.exporter import export_tle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import build_table as bt  # noqa: E402
from src import schema  # noqa: E402
from src.orbit import EARTH_EQUATORIAL_RADIUS_KM, MU_WGS72, build_decay_intervals, clean_tle_history, tle_to_row  # noqa: E402
from src.simulate import planted_storms, synthetic_spaceweather  # noqa: E402

START = pd.Timestamp("2024-04-01")
N_DAYS = 70
EPOCH_1949 = pd.Timestamp("1949-12-31")


def tle_lines(norad_id, epoch, a_km, inc_deg, raan_deg):
    """One TLE (line1, line2) for a near-circular orbit with semi-major axis a_km at epoch."""
    n = np.sqrt(MU_WGS72 / a_km**3) * 60          # rad/min
    sat = Satrec()
    sat.sgp4init(WGS72, "i", norad_id, (epoch - EPOCH_1949) / pd.Timedelta(days=1), 1e-4, 0.0, 0.0,
                 0.002, 0.0, np.radians(inc_deg), 0.0, n, np.radians(raan_deg))
    return export_tle(sat)


def fake_cohort(seed=0):
    """(intervals, tles, sw, storms) in the format of build_table.load_inputs."""
    rng = np.random.default_rng(seed)
    storms = planted_storms(START + pd.to_timedelta([20.25, 45.25], unit="D"))
    sw = synthetic_spaceweather(START, START + pd.Timedelta(days=N_DAYS + 1), storms, rng)
    objects = [(90001, 450.0, 51.6, 60.0, "main"), (90002, 550.0, 98.0, 20.0, "main"),
               (90003, 650.0, 82.0, 10.0, "main"), (90004, 1100.0, 74.0, 2.0, "high_altitude")]
    rows = []
    for norad_id, alt, inc, rate_m, group in objects:
        days = np.sort(rng.uniform(0, N_DAYS, int(N_DAYS * 1.5)))
        for d in days:
            epoch = START + pd.Timedelta(days=float(d))
            a = EARTH_EQUATORIAL_RADIUS_KM + alt - rate_m * d / 1000 + rng.normal(0, 0.01)
            l1, l2 = tle_lines(norad_id, epoch, a, inc, 0.0)
            rows.append({**tle_to_row(l1, l2), "line1": l1, "line2": l2, "cohort_group": group})
    tles = pd.DataFrame(rows)
    tles, _ = clean_tle_history(tles)
    parts = []
    for group, g in tles.groupby("cohort_group"):
        iv, _ = build_decay_intervals(g, storms)
        parts.append(iv.assign(cohort_group=group))
    return pd.concat(parts, ignore_index=True), tles, sw, storms


@pytest.fixture(scope="module")
def cohort():
    return fake_cohort()


@pytest.fixture(scope="module")
def built(cohort, tmp_path_factory):
    chunk_dir = tmp_path_factory.mktemp("chunks")
    table, drag, gate = bt.build_table(inputs=cohort, chunk_dir=chunk_dir, save=False, log=None)
    return table, drag, gate, chunk_dir


def test_schema_columns(built):
    table = built[0]
    assert schema.problems(table) == []
    assert list(table.columns[:len(schema.COLUMNS)]) == list(schema.COLUMNS)
    for col in ["Y_total", "climb_flag", "hampel_outlier", "drag_source", "drag_unstable", "role"]:
        assert col in table
    assert (table["is_active"] == 0).all()
    assert not any(f in c.lower() for c in table.columns for f in schema.FORBIDDEN)


def test_no_interval_across_an_onset(built, cohort):
    storms = cohort[3]
    assert not schema.bridges_onset(built[0], storms).any()


def test_bridging_interval_is_refused(cohort, tmp_path):
    intervals, tles, sw, storms = cohort
    onset = storms["main_onset"].iloc[0]
    bad = intervals.iloc[[0]].assign(t_start=onset - pd.Timedelta(hours=6), t_end=onset + pd.Timedelta(hours=6),
                                     t_mid=onset, dt_days=0.5)
    with pytest.raises(ValueError, match="main_onset"):
        bt.check_no_bridges(pd.concat([intervals, bad]), storms)


def test_event_day_range_and_link(built, cohort):
    table, storms = built[0], cohort[3]
    linked = table["storm_id"].notna()
    lo, hi = schema.EVENT_WINDOW
    assert linked.any() and (~linked).any()
    assert table.loc[linked, "event_day"].between(lo, hi).all()
    assert table.loc[~linked, "event_day"].isna().all()
    onset = table.loc[linked, "storm_id"].map(storms.set_index("storm_id")["main_onset"])
    day = (table.loc[linked, "t_mid"] - onset).dt.total_seconds() / 86400
    assert np.allclose(day, table.loc[linked, "event_day"])


def test_climbs_kept_and_droppable(built, cohort, tmp_path):
    table = built[0]
    climbs = table["climb_flag"] == 1
    assert climbs.any()                                   # noise makes some passive climbs
    # Climbs stay in the main sample unless the Hampel filter flags them
    assert (table.loc[climbs & (table["hampel_outlier"] == 0), "tle_ok"] == 1).all()
    no_climbs, _, _ = bt.build_table(drop_climbs=True, inputs=cohort, chunk_dir=built[3], save=False, log=None)
    assert (no_climbs["climb_flag"] == 0).all()
    assert len(no_climbs) == (~climbs).sum()


def test_high_altitude_convention(built):
    table, drag = built[0], built[1]
    high = table["cohort_group"] == "high_altitude"
    assert high.any()
    assert table.loc[high, "Y_log_ratio"].isna().all()
    assert table.loc[~high, "Y_log_ratio"].notna().any()
    # Fit kept when stable, cohort median of the main objects when not
    src = table.loc[high, "drag_source"].iloc[0]
    hid = table.loc[high, "norad_id"].iloc[0]
    assert src == ("cohort_median" if drag.loc[hid, "unstable"] else "fit")
    if src == "cohort_median":
        main_ids = table.loc[~high, "norad_id"].unique()
        assert np.isclose(table.loc[high, "drag_sensitivity"].iloc[0], drag.loc[main_ids, "B"].median())


def test_unstable_high_alt_falls_back(cohort, tmp_path):
    """Too few quiet fit intervals at high altitude -> cohort median, marked as such."""
    intervals, tles, sw, storms = cohort
    high_id = intervals.loc[intervals["cohort_group"] == "high_altitude", "norad_id"].iloc[0]
    keep = (intervals["norad_id"] != high_id) | (intervals["t_start"] < START + pd.Timedelta(days=12))
    drivers = bt.cohort_drivers(tles, intervals[keep].rename(columns={"alt_km": "alt_km_tle"}), sw, storms,
                                chunk_dir=tmp_path, log=None)
    rates, drag = bt.cohort_physics(drivers, storms)
    assert drag.loc[high_id, "unstable"] == 1 and drag.loc[high_id, "drag_source"] == "cohort_median"
    main_ids = drag.index[drag.index != high_id]
    assert np.isclose(rates.loc[rates["norad_id"] == high_id, "B_msis21"].iloc[0], drag.loc[main_ids, "B"].median())


def test_drag_positive_and_total_effect(built):
    table = built[0]
    main = table[table["cohort_group"] == "main"]
    assert (main["drag_sensitivity"] > 0).all()
    # Outside every event window the quiet-ap prediction is the actual-ap one
    out = main["storm_id"].isna()
    assert np.allclose(main.loc[out, "Y_total"], main.loc[out, "Y_residual"])


def test_cache_resumes(built, cohort):
    chunk_dir = built[3]
    assert len(list(Path(chunk_dir).glob("chunk_*_drivers.parquet"))) == 1
    stamp = {p: p.stat().st_mtime for p in Path(chunk_dir).glob("*.parquet")}
    again, _, _ = bt.build_table(inputs=cohort, chunk_dir=chunk_dir, save=False, log=None)
    assert {p: p.stat().st_mtime for p in Path(chunk_dir).glob("*.parquet")} == stamp
    pd.testing.assert_frame_equal(again, built[0])


def test_drag_bootstrap_covers_truth():
    """Known B, noisy observed decay: the bootstrap interval covers B; few intervals -> unstable."""
    rng = np.random.default_rng(1)
    rows = []
    for norad_id, n in [(1, 300), (2, 10)]:
        t = START + pd.to_timedelta(np.arange(n) * 1.0, unit="D")
        driver = -rng.uniform(0.5, 1.5, n)
        rows.append(pd.DataFrame({"norad_id": norad_id, "t_mid": t, "dt_days": 1.0, "role": "fit",
                                  "driver_msis21": driver,
                                  "da_dt_km_per_day": 0.02 * driver + rng.normal(0, 0.002, n)}))
    drag = bt.drag_bootstrap(pd.concat(rows, ignore_index=True))
    assert drag.loc[1, "ci_lo"] < 0.02 < drag.loc[1, "ci_hi"]
    assert drag.loc[1, "unstable"] == 0
    assert drag.loc[2, "unstable"] == 1
