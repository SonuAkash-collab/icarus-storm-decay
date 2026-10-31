"""Tests for src/objects.py. Run with:  pytest

Inline fixture records, no network. The two download tests at the end skip without
network access (and, for DISCOSweb, without DISCOS_TOKEN in .env).
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import requests

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import objects  # noqa: E402
from src.orbit import decay_rates  # noqa: E402
from src.schema import OBJECT_COLUMNS, validate_objects  # noqa: E402

# Trimmed CelesTrak SATCAT JSON records (field names as in records.php?FORMAT=JSON)
SATCAT = {
    41341: {"OBJECT_NAME": "CZ-4B R/B", "NORAD_CAT_ID": 41341, "OBJECT_TYPE": "R/B",
            "LAUNCH_DATE": "2016-03-30", "DECAY_DATE": None},
    22285: {"OBJECT_NAME": "SL-16 R/B", "NORAD_CAT_ID": 22285, "OBJECT_TYPE": "R/B",
            "LAUNCH_DATE": "1992-11-17", "DECAY_DATE": ""},
    30000: {"OBJECT_NAME": "FENGYUN 1C DEB", "NORAD_CAT_ID": 30000, "OBJECT_TYPE": "DEB",
            "LAUNCH_DATE": "1999-05-10", "DECAY_DATE": "2024-09-01"},
    40001: {"OBJECT_NAME": "SAT A", "NORAD_CAT_ID": 40001, "OBJECT_TYPE": "PAY",
            "LAUNCH_DATE": "2014-06-01", "DECAY_DATE": None},
    40002: {"OBJECT_NAME": "SAT B", "NORAD_CAT_ID": 40002, "OBJECT_TYPE": "PAY",
            "LAUNCH_DATE": "2014-06-01", "DECAY_DATE": None},
    40003: {"OBJECT_NAME": "SAT C", "NORAD_CAT_ID": 40003, "OBJECT_TYPE": "PAY",
            "LAUNCH_DATE": "2015-01-01", "DECAY_DATE": None},
}
# Trimmed DISCOSweb object attributes (mass in kg, cross-sections in m^2)
DISCOS = {
    41341: {"satno": 41341, "mass": 2000.0, "xSectAvg": 12.0, "xSectMin": 7.0, "xSectMax": 20.0},
    22285: {"satno": 22285, "mass": 9000.0, "xSectAvg": 36.0},
    30000: {"satno": 30000, "mass": None, "xSectAvg": 0.05},
    40001: {"satno": 40001, "mass": 0.0, "xSectAvg": 2.0},
    40002: None,
}


def _intervals(norad_id, step_day=None, step_km=1.0, spike_day=None, storm_day=None, days=40, seed=0):
    """Decay intervals of a synthetic object: TLE every 8 h, 50 m/day decay, 20 m TLE
    scatter, optional lasting rise (maneuver), single outlier TLE (spike) or 2-day storm
    with 1 km/day extra decay."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2024-05-01") + pd.to_timedelta(np.arange(0, days, 1 / 3), unit="D")
    day = (t - t[0]).total_seconds().to_numpy() / 86400
    a = 7000.0 - 0.05 * day + rng.normal(0, 0.02, len(t))
    if storm_day is not None:
        a -= np.clip(day - storm_day, 0, 2) * 1.0
    if step_day is not None:
        a += np.where(day >= step_day, step_km, 0.0)
    if spike_day is not None:
        a[np.argmin(np.abs(day - spike_day))] += 1.0
    tles = pd.DataFrame({"norad_id": norad_id, "epoch": t, "a_km": a})
    return decay_rates(tles)


# --- SATCAT and DISCOSweb parsing ----------------------------------------------

def test_parse_satcat_maps_types_and_dates():
    out = objects.parse_satcat({k: SATCAT[k] for k in [41341, 30000, 40001]}).set_index("norad_id")
    assert out.loc[41341, "object_type"] == "rocket body"
    assert out.loc[30000, "object_type"] == "debris"
    assert out.loc[40001, "object_type"] == "payload"
    assert out.loc[30000, "decay_date"] == pd.Timestamp("2024-09-01")
    assert pd.isna(out.loc[41341, "decay_date"])
    assert out.loc[41341, "launch_date"] == pd.Timestamp("2016-03-30")


def test_missing_satcat_record_is_unknown():
    with pytest.warns(UserWarning, match="No SATCAT record"):
        out = objects.parse_satcat({99999: None})
    assert out["object_type"].iloc[0] == "unknown"


def test_area_to_mass_and_catalog_drag():
    out = objects.add_area_to_mass(objects.parse_discos(DISCOS)).set_index("norad_id")
    assert out.loc[41341, "area_to_mass"] == pytest.approx(12.0 / 2000.0)
    assert out.loc[41341, "drag_sensitivity_catalog"] == pytest.approx(2.2 * 12.0 / 2000.0)
    # Missing or zero mass, or no DISCOSweb object: no area-to-mass
    assert out.loc[[30000, 40001, 40002], "area_to_mass"].isna().all()


# --- maneuvers -----------------------------------------------------------------

def test_detects_lasting_rise():
    events = objects.detect_maneuvers(_intervals(1, step_day=20))
    assert len(events) == 1
    assert abs(events["t_maneuver"].iloc[0] - pd.Timestamp("2024-05-21")) <= pd.Timedelta(hours=8)
    assert events["rise_km"].iloc[0] == pytest.approx(1.0, abs=0.2)


def test_ignores_decay_noise_and_single_outlier():
    assert objects.detect_maneuvers(_intervals(1)).empty
    assert objects.detect_maneuvers(_intervals(1, spike_day=15)).empty


def test_storm_is_not_a_maneuver():
    assert objects.detect_maneuvers(_intervals(1, storm_day=15)).empty
    # A maneuver a week after the storm is still found
    events = objects.detect_maneuvers(_intervals(1, storm_day=15, step_day=24))
    assert len(events) == 1 and events["t_maneuver"].iloc[0].day == 25


def test_small_rise_below_threshold_is_not_a_maneuver():
    # 0.1 km lasting rise: above the 20 m scatter, below the 0.2 km floor
    assert objects.detect_maneuvers(_intervals(1, step_day=20, step_km=0.1)).empty


def test_gap_in_intervals_is_not_a_maneuver():
    # A missing interval (e.g. one bridging a storm onset) hides a 1 km rise: unknown, not flagged
    iv = _intervals(1, step_day=20)
    across = (iv["t_start"] < pd.Timestamp("2024-05-21")) & (iv["t_end"] >= pd.Timestamp("2024-05-21"))
    assert across.sum() == 1
    assert objects.detect_maneuvers(iv[~across]).empty


def test_classify_activity():
    iv = pd.concat([_intervals(40001, step_day=20), _intervals(40002, seed=1),
                    _intervals(41341, step_day=10, seed=2)])
    frame = objects.parse_satcat({k: SATCAT[k] for k in [41341, 30000, 40001, 40002, 40003]})
    out = objects.classify_activity(frame, iv).set_index("norad_id")
    # Rocket body stays passive even with a rise in its intervals; debris passive without any
    assert out.loc[[41341, 30000], "is_passive"].tolist() == [1, 1]
    assert out.loc[41341, "maneuver_checked"] == 0
    # Payload with a maneuver: active; without: passive; without intervals: active
    assert out.loc[40001, ["n_maneuvers", "is_active"]].tolist() == [1, 1]
    assert out.loc[40002, ["n_maneuvers", "is_passive"]].tolist() == [0, 1]
    assert out.loc[40003, ["maneuver_checked", "is_active"]].tolist() == [0, 1]
    assert (out["is_active"] + out["is_passive"] == 1).all()


# --- objects table -------------------------------------------------------------

def test_assemble_objects_matches_schema(tmp_path):
    iv = pd.concat([_intervals(40001, step_day=20), _intervals(40002, seed=1)])
    with pytest.warns(UserWarning):
        out = objects.assemble_objects({**SATCAT, 50000: None}, DISCOS, iv)
    assert list(out.columns) == list(OBJECT_COLUMNS)
    assert out["norad_id"].tolist() == sorted(out["norad_id"])
    assert len(out) == 7
    assert not any("bstar" in c.lower() for c in out.columns)
    for c, (dtype, *_) in OBJECT_COLUMNS.items():
        assert str(out[c].dtype) == dtype, c
    path = tmp_path / "objects.parquet"
    out.to_parquet(path, index=False)
    back = objects.load_objects(path)
    pd.testing.assert_frame_equal(back, out)


def test_validate_objects_rejects_active_rocket_body():
    out = objects.assemble_objects({41341: SATCAT[41341]}, DISCOS)
    out.loc[0, ["is_passive", "is_active"]] = [0, 1]
    with pytest.raises(ValueError, match="always passive"):
        validate_objects(out)


# --- downloads with a fake session (cache, paging, rate limit) -----------------

class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}

    def json(self):
        if isinstance(self.payload, str):
            raise ValueError("not JSON")
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        return self.responses.pop(0)


def test_fetch_satcat_caches(tmp_path):
    session = FakeSession([FakeResponse([SATCAT[41341]]), FakeResponse("No SATCAT records found")])
    out = objects.fetch_satcat([41341, 99999], session=session, raw_dir=tmp_path, pause=0)
    assert out[41341]["OBJECT_TYPE"] == "R/B" and out[99999] is None
    assert session.calls[0][1] == {"CATNR": 41341, "FORMAT": "JSON"}
    # Second call is served from data/raw (here tmp_path) without any request
    again = objects.fetch_satcat([41341, 99999], session=FakeSession([]), raw_dir=tmp_path)
    assert again == out


def test_fetch_discos_waits_out_429_and_follows_pages(tmp_path):
    waits = []
    page1 = {"data": [{"attributes": DISCOS[41341]}], "links": {"next": "/api/objects?page[number]=2"}}
    page2 = {"data": [{"attributes": DISCOS[22285]}], "links": {"next": None}}
    session = FakeSession([FakeResponse({}, 429, {"Retry-After": "7"}), FakeResponse(page1),
                           FakeResponse(page2)])
    out = objects.fetch_discos([41341, 22285, 40002], token="t", session=session, raw_dir=tmp_path,
                               sleep=waits.append, min_interval=3.0)
    assert out[41341]["mass"] == 2000.0 and out[22285]["xSectAvg"] == 36.0 and out[40002] is None
    url, params, headers = session.calls[0]
    assert params["filter"] == "in(satno,(41341,22285,40002))"
    assert headers["Authorization"] == "Bearer t" and headers["DiscosWeb-Api-Version"] == "2"
    assert session.calls[2][0] == "https://discosweb.esoc.esa.int/api/objects?page[number]=2"
    assert waits == [7.0, 3.0]          # Retry-After, then spacing before the second page
    # Cached: no token or request needed now
    assert objects.fetch_discos([41341, 40002], token=None, session=FakeSession([]), raw_dir=tmp_path) \
        == {41341: out[41341], 40002: None}


def test_fetch_discos_needs_token(tmp_path, monkeypatch):
    monkeypatch.setattr(objects, "discos_token", lambda: None)
    with pytest.raises(RuntimeError, match="DISCOS_TOKEN"):
        objects.fetch_discos([41341], raw_dir=tmp_path, session=FakeSession([]))


# --- fitted vs catalog drag ----------------------------------------------------

def test_compare_drag_rocket_bodies_only(tmp_path):
    catalog = pd.DataFrame({"norad_id": [1, 2, 3, 4, 5],
                            "object_type": ["rocket body"] * 4 + ["debris"],
                            "drag_sensitivity_catalog": [0.01, 0.02, 0.03, 0.04, 0.5]})
    measured = pd.Series([0.012, 0.018, 0.035, 0.05, 0.001], index=[1, 2, 3, 4, 5])
    res, fig = objects.compare_drag(measured, catalog, out_path=tmp_path / "drag.png")
    assert res["n"] == 4 and res["rho"] == pytest.approx(1.0)
    assert (tmp_path / "drag.png").exists()
    frame = measured.rename("drag_sensitivity").rename_axis("norad_id").reset_index()
    res2, _ = objects.compare_drag(frame, catalog, object_type=None)
    assert res2["n"] == 5


# --- live downloads (skip offline) ---------------------------------------------

def _online(url):
    try:
        requests.head(url, timeout=10)
        return True
    except requests.RequestException:
        return False


def test_satcat_download(tmp_path):
    if not _online(objects.SATCAT_URL):
        pytest.skip("No network access to CelesTrak")
    out = objects.parse_satcat(objects.fetch_satcat([41341], raw_dir=tmp_path))
    assert out["object_type"].iloc[0] == "rocket body"


def test_discos_download(tmp_path):
    token = objects.discos_token()
    if not token:
        pytest.skip("No DISCOS_TOKEN in .env")
    if not _online(objects.DISCOS_URL):
        pytest.skip("No network access to DISCOSweb")
    out = objects.parse_discos(objects.fetch_discos([41341], token=token, raw_dir=tmp_path))
    assert out["mass_kg"].iloc[0] > 0 and out["xsect_avg_m2"].iloc[0] > 0
