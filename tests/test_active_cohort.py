import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.objects import classify_activity
import src.active_cohort as ac
from src.active_cohort import flag_free_drag_payloads
from src.spacetrack_io import select_active_cohort


def test_operational_payload_is_active_without_a_maneuver():
    objects = pd.DataFrame({
        "norad_id": [1, 2], "object_type": ["payload", "payload"],
        "ops_status_code": ["+", "D"],
    })
    intervals = pd.DataFrame({"norad_id": [1, 2], "t_start": pd.to_datetime(["2024-01-01"] * 2),
                              "t_end": pd.to_datetime(["2024-01-02"] * 2), "da_km": [-0.1, -0.1]})
    out = classify_activity(objects, intervals).set_index("norad_id")
    assert out.loc[1, "is_active"] == 1
    assert out.loc[2, "is_passive"] == 1


def test_starlink_name_is_low_thrust_active_fallback():
    objects = pd.DataFrame({"norad_id": [1], "object_type": ["payload"],
                            "object_name": ["STARLINK-1008"]})
    out = classify_activity(objects, pd.DataFrame()).iloc[0]
    assert out["is_active"] == 1


def test_flags_long_starlink_free_drag():
    objects = pd.DataFrame({"norad_id": [1], "object_name": ["STARLINK-1008"]})
    starts = pd.date_range("2024-01-01", periods=30)
    intervals = pd.DataFrame({"norad_id": 1, "t_start": starts,
                              "t_end": starts + pd.Timedelta(days=1), "da_km": -0.05})
    assert flag_free_drag_payloads(intervals, objects) == {1}



def test_flagged_starlink_stays_active(monkeypatch):
    """The active_cohort guard is informational: build_table.free_drag_flags decides
    free_drag, so a flagged Starlink keeps is_active = 1 and all its intervals."""
    objects = pd.DataFrame({"norad_id": [1], "object_type": ["payload"],
                            "object_name": ["STARLINK-1008"]})
    starts = pd.date_range("2024-01-01", periods=30)
    intervals = pd.DataFrame({"norad_id": 1, "t_start": starts,
                              "t_end": starts + pd.Timedelta(days=1), "da_km": -0.05})
    monkeypatch.setattr(ac, "tles_to_frame", lambda records: pd.DataFrame())
    monkeypatch.setattr(ac, "clean_tle_history", lambda tles, **kw: (tles, pd.DataFrame()))
    monkeypatch.setattr(ac, "build_decay_intervals", lambda cleaned, storms, **kw: (intervals, pd.DataFrame()))
    active, activity, *_ = ac.build_active_intervals([], pd.DataFrame(), objects)
    row = activity.set_index("norad_id").loc[1]
    assert row["free_drag_flag"] == 1
    assert row["is_active"] == 1 and row["is_passive"] == 0
    assert len(active) == len(intervals) and (active["is_active"] == 1).all()

def test_select_active_cohort_filters_payloads():
    records = {"2024-01-01": [
        {"NORAD_CAT_ID": 1, "OBJECT_NAME": "STARLINK-1", "OBJECT_TYPE": "PAYLOAD",
         "ECCENTRICITY": 0.001, "PERIAPSIS": 500, "APOAPSIS": 510},
        {"NORAD_CAT_ID": 2, "OBJECT_NAME": "DEBRIS", "OBJECT_TYPE": "DEBRIS",
         "ECCENTRICITY": 0.001, "PERIAPSIS": 500, "APOAPSIS": 510},
    ]}
    manifest, _ = select_active_cohort(records, min_snapshots=1, target_size=1)
    assert len(manifest) == 1
    assert manifest.iloc[0]["norad_id"] == 1
