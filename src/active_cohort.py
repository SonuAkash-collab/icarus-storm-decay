"""Build active-satellite intervals for the operator offset ratio."""

import pandas as pd

from src.objects import classify_activity
from src.orbit import build_decay_intervals, clean_tle_history, tles_to_frame


def build_active_intervals(records, storms, objects, **cleaning_kwargs):
    """Parse payload TLE records, detect maneuvers, and return active intervals.

    ``objects`` must contain ``norad_id`` and ``object_type``; payload rows with at
    least one detected maneuver are marked active. TLE intervals are built with
    passive climb filtering disabled because station-keeping rises are the signal
    needed by the offset-ratio analysis.
    """
    tles = tles_to_frame(records)
    cleaned, cleaning_report = clean_tle_history(tles, passive=False, **cleaning_kwargs)
    intervals, interval_report = build_decay_intervals(cleaned, storms, passive=False)
    activity = classify_activity(objects[["norad_id", "object_type"]], intervals)
    active_ids = activity.loc[activity["is_active"].eq(1), "norad_id"]
    active = intervals[intervals["norad_id"].isin(active_ids)].merge(
        activity[["norad_id", "is_active", "n_maneuvers"]], on="norad_id", how="left"
    )
    return active, activity, cleaning_report, interval_report
