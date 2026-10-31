"""Build active-satellite intervals for the operator offset ratio."""

import pandas as pd

from src.objects import classify_activity
from src.orbit import build_decay_intervals, clean_tle_history, tles_to_frame


def flag_free_drag_payloads(intervals, objects, min_days=21.0):
    """Return Starlink IDs with sustained decay and no station-keeping evidence.

    These are treated as failed/deorbiting payloads rather than operator-response
    observations. The conservative rule requires at least ``min_days`` of coverage
    and no positive maneuver event in the interval series.
    """
    starlink = objects[objects.get("object_name", "").astype("string").str.upper().str.startswith("STARLINK")]
    if starlink.empty or intervals.empty:
        return set()
    out = set()
    for norad_id, group in intervals[intervals["norad_id"].isin(starlink["norad_id"])].groupby("norad_id"):
        span = (group["t_end"].max() - group["t_start"].min()).total_seconds() / 86400
        if span >= min_days and (group["da_km"] >= 0).sum() == 0:
            out.add(int(norad_id))
    return out


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
    columns = ["norad_id", "object_type"]
    if "object_name" in objects:
        columns.append("object_name")
    if "ops_status_code" in objects:
        columns.append("ops_status_code")
    activity = classify_activity(objects[columns], intervals)
    free_drag = flag_free_drag_payloads(intervals, objects, min_days=21.0)
    if free_drag:
        activity.loc[activity["norad_id"].isin(free_drag), ["is_active", "is_passive"]] = [0, 1]
        activity["free_drag_flag"] = activity["norad_id"].isin(free_drag).astype("int64")
    else:
        activity["free_drag_flag"] = 0
    active_ids = activity.loc[activity["is_active"].eq(1), "norad_id"]
    active = intervals[intervals["norad_id"].isin(active_ids)].merge(
        activity[["norad_id", "is_active", "n_maneuvers"]], on="norad_id", how="left"
    )
    return active, activity, cleaning_report, interval_report
