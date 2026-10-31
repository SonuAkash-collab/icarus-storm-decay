"""Build active-satellite intervals for the operator offset ratio.

One command downloads (or reads from cache) everything and writes the active intervals:

    python -m src.active_cohort

Steps (all downloads cached under data/raw/, outputs under data/processed/):
1. Payload snapshots on the passive cohort's 9 sample dates (spacetrack_io.find_candidates
   with object type PAYLOAD, 400-800 km, ecc < 0.01), cached per date in
   data/raw/active_candidates/.
2. Cohort: N_STARLINK Starlink and N_OTHER other payloads, each picked with
   spacetrack_io.select_active_cohort (coverage on at least 8 of 9 dates, spread over
   100 km altitude bands). Starlink and the rest are selected separately because
   Starlink is ~70% of the payloads in this band and would otherwise fill the cohort.
3. Full TLE history Apr 2024 - Apr 2025 (data/raw/tle_active_cohort.json), cleaning and
   intervals (build_active_intervals), activity from objects.classify_activity.

Outputs:
    data/processed/active_cohort_manifest.parquet    selected objects (+ object_name)
    data/processed/active_tles_cleaned.parquet       cleaned TLEs with line1/line2
    data/processed/active_activity.parquet           per object: is_active, n_maneuvers
    data/processed/orbit_intervals_actives.parquet   intervals of the active objects
"""

import argparse
import json

import pandas as pd

from src.objects import classify_activity
from src.orbit import build_decay_intervals, clean_tle_history, tles_to_frame
from src.spacetrack_io import RAW_DIR, download_tle_history, find_candidates, select_active_cohort

PROCESSED_DIR = RAW_DIR.parent / "processed"
SAMPLE_DATES_PATH = RAW_DIR / "piyush" / "processed" / "passive_cohort_sample_dates.json"
SAMPLE_DATES = ["2024-04-15", "2024-05-15", "2024-06-15", "2024-08-15", "2024-10-15",
                "2024-12-15", "2025-01-15", "2025-03-15", "2025-04-15"]
CANDIDATE_DIR = RAW_DIR / "active_candidates"
TLE_FILE = "tle_active_cohort.json"
START, END = "2024-04-01", "2025-04-30"
N_STARLINK = 25
N_OTHER = 25

MANIFEST_PATH = PROCESSED_DIR / "active_cohort_manifest.parquet"
TLES_PATH = PROCESSED_DIR / "active_tles_cleaned.parquet"
ACTIVITY_PATH = PROCESSED_DIR / "active_activity.parquet"
INTERVALS_PATH = PROCESSED_DIR / "orbit_intervals_actives.parquet"

# Snapshot fields kept in the cache (the full records are ~20k rows per date)
SNAPSHOT_FIELDS = ["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "EPOCH", "ECCENTRICITY",
                   "PERIAPSIS", "APOAPSIS", "INCLINATION"]


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
    active_ids = activity.loc[activity["is_active"].eq(1), "norad_id"]
    active = intervals[intervals["norad_id"].isin(active_ids)].merge(
        activity[["norad_id", "is_active", "n_maneuvers"]], on="norad_id", how="left"
    )
    return active, activity, cleaning_report, interval_report, cleaned


def sample_dates():
    """The passive cohort's sample dates (Piyush's file if present, else the same list)."""
    if SAMPLE_DATES_PATH.exists():
        return json.loads(SAMPLE_DATES_PATH.read_text())
    return SAMPLE_DATES


def payload_snapshots(dates=None, client=None, force=False):
    """{date: payload records} for the sample dates, one Space-Track query per uncached date."""
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for day in dates or sample_dates():
        path = CANDIDATE_DIR / f"{day}.json"
        if path.exists() and not force:
            out[day] = json.loads(path.read_text())
            continue
        records = find_candidates(day, object_types=("PAYLOAD",), client=client)
        records = [{k: r.get(k) for k in SNAPSHOT_FIELDS} for r in records]
        path.write_text(json.dumps(records))
        out[day] = records
    return out


def is_starlink(names):
    return pd.Series(names, dtype="string").str.upper().str.startswith("STARLINK").fillna(False).to_numpy()


def select_cohort(records_by_date, n_starlink=N_STARLINK, n_other=N_OTHER, **kwargs):
    """Manifest of the selected payloads: n_starlink Starlink + n_other others, each group
    picked by select_active_cohort (coverage, altitude bands). Adds object_name and group."""
    names = {}
    for records in records_by_date.values():
        for r in records:
            names[int(r["NORAD_CAT_ID"])] = r.get("OBJECT_NAME")
    parts = []
    for group, n, keep in (("starlink", n_starlink, True), ("other", n_other, False)):
        subset = {d: [r for r in recs if bool(is_starlink([r.get("OBJECT_NAME")])[0]) == keep]
                  for d, recs in records_by_date.items()}
        manifest, _ = select_active_cohort(subset, target_size=n, **kwargs)
        parts.append(manifest[manifest["selected"]].assign(group=group))
    out = pd.concat(parts, ignore_index=True)
    out["object_name"] = out["norad_id"].map(names).astype("string")
    out["object_type"] = "payload"
    return out


def build_active_cohort(force=False, client=None, log=print):
    """Steps 1-3 of the module notes. Returns (intervals, activity, cleaned TLEs, manifest)."""
    from src.storms import load_storms
    if MANIFEST_PATH.exists() and not force:
        manifest = pd.read_parquet(MANIFEST_PATH)
    else:
        manifest = select_cohort(payload_snapshots(client=client))
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        manifest.to_parquet(MANIFEST_PATH, index=False)
    log(f"cohort: {(manifest['group'] == 'starlink').sum()} Starlink + {(manifest['group'] == 'other').sum()} other")
    records = download_tle_history(manifest["norad_id"].tolist(), START, END, out_name=TLE_FILE,
                                   force=force, client=client)
    intervals, activity, clean_rep, int_rep, cleaned = build_active_intervals(records, load_storms(), manifest)
    intervals.to_parquet(INTERVALS_PATH, index=False)
    activity.to_parquet(ACTIVITY_PATH, index=False)
    cleaned.to_parquet(TLES_PATH, index=False)
    log(clean_rep.to_string(index=False))
    log(int_rep.to_string(index=False))
    log(f"active: {int(activity['is_active'].sum())} of {len(activity)} objects, {len(intervals):,} intervals")
    return intervals, activity, cleaned, manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="re-select and re-download")
    build_active_cohort(force=p.parse_args(argv).force)


if __name__ == "__main__":
    main()
