"""Download TLE history from Space-Track.

Credentials are read from a local .env file (never committed):
    SPACETRACK_USER=...
    SPACETRACK_PASS=...
"""

import json
import os
from pathlib import Path

import pandas as pd
import spacetrack.operators as op
from dotenv import load_dotenv
from spacetrack import SpaceTrackClient

# Project root = one folder above src/
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def get_client():
    """Log in to Space-Track with the credentials in .env."""
    load_dotenv(ROOT / ".env")
    user = os.getenv("SPACETRACK_USER")
    password = os.getenv("SPACETRACK_PASS")
    if not user or not password:
        raise RuntimeError("Fill in SPACETRACK_USER and SPACETRACK_PASS in .env")
    return SpaceTrackClient(identity=user, password=password)


def download_tle_history(norad_ids, start, end, out_name=None, force=False, client=None):
    """Download all historical TLEs (GP history) for some objects.

    norad_ids: list of NORAD catalog numbers
    start, end: dates as 'YYYY-MM-DD' strings (end is inclusive)
    Returns the list of records (dicts) and saves them as JSON in data/raw/.
    If that file already exists it is loaded instead (use force=True to re-download).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if out_name is None:
        ids = "_".join(str(i) for i in norad_ids)
        out_name = f"tle_{ids}_{start}_{end}.json"
    path = RAW_DIR / out_name

    # Reuse an earlier download instead of querying Space-Track again
    if path.exists() and not force:
        with open(path) as f:
            return json.load(f)

    client = client or get_client()
    records = client.gp_history(
        norad_cat_id=list(norad_ids),
        epoch=op.inclusive_range(start, end),
        orderby=["norad_cat_id", "epoch"],
        format="json",
    )
    # The client returns a JSON string; turn it into Python objects
    if isinstance(records, str):
        records = json.loads(records)

    with open(path, "w") as f:
        json.dump(records, f)
    return records


def find_candidates(day, object_types=("ROCKET BODY", "DEBRIS"),
                    min_perigee=400, max_apogee=800, max_ecc=0.01, client=None):
    """List objects whose orbit on a given day was near-circular and in an altitude band.

    Uses GP history for a one-day window, so it describes the orbit at that time
    (not today). Perigee/apogee are altitudes in km as given by Space-Track. The
    defaults implement the project's 400-800 km passive-object screen; call this
    for multiple dates to measure coverage across the study year.
    """
    client = client or get_client()
    next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    records = client.gp_history(
        epoch=op.inclusive_range(day, next_day),
        object_type=list(object_types),
        eccentricity=op.less_than(max_ecc),
        periapsis=op.greater_than(min_perigee),
        apoapsis=op.less_than(max_apogee),
        orderby="norad_cat_id",
        format="json",
    )
    if isinstance(records, str):
        records = json.loads(records)
    return records


def select_passive_cohort(
    records_by_date,
    *,
    min_perigee=400,
    max_apogee=800,
    max_ecc=0.01,
    min_snapshots=8,
    target_size=300,
):
    """Rank a sampled set of Space-Track candidate snapshots into a cohort manifest.

    ``records_by_date`` maps each sampled ISO date to the records returned by
    ``find_candidates``. A candidate must be a debris object or rocket body, stay
    inside the requested perigee/apogee and eccentricity bounds on every sampled
    date where it appears, and appear on at least ``min_snapshots`` dates.
    Candidates are ranked by number of qualifying snapshots, then NORAD ID; the
    first ``target_size`` are marked selected. The returned manifest includes all
    candidate rows so the selection and coverage can be reviewed before downloading
    their full TLE histories.
    """
    if min_snapshots < 1 or target_size < 1:
        raise ValueError("min_snapshots and target_size must be positive")
    passive_types = {"DEBRIS", "ROCKET BODY"}
    rows = []
    for sample_date, records in records_by_date.items():
        for record in records:
            try:
                norad_id = int(record["NORAD_CAT_ID"])
                object_type = str(record.get("OBJECT_TYPE", "")).strip().upper()
                ecc = float(record.get("ECCENTRICITY"))
                perigee = float(record.get("PERIAPSIS", record.get("PERIGEE")))
                apogee = float(record.get("APOAPSIS", record.get("APOGEE")))
            except (KeyError, TypeError, ValueError):
                continue
            if object_type not in passive_types:
                continue
            if ecc >= max_ecc or perigee < min_perigee or apogee > max_apogee:
                continue
            rows.append({
                "norad_id": norad_id,
                "object_type": object_type,
                "sample_date": str(sample_date),
                "ecc": ecc,
                "perigee_km": perigee,
                "apogee_km": apogee,
            })

    columns = ["norad_id", "object_type", "snapshots", "coverage_fraction",
               "ecc_median", "perigee_median_km", "apogee_median_km", "selected"]
    if not rows:
        return pd.DataFrame(columns=columns), pd.DataFrame([
            {"step": "sample dates queried", "count": len(records_by_date)},
            {"step": "objects meeting passive/orbit/coverage filters", "count": 0},
            {"step": "selected", "count": 0},
        ])

    snapshots = pd.DataFrame(rows)
    n_dates = max(len(records_by_date), 1)
    manifest = (snapshots.groupby("norad_id", as_index=False)
                .agg(object_type=("object_type", lambda x: x.mode().iat[0]),
                     snapshots=("sample_date", "nunique"),
                     ecc_median=("ecc", "median"),
                     perigee_median_km=("perigee_km", "median"),
                     apogee_median_km=("apogee_km", "median")))
    manifest["coverage_fraction"] = manifest["snapshots"] / n_dates
    manifest = manifest[manifest["snapshots"] >= min_snapshots]
    manifest = manifest.sort_values(["snapshots", "norad_id"], ascending=[False, True]).reset_index(drop=True)
    manifest["selected"] = False
    manifest.loc[:target_size - 1, "selected"] = True
    manifest = manifest[columns]
    report = pd.DataFrame([
        {"step": "sample dates queried", "count": len(records_by_date)},
        {"step": "objects meeting passive/orbit/coverage filters", "count": len(manifest)},
        {"step": "selected", "count": int(manifest["selected"].sum())},
    ])
    return manifest, report

