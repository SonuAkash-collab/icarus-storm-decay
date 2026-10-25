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
                    min_perigee=450, max_apogee=550, max_ecc=0.01, client=None):
    """List objects whose orbit on a given day was near-circular and in an altitude band.

    Uses GP history for a one-day window, so it describes the orbit at that time
    (not today). Perigee/apogee are altitudes in km as given by Space-Track.
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

