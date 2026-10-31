"""Object catalog (task B3): one row per object, joined to the analysis table on norad_id.

Sources:
- CelesTrak SATCAT (no login): object type, name, launch and decay date.
  https://celestrak.org/satcat/records.php?CATNR=<id>&FORMAT=JSON, one object per request.
- ESA DISCOSweb (token in .env as DISCOS_TOKEN, never committed): mass (kg) and average
  cross-section xSectAvg (m^2). API v2, JSON:API format, objects endpoint filtered on
  satno, at most 100 objects per page. The API allows a limited number of requests per
  minute and answers 429 with a Retry-After header when that is exceeded, so requests
  are spaced and 429s are waited out.
Raw responses are cached in data/raw/ and reused unless force=True.

From these:
- area_to_mass = xSectAvg / mass (m^2/kg) and drag_sensitivity_catalog = 2.2 * area_to_mass,
  the catalog counterpart of the quiet-time fitted B = Cd*A/m in src/physics.py (Cd = 2.2
  is the usual flat-plate value). It is the fallback for objects whose fit is unusable
  (high-altitude group, see B4-B5 in docs/PLAN.md).
- is_passive / is_active: debris and rocket bodies are passive. A payload (or an object of
  unknown type) is passive only if its decay intervals were searched and no maneuver was
  found (detect_maneuvers). One without intervals counts as active, so it never enters
  the passive sample unchecked.

Column names and units live in schema.OBJECT_COLUMNS. BSTAR is not used anywhere.
"""

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.schema import OBJECT_COLUMNS, validate_objects

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OBJECTS_PATH = PROCESSED_DIR / "objects.parquet"
# Interval table from the A2-A4 pipeline (orbit.build_decay_intervals)
INTERVALS_PATH = PROCESSED_DIR / "orbit_intervals_a2_a4.parquet"

SATCAT_URL = "https://celestrak.org/satcat/records.php"
SATCAT_CACHE = "satcat_records.json"
SATCAT_PAUSE_S = 0.5                     # be polite to CelesTrak between uncached requests
SATCAT_TYPES = {"DEB": "debris", "R/B": "rocket body", "PAY": "payload", "UNK": "unknown"}
PASSIVE_TYPES = {"debris", "rocket body"}

DISCOS_URL = "https://discosweb.esoc.esa.int/api/objects"
DISCOS_CACHE = "discos_objects.json"
DISCOS_API_VERSION = "2"
DISCOS_PAGE_SIZE = 100                   # API maximum
DISCOS_MIN_INTERVAL_S = 3.0              # spacing between requests (~20 per minute)
DISCOS_MAX_RETRIES = 5

CD_CATALOG = 2.2                         # drag coefficient for the catalog drag sensitivity

# Maneuver detection defaults (detect_maneuvers)
MANEUVER_WINDOW_DAYS = 2.0
MANEUVER_K = 5.0
MANEUVER_MIN_RISE_KM = 0.2
MANEUVER_MIN_POINTS = 3
MANEUVER_TREND_DAYS = 7.0


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


# ---------------------------------------------------------------------------
# 1. CelesTrak SATCAT
# ---------------------------------------------------------------------------

def fetch_satcat(norad_ids, force=False, session=None, raw_dir=None, pause=SATCAT_PAUSE_S):
    """SATCAT record (dict) per NORAD ID, None if CelesTrak has no record.

    Records are cached in data/raw/satcat_records.json, keyed by NORAD ID as a string;
    only IDs missing from the cache are requested (all of them with force=True).
    """
    path = Path(raw_dir or RAW_DIR) / SATCAT_CACHE
    cache = {} if force else _load_cache(path)
    todo = [int(i) for i in norad_ids if str(int(i)) not in cache]
    session = session or (requests.Session() if todo else None)
    try:
        for n, norad_id in enumerate(todo):
            if n:
                time.sleep(pause)
            resp = session.get(SATCAT_URL, params={"CATNR": norad_id, "FORMAT": "JSON"}, timeout=60)
            resp.raise_for_status()
            try:
                records = resp.json()
            except ValueError:          # plain-text "No SATCAT records found"
                records = []
            cache[str(norad_id)] = records[0] if records else None
    finally:
        if todo:
            _save_cache(path, cache)
    return {int(i): cache.get(str(int(i))) for i in norad_ids}


def parse_satcat(records):
    """SATCAT records {norad_id: dict or None} -> norad_id, object_name, object_type,
    launch_date, decay_date. Objects without a record get type 'unknown'."""
    rows = []
    for norad_id, rec in records.items():
        rec = rec or {}
        rows.append({
            "norad_id": int(norad_id),
            "object_name": rec.get("OBJECT_NAME"),
            "object_type": SATCAT_TYPES.get(str(rec.get("OBJECT_TYPE", "")).strip().upper(), "unknown"),
            "launch_date": pd.to_datetime(rec.get("LAUNCH_DATE") or None),
            "decay_date": pd.to_datetime(rec.get("DECAY_DATE") or None),
        })
    missing = [r["norad_id"] for r in rows if r["object_name"] is None]
    if missing:
        warnings.warn(f"No SATCAT record for {len(missing)} objects, typed 'unknown': {missing[:10]}")
    out = pd.DataFrame(rows, columns=["norad_id", "object_name", "object_type", "launch_date", "decay_date"])
    out["launch_date"] = pd.to_datetime(out["launch_date"])
    out["decay_date"] = pd.to_datetime(out["decay_date"])
    return out


# ---------------------------------------------------------------------------
# 2. ESA DISCOSweb
# ---------------------------------------------------------------------------

def discos_token():
    """DISCOSweb token from .env (DISCOS_TOKEN), or None."""
    load_dotenv(ROOT / ".env")
    return os.getenv("DISCOS_TOKEN") or None


def _discos_get(session, url, params, headers, sleep, max_retries=DISCOS_MAX_RETRIES):
    """GET with the rate limit respected: on 429 wait Retry-After seconds and try again."""
    for _ in range(max_retries + 1):
        resp = session.get(url, params=params, headers=headers, timeout=60)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        sleep(float(resp.headers.get("Retry-After", 60)))
    raise RuntimeError(f"DISCOSweb still rate-limited after {max_retries} retries")


def fetch_discos(norad_ids, token=None, force=False, session=None, raw_dir=None, sleep=time.sleep,
                 min_interval=DISCOS_MIN_INTERVAL_S):
    """DISCOSweb object attributes (dict) per NORAD ID, None if DISCOSweb has no object.

    Cached in data/raw/discos_objects.json; only uncached IDs are requested, in batches of
    DISCOS_PAGE_SIZE with filter=in(satno,(...)), following links.next if the reply is paged.
    """
    path = Path(raw_dir or RAW_DIR) / DISCOS_CACHE
    cache = {} if force else _load_cache(path)
    todo = [int(i) for i in norad_ids if str(int(i)) not in cache]
    if todo:
        token = token or discos_token()
        if not token:
            raise RuntimeError("Fill in DISCOS_TOKEN in .env")
        session = session or requests.Session()
        headers = {"Authorization": f"Bearer {token}", "DiscosWeb-Api-Version": DISCOS_API_VERSION}
        n_requests = 0
        try:
            for k in range(0, len(todo), DISCOS_PAGE_SIZE):
                batch = todo[k:k + DISCOS_PAGE_SIZE]
                url = DISCOS_URL
                params = {"filter": f"in(satno,({','.join(map(str, batch))}))",
                          "page[size]": DISCOS_PAGE_SIZE}
                found = {}
                while url:
                    if n_requests:
                        sleep(min_interval)
                    reply = _discos_get(session, url, params, headers, sleep)
                    n_requests += 1
                    for obj in reply.get("data", []):
                        attrs = obj.get("attributes", {})
                        if attrs.get("satno") is not None:
                            found[str(int(attrs["satno"]))] = attrs
                    nxt = (reply.get("links") or {}).get("next")
                    # links.next is a full URL or a path with its query already in it
                    url = (nxt if nxt.startswith("http") else
                           "https://discosweb.esoc.esa.int" + nxt) if nxt else None
                    params = None
                for norad_id in batch:
                    cache[str(norad_id)] = found.get(str(norad_id))
        finally:
            _save_cache(path, cache)
    return {int(i): cache.get(str(int(i))) for i in norad_ids}


def parse_discos(records):
    """DISCOSweb attributes {norad_id: dict or None} -> norad_id, mass_kg, xsect_avg_m2."""
    rows = []
    for norad_id, attrs in records.items():
        attrs = attrs or {}
        rows.append({"norad_id": int(norad_id),
                     "mass_kg": attrs.get("mass"),
                     "xsect_avg_m2": attrs.get("xSectAvg")})
    out = pd.DataFrame(rows, columns=["norad_id", "mass_kg", "xsect_avg_m2"])
    for c in ["mass_kg", "xsect_avg_m2"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        out.loc[out[c] <= 0, c] = np.nan      # zero mass or area means "not known"
    return out


def add_area_to_mass(df, cd=CD_CATALOG):
    """area_to_mass = xsect_avg_m2 / mass_kg (m^2/kg), drag_sensitivity_catalog = cd * area_to_mass.
    NaN unless both mass and cross-section are known."""
    df = df.copy()
    df["area_to_mass"] = df["xsect_avg_m2"] / df["mass_kg"]
    df["drag_sensitivity_catalog"] = cd * df["area_to_mass"]
    return df


# ---------------------------------------------------------------------------
# 3. Maneuvers and passive / active
# ---------------------------------------------------------------------------

def _a_segments(g):
    """Relative semi-major axis (km) at the TLE epochs of one object, rebuilt from the
    interval changes da_km. A gap (an interval left out, e.g. one bridging a storm onset)
    starts a new segment, because the change across it is unknown."""
    g = g.sort_values("t_start")
    new_seg = (g["t_start"] != g["t_end"].shift()).to_numpy()
    starts = np.flatnonzero(new_seg).tolist() + [len(g)]
    t0, t1, da = g["t_start"].to_numpy(), g["t_end"].to_numpy(), g["da_km"].to_numpy(dtype=float)
    for s, e in zip(starts[:-1], starts[1:]):
        times = np.r_[t0[s], t1[s:e]]
        yield times, np.r_[0.0, np.cumsum(da[s:e])]


def _upper_sd(x):
    """Noise scale from the upper half only: (75th - 50th percentile) / 0.6745, ignoring
    NaN (= sd for normal noise). Storms push the rise statistic down, never up, so they
    stay out of it; a few maneuver epochs hardly move the 75th percentile."""
    q50, q75 = np.nanpercentile(x, [50, 75])
    return float(q75 - q50) / 0.6745


def detect_maneuvers(intervals, window_days=MANEUVER_WINDOW_DAYS, k=MANEUVER_K,
                     min_rise_km=MANEUVER_MIN_RISE_KM, min_points=MANEUVER_MIN_POINTS,
                     trend_days=MANEUVER_TREND_DAYS):
    """Sustained semi-major-axis rises (maneuvers) in decay intervals.

    intervals: orbit.build_decay_intervals / orbit.decay_rates output (norad_id, t_start,
    t_end, da_km). Per object, a(t) is rebuilt from da_km and at every TLE epoch the median
    of a over the next `window_days` is compared with the median over the previous
    `window_days` (at least `min_points` TLEs on each side). The decay expected between
    the two windows is added back (local decay rate = median of the ~1-day rates starting
    within +-`trend_days`; a median, so a storm burst does not move it), so the rise is
    measured against the decay trend: drag cannot produce it. A rise of more than
    max(min_rise_km, k * noise) is a maneuver; noise is the scatter of this rise across
    the object's epochs, taken from its upper half (storms only push it down), so it
    covers both the TLE scatter and the error of the local decay rate. Medians make it
    ignore a single outlier TLE (up and straight back down), so only a rise that lasts
    counts. Neighbouring flagged epochs (within `window_days`) are one maneuver, stamped
    at its largest rise.

    Synthetic check (100 seeds, 120 days, TLE every 8 h): 50 m/day decay, 20 m scatter:
    no false alarms, no single outlier flagged, 0.3 km rises found 100/100. Decay rising
    to 400 m/day with a 2-day storm (+1 km/day) every 30 days, 40 m scatter: no false
    alarms, 1 km rises found 99/100, 0.5 km rises mostly missed.

    Returns one row per maneuver: norad_id, t_maneuver, rise_km, threshold_km.
    """
    w = np.timedelta64(int(window_days * 86400e9), "ns")
    tw = np.timedelta64(int(trend_days * 86400e9), "ns")
    rows = []
    for norad_id, g in intervals.groupby("norad_id"):
        for times, a in _a_segments(g):
            if len(a) < 2 * min_points:
                continue
            times = times.astype("datetime64[ns]")
            days = (times - times[0]) / np.timedelta64(1, "D")
            # Decay rate from each epoch to the first epoch >= 1 day later (8 h rates are noisy)
            ahead = np.searchsorted(days, days + 1.0, side="left")
            ok = ahead < len(a)
            rate = (a[ahead[ok]] - a[ok]) / (days[ahead[ok]] - days[ok])
            lo = np.searchsorted(times, times - w, side="left")
            hi = np.searchsorted(times, times + w, side="right")
            # Rate j starts at epoch j; take those starting within +-trend_days
            t_lo = np.searchsorted(times, times - tw, side="left")
            t_hi = np.minimum(np.searchsorted(times, times + tw, side="right"), len(rate))
            rise = np.full(len(a), np.nan)
            for i in range(len(a)):
                if i - lo[i] >= min_points and hi[i] - i >= min_points and t_hi[i] > t_lo[i]:
                    expected = np.median(rate[t_lo[i]:t_hi[i]]) * (
                        np.median(days[i:hi[i]]) - np.median(days[lo[i]:i]))
                    rise[i] = np.median(a[i:hi[i]]) - np.median(a[lo[i]:i]) - expected
            if np.isnan(rise).all():
                continue
            threshold = max(min_rise_km, k * _upper_sd(rise))
            flagged = np.flatnonzero(rise > threshold)
            if not len(flagged):
                continue
            for group in np.split(flagged, np.flatnonzero(np.diff(times[flagged]) > w) + 1):
                i = group[np.argmax(rise[group])]
                rows.append({"norad_id": int(norad_id), "t_maneuver": pd.Timestamp(times[i]),
                             "rise_km": float(rise[i]), "threshold_km": threshold})
    return pd.DataFrame(rows, columns=["norad_id", "t_maneuver", "rise_km", "threshold_km"])


def classify_activity(objects, intervals=None, **kwargs):
    """Add maneuver_checked, n_maneuvers, is_passive, is_active to a frame with norad_id
    and object_type. Debris and rocket bodies are passive; any other object is passive
    only if it has decay intervals and detect_maneuvers finds none in them."""
    out = objects.copy()
    # Operational payloads are active even when continuous low-thrust station
    # keeping produces no discrete semi-major-axis jump.
    ops = out.get("ops_status_code", pd.Series(index=out.index, dtype="string"))
    operational = ops.astype("string").str.strip().str.upper().isin({"+", "P", "B", "S", "X"})
    # Space-Track snapshots do not always expose OPS_STATUS_CODE; Starlink's
    # operator naming is an explicit low-thrust operational fallback.
    names = out.get("object_name", pd.Series(index=out.index, dtype="string"))
    operational = (operational | names.astype("string").str.upper().str.startswith("STARLINK")).fillna(False)
    out["maneuver_checked"] = 0
    out["n_maneuvers"] = 0
    if intervals is not None and len(intervals):
        searched = ~out["object_type"].isin(PASSIVE_TYPES) & out["norad_id"].isin(intervals["norad_id"])
        out["maneuver_checked"] = searched.astype("int64")
        check = out.loc[searched, "norad_id"]
        events = detect_maneuvers(intervals[intervals["norad_id"].isin(check)], **kwargs)
        counts = events.groupby("norad_id").size()
        out["n_maneuvers"] = out["norad_id"].map(counts).fillna(0).astype("int64")
    by_type = out["object_type"].isin(PASSIVE_TYPES)
    quiet_payload = (out["maneuver_checked"] == 1) & (out["n_maneuvers"] == 0)
    out["is_passive"] = (by_type | (quiet_payload & ~operational)).astype("int64")
    out["is_active"] = 1 - out["is_passive"]
    return out


# ---------------------------------------------------------------------------
# 4. Objects table
# ---------------------------------------------------------------------------

def assemble_objects(satcat_records, discos_records=None, intervals=None, **kwargs):
    """Objects table from already-fetched records (no network): SATCAT fields, DISCOSweb
    mass and cross-section, area-to-mass, maneuvers and passive / active. Validated with
    schema.validate_objects."""
    objects = parse_satcat(satcat_records)
    discos = parse_discos(discos_records or {})
    objects = objects.merge(discos, on="norad_id", how="left")
    objects = add_area_to_mass(objects)
    objects = classify_activity(objects, intervals, **kwargs)
    return validate_objects(objects[list(OBJECT_COLUMNS)].sort_values("norad_id").reset_index(drop=True))


def build_objects(norad_ids, intervals=None, out_path=None, force=False, token=None, session=None,
                  **kwargs):
    """Download (or read from cache) SATCAT and DISCOSweb for norad_ids, build the objects
    table and save it to data/processed/objects.parquet.

    intervals: decay intervals for the maneuver search; by default the A2-A4 table
    (INTERVALS_PATH) if it exists. Without them every payload counts as active.
    """
    norad_ids = sorted({int(i) for i in norad_ids})
    if intervals is None and INTERVALS_PATH.exists():
        intervals = pd.read_parquet(INTERVALS_PATH)
    if intervals is None:
        warnings.warn(f"No decay intervals ({INTERVALS_PATH.name}); payloads are counted as active")
    satcat = fetch_satcat(norad_ids, force=force, session=session)
    discos = fetch_discos(norad_ids, token=token, force=force, session=session)
    objects = assemble_objects(satcat, discos, intervals, **kwargs)
    path = Path(out_path or OBJECTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    objects.to_parquet(path, index=False)
    return objects


def load_objects(path=None):
    """Read and validate the saved objects table."""
    return validate_objects(pd.read_parquet(path or OBJECTS_PATH))


# ---------------------------------------------------------------------------
# 5. Fitted vs catalog drag sensitivity
# ---------------------------------------------------------------------------

def compare_drag(measured, catalog, object_type="rocket body", ax=None, out_path=None):
    """Spearman correlation and scatter plot of fitted vs catalog drag sensitivity.

    measured: fitted B per object (m^2/kg), a Series indexed by norad_id (as returned by
    physics.fit_drag_sensitivity) or a frame with norad_id and drag_sensitivity.
    catalog: objects table (norad_id, object_type, drag_sensitivity_catalog).
    Only objects of `object_type` with both values > 0 are used (rocket bodies by default:
    debris has no reliable mass or area). Rank correlation, because the catalog uses one
    Cd and an orientation-averaged area, so only the ordering is expected to agree.

    Returns (result, fig): result has rho, p_value, n and the matched table.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    from src.plots import DATA_COLOR, WEATHER_COLOR

    if isinstance(measured, pd.Series):
        measured = measured.rename("drag_sensitivity").rename_axis("norad_id").reset_index()
    both = measured[["norad_id", "drag_sensitivity"]].merge(
        catalog[["norad_id", "object_type", "drag_sensitivity_catalog"]], on="norad_id")
    if object_type is not None:
        both = both[both["object_type"] == object_type]
    both = both[(both["drag_sensitivity"] > 0) & (both["drag_sensitivity_catalog"] > 0)].reset_index(drop=True)
    n = len(both)
    rho, p = (stats.spearmanr(both["drag_sensitivity_catalog"], both["drag_sensitivity"])
              if n >= 3 else (np.nan, np.nan))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    ax.scatter(both["drag_sensitivity_catalog"], both["drag_sensitivity"], s=18, color=DATA_COLOR)
    if n:
        values = both[["drag_sensitivity", "drag_sensitivity_catalog"]].to_numpy()
        lims = [values.min() / 1.5, values.max() * 1.5]
        ax.plot(lims, lims, color=WEATHER_COLOR, lw=0.8, ls="--", label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False)
    ax.set_xlabel(f"Catalog {CD_CATALOG} x A/m (m$^2$/kg)")
    ax.set_ylabel("Fitted quiet-time B (m$^2$/kg)")
    ax.set_title(f"{object_type or 'all objects'}: Spearman rho = {rho:.2f} (n = {n})")
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return {"rho": float(rho), "p_value": float(p), "n": n, "table": both}, fig
