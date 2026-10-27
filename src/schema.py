"""Shared analysis table: one row = one decay interval of one object.

Both causal branches (event study / offset ratio and double ML / vulnerability map)
read this table, and the synthetic generator (src/synthetic.py) writes it, so the
column names, units and meanings live here and nowhere else.

Columns (who builds them: A = orbit data, B = space weather + physics):

    norad_id          A  object ID
    t_mid             A  interval midpoint (naive UTC, like orbit.decay_rates)
    dt_days           A  interval length (days), > 0
    alt_km            A  mean altitude over the interval (km)
    inc_deg           A  inclination (deg)
    local_solar_time  A  local solar time of the ascending node (hours, 0-24)
    day_of_year       A  day of year of t_mid (1-366), for season
    ecc               A  mean eccentricity over the interval. TLEs are fitted mean
                         elements, not altitudes: on oval orbits the fit swings with
                         the argument of perigee and fakes altitude drops, so the
                         main analysis keeps only ecc < MAX_ECC (0.01)
    tle_ok            A  1 if both TLEs of the interval passed A3 cleaning (no
                         cross-tag inclination/RAAN jump, not a rolling-median
                         outlier, no climb of a passive object), else 0. Rows with
                         0 stay in the table so the share removed can be reported
    is_active         B  0 = passive (debris, rocket body, dead satellite), 1 = active
    Y_residual        B  observed minus predicted da/dt (m/day) with MSIS 2.1. Negative =
                         more decay than the physics model predicts. physics.py works in
                         km/day, so this is resid_msis21 * 1000.
    Y_residual_nrl    B  the same with NRLMSISE-00 (resid_nrlmsise00 * 1000), robustness
    Y_log_ratio       B  log(observed / predicted); NaN when the object is not seen
                         to decay (observed >= 0), as in physics.add_predictions
    T_ap_int          B  ap integrated over the interval (nT * day), continuous treatment
    T_storm           B  1 if the interval overlaps [main_onset, recovery_end] of a storm
    storm_id          B  storm whose event window (-10..+20 days around main_onset)
                         contains t_mid; <NA> if none
    storm_class       B  class of that storm (storms.storm_class); <NA> if none
    event_day         B  days from that storm's main_onset to t_mid; NaN if none
    f107, f107_81d    B  daily and 81-day F10.7 (sfu), Sun UV confounders
    drag_sensitivity  B  quiet-time drag sensitivity B = Cd*A/m (m^2/kg), physics.py.
                         May be missing for active satellites (manoeuvres spoil the
                         quiet-time fit); sample_report counts them

Rules the table enforces (see validate):
- BSTAR and air density are never columns: BSTAR jumps during storms and density is
  the mediator, so neither may be outcome or control.
- Day 0 = main_onset from storms.py; event window is -10 to +20 days.
- Intervals never bridge a main_onset (task A4); bridges_onset checks it.
- Analysis code takes its rows from main_sample (passive, near-circular, clean TLEs).
- Extra columns are allowed (e.g. true_effect in synthetic data), missing ones are not.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# name -> (pandas dtype, unit, built by, meaning)
COLUMNS = {
    "norad_id":         ("int64",          "",              "A", "Object ID"),
    "t_mid":            ("datetime64[ns]", "UTC",           "A", "Interval midpoint"),
    "dt_days":          ("float64",        "day",           "A", "Interval length"),
    "alt_km":           ("float64",        "km",            "A", "Mean altitude"),
    "inc_deg":          ("float64",        "deg",           "A", "Inclination"),
    "local_solar_time": ("float64",        "h",             "A", "Local solar time of the ascending node"),
    "day_of_year":      ("int64",          "day",           "A", "Day of year of t_mid"),
    "ecc":              ("float64",        "",              "A", "Mean eccentricity"),
    "tle_ok":           ("int64",          "",              "A", "1 = both TLEs passed cleaning"),
    "is_active":        ("int64",          "",              "B", "0 = passive, 1 = active"),
    "Y_residual":       ("float64",        "m/day",         "B", "Observed minus predicted decay"),
    "Y_residual_nrl":   ("float64",        "m/day",         "B", "Observed minus predicted decay (NRLMSISE-00)"),
    "Y_log_ratio":      ("float64",        "",              "B", "log(observed / predicted)"),
    "T_ap_int":         ("float64",        "nT*day",        "B", "ap integrated over the interval"),
    "T_storm":          ("int64",          "",              "B", "1 if inside a storm window"),
    "storm_id":         ("Int64",          "",              "B", "Storm link (nullable)"),
    "storm_class":      ("string",         "",              "B", "Storm class (nullable)"),
    "event_day":        ("float64",        "day",           "B", "Days from main_onset"),
    "f107":             ("float64",        "sfu",           "B", "Daily F10.7"),
    "f107_81d":         ("float64",        "sfu",           "B", "81-day mean F10.7"),
    "drag_sensitivity": ("float64",        "m^2/kg",        "B", "Quiet-time drag sensitivity (replaces BSTAR)"),
}
KEY = ["norad_id", "t_mid"]

OUTCOMES = ["Y_residual", "Y_residual_nrl", "Y_log_ratio"]
TREATMENTS = ["T_ap_int", "T_storm"]
# Adjustment set from the causal graph (src/dag.py checks it)
CONFOUNDERS = ["f107", "f107_81d", "day_of_year", "alt_km", "inc_deg",
               "local_solar_time", "drag_sensitivity"]
# Never allowed in the table, in any capitalisation
FORBIDDEN = ["bstar", "rho", "density", "air_density"]

STORM_CLASSES = ["moderate", "strong", "severe", "extreme"]
EVENT_WINDOW = (-10, 20)    # days around main_onset
MAX_ECC = 0.01              # near-circular orbits only in the main analysis
HIGH_ALT_MIN_KM = 1000.0    # objects at or above this are the high-altitude control, not the main sample
NOT_NULL = ["norad_id", "t_mid", "dt_days", "alt_km", "inc_deg", "local_solar_time",
            "day_of_year", "ecc", "tle_ok", "is_active", "T_ap_int", "T_storm", "f107", "f107_81d"]


def empty_frame():
    """Zero-row table with every column and dtype."""
    return pd.DataFrame({c: pd.Series(dtype=dtype) for c, (dtype, *_) in COLUMNS.items()})


def describe():
    """Column, dtype, unit, builder and meaning as a table (for docs and the paper)."""
    return pd.DataFrame([(c, *v) for c, v in COLUMNS.items()],
                        columns=["column", "dtype", "unit", "built_by", "meaning"])


def coerce(df):
    """Cast schema columns to their dtypes and put them first (extra columns kept, after)."""
    df = df.copy()
    for c, (dtype, *_) in COLUMNS.items():
        if c in df:
            df[c] = df[c].astype(dtype)
    extra = [c for c in df.columns if c not in COLUMNS]
    return df[[c for c in COLUMNS if c in df] + extra]


def problems(df):
    """List of schema violations (empty list = valid)."""
    out = []
    missing = [c for c in COLUMNS if c not in df]
    if missing:
        out.append(f"missing columns: {missing}")
    bad = [c for c in df.columns if any(f in c.lower() for f in FORBIDDEN)]
    if bad:
        out.append(f"forbidden columns (BSTAR / density): {bad}")
    if missing:
        return out

    nulls = [c for c in NOT_NULL if df[c].isna().any()]
    if nulls:
        out.append(f"nulls in required columns: {nulls}")
    if (df["drag_sensitivity"].isna() & (df["is_active"] == 0)).any():
        out.append("drag_sensitivity may only be missing for active satellites")
    if (df["dt_days"] <= 0).any():
        out.append("dt_days must be > 0")
    if df.duplicated(KEY).any():
        out.append(f"duplicate {KEY} rows")
    for c in ["is_active", "T_storm", "tle_ok"]:
        if not df[c].dropna().isin([0, 1]).all():
            out.append(f"{c} must be 0 or 1")
    if not df["day_of_year"].dropna().between(1, 366).all():
        out.append("day_of_year must be in 1..366")
    if not df["local_solar_time"].dropna().between(0, 24).all():
        out.append("local_solar_time must be in 0..24 h")
    if not df["ecc"].dropna().between(0, 1, inclusive="left").all():
        out.append("ecc must be in [0, 1)")
    if (df["T_ap_int"] < 0).any():
        out.append("T_ap_int must be >= 0")

    # Storm link: storm_id, storm_class and event_day are all set or all missing
    linked = df["storm_id"].notna()
    if not (df["event_day"].notna() == linked).all() or not (df["storm_class"].notna() == linked).all():
        out.append("storm_id, storm_class and event_day must be missing together")
    if not df["storm_class"].dropna().isin(STORM_CLASSES).all():
        out.append(f"storm_class must be one of {STORM_CLASSES}")
    lo, hi = EVENT_WINDOW
    if not df["event_day"].dropna().between(lo, hi).all():
        out.append(f"event_day must be in {lo}..{hi}")
    if (df["T_storm"].eq(1) & ~linked).any():
        out.append("T_storm = 1 rows need a storm_id")
    return out


def validate(df):
    """Raise ValueError listing every problem; return the table with schema dtypes."""
    found = problems(df)
    if found:
        raise ValueError("Schema check failed:\n- " + "\n- ".join(found))
    return coerce(df)


def _clean_passive(df, max_ecc):
    return ((df["is_active"] == 0) & (df["ecc"] < max_ecc) & (df["tle_ok"] == 1)
            & df["drag_sensitivity"].notna())


def main_sample(df, max_ecc=MAX_ECC, max_alt_km=HIGH_ALT_MIN_KM):
    """Rows for the main analysis: passive objects below the high-altitude control,
    near-circular orbits, clean TLEs."""
    return df[_clean_passive(df, max_ecc) & (df["alt_km"] < max_alt_km)]


def high_alt_sample(df, max_ecc=MAX_ECC, min_alt_km=HIGH_ALT_MIN_KM):
    """Rows of the high-altitude control: the main-sample filters, but at or above
    min_alt_km, where the storm effect should be about zero."""
    return df[_clean_passive(df, max_ecc) & (df["alt_km"] >= min_alt_km)]


def sample_report(df, max_ecc=MAX_ECC, max_alt_km=HIGH_ALT_MIN_KM):
    """How many rows each main-sample filter removes (count and share), applied in order."""
    steps = [("active satellite", df["is_active"] == 1),
             (f"high-altitude control (alt >= {max_alt_km:g} km)", df["alt_km"] >= max_alt_km),
             (f"ecc >= {max_ecc}", df["ecc"] >= max_ecc),
             ("TLE failed cleaning", df["tle_ok"] == 0),
             ("no drag_sensitivity", df["drag_sensitivity"].isna())]
    keep = pd.Series(True, index=df.index)
    rows = []
    for name, drop in steps:
        removed = int((keep & drop).sum())
        rows.append({"filter": name, "removed": removed, "share": removed / len(df)})
        keep &= ~drop
    rows.append({"filter": "kept", "removed": int(keep.sum()), "share": keep.mean()})
    return pd.DataFrame(rows)


def bridges_onset(df, storms):
    """Boolean Series: True where an interval spans a storm's main_onset (should be none).

    Interval ends are rebuilt as t_mid -/+ dt_days / 2, with 1 s tolerance for rounding.
    """
    half = pd.to_timedelta(df["dt_days"] / 2, unit="D") - pd.Timedelta(seconds=1)
    start, end = df["t_mid"] - half, df["t_mid"] + half
    out = pd.Series(False, index=df.index)
    for onset in storms["main_onset"]:
        out |= (start < onset) & (end > onset)
    return out


def save_table(df, path=None):
    """Validate and save as Parquet (default data/processed/analysis_table.parquet)."""
    if path is None:
        path = PROCESSED_DIR / "analysis_table.parquet"
    df = validate(df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_table(path=None):
    """Read and validate the saved analysis table."""
    if path is None:
        path = PROCESSED_DIR / "analysis_table.parquet"
    return validate(pd.read_parquet(path))


def link_storms(t_start, t_end, storms):
    """Storm columns T_storm, storm_id, storm_class, event_day for a set of intervals.

    t_start, t_end: interval ends (naive UTC); storms: catalog from storms.storm_catalog
    (columns storm_id, main_onset, class, recovery_end). The interval is linked to the
    storm whose event window (-10..+20 days around main_onset) contains its midpoint;
    if two windows overlap, the storm with the nearest main_onset wins. T_storm = 1 if
    the interval overlaps [main_onset, recovery_end] of that storm.
    """
    t_start = pd.DatetimeIndex(t_start)
    t_end = pd.DatetimeIndex(t_end)
    t_mid = t_start + (t_end - t_start) / 2
    n = len(t_mid)
    lo, hi = EVENT_WINDOW
    storm_id = pd.array([pd.NA] * n, dtype="Int64")
    storm_class = pd.array([pd.NA] * n, dtype="string")
    event_day = np.full(n, np.nan)
    t_storm = np.zeros(n, dtype="int64")
    best = np.full(n, np.inf)
    for s in storms.to_dict("records"):
        day = (t_mid - s["main_onset"]).total_seconds().to_numpy() / 86400
        take = (day >= lo) & (day <= hi) & (np.abs(day) < best)
        best[take] = np.abs(day[take])
        storm_id[take] = s["storm_id"]
        storm_class[take] = s["class"]
        event_day[take] = day[take]
        inside = (t_end > s["main_onset"]) & (t_start < s["recovery_end"])
        t_storm[take] = inside[take]
    return pd.DataFrame({"T_storm": t_storm, "storm_id": storm_id,
                         "storm_class": storm_class, "event_day": event_day})
