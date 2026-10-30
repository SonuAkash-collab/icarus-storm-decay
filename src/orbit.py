"""Orbit quantities from TLEs: epoch, eccentricity, Brouwer mean semi-major axis, decay rate.

We use python-sgp4. `Satrec.no_kozai` is the Brouwer mean motion (rad/min) that SGP4
recovers from the TLE's Kozai mean motion. From it: a = (mu / n^2)^(1/3).
BSTAR is deliberately not used anywhere.
"""

import numpy as np
import pandas as pd
from sgp4.api import Satrec, WGS72

# SGP4 uses WGS-72 constants, so we use the same mu (km^3/s^2) for consistency
MU_WGS72 = 398600.8
EARTH_EQUATORIAL_RADIUS_KM = 6378.137


def semi_major_axis_km(no_kozai_rad_per_min, mu=MU_WGS72):
    """Semi-major axis (km) from mean motion in rad/min: a = (mu / n^2)^(1/3)."""
    n = no_kozai_rad_per_min / 60.0  # rad/s
    return (mu / n**2) ** (1.0 / 3.0)


def _solar_right_ascension_deg(timestamp):
    """Low-precision apparent solar right ascension for local-time features."""
    jd = pd.Timestamp(timestamp).to_julian_date()
    days = jd - 2451545.0
    mean_longitude = np.radians((280.460 + 0.9856474 * days) % 360.0)
    mean_anomaly = np.radians((357.528 + 0.9856003 * days) % 360.0)
    ecliptic_longitude = (mean_longitude + np.radians(1.915) * np.sin(mean_anomaly)
                          + np.radians(0.020) * np.sin(2 * mean_anomaly))
    obliquity = np.radians(23.439 - 0.0000004 * days)
    right_ascension = np.arctan2(np.cos(obliquity) * np.sin(ecliptic_longitude),
                                 np.cos(ecliptic_longitude))
    return float(np.degrees(right_ascension) % 360.0)


def tle_to_row(line1, line2):
    """Parse one TLE and return its epoch, eccentricity, inclination and Brouwer mean a."""
    sat = Satrec.twoline2rv(line1, line2, WGS72)
    # Epoch as a Julian date -> pandas UTC timestamp (rounded to the microsecond)
    jd = sat.jdsatepoch + sat.jdsatepochF
    epoch = pd.Timestamp("2000-01-01 12:00:00") + pd.to_timedelta(jd - 2451545.0, unit="D")
    return {
        "norad_id": sat.satnum,
        "epoch": epoch.round("us"),
        "ecc": sat.ecco,
        "incl_deg": np.degrees(sat.inclo),
        "raan_deg": np.degrees(sat.nodeo) % 360.0,
        "local_solar_time": ((12.0 + (np.degrees(sat.nodeo) - _solar_right_ascension_deg(epoch)) / 15.0) % 24.0),
        "a_km": semi_major_axis_km(sat.no_kozai),
    }


def tles_to_frame(records):
    """Turn Space-Track GP records (dicts with TLE_LINE1/TLE_LINE2) into a DataFrame.

    Keeps CREATION_DATE (when Space-Track published the TLE) so duplicates can be resolved,
    and the two TLE lines so the orbit can be propagated later (src/physics.py).
    """
    rows = []
    for rec in records:
        row = tle_to_row(rec["TLE_LINE1"], rec["TLE_LINE2"])
        row["created"] = pd.Timestamp(rec.get("CREATION_DATE")) if rec.get("CREATION_DATE") else pd.NaT
        row["line1"] = rec["TLE_LINE1"]
        row["line2"] = rec["TLE_LINE2"]
        rows.append(row)
    return pd.DataFrame(rows)


def drop_duplicate_epochs(df, tol=pd.Timedelta(seconds=1)):
    """Keep one TLE per (object, epoch): the latest published one.

    Space-Track sometimes re-issues a TLE with an epoch shifted by a few milliseconds
    (TLE epochs only have ~1 ms precision). Epochs closer than `tol` are therefore
    treated as the same epoch. If no publication time is known, the row appearing
    last in the input wins.
    """
    df = df.reset_index(drop=True)
    df["_order"] = np.arange(len(df))
    df = df.sort_values(["norad_id", "epoch", "_order"])

    # Group id: a new group starts when the object changes or the epoch gap >= tol
    new_group = (df["norad_id"] != df["norad_id"].shift()) | (df["epoch"].diff() >= tol)
    df["_group"] = new_group.cumsum()

    # Within each group, keep the latest published row
    sort_cols = ["_group"] + (["created"] if "created" in df else []) + ["_order"]
    df = df.sort_values(sort_cols, na_position="first")
    df = df.drop_duplicates(subset="_group", keep="last")
    df = df.sort_values(["norad_id", "epoch"])
    return df.drop(columns=["_order", "_group"]).reset_index(drop=True)


def _raan_rate_residual_deg_per_day(raan_deg, epoch, window):
    """Difference between each RAAN drift rate and its local median rate.

    RAAN itself precesses by several degrees per day in LEO, so a raw RAAN jump is
    not a valid cross-tag check. This detects a discontinuity from that object's
    usual local precession instead.
    """
    unwrapped = np.degrees(np.unwrap(np.radians(raan_deg.to_numpy(dtype=float))))
    dt_days = epoch.diff().dt.total_seconds().to_numpy() / 86400.0
    rate = pd.Series(np.r_[np.nan, np.diff(unwrapped) / dt_days[1:]], index=raan_deg.index)
    baseline = rate.rolling(window, center=True, min_periods=3).median()
    return (rate - baseline).abs()


def clean_tle_history(
    df,
    *,
    passive=True,
    max_inc_jump_deg=1.0,
    max_raan_rate_residual_deg_per_day=2.0,
    rolling_window=11,
    rolling_mad_threshold=8.0,
    outlier_floor_km=2.0,
    duplicate_tolerance=pd.Timedelta(seconds=1),
):
    """Deduplicate TLEs and flag local fit jumps/outliers without deleting rows.

    Returns ``(cleaned, report)``. ``cleaned.tle_ok`` is a point-level flag; the
    interval builder below additionally rejects an interval if either endpoint
    failed or a passive object's mean semi-major axis increased beyond its
    configured climb tolerance. Thresholds are keyword arguments so the team can
    report and sensitivity-check its cleaning choices.

    ``rolling_window`` is a centered count of neighboring TLEs for each object.
    The rolling median check uses the local median absolute deviation (MAD), with
    ``outlier_floor_km`` preventing a zero-MAD window from flagging tiny rounding
    differences.
    """
    required = {"norad_id", "epoch", "a_km", "ecc", "incl_deg"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"TLE frame is missing required columns: {sorted(missing)}")
    if max_inc_jump_deg <= 0 or max_raan_rate_residual_deg_per_day <= 0:
        raise ValueError("inclination and RAAN thresholds must be positive")
    if rolling_window < 3 or rolling_window % 2 == 0:
        raise ValueError("rolling_window must be an odd integer >= 3")
    if rolling_mad_threshold <= 0 or outlier_floor_km < 0:
        raise ValueError("outlier thresholds must be nonnegative, with MAD threshold > 0")

    before = len(df)
    clean = drop_duplicate_epochs(df, tol=duplicate_tolerance).copy()
    clean["tle_ok"] = 1
    reasons = {"inclination_jump": 0, "raan_jump": 0, "rolling_median_outlier": 0}
    half = rolling_window // 2

    for _, indices in clean.groupby("norad_id", sort=False).groups.items():
        g = clean.loc[indices].sort_values("epoch")
        bad = pd.Series(False, index=g.index)

        inc_jump = g["incl_deg"].diff().abs() > max_inc_jump_deg
        reasons["inclination_jump"] += int(inc_jump.sum())
        bad |= inc_jump

        if "raan_deg" in g:
            raan_jump = (_raan_rate_residual_deg_per_day(
                g["raan_deg"], g["epoch"], rolling_window
            ) > max_raan_rate_residual_deg_per_day)
            reasons["raan_jump"] += int(raan_jump.sum())
            bad |= raan_jump

        median = g["a_km"].rolling(rolling_window, center=True, min_periods=3).median()
        abs_dev = (g["a_km"] - median).abs()
        mad = abs_dev.rolling(rolling_window, center=True, min_periods=3).median()
        threshold = np.maximum(outlier_floor_km, rolling_mad_threshold * 1.4826 * mad)
        outlier = abs_dev > threshold
        reasons["rolling_median_outlier"] += int(outlier.sum())
        bad |= outlier

        clean.loc[g.index, "tle_ok"] = (~bad).astype("int64")

    report = pd.DataFrame([
        {"step": "input TLEs", "count": before},
        {"step": "duplicate epochs removed", "count": before - len(clean)},
        {"step": "inclination jumps flagged", "count": reasons["inclination_jump"]},
        {"step": "RAAN jumps flagged", "count": reasons["raan_jump"]},
        {"step": "rolling-median outliers flagged", "count": reasons["rolling_median_outlier"]},
        {"step": "TLEs retained by point-level checks", "count": int(clean["tle_ok"].sum())},
    ])
    return clean.reset_index(drop=True), report


def build_decay_intervals(
    tles,
    storms,
    *,
    passive=True,
    climb_tolerance_km=0.0,
):
    """Build adjacent-TLE decay intervals and omit any interval crossing an onset.

    ``storms`` must contain ``main_onset`` timestamps. Intervals are retained with
    ``tle_ok=0`` when a point-level check failed or a passive object's semi-major
    axis climbed beyond ``climb_tolerance_km``; this keeps the cleaning share
    auditable. Intervals spanning a storm onset are omitted entirely so later
    physics joins cannot use a decay rate that averages across day 0.
    """
    if "tle_ok" not in tles:
        raise ValueError("Call clean_tle_history first; expected a 'tle_ok' column")
    if "main_onset" not in storms:
        raise ValueError("storms must contain a 'main_onset' column")
    if climb_tolerance_km < 0:
        raise ValueError("climb_tolerance_km must be >= 0")

    onsets = pd.DatetimeIndex(pd.to_datetime(storms["main_onset"]))
    rows = []
    omitted_onset = 0
    climb_flags = 0
    invalid_endpoint = 0
    for norad_id, group in tles.sort_values("epoch").groupby("norad_id", sort=False):
        g = group.sort_values("epoch").reset_index(drop=True)
        for i in range(len(g) - 1):
            left, right = g.iloc[i], g.iloc[i + 1]
            t0, t1 = pd.Timestamp(left["epoch"]), pd.Timestamp(right["epoch"])
            if t1 <= t0:
                continue
            # An onset exactly at an interval boundary does not lie inside it.
            if ((onsets > t0) & (onsets < t1)).any():
                omitted_onset += 1
                continue

            dt_days = (t1 - t0).total_seconds() / 86400.0
            da_km = float(right["a_km"] - left["a_km"])
            climb = bool(passive and da_km > climb_tolerance_km)
            endpoints_ok = bool(left["tle_ok"] == 1 and right["tle_ok"] == 1)
            climb_flags += int(climb)
            invalid_endpoint += int(not endpoints_ok)
            midpoint = t0 + (t1 - t0) / 2
            rows.append({
                "norad_id": int(norad_id),
                "t_start": t0,
                "t_end": t1,
                "t_mid": midpoint,
                "dt_days": dt_days,
                "da_km": da_km,
                "da_dt_km_per_day": da_km / dt_days,
                "alt_km": (float(left["a_km"]) + float(right["a_km"])) / 2 - EARTH_EQUATORIAL_RADIUS_KM,
                "ecc": (float(left["ecc"]) + float(right["ecc"])) / 2,
                "inc_deg": (float(left["incl_deg"]) + float(right["incl_deg"])) / 2,
                "local_solar_time": float(np.degrees(np.angle(
                    np.exp(1j * np.radians(float(left["local_solar_time"]) * 15.0))
                    + np.exp(1j * np.radians(float(right["local_solar_time"]) * 15.0)))
                ) % 360.0) / 15.0,
                "day_of_year": int(midpoint.dayofyear),
                "tle_ok": int(endpoints_ok and not climb),
                "climb_flag": int(climb),
            })
    columns = ["norad_id", "t_start", "t_end", "t_mid", "dt_days", "da_km",
               "da_dt_km_per_day", "alt_km", "ecc", "inc_deg", "local_solar_time",
               "day_of_year", "tle_ok", "climb_flag"]
    intervals = pd.DataFrame(rows, columns=columns)
    report = pd.DataFrame([
        {"step": "intervals built", "count": len(intervals)},
        {"step": "intervals crossing storm onset omitted", "count": omitted_onset},
        {"step": "passive climb intervals flagged", "count": climb_flags},
        {"step": "intervals with failed endpoint checks", "count": invalid_endpoint},
        {"step": "intervals passing all A checks", "count": int(intervals["tle_ok"].sum())},
    ])
    return intervals, report


def decay_rates(df):
    """Decay rate between consecutive TLEs of each object.

    da_dt_km_per_day = (a[i+1] - a[i]) / (t[i+1] - t[i]), stamped at the midpoint time.
    Negative values mean the orbit is shrinking. Duplicates must already be removed.
    """
    out = []
    for norad_id, g in df.sort_values("epoch").groupby("norad_id"):
        t0 = g["epoch"].iloc[:-1].to_numpy()
        t1 = g["epoch"].iloc[1:].to_numpy()
        a0 = g["a_km"].iloc[:-1].to_numpy()
        a1 = g["a_km"].iloc[1:].to_numpy()
        dt_days = (t1 - t0) / np.timedelta64(1, "D")
        out.append(pd.DataFrame({
            "norad_id": norad_id,
            "t_start": t0,
            "t_end": t1,
            "t_mid": t0 + (t1 - t0) / 2,
            "dt_days": dt_days,
            "da_km": a1 - a0,
            "da_dt_km_per_day": (a1 - a0) / dt_days,
        }))
    rates = pd.concat(out, ignore_index=True)
    if (rates["dt_days"] <= 0).any():
        raise ValueError("Found an interval with dt <= 0; remove duplicate epochs first")
    return rates


def mean_decay_before_after(rates, event_time, days=7):
    """Time-weighted mean decay rate (km/day) in the `days` before and after an event.

    Only intervals that lie fully on one side of the event are used, so no interval
    crosses the storm onset. Time-weighted mean = sum(da) / sum(dt).
    """
    event_time = pd.Timestamp(event_time)
    win = pd.Timedelta(days=days)
    before = rates[(rates["t_start"] >= event_time - win) & (rates["t_end"] <= event_time)]
    after = rates[(rates["t_start"] >= event_time) & (rates["t_end"] <= event_time + win)]

    def weighted(x):
        return x["da_km"].sum() / x["dt_days"].sum() if len(x) else np.nan

    rows = []
    for norad_id in sorted(rates["norad_id"].unique()):
        b = before[before["norad_id"] == norad_id]
        a = after[after["norad_id"] == norad_id]
        # Rough standard error of the change, from the scatter of interval rates.
        # Consecutive TLE fits share tracking data, so treat this as a lower bound.
        se = np.sqrt(b["da_dt_km_per_day"].var() / len(b) + a["da_dt_km_per_day"].var() / len(a))
        rows.append({"norad_id": norad_id,
                     "before_km_per_day": weighted(b), "n_before": len(b),
                     "after_km_per_day": weighted(a), "n_after": len(a),
                     "se_change": se})
    res = pd.DataFrame(rows)
    res["change_km_per_day"] = res["after_km_per_day"] - res["before_km_per_day"]
    res["ratio_after_before"] = res["after_km_per_day"] / res["before_km_per_day"]
    return res
