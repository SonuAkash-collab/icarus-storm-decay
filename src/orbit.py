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
DEFAULT_MAX_GAP_DAYS = 5.0
DEFAULT_CLIMB_NOISE_MULTIPLIER = 3.0


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


def tle_noise_by_object(tles, rolling_window=11, min_noise_km=0.01):
    """Estimate each object's semi-major-axis TLE noise in km.

    The estimate is 1.4826 times the local rolling-median residual MAD. This
    leaves the physical decay trend in place while measuring the point-to-point
    fitting noise around it. A small floor avoids treating a perfectly smooth
    synthetic or very short history as noise-free.
    """
    required = {"norad_id", "epoch", "a_km"}
    missing = required.difference(tles.columns)
    if missing:
        raise ValueError(f"TLE frame is missing required columns: {sorted(missing)}")
    if rolling_window < 3 or rolling_window % 2 == 0:
        raise ValueError("rolling_window must be an odd integer >= 3")
    if min_noise_km <= 0:
        raise ValueError("min_noise_km must be > 0")

    estimates = {}
    for norad_id, group in tles.sort_values("epoch").groupby("norad_id", sort=False):
        a = group["a_km"].reset_index(drop=True)
        trend = a.rolling(rolling_window, center=True, min_periods=3).median()
        residual = (a - trend).dropna()
        if len(residual):
            centered = residual - residual.median()
            noise = 1.4826 * centered.abs().median()
        else:
            noise = np.nan
        estimates[int(norad_id)] = max(float(noise) if pd.notna(noise) else 0.0,
                                  min_noise_km)
    return pd.Series(estimates, name="tle_noise_km", dtype="float64")


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
        {"step": "TLEs failed point-level checks", "count": int((clean["tle_ok"] == 0).sum())},
        {"step": "TLEs retained by point-level checks", "count": int(clean["tle_ok"].sum())},
    ])
    return clean.reset_index(drop=True), report


def cleaning_sensitivity(tles, settings=None):
    """Report point-level TLE cleaning shares under named threshold settings.

    The default tight setting is deliberately stricter than the production setting
    and is a diagnostic report, not a replacement analysis filter.
    """
    if settings is None:
        settings = {
            "baseline": {},
            "tight": {
                "max_inc_jump_deg": 0.25,
                "max_raan_rate_residual_deg_per_day": 0.5,
                "rolling_mad_threshold": 5.0,
                "outlier_floor_km": 0.25,
            },
        }
    rows = []
    for name, kwargs in settings.items():
        cleaned, report = clean_tle_history(tles, **kwargs)
        reason_counts = report.set_index("step")["count"]
        rows.append({
            "setting": name,
            "tle_count": len(cleaned),
            "flagged": int((cleaned["tle_ok"] == 0).sum()),
            "flagged_share": float((cleaned["tle_ok"] == 0).mean()),
            "inclination_jumps": int(reason_counts["inclination jumps flagged"]),
            "raan_jumps": int(reason_counts["RAAN jumps flagged"]),
            "rolling_outliers": int(reason_counts["rolling-median outliers flagged"]),
        })
    return pd.DataFrame(rows)


def flagged_tles_by_event_day(tles, storms, settings=None):
    """Return tight-setting flagged TLEs with storm-relative timing.

    ``event_day`` is measured from the nearest storm ``main_onset`` when the TLE
    falls in the configured -10..+20-day event window. ``period`` distinguishes
    the active storm interval, its event-window pre/post days, and calm time. This
    makes it possible to check that the diagnostic filter is not selectively
    deleting storm-time orbit response.
    """
    cleaned, _ = clean_tle_history(tles, **(settings or {
        "max_inc_jump_deg": 0.25,
        "max_raan_rate_residual_deg_per_day": 0.5,
        "rolling_mad_threshold": 5.0,
        "outlier_floor_km": 0.25,
    }))
    flagged = cleaned.loc[cleaned["tle_ok"].eq(0)].copy()
    onset = pd.to_datetime(storms["main_onset"]).sort_values().to_numpy()
    storm_ids = storms.sort_values("main_onset")["storm_id"].to_numpy()
    recovery = pd.to_datetime(storms.sort_values("main_onset")["recovery_end"]).to_numpy()
    event_days, ids, periods = [], [], []
    for timestamp in pd.to_datetime(flagged["epoch"]):
        distances = np.abs((onset - timestamp.to_datetime64()) / np.timedelta64(1, "D"))
        nearest = int(np.argmin(distances))
        day = float((timestamp.to_datetime64() - onset[nearest]) / np.timedelta64(1, "D"))
        if -10 <= day <= 20:
            event_days.append(day)
            ids.append(int(storm_ids[nearest]))
            if timestamp.to_datetime64() <= recovery[nearest]:
                periods.append("storm")
            elif day < 0:
                periods.append("pre_event_window")
            else:
                periods.append("post_event_window")
        else:
            event_days.append(np.nan)
            ids.append(pd.NA)
            periods.append("calm")
    flagged["storm_id"] = pd.array(ids, dtype="Int64")
    flagged["event_day"] = event_days
    flagged["period"] = periods
    return flagged[["norad_id", "epoch", "storm_id", "event_day", "period", "tle_ok"]]


def build_decay_intervals(
    tles,
    storms,
    *,
    passive=True,
    climb_tolerance_km=None,
    climb_noise_multiplier=DEFAULT_CLIMB_NOISE_MULTIPLIER,
    max_gap_days=DEFAULT_MAX_GAP_DAYS,
):
    """Build adjacent-TLE decay intervals and omit any interval crossing an onset.

    ``storms`` must contain ``main_onset`` timestamps. Intervals longer than
    ``max_gap_days`` and those spanning an onset are omitted. Intervals are retained
    with ``tle_ok=0`` when a point-level check failed or a passive object's mean
    semi-major axis climbed by more than three times its robust TLE-noise estimate.
    This keeps the cleaning share auditable without treating a tiny fitted-element
    rise as physical climb.
    """
    if "tle_ok" not in tles:
        raise ValueError("Call clean_tle_history first; expected a 'tle_ok' column")
    if "main_onset" not in storms:
        raise ValueError("storms must contain a 'main_onset' column")
    if climb_tolerance_km is not None and climb_tolerance_km < 0:
        raise ValueError("climb_tolerance_km must be >= 0")
    if climb_noise_multiplier <= 0 or max_gap_days <= 0:
        raise ValueError("climb_noise_multiplier and max_gap_days must be > 0")

    onsets = pd.DatetimeIndex(pd.to_datetime(storms["main_onset"]))
    if climb_tolerance_km is None:
        noise = tle_noise_by_object(tles)
        tolerances = noise * climb_noise_multiplier
    else:
        tolerances = pd.Series(float(climb_tolerance_km),
                               index=tles["norad_id"].unique(), dtype="float64")
    rows = []
    omitted_onset = 0
    omitted_gap = 0
    climb_flags = 0
    invalid_endpoint = 0
    for norad_id, group in tles.sort_values("epoch").groupby("norad_id", sort=False):
        g = group.sort_values("epoch").reset_index(drop=True)
        for i in range(len(g) - 1):
            left, right = g.iloc[i], g.iloc[i + 1]
            t0, t1 = pd.Timestamp(left["epoch"]), pd.Timestamp(right["epoch"])
            if t1 <= t0:
                continue
            dt_days = (t1 - t0).total_seconds() / 86400.0
            if dt_days > max_gap_days:
                omitted_gap += 1
                continue
            # An onset exactly at an interval boundary does not lie inside it.
            if ((onsets > t0) & (onsets < t1)).any():
                omitted_onset += 1
                continue

            da_km = float(right["a_km"] - left["a_km"])
            tolerance = float(tolerances.loc[int(norad_id)])
            climb = bool(passive and da_km > tolerance)
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
                "climb_tolerance_km": tolerance,
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
               "day_of_year", "climb_tolerance_km", "tle_ok", "climb_flag"]
    intervals = pd.DataFrame(rows, columns=columns)
    report = pd.DataFrame([
        {"step": "intervals built", "count": len(intervals)},
        {"step": f"interval gaps > {max_gap_days:g} days omitted", "count": omitted_gap},
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
