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


def semi_major_axis_km(no_kozai_rad_per_min, mu=MU_WGS72):
    """Semi-major axis (km) from mean motion in rad/min: a = (mu / n^2)^(1/3)."""
    n = no_kozai_rad_per_min / 60.0  # rad/s
    return (mu / n**2) ** (1.0 / 3.0)


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
        "a_km": semi_major_axis_km(sat.no_kozai),
    }


def tles_to_frame(records):
    """Turn Space-Track GP records (dicts with TLE_LINE1/TLE_LINE2) into a DataFrame.

    Keeps CREATION_DATE (when Space-Track published the TLE) so duplicates can be resolved.
    """
    rows = []
    for rec in records:
        row = tle_to_row(rec["TLE_LINE1"], rec["TLE_LINE2"])
        row["created"] = pd.Timestamp(rec.get("CREATION_DATE")) if rec.get("CREATION_DATE") else pd.NaT
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
