"""Reference-orbit parsing and TLE decay validation.

The TU Graz ITSG operational archive publishes GRACE-FO and Swarm reduced-dynamic
orbits in a GROOPS text format: MJD, Cartesian position in metres, and velocity in
metres/second. We derive osculating semi-major axis with the vis-viva equation and
compare its daily trend with the TLE mean semi-major axis.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from src.orbit import MU_WGS72


def parse_groops_orbit(path):
    """Read a GROOPS reduced-dynamic orbit file into a time-indexed DataFrame."""
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", errors="replace") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith(("#", "groops")):
                continue
            values = line.split()
            if len(values) < 7:
                continue
            try:
                mjd = float(values[0])
                position_m = np.asarray([float(value) for value in values[1:4]])
                velocity_m_s = np.asarray([float(value) for value in values[4:7]])
            except ValueError:
                continue
            radius_km = np.linalg.norm(position_m) / 1000.0
            speed_km_s = np.linalg.norm(velocity_m_s) / 1000.0
            semi_major_axis_km = 1.0 / (2.0 / radius_km - speed_km_s**2 / MU_WGS72)
            timestamp = pd.Timestamp("1858-11-17") + pd.to_timedelta(mjd, unit="D")
            rows.append({
                "time": timestamp,
                "a_km_precise": semi_major_axis_km,
                "radius_km": radius_km,
            })
    if not rows:
        raise ValueError(f"No orbit state rows found in {path}")
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def daily_precise_axis(precise, freq="D"):
    """Daily median precise semi-major axis from parsed state rows."""
    data = precise.set_index("time")["a_km_precise"].resample(freq).median().dropna()
    return data.rename("a_km_precise").reset_index()


def estimate_orbit_period_minutes(precise, min_period_minutes=70.0, max_period_minutes=130.0):
    """Estimate the revolution period from repeated radius minima."""
    data = precise.sort_values("time").drop_duplicates("time")
    if "radius_km" not in data or len(data) < 10:
        raise ValueError("Need unique precise states with radius_km to estimate period")
    radius = data["radius_km"].to_numpy()
    times = data["time"].to_numpy()
    minima = np.flatnonzero((radius[1:-1] <= radius[:-2]) & (radius[1:-1] < radius[2:])) + 1
    gaps = np.diff(times[minima]).astype("timedelta64[s]").astype(float) / 60.0
    gaps = gaps[(gaps >= min_period_minutes) & (gaps <= max_period_minutes)]
    if not len(gaps):
        raise ValueError("Could not identify a plausible orbital period")
    return float(np.median(gaps))


def orbit_averaged_precise_axis(precise, period_minutes=None):
    """Average precise osculating semi-major axis over one orbital period.

    A daily median still contains the within-orbit oscillation. Binning the 10 s
    states into successive orbital periods removes that oscillation before daily
    decay rates are computed.
    """
    if period_minutes is None:
        period_minutes = estimate_orbit_period_minutes(precise)
    if period_minutes <= 0:
        raise ValueError("period_minutes must be positive")
    data = precise.sort_values("time").copy()
    elapsed_minutes = (data["time"] - data["time"].iloc[0]).dt.total_seconds() / 60.0
    data["orbit_number"] = np.floor(elapsed_minutes / period_minutes).astype(int)
    return (data.groupby("orbit_number", as_index=False)
            .agg(time=("time", "median"), a_km_precise=("a_km_precise", "median")))


def daily_decay_rates(axis, value_column, smoothing_days=1):
    """Return daily d(a)/dt in km/day from an orbit-averaged or TLE axis table."""
    if smoothing_days < 1:
        raise ValueError("smoothing_days must be positive")
    values = axis.set_index("time")[value_column].resample("D").median().dropna()
    if smoothing_days > 1:
        values = values.rolling(smoothing_days, center=True, min_periods=1).median()
    return values.diff().rename("decay_km_per_day").dropna().reset_index()


def compare_decay_rates(tles, precise, period_minutes=94.0, smoothing_days=1, max_lag_days=3):
    """Compare daily TLE and orbit-averaged precise decay rates with a lag sweep."""
    tle = tles[["epoch", "a_km"]].rename(columns={"epoch": "time", "a_km": "a_km_tle"})
    tle["time"] = pd.to_datetime(tle["time"])
    tle = tle.groupby(tle["time"].dt.floor("D"), as_index=False)["a_km_tle"].median()
    tle = tle.rename(columns={"time": "time"})
    precise_axis = orbit_averaged_precise_axis(precise, period_minutes)
    precise_daily = daily_decay_rates(precise_axis, "a_km_precise", smoothing_days)
    tle_daily = daily_decay_rates(tle, "a_km_tle", smoothing_days)
    rates = tle_daily.rename(columns={"decay_km_per_day": "tle_decay_km_per_day"}).merge(
        precise_daily.rename(columns={"decay_km_per_day": "precise_decay_km_per_day"}),
        on="time", how="inner",
    )
    lag_rows = []
    for lag in range(-max_lag_days, max_lag_days + 1):
        shifted = precise_daily.copy()
        shifted["time"] = shifted["time"] + pd.Timedelta(days=lag)
        joined = tle_daily.merge(shifted, on="time")
        corr = joined.iloc[:, 1].corr(joined.iloc[:, 2]) if len(joined) >= 3 else np.nan
        lag_rows.append({"lag_days": lag, "n_days": len(joined), "correlation": corr})
    lags = pd.DataFrame(lag_rows)
    return rates, lags


def compare_tle_precise(tles, precise, smoothing_days=3):
    """Compare daily TLE and precise-orbit semi-major-axis trends.

    Absolute levels have a mean-element versus osculating-element offset, so the
    agreement metrics remove each series' median before calculating bias/RMSE. The
    returned table retains the aligned levels and centred values for plotting.
    """
    if smoothing_days < 1:
        raise ValueError("smoothing_days must be positive")
    tle = tles[["epoch", "a_km"]].copy()
    tle["time"] = pd.to_datetime(tle["epoch"]).dt.floor("D")
    tle = tle.groupby("time", as_index=False)["a_km"].median()
    ref = daily_precise_axis(precise)
    merged = tle.merge(ref, on="time", how="inner").sort_values("time")
    if len(merged) < 3:
        raise ValueError("Need at least three overlapping daily TLE/reference points")
    merged["tle_smooth"] = merged["a_km"].rolling(smoothing_days, center=True, min_periods=1).median()
    merged["precise_smooth"] = merged["a_km_precise"].rolling(smoothing_days, center=True, min_periods=1).median()
    merged["tle_centered"] = merged["tle_smooth"] - merged["tle_smooth"].median()
    merged["precise_centered"] = merged["precise_smooth"] - merged["precise_smooth"].median()
    delta = merged["tle_centered"] - merged["precise_centered"]
    slope_tle = np.polyfit(np.arange(len(merged)), merged["tle_centered"], 1)[0]
    slope_precise = np.polyfit(np.arange(len(merged)), merged["precise_centered"], 1)[0]
    summary = pd.DataFrame([{
        "n_days": len(merged),
        "median_level_offset_km": float((merged["a_km"] - merged["a_km_precise"]).median()),
        "centred_rmse_km": float(np.sqrt(np.mean(delta**2))),
        "centred_correlation": float(merged["tle_centered"].corr(merged["precise_centered"])),
        "tle_slope_km_per_day": float(slope_tle),
        "precise_slope_km_per_day": float(slope_precise),
        "slope_difference_km_per_day": float(slope_tle - slope_precise),
    }])
    return merged, summary


def load_orbit_files(paths):
    """Parse and concatenate multiple daily precise-orbit files."""
    frames = [parse_groops_orbit(Path(path)) for path in paths]
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates("time")
            .sort_values("time")
            .reset_index(drop=True))
