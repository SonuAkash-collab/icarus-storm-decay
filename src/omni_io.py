"""Download and parse hourly OMNI2 data (space weather indices) from NASA SPDF.

Format reference: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
Each line is one hour with 55 whitespace-separated words.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"

# Word number (1-based, from omni2.text) -> (our column name, fill value)
COLUMNS = {
    1: ("year", None),
    2: ("doy", None),
    3: ("hour", None),
    39: ("kp10", 99),       # Kp * 10 (e.g. 37 means Kp = 3.7)
    41: ("dst", 99999),     # Dst index, nT
    50: ("ap", 999),        # ap index, nT (linear scale)
    51: ("f107", 999.9),    # F10.7 solar radio flux, sfu
}
N_WORDS = 55


def download_omni2(year=2024, force=False):
    """Download omni2_<year>.dat into data/raw/ (skipped if already there)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"omni2_{year}.dat"
    if force or not path.exists():
        resp = requests.get(BASE_URL.format(year=year), timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def parse_omni2(path):
    """Parse an OMNI2 hourly file into a DataFrame indexed by UTC time.

    Columns: dst (nT), ap (nT), kp (0-9), f107 (sfu). Fill values become NaN.
    """
    table = np.loadtxt(path)
    if table.shape[1] != N_WORDS:
        raise ValueError(f"Expected {N_WORDS} columns, got {table.shape[1]}")

    df = pd.DataFrame({name: table[:, word - 1] for word, (name, _) in COLUMNS.items()})

    # Replace fill values with NaN
    for word, (name, fill) in COLUMNS.items():
        if fill is not None:
            df.loc[np.isclose(df[name], fill), name] = np.nan

    # Build the timestamp from year + day-of-year + hour
    df["time"] = (pd.to_datetime(df["year"].astype(int).astype(str), format="%Y")
                  + pd.to_timedelta(df["doy"] - 1, unit="D")
                  + pd.to_timedelta(df["hour"], unit="h"))
    df["kp"] = df["kp10"] / 10.0
    return df.set_index("time")[["dst", "ap", "kp", "f107"]]


def load_omni2(year=2024):
    """Download (if needed) and parse one year of OMNI2."""
    return parse_omni2(download_omni2(year))


# ---------------------------------------------------------------------------
# Full study period: hourly space-weather table
# ---------------------------------------------------------------------------

PROCESSED_DIR = ROOT / "data" / "processed"
PERIOD_START = pd.Timestamp("2024-04-01 00:00")
PERIOD_END = pd.Timestamp("2025-04-30 23:00")


FLARE_RATIO = 1.5   # daily F10.7 above 1.5x its 7-day centred median = flare spike


def remove_f107_flare_spikes(f107_hourly, ratio=FLARE_RATIO):
    """Replace one-day F10.7 spikes caused by solar flares.

    The daily F10.7 is measured around local noon at Penticton. If a flare happens
    during the measurement the value jumps for that one day (e.g. 412.9 sfu on
    30 Jul 2024, between ~230 and ~242). Such a day is not the UV background we want.
    A day counts as a spike if it is above `ratio` x the median of the 7 days centred
    on it; it is replaced by the mean of the day before and the day after.
    Returns (cleaned hourly series, boolean hourly flag).
    """
    daily = f107_hourly.resample("D").mean()
    spike = daily > ratio * daily.rolling(7, center=True, min_periods=4).median()
    neighbours = (daily.shift(1) + daily.shift(-1)) / 2
    cleaned_daily = daily.where(~spike, neighbours)
    for day in daily.index[spike]:
        print(f"F10.7 flare spike on {day.date()}: {daily[day]:.1f} -> {cleaned_daily[day]:.1f} sfu")
    days = f107_hourly.index.floor("D")
    return (pd.Series(cleaned_daily.reindex(days).to_numpy(), index=f107_hourly.index),
            pd.Series(spike.reindex(days).to_numpy(), index=f107_hourly.index))


def f107_81day_mean(f107_hourly):
    """81-day centred mean of daily F10.7, put back on the hourly time grid.

    OMNI2 repeats the daily F10.7 value in every hour of the day, so the daily value is
    the mean of that day's hours. A day only gets a value if all 81 days are present.
    """
    daily = f107_hourly.resample("D").mean()
    mean81 = daily.rolling(81, center=True, min_periods=81).mean()
    # Every hour of a day gets that day's 81-day mean
    return mean81.reindex(f107_hourly.index.floor("D")).to_numpy()


def build_spaceweather(start=PERIOD_START, end=PERIOD_END, out_path=None):
    """Hourly table of ap, dst, kp, f107, f107_81d, f107_flare for start..end (UTC, inclusive).

    Loads every OMNI2 year needed (plus 40 days on each side for the 81-day mean),
    puts everything on a complete hourly grid, prints any missing hours, and saves
    a Parquet file (default data/processed/spaceweather.parquet).
    Times are naive timestamps in UTC, like the TLE epochs in src/orbit.py.
    Kp is stored as given (3-hour value repeated each hour); never average it.
    One-day F10.7 flare spikes are replaced first (f107_flare marks those hours).
    """
    pad = pd.Timedelta(days=40)
    years = range((start - pad).year, (end + pad).year + 1)
    omni = pd.concat([load_omni2(y) for y in years]).sort_index()
    omni = omni[~omni.index.duplicated(keep="last")]

    # Complete hourly grid, so missing hours show up as NaN rows
    full_grid = pd.date_range(start - pad, end + pad, freq="h")
    omni = omni.reindex(full_grid)
    omni["f107"], omni["f107_flare"] = remove_f107_flare_spikes(omni["f107"])
    omni["f107_81d"] = f107_81day_mean(omni["f107"])

    table = omni.loc[start:end, ["ap", "dst", "kp", "f107", "f107_81d", "f107_flare"]]
    table.index.name = "time"

    missing = table[table.isna().any(axis=1)]
    if len(missing) == 0:
        print(f"No missing hours ({len(table)} hours, {start} to {end}).")
    else:
        print(f"{len(missing)} hours with missing values:")
        print(missing.isna().sum().to_string())
        print(missing.index.to_series().to_string(index=False))

    table = table.reset_index()
    if out_path is None:
        out_path = PROCESSED_DIR / "spaceweather.parquet"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    return table


def load_spaceweather(path=None):
    """Read the saved hourly table back, indexed by time."""
    if path is None:
        path = PROCESSED_DIR / "spaceweather.parquet"
    return pd.read_parquet(path).set_index("time")
