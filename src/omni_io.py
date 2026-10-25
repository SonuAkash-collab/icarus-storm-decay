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
