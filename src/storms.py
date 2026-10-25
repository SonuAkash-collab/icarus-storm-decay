"""Geomagnetic storm catalog from hourly Dst.

A storm is a period with Dst below -50 nT. Dips that are less than 3 days apart
(from the last hour below -50 nT to the next first hour below -50 nT) are one event.

Per storm:
- onset_time:   start of the Dst drop = hour of the highest Dst in the 24 h before Dst
                first goes below -50 nT (often the sudden commencement bump)
- main_onset:   start of the drop that leads to the peak = hour of the highest Dst in the
                24 h before the peak (never earlier than onset_time). Differs from
                onset_time when an earlier, weaker dip was merged into the same event
                (e.g. Oct 2024: onset_time Oct 6, main_onset Oct 10).
- peak_time:    hour of the lowest Dst
- min_dst:      lowest Dst (nT)
- class:        moderate (-50 to -100), strong (-100 to -200), severe (-200 to -350),
                extreme (below -350)
- recovery_end: first hour after the last hour below -50 nT where Dst is back above -30 nT
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

STORM_THRESHOLD = -50.0     # nT
RECOVERY_LEVEL = -30.0      # nT
MERGE_GAP = pd.Timedelta(days=3)
ONSET_LOOKBACK = pd.Timedelta(hours=24)


def storm_class(min_dst):
    """Class from the lowest Dst of a storm (boundary values go to the weaker class)."""
    if min_dst < -350:
        return "extreme"
    if min_dst < -200:
        return "severe"
    if min_dst < -100:
        return "strong"
    return "moderate"


def find_dips(dst, threshold=STORM_THRESHOLD, merge_gap=MERGE_GAP):
    """List of (first, last) times below `threshold`, with nearby dips merged.

    dst: hourly Series indexed by time. Two dips are merged when the time between the
    end of one and the start of the next is less than `merge_gap`.
    """
    below = dst < threshold
    times = dst.index[below]
    if len(times) == 0:
        return []

    dips = [[times[0], times[0]]]
    for t in times[1:]:
        if t - dips[-1][1] < merge_gap:
            dips[-1][1] = t           # same event: extend it
        else:
            dips.append([t, t])       # new event
    return [tuple(d) for d in dips]


def storm_catalog(dst, out_path=None):
    """Build the storm catalog from an hourly Dst Series (index = UTC time).

    If out_path is given (or "default"), the catalog is also saved as Parquet;
    "default" means data/processed/storms.parquet.
    """
    dst = dst.sort_index()
    rows = []
    prev_end = dst.index[0]
    for first, last in find_dips(dst):
        event = dst.loc[first:last]
        peak_time = event.idxmin()

        # Onset: highest Dst in the 24 h before the first hour below -50 nT, not
        # reaching back into the previous storm. Ties -> the latest such hour.
        before = dst.loc[max(first - ONSET_LOOKBACK, prev_end):first]
        onset_time = before[before == before.max()].index[-1]

        # Main onset: same idea, but looking back from the peak
        before_peak = dst.loc[max(peak_time - ONSET_LOOKBACK, onset_time):peak_time]
        main_onset = before_peak[before_peak == before_peak.max()].index[-1]

        # Recovery: first hour after the event where Dst is back above -30 nT
        after = dst.loc[last:]
        recovered = after[after > RECOVERY_LEVEL]
        recovery_end = recovered.index[0] if len(recovered) else pd.NaT

        rows.append({
            "onset_time": onset_time,
            "main_onset": main_onset,
            "peak_time": peak_time,
            "min_dst": float(event.min()),
            "class": storm_class(event.min()),
            "recovery_end": recovery_end,
            "hours_below_50": int((event < STORM_THRESHOLD).sum()),
        })
        prev_end = recovery_end if pd.notna(recovery_end) else last

    storms = pd.DataFrame(rows)
    storms.insert(0, "storm_id", np.arange(1, len(storms) + 1))

    if out_path is not None:
        if out_path == "default":
            out_path = PROCESSED_DIR / "storms.parquet"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        storms.to_parquet(out_path, index=False)
    return storms


def load_storms(path=None):
    """Read the saved storm catalog."""
    if path is None:
        path = PROCESSED_DIR / "storms.parquet"
    return pd.read_parquet(path)
