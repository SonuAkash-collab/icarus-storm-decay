"""Plotting helpers for the sanity check."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Muted colors: one data color, gray for space weather, red-ish for the storm marker
DATA_COLOR = "#2a78d6"
WEATHER_COLOR = "#52514e"
EVENT_COLOR = "#e34948"


def plot_object_with_storm(name, tles, rates, omni, event_time, start, end, out_path=None):
    """Four stacked panels on one time axis: a(t), decay rate, Dst, ap.

    tles: rows for one object (epoch, a_km); rates: its decay rates (t_mid, da_dt_km_per_day);
    omni: hourly DataFrame with dst and ap. A vertical line marks the storm onset.
    """
    tles = tles[(tles["epoch"] >= start) & (tles["epoch"] <= end)]
    rates = rates[(rates["t_mid"] >= start) & (rates["t_mid"] <= end)]
    omni = omni.loc[start:end]

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.5, 1.5]})

    ax = axes[0]
    ax.plot(tles["epoch"], tles["a_km"], color=DATA_COLOR, lw=1.5, marker="o", ms=2.5)
    ax.set_ylabel("Brouwer mean a (km)")
    ax.ticklabel_format(axis="y", useOffset=False)

    ax = axes[1]
    ax.plot(rates["t_mid"], rates["da_dt_km_per_day"] * 1000, color=DATA_COLOR, lw=1, marker="o", ms=2.5)
    ax.axhline(0, color=WEATHER_COLOR, lw=0.8)
    ax.set_ylabel("da/dt (m/day)")

    axes[2].plot(omni.index, omni["dst"], color=WEATHER_COLOR, lw=1)
    axes[2].set_ylabel("Dst (nT)")
    axes[3].plot(omni.index, omni["ap"], color=WEATHER_COLOR, lw=1)
    axes[3].set_ylabel("ap (nT)")

    for ax in axes:
        ax.axvline(event_time, color=EVENT_COLOR, lw=1, ls="--")
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].text(event_time, 1.0, " storm onset", color=EVENT_COLOR, va="bottom",
                 transform=axes[0].get_xaxis_transform(), fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0].set_title(name, loc="left", fontsize=12, pad=16)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig
