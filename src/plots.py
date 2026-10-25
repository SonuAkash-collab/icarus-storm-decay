"""Plotting helpers (sanity check, space weather)."""

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


def _clean_axes(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_dst_with_storms(sw, storms, labels=None, out_path=None):
    """Hourly Dst over the whole period; each storm shaded from onset to recovery_end.

    sw: hourly table indexed by time (needs dst); storms: catalog from src/storms.py.
    labels: {storm_id: text} for storms to label at their peak.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for _, s in storms.iterrows():
        ax.axvspan(s["onset_time"], s["recovery_end"], color=EVENT_COLOR, alpha=0.15, lw=0)
    ax.plot(sw.index, sw["dst"], color=WEATHER_COLOR, lw=0.7)
    ax.axhline(-50, color=EVENT_COLOR, lw=0.8, ls="--")
    ax.text(sw.index[0], -50, " -50 nT", color=EVENT_COLOR, va="bottom", fontsize=8)

    for storm_id, text in (labels or {}).items():
        s = storms.set_index("storm_id").loc[storm_id]
        ax.annotate(f"{text}\n{s['min_dst']:.0f} nT", (s["peak_time"], s["min_dst"]),
                    xytext=(12, 0), textcoords="offset points", va="center", fontsize=9)

    ax.set_ylabel("Dst (nT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_title(f"Dst with the {len(storms)} storms (Dst < -50 nT) shaded, onset to recovery",
                 loc="left", fontsize=12)
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_f107(sw, out_path=None):
    """Daily F10.7 with its 81-day centred mean."""
    daily = sw[["f107", "f107_81d"]].resample("D").first()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(daily.index, daily["f107"], color=WEATHER_COLOR, lw=0.8, label="daily F10.7")
    ax.plot(daily.index, daily["f107_81d"], color=DATA_COLOR, lw=2, label="81-day centred mean")
    ax.set_ylabel("F10.7 (sfu)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Solar UV background: F10.7", loc="left", fontsize=12)
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig
