"""Plotting helpers (sanity check, space weather, physics-model comparison, cohort physics, actives)."""

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


# Physics-model comparison (notebook 03)
MODEL_STYLE = {"nrlmsise00": ("NRLMSISE-00", "#e8a33d"), "msis21": ("MSIS 2.1", "#2e9e6b")}


def plot_observed_vs_predicted(rates, storms, title, start=None, end=None, out_path=None):
    """Two panels for one object: observed and predicted da/dt, and the residual.

    rates: output of physics.run_physics for ONE object (t_mid, da_dt_km_per_day,
    pred_<model>, resid_<model>). Storms are shaded from main_onset to recovery_end.
    start/end: optional zoom window.
    """
    if start is not None:
        rates = rates[(rates["t_mid"] >= start) & (rates["t_mid"] <= end)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    for ax in axes:
        for _, s in storms.iterrows():
            if start is None or (s["recovery_end"] >= start and s["main_onset"] <= end):
                ax.axvspan(s["main_onset"], s["recovery_end"], color=EVENT_COLOR, alpha=0.12, lw=0)

    ax = axes[0]
    ax.plot(rates["t_mid"], rates["da_dt_km_per_day"] * 1000, color=DATA_COLOR, lw=0.6,
            marker="o", ms=2, label="observed (TLE)")
    for model, (label, color) in MODEL_STYLE.items():
        ax.plot(rates["t_mid"], rates[f"pred_{model}"] * 1000, color=color, lw=1.3, label=f"predicted, {label}")
    ax.set_ylabel("da/dt (m/day)")
    ax.legend(frameon=False, loc="lower left", fontsize=9, ncol=3)

    ax = axes[1]
    for model, (label, color) in MODEL_STYLE.items():
        ax.plot(rates["t_mid"], rates[f"resid_{model}"] * 1000, color=color, lw=0.8, marker="o", ms=1.5,
                label=label)
    ax.axhline(0, color=WEATHER_COLOR, lw=0.8)
    ax.set_ylabel("observed - predicted\n(m/day, < 0 = extra decay)")

    fmt = "%b %d" if start is not None else "%b\n%Y"
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    axes[0].set_title(title, loc="left", fontsize=12)
    for ax in axes:
        _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_outliers(rates, flags, model, title, out_path=None):
    """Residual of one object over the year, Hampel outliers marked.

    rates: physics output for ONE object; flags: boolean array (physics.flag_outliers).
    """
    r = rates[f"resid_{model}"] * 1000
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(rates["t_mid"], r, color=DATA_COLOR, lw=0.5, marker="o", ms=1.5, label="kept")
    ax.plot(rates["t_mid"][flags], r[flags], "x", color=EVENT_COLOR, ms=6, label="Hampel outlier")
    ax.axhline(0, color=WEATHER_COLOR, lw=0.8)
    ax.set_ylabel("observed - predicted (m/day)")
    ax.set_title(title, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_smoothing_zoom(rates, variants, t0, title, days=(-10, 10), out_path=None):
    """Observed decay and the prediction smoothed by different amounts around one storm.

    rates: physics output for ONE object with pred_<name> for every name in `variants`
    ({column suffix: label}); t0: main_onset. The days 0..+3 window is shaded.
    """
    import pandas as pd

    start, end = t0 + pd.Timedelta(days=days[0]), t0 + pd.Timedelta(days=days[1])
    r = rates[(rates["t_mid"] >= start) & (rates["t_mid"] <= end)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(t0, t0 + pd.Timedelta(days=3), color=EVENT_COLOR, alpha=0.10, lw=0)
    ax.plot(r["t_mid"], r["da_dt_km_per_day"] * 1000, color=DATA_COLOR, lw=0.6, marker="o", ms=2,
            label="observed (TLE)")
    shades = plt.cm.Greys([0.95, 0.75, 0.55, 0.4, 0.3])
    for (name, label), color in zip(variants.items(), shades):
        ax.plot(r["t_mid"], r[f"pred_{name}"] * 1000, color=color, lw=1.3, label=label)
    ax.set_ylabel("da/dt (m/day)")
    ax.set_title(title, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_storm_changes(per_storm, title, out_path=None):
    """Baseline-relative residual per storm (dots per object, bar = mean), storms
    ordered by min Dst. per_storm: physics.storm_baseline_residual with change in m/day."""
    order = per_storm.groupby("storm_id")["min_dst"].first().sort_values().index
    pos = {sid: i for i, sid in enumerate(order)}
    means = per_storm.groupby("storm_id")["change"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(order)), means, color=DATA_COLOR, alpha=0.35, width=0.7, label="mean of objects")
    for marker, (nid, g) in zip("os^v", per_storm.groupby("norad_id")):
        ax.plot(g["storm_id"].map(pos), g["change"], marker, color=DATA_COLOR, ms=4, label=f"NORAD {nid}")
    ax.axhline(0, color=WEATHER_COLOR, lw=0.8)
    dst = per_storm.groupby("storm_id")["min_dst"].first().reindex(order)
    ax.set_xticks(range(len(order)), [f"{s}\n{d:.0f}" for s, d in zip(order, dst)], fontsize=8)
    ax.set_xlabel("storm id / min Dst (nT)")
    ax.set_ylabel("days 0..+3 minus days -10..-3\n(m/day, > 0 = over-prediction)")
    ax.set_title(title, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_drag_sensitivity(drag, alt_km, out_path=None):
    """Quiet-time drag sensitivity per object: histogram (log scale) and B against
    altitude with the bootstrap 95% interval; unstable objects in red.

    drag: build_table.drag_bootstrap output (fit objects only); alt_km: Series of mean
    altitude by norad_id.
    """
    import numpy as np

    d = drag.join(alt_km.rename("alt_km"))
    bad = d["unstable"] == 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(np.log10(d["B"]), bins=30, color=DATA_COLOR, alpha=0.8)
    ax1.set_xlabel("log10 drag sensitivity B (m^2/kg)")
    ax1.set_ylabel("objects")
    for mask, color, label in ((~bad, DATA_COLOR, "stable"), (bad, EVENT_COLOR, "unstable")):
        x = d[mask]
        ax2.errorbar(x["alt_km"], x["B"], yerr=[x["B"] - x["ci_lo"].clip(lower=1e-6), x["ci_hi"] - x["B"]],
                     fmt="o", ms=3, lw=0.6, color=color, label=f"{label} ({mask.sum()})")
    ax2.set_yscale("log")
    ax2.set_xlabel("mean altitude (km)")
    ax2.set_ylabel("B with 95% bootstrap interval")
    ax2.legend(frameon=False, fontsize=9)
    for ax in (ax1, ax2):
        _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_holdout_t(per_object, title, out_path=None):
    """Histogram of per-object held-out t values against the standard normal curve."""
    import numpy as np

    t = per_object["t"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(t, bins=np.arange(-5, 5.25, 0.5), density=True, color=DATA_COLOR, alpha=0.7,
            label=f"{len(t)} objects")
    x = np.linspace(-5, 5, 200)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color=WEATHER_COLOR, lw=1.2, label="N(0, 1)")
    ax.set_xlabel("held-out quiet residual / weekly-block SE")
    ax.set_ylabel("density")
    ax.set_title(title, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_storm_residual_vs_alt(windows, title, out_path=None):
    """Days 0..+3 residual per object against altitude for some storms.

    windows: {label: build_table.storm_window_residual frame}; one panel per outcome
    (Y_residual = model error, Y_total = total storm effect).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, col, name in zip(axes, ["Y_residual", "Y_total"],
                             ["model error (Y_residual)", "total storm effect (Y_total)"]):
        for (label, w), color in zip(windows.items(), [DATA_COLOR, EVENT_COLOR, WEATHER_COLOR]):
            ax.plot(w["alt_km"], w[col], "o", ms=3, alpha=0.7, color=color, label=f"{label} ({len(w)})")
        ax.axhline(0, color=WEATHER_COLOR, lw=0.8)
        ax.set_xlabel("mean altitude (km)")
        ax.set_title(name, loc="left", fontsize=11)
        _clean_axes(ax)
    axes[0].set_ylabel("days 0..+3 mean (m/day)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_event_day_profile(profiles, title, ylabel, out_path=None):
    """Mean outcome by event day with +/- 1.96 SE bands (SE across storms).

    profiles: {label: build_table.event_day_profile frame}.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for (label, p), color in zip(profiles.items(), [DATA_COLOR, EVENT_COLOR, WEATHER_COLOR]):
        ax.plot(p.index, p["mean"], marker="o", ms=3, lw=1, color=color, label=label)
        ax.fill_between(p.index, p["mean"] - 1.96 * p["se"], p["mean"] + 1.96 * p["se"], color=color, alpha=0.12, lw=0)
    ax.axhline(0, color=WEATHER_COLOR, lw=0.8)
    ax.axvline(0, color=EVENT_COLOR, lw=0.8, ls="--")
    ax.set_xlabel("days from main_onset")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_event_study(result, title, ylabel="effect vs day -1 (m/day)", out_path=None):
    """Stacked event-study coefficients by event day with 95% wild-cluster-bootstrap CIs.

    result: event_study.event_study output (index k = days from main_onset, columns
    coef, ci_lo, ci_hi, n_storms). Day -1 is the reference (fixed at 0); the short-run
    window (days 0-2) is shaded.
    """
    r = result.sort_index()
    ref = r.index[(r["coef"] == 0) & (r["ci_lo"] == 0) & (r["ci_hi"] == 0)]
    est = r.drop(index=ref)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(-0.5, 2.5, color=EVENT_COLOR, alpha=0.07, lw=0)
    ax.axhline(0, color=WEATHER_COLOR, lw=0.8)
    ax.axvline(0, color=EVENT_COLOR, lw=0.8, ls="--")
    ax.errorbar(est.index, est["coef"], yerr=[est["coef"] - est["ci_lo"], est["ci_hi"] - est["coef"]],
                fmt="o", ms=4, color=DATA_COLOR, ecolor=DATA_COLOR, elinewidth=1, capsize=0)
    ax.plot(ref, [0] * len(ref), "o", ms=4, mfc="white", mec=DATA_COLOR, label="reference day")
    ax.text(1, 1.0, "days 0-2", color=EVENT_COLOR, ha="center", va="bottom", fontsize=9,
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("days from main_onset")
    ax.set_ylabel(ylabel)
    n = result.attrs.get("n_clusters")
    ax.set_title(title + (f" ({n} storms)" if n else ""), loc="left", fontsize=12, pad=14)
    ax.set_xticks(range(int(r.index.min()), int(r.index.max()) + 1, 2))
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Active satellites (notebooks/07)
# ---------------------------------------------------------------------------

def plot_catalog_scale(bodies, actives, out_path=None):
    """Fitted / catalog drag sensitivity of rocket bodies against altitude, with the
    scale given to each active satellite (build_table.catalog_scale).

    bodies: frame with alt_km and ratio (one row per rocket body); actives: frame with
    alt_km and scale (one row per active satellite).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(bodies["alt_km"], bodies["ratio"], s=12, color=WEATHER_COLOR, alpha=0.6,
               label=f"rocket bodies ({len(bodies)})")
    ax.scatter(actives["alt_km"], actives["scale"], s=40, marker="D", color=DATA_COLOR,
               label="scale used for actives (median within +-50 km)")
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_xlabel("mean altitude (km)")
    ax.set_ylabel("fitted B / catalog B")
    ax.legend(frameon=False, fontsize=9)
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_active_altitudes(active, ncols=5, out_path=None):
    """Altitude (TLE mean a minus Earth radius) of every active satellite over the year,
    rows flagged by build_table.free_drag_flags in red (left out of the offset ratio).

    active: analysis-table rows with is_active = 1 (t_mid, alt_km_tle, free_drag, object_name).
    """
    import numpy as np

    groups = list(active.groupby("norad_id"))
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (norad_id, g) in zip(axes, groups):
        g = g.sort_values("t_mid")
        ax.plot(g["t_mid"], g["alt_km_tle"], lw=0.6, color=WEATHER_COLOR)
        bad = g[g["free_drag"] == 1]
        ax.scatter(bad["t_mid"], bad["alt_km_tle"], s=1, color=EVENT_COLOR)
        ax.set_title(f"{g['object_name'].iat[0]} ({norad_id})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        _clean_axes(ax)
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    fig.supylabel("altitude (km)")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120)
    return fig


def plot_match_gap(passive, active, out_path=None):
    """Inclination against altitude for the passive pool and the active satellites:
    shows whether actives have passive twins nearby (offset_ratio.match).

    passive, active: one row per object with alt_km and inc_deg.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(passive["alt_km"], passive["inc_deg"], s=10, color=WEATHER_COLOR, alpha=0.6,
               label=f"passive main sample ({len(passive)})")
    ax.scatter(active["alt_km"], active["inc_deg"], s=30, marker="D", color=DATA_COLOR,
               label=f"active satellites ({len(active)})")
    ax.set_xlabel("mean altitude (km)")
    ax.set_ylabel("inclination (deg)")
    ax.legend(frameon=False, fontsize=9)
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_offset_model_storms(pairs, storms, pred="P", out_path=None):
    """Observed vs predicted storm loss per storm (duck + recovery windows, summed over
    active satellites, km). On the diagonal: offset 0 (no operator response); on y = 0:
    offset 1 (nothing lost). pairs: offset_model pairs; pred: "P" or "Pc"."""
    import numpy as np

    per = pairs.groupby("storm_id")[["O_duck", "O_rec", f"{pred}_duck", f"{pred}_rec"]].sum() / 1000
    per = per.join(storms.set_index("storm_id")[["class"]])
    x, y = per[f"{pred}_duck"] + per[f"{pred}_rec"], per["O_duck"] + per["O_rec"]
    fig, ax = plt.subplots(figsize=(6, 5))
    lim = np.nanmax(np.abs(np.r_[x, y])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=WEATHER_COLOR, lw=0.8, ls="--", label="offset 0 (observed = predicted)")
    ax.axhline(0, color=EVENT_COLOR, lw=0.8, ls=":", label="offset 1 (nothing lost)")
    ax.scatter(x, y, s=30, color=DATA_COLOR)
    for sid, xi, yi in zip(per.index, x, y):
        ax.annotate(str(sid), (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("predicted storm loss (km, summed over satellites)")
    ax.set_ylabel("observed loss vs baseline (km)")
    ax.legend(frameon=False, fontsize=8)
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_offset_model_summary(summary, placebo, stat="offset_total", out_path=None):
    """Forest plot of one offset statistic: every run of offset_model.run and the placebo,
    with 95% CIs. summary / placebo: the stacked frames from offset_model.run."""
    rows = [(f"{a} / {b}", r) for (a, b, s), r in summary.iterrows() if s == stat]
    rows += [(f"placebo ({a}) / {b}", r) for (a, b, s), r in placebo.iterrows() if s == stat]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1))
    for i, (label, r) in enumerate(rows):
        color = EVENT_COLOR if label.startswith("placebo") else DATA_COLOR
        ax.errorbar(r["est"], i, xerr=[[r["est"] - r["ci_lo"]], [r["ci_hi"] - r["est"]]], fmt="o", ms=4,
                    color=color, elinewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=8)
    ax.invert_yaxis()
    for v in (0, 1):
        ax.axvline(v, color=WEATHER_COLOR, lw=0.6, ls=":")
    ax.set_xlabel(f"{stat} (95% CI)")
    _clean_axes(ax)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig
