"""Figure pack for the first review: seven figures and SUMMARY.md in results/review/.

    python -m src.review_figures

Needs data/processed/{analysis_table, storms, spaceweather}.parquet, objects.parquet and
the track C outputs in results/causal (python -m src.causal_run). Each figure function
takes plain frames and returns (figure, key numbers); the key numbers feed SUMMARY.md, so
the summary is never typed by hand. Existing plotting code in src/plots.py is reused.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import physics, plots
from src.objects import compare_drag
from src.schema import main_sample

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CAUSAL = ROOT / "results" / "causal"
OUT_DIR = ROOT / "results" / "review"
GRACE_IMAGE = ROOT / "results" / "a7_oct2024_decay_rate_lag_check.png"
DPI = 150
CLASSES = ["moderate", "strong", "severe", "extreme"]
POST = (0, 3)                  # days of the storm window (same as physics.POST_DAYS)
PRE = (-10, -3)                # quiet baseline days
# From Piyush's A7 run (docs/PLAN.md, Oct 31): the precise-orbit files are not in the repo.
GRACE_R = 0.70


def set_style():
    """One look for all figures (the helpers in plots.py set titles and axes themselves)."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "legend.fontsize": 9, "figure.dpi": DPI, "savefig.dpi": DPI,
                         "savefig.bbox": "tight"})


def _save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def storm_by_class(storms, cls):
    """First storm of a class (the extreme class has one: May 2024)."""
    return storms[storms["class"] == cls].iloc[0]


# ---------------------------------------------------------------------------
# 1. May 2024 decay rate across the cohort
# ---------------------------------------------------------------------------

def fig_may_cohort(table, storms, days=(-7, 10)):
    """Decay rate of every cohort interval around the May 2024 main_onset, with daily
    median and quartiles. Intervals never cross main_onset (checked in build_table)."""
    s = storm_by_class(storms, "extreme")
    t0 = s["main_onset"]
    rows = main_sample(table)
    rows = rows[rows["t_mid"].between(t0 + pd.Timedelta(days=days[0]), t0 + pd.Timedelta(days=days[1]))]
    daily = rows.groupby(rows["t_mid"].dt.floor("D"))["obs_decay"].quantile([0.25, 0.5, 0.75]).unstack()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(rows["t_mid"], rows["obs_decay"], s=4, alpha=0.2, color=plots.DATA_COLOR, lw=0,
               label="one decay interval")
    x = daily.index + pd.Timedelta(hours=12)
    ax.fill_between(x, daily[0.25], daily[0.75], color=plots.WEATHER_COLOR, alpha=0.25, lw=0,
                    label="daily quartiles")
    ax.plot(x, daily[0.5], color=plots.WEATHER_COLOR, lw=1.5, label="daily median")
    ax.axvline(t0, color=plots.EVENT_COLOR, lw=1.2, ls="--")
    ax.text(t0, 1.0, " main_onset", color=plots.EVENT_COLOR, va="bottom", fontsize=9,
            transform=ax.get_xaxis_transform())
    ax.set_ylim(-650, 100)
    ax.set_ylabel("decay rate da/dt (m/day)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_title(f"May 2024 storm: decay rate of {rows['norad_id'].nunique()} passive objects "
                 f"(Dst min {s['min_dst']:.0f} nT)", loc="left", pad=14)
    ax.legend(frameon=False, loc="lower right", markerscale=3)
    plots._clean_axes(ax)
    fig.tight_layout()

    day = lambda a, b: rows[rows["t_mid"].between(t0 + pd.Timedelta(days=a), t0 + pd.Timedelta(days=b))]
    before, after = day(-3, 0)["obs_decay"].median(), day(0, 1)["obs_decay"].median()
    return fig, {"objects": rows["norad_id"].nunique(), "before": before, "after": after,
                 "ratio": after / before}


# ---------------------------------------------------------------------------
# 2. Storm catalog
# ---------------------------------------------------------------------------

def fig_storm_catalog(sw, storms):
    labels = {int(storm_by_class(storms, "extreme")["storm_id"]): "May 2024"}
    for _, s in storms[storms["class"] == "severe"].iterrows():
        labels[int(s["storm_id"])] = s["peak_time"].strftime("%b %Y")
    fig = plots.plot_dst_with_storms(sw, storms, labels=labels)
    fig.axes[0].set_ylabel("Dst (nT), hourly")
    counts = storms["class"].value_counts()
    return fig, {"n": len(storms), **{c: int(counts.get(c, 0)) for c in CLASSES},
                 "min_dst": storms["min_dst"].min()}


# ---------------------------------------------------------------------------
# 3. Physics model check
# ---------------------------------------------------------------------------

def holdout_number(table):
    """Held-out quiet residual (m/day) of MSIS 2.1, pooled over the main cohort: dt-weighted
    mean, SE from calendar-week blocks (same as build_table.holdout_gate)."""
    h = table[(table["role"] == "holdout") & (table["hampel_outlier"] == 0) & (table["is_active"] == 0)
              & (table["cohort_group"] == "main")]
    w = h["dt_days"]
    mean = (h["Y_residual"] * w).sum() / w.sum()
    se = physics.block_se(h["Y_residual"], h["t_mid"])
    return mean, se, h["norad_id"].nunique()


def pick_examples(table, objects, storms, n=2):
    """Two rocket bodies with the most clean intervals around May 2024, one near 500 km and
    one near 650 km, so the panels show different altitudes."""
    t0 = storm_by_class(storms, "extreme")["main_onset"]
    rows = main_sample(table)
    rows = rows[rows["t_mid"].between(t0 - pd.Timedelta(days=10), t0 + pd.Timedelta(days=10))
                & (rows["hampel_outlier"] == 0)]
    bodies = objects.loc[objects["object_type"] == "rocket body", "norad_id"]
    g = rows[rows["norad_id"].isin(bodies)].groupby("norad_id").agg(n=("t_mid", "size"), alt=("alt_km", "mean"))
    picks = []
    for lo, hi in ((450, 550), (600, 700)):
        sub = g[g["alt"].between(lo, hi)]
        picks.append(int(sub["n"].idxmax()))
    return picks[:n]


def fig_physics_check(table, storms, objects, days=(-10, 10)):
    """Observed vs MSIS 2.1 predicted da/dt for two example objects through the May 2024
    storm. The held-out quiet residual of the whole cohort is one number in the title."""
    s = storm_by_class(storms, "extreme")
    t0 = s["main_onset"]
    ids = pick_examples(table, objects, storms)
    names = objects.set_index("norad_id")["object_name"]
    mean, se, n_obj = holdout_number(table)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax, nid in zip(axes, ids):
        g = table[(table["norad_id"] == nid) & table["t_mid"].between(
            t0 + pd.Timedelta(days=days[0]), t0 + pd.Timedelta(days=days[1]))
            & (table["tle_ok"] == 1) & (table["hampel_outlier"] == 0)].sort_values("t_mid")
        ax.axvspan(t0, s["recovery_end"], color=plots.EVENT_COLOR, alpha=0.12, lw=0)
        ax.plot(g["t_mid"], g["obs_decay"], color=plots.DATA_COLOR, lw=0.6, marker="o", ms=3,
                label="observed (TLE)")
        ax.plot(g["t_mid"], g["pred_decay"], color=plots.MODEL_STYLE["msis21"][1], lw=1.5,
                label="predicted (MSIS 2.1, actual ap)")
        ax.set_ylabel("da/dt (m/day)")
        ax.set_title(f"{names.get(nid, '')} (NORAD {nid}, {g['alt_km'].mean():.0f} km)", loc="left",
                     fontsize=10)
        plots._clean_axes(ax)
    axes[0].legend(frameon=False, loc="lower left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle(f"Physics check, May 2024 storm (shaded). Held-out quiet residual, {n_obj} objects: "
                 f"{mean:+.2f} m/day (SE {se:.2f})", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return fig, {"holdout": mean, "holdout_se": se, "objects": n_obj, "examples": ids}


# ---------------------------------------------------------------------------
# 4. Event study
# ---------------------------------------------------------------------------

def parse_short_run(summary_path=CAUSAL / "summary.md"):
    """Short-run effects (days 0-2) per outcome from results/causal/summary.md."""
    text = Path(summary_path).read_text()
    out = {}
    for outcome in ("Y_total", "Y_residual"):
        section = text.split(f"## Event study: {outcome}")[1].split("\n## ")[0]
        m = re.search(r"Short-run effect \(days 0-2, m/day\): (-?[\d.]+) \(95% CI (-?[\d.]+) to (-?[\d.]+)\)",
                      section)
        out[outcome] = tuple(float(v) for v in m.groups())
    return out


def fig_event_study(causal_dir=CAUSAL):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for ax, outcome, title in zip(
            axes, ("Y_total", "Y_residual"),
            ("Total storm effect (Y_total): observed minus prediction at quiet ap",
             "Model error (Y_residual): observed minus MSIS 2.1 at actual ap")):
        res = pd.read_csv(Path(causal_dir) / outcome / "event_study.csv", index_col="k")
        plots.plot_event_study(res, f"{title} ({int(res['n_storms'].max())} storms)", ax=ax)
        ax.set_xlabel("")
    axes[-1].set_xlabel("days from main_onset")
    fig.tight_layout()
    return fig, parse_short_run(Path(causal_dir) / "summary.md")


# ---------------------------------------------------------------------------
# 5. Model error by storm class
# ---------------------------------------------------------------------------

def storm_class_error(table, storms, outcome="Y_residual", pre=PRE, post=POST, min_days=2.0):
    """Change in `outcome` (days post minus days pre, m/day) per storm: mean over objects of
    the per-object dt-weighted window means. Objects need > min_days of coverage per window."""
    rows = main_sample(table)
    rows = rows[rows["hampel_outlier"] == 0]
    out = []
    for s in storms.to_dict("records"):
        # days from this storm's main_onset (table.event_day follows the nearest storm, which
        # would hide the pre-period of a storm that follows another one closely)
        day = (rows["t_mid"] - s["main_onset"]) / pd.Timedelta(days=1)
        g = rows[day.between(-12, 5)].assign(event_day=day, yw=lambda d: d[outcome] * d["dt_days"])
        win = {}
        for name, (a, b) in (("pre", pre), ("post", post)):
            w = g[g["event_day"].between(a, b)].groupby("norad_id")[["yw", "dt_days"]].sum()
            win[name] = (w["yw"] / w["dt_days"])[w["dt_days"] >= min_days]
        d = (win["post"] - win["pre"]).dropna()
        if len(d):
            out.append({"storm_id": s["storm_id"], "storm_class": s["class"], "change": d.mean(),
                        "se_objects": d.std() / np.sqrt(len(d)) if len(d) > 1 else np.nan, "n_objects": len(d)})
    return pd.DataFrame(out)


def class_summary(per_storm):
    """Mean over storms by class. CI: t interval over storms when a class has >= 3 storms;
    with fewer (severe 2, extreme 1) a t interval is meaningless, so the CI is the
    across-object interval of those storms (their per-storm SEs combined)."""
    rows = []
    for cls in CLASSES:
        g = per_storm[per_storm["storm_class"] == cls]
        if g.empty:
            continue
        n = len(g)
        mean = g["change"].mean()
        if n >= 3:
            half, basis = stats.t.ppf(0.975, n - 1) * g["change"].std() / np.sqrt(n), "storms"
        else:
            half, basis = 1.96 * np.sqrt((g["se_objects"] ** 2).sum()) / n, "objects"
        rows.append({"storm_class": cls, "mean": mean, "ci_lo": mean - half, "ci_hi": mean + half,
                     "n_storms": n, "ci_basis": basis})
    return pd.DataFrame(rows).set_index("storm_class")


def fig_class_error(table, storms):
    per = storm_class_error(table, storms)
    cs = class_summary(per)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = {c: i for i, c in enumerate(CLASSES)}
    ax.axhline(0, color=plots.WEATHER_COLOR, lw=0.8)
    ax.scatter(per["storm_class"].map(x), per["change"],
               s=14, color=plots.WEATHER_COLOR, alpha=0.5, label="one storm")
    for cls, r in cs.iterrows():
        solid = r["ci_basis"] == "storms"
        ax.errorbar(x[cls] + 0.18, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                    fmt="o", ms=6, color=plots.DATA_COLOR if solid else plots.EVENT_COLOR, capsize=4)
    ax.set_xticks(range(len(CLASSES)), [f"{c}\n({int(cs.loc[c, 'n_storms'])} storm"
                                        f"{'s' if cs.loc[c, 'n_storms'] > 1 else ''})" for c in CLASSES])
    ax.set_ylabel("Y_residual, days 0..+3 minus days -10..-3 (m/day)\n> 0: less decay than MSIS predicts")
    ax.set_title("Model error by storm class (blue: 95% CI over storms; red: fewer than 3 storms, CI over objects)",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    plots._clean_axes(ax)
    fig.tight_layout()
    cs.attrs["severe_storms"] = per.loc[per["storm_class"] == "severe", "change"].round(1).tolist()
    return fig, cs


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------

def fig_validation(table, objects, grace_image=GRACE_IMAGE):
    """Left: GRACE-FO-1 Oct 2024 TLE vs precise-orbit daily decay rate (Piyush's A7 figure,
    drawn from his saved image because the precise-orbit files are not in the repo).
    Right: fitted vs catalog drag sensitivity of the rocket bodies."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    img = mpimg.imread(grace_image)
    h, w = img.shape[:2]
    ax_l.imshow(img[: int(h * 0.475), : int(w * 0.49)])        # top-left panel: GRACE-FO-1 decay rates
    ax_l.axis("off")
    ax_l.set_title(f"GRACE-FO-1, Oct 2024: TLE vs precise-orbit daily decay (r = {GRACE_R:.2f} at lag 0)",
                   loc="left", fontsize=11)
    passive = table[table["is_active"] == 0].groupby("norad_id")["drag_sensitivity"].first()
    res, _ = compare_drag(passive, objects, ax=ax_r)
    ax_r.set_title("")
    ax_r.set_title(f"Rocket bodies: fitted vs catalog drag\nSpearman rho = {res['rho']:.2f} (n = {res['n']})",
                   loc="left", fontsize=11)
    plots._clean_axes(ax_r)
    fig.tight_layout()
    return fig, {"rho": res["rho"], "n": res["n"], "grace_r": GRACE_R}


# ---------------------------------------------------------------------------
# 7. Operator offset
# ---------------------------------------------------------------------------

def fig_offset(storms, causal_dir=CAUSAL):
    pairs = pd.read_csv(Path(causal_dir) / "offset_model_pairs.csv")
    model = pd.read_csv(Path(causal_dir) / "offset_model.csv")
    m = model[(model["storms"] == "all storms") & (model["prediction"] == "uncorrected")].set_index("statistic")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1]})
    plots.plot_offset_model_storms(pairs, storms, ax=ax_l)
    ax_l.set_title("Starlink: observed vs predicted storm loss", loc="left", fontsize=11)
    ax_l.set_xlabel("predicted storm loss from physics (km, summed over satellites)")
    ax_l.set_ylabel("observed loss vs own baseline (km)")

    stats_ = [("duck_part", "ducking\n(during storm)"), ("rec_part", "recovery\n(10 days after)"),
              ("offset_total", "total offset")]
    for i, (key, label) in enumerate(stats_):
        r = m.loc[key]
        ax_r.errorbar(i, r["est"], yerr=[[r["est"] - r["ci_lo"]], [r["ci_hi"] - r["est"]]], fmt="o", ms=7,
                      color=plots.DATA_COLOR, capsize=4)
        ax_r.annotate(f"{r['est']:.2f}", (i, r["est"]), xytext=(10, 0), textcoords="offset points", fontsize=9,
                      va="center")
    ax_r.axhline(0, color=plots.WEATHER_COLOR, lw=0.8)
    ax_r.axhline(1, color=plots.WEATHER_COLOR, lw=0.8, ls=":")
    ax_r.set_xticks(range(3), [l for _, l in stats_])
    ax_r.set_xlim(-0.5, 2.5)
    ax_r.set_ylabel("share of predicted storm loss avoided (95% CI)")
    ax_r.set_title(f"Operator offset ratio ({int(m.loc['offset_total', 'n_storms'])} storms, "
                   f"{int(m.loc['offset_total', 'n_objects'])} satellites)", loc="left", fontsize=11)
    plots._clean_axes(ax_r)
    fig.tight_layout()
    per = pairs.groupby("storm_id")[["O_duck", "O_rec", "P_duck", "P_rec"]].sum()
    smaller = int(((per["O_duck"] + per["O_rec"]) > (per["P_duck"] + per["P_rec"])).sum())   # losses are negative
    out = {k: tuple(m.loc[k, ["est", "ci_lo", "ci_hi"]]) for k in ("duck_part", "rec_part", "offset_total")}
    out["storms_smaller"] = (smaller, len(per))
    return fig, out


# ---------------------------------------------------------------------------
# Summary and driver
# ---------------------------------------------------------------------------

def summary_text(n):
    cs = n["class"]
    ext, sev = cs.loc["extreme"], cs.loc["severe"]
    f = lambda t: f"{t[0]:+.1f} m/day (95% CI {t[1]:+.1f} to {t[2]:+.1f})"
    o = n["offset"]
    g = lambda t: f"{t[0]:.2f} (95% CI {t[1]:.2f} to {t[2]:.2f})"
    return "\n".join([
        "# Review figure pack",
        "",
        "Made by `python -m src.review_figures` from the analysis table and results/causal. "
        "One line per figure with its key number.",
        "",
        f"1. `fig1_may2024_cohort.png`: the median decay rate of {n['may']['objects']} passive objects goes from "
        f"{n['may']['before']:.0f} m/day (3 days before main_onset) to {n['may']['after']:.0f} m/day "
        f"(first day after), {n['may']['ratio']:.1f}x.",
        f"2. `fig2_storm_catalog.png`: {n['catalog']['n']} storms with Dst < -50 nT (Apr 2024 - Apr 2025): "
        f"{n['catalog']['moderate']} moderate, {n['catalog']['strong']} strong, {n['catalog']['severe']} severe, "
        f"{n['catalog']['extreme']} extreme; deepest Dst {n['catalog']['min_dst']:.0f} nT (May 2024).",
        f"3. `fig3_physics_check.png`: held-out quiet residual of MSIS 2.1 on {n['physics']['objects']} objects "
        f"{n['physics']['holdout']:+.2f} m/day (SE {n['physics']['holdout_se']:.2f}), i.e. zero within noise; "
        f"example objects NORAD {n['physics']['examples'][0]} and {n['physics']['examples'][1]}.",
        f"4. `fig4_event_study.png`: days 0-2 effect, total storm effect {f(n['event']['Y_total'])}; "
        f"model error {f(n['event']['Y_residual'])}. Day -1 is the reference.",
        f"5. `fig5_model_error_by_class.png`: mean Y_residual change (days 0..+3 minus days -10..-3) by class: "
        + ", ".join(f"{c} {cs.loc[c, 'mean']:+.1f} ({int(cs.loc[c, 'n_storms'])})" for c in cs.index)
        + f" m/day (storms in brackets). Extreme (one storm, CI over objects) {ext['mean']:+.1f} "
        f"[{ext['ci_lo']:+.1f}, {ext['ci_hi']:+.1f}]; severe (two storms, CI over objects) {sev['mean']:+.1f} "
        f"[{sev['ci_lo']:+.1f}, {sev['ci_hi']:+.1f}], but the two severe storms disagree "
        f"({cs.attrs['severe_storms'][0]:+.1f} and {cs.attrs['severe_storms'][1]:+.1f}). Positive = MSIS predicts more decay than observed; "
        "caveat from notebook 04: the May 2024 value vanishes with 3-day smoothing (TLE blur).",
        f"6. `fig6_validation.png`: GRACE-FO-1 Oct 2024 TLE vs precise decay rate r = {n['valid']['grace_r']:.2f} "
        f"(Piyush's A7 result; the left panel is cut from his saved figure); fitted vs catalog drag of rocket bodies "
        f"Spearman rho = {n['valid']['rho']:.2f} (n = {n['valid']['n']}).",
        f"7. `fig7_operator_offset.png`: Starlink total offset {g(o['offset_total'])}, of which ducking part "
        f"{g(o['duck_part'])} and recovery part {g(o['rec_part'])}; "
        f"observed loss is smaller than predicted in {n['offset_storms'][0]} of {n['offset_storms'][1]} storms.",
        "",
    ])


def main(out_dir=OUT_DIR):
    set_style()
    out_dir = Path(out_dir)
    table = pd.read_parquet(PROCESSED / "analysis_table.parquet")
    storms = pd.read_parquet(PROCESSED / "storms.parquet")
    sw = pd.read_parquet(PROCESSED / "spaceweather.parquet")
    if "time" in sw.columns:
        sw = sw.set_index("time")
    objects = pd.read_parquet(PROCESSED / "objects.parquet")

    numbers = {}
    for key, name, make in [
        ("may", "fig1_may2024_cohort.png", lambda: fig_may_cohort(table, storms)),
        ("catalog", "fig2_storm_catalog.png", lambda: fig_storm_catalog(sw, storms)),
        ("physics", "fig3_physics_check.png", lambda: fig_physics_check(table, storms, objects)),
        ("event", "fig4_event_study.png", lambda: fig_event_study()),
        ("class", "fig5_model_error_by_class.png", lambda: fig_class_error(table, storms)),
        ("valid", "fig6_validation.png", lambda: fig_validation(table, objects)),
        ("offset", "fig7_operator_offset.png", lambda: fig_offset(storms)),
    ]:
        fig, numbers[key] = make()
        if key == "offset":
            numbers["offset_storms"] = numbers[key].pop("storms_smaller")
        print("wrote", _save(fig, name, out_dir))
    (out_dir / "SUMMARY.md").write_text(summary_text(numbers))
    print("wrote", out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
