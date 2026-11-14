"""Review figure pack: nine figures and SUMMARY.md in results/review/.

    python -m src.review_figures

Needs data/processed/{analysis_table, storms, spaceweather}.parquet, objects.parquet, the
cleaned cohort TLEs (fig 8, src/extreme_check.py), the track C outputs in results/causal
(python -m src.causal_run) and the track D outputs in results/dml (python -m src.dml_run,
src.dml_refute, src.storm_class for Y_residual, src.vulnerability_map with and without
--per-drag). Each figure function
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

from src import extreme_check, physics, plots
from src import vulnerability_map as vm
from src.dml_refute import placebo_p
from src.objects import compare_drag
from src.schema import main_sample

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CAUSAL = ROOT / "results" / "causal"
DML = ROOT / "results" / "dml"
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
# 8. Extreme-storm check (src/extreme_check.py, notebooks/09)
# ---------------------------------------------------------------------------

def extreme_numbers(out):
    """Key numbers of extreme_check.run for SUMMARY.md, keyed by storm name (e.g. 'May 2024')."""
    smooth = f"MSIS 2.1 smoothed {max(extreme_check.SMOOTH_DAYS)} d"
    calm = out["tables"][0]
    tri = lambda r, c, lo, hi: (r[c], r[lo], r[hi])     # noqa: E731
    storms = {}
    for sid, name in out["names"].items():
        short = name.split(" (")[0]
        c = calm["storm"].loc[sid]
        alt = calm["storm_alt"][calm["storm_alt"]["storm_id"] == sid]
        storms[short] = {
            "class": name.split("(")[1].rstrip(")"),
            "raw": tri(out["by_storm"].loc[("MSIS 2.1", sid)], "mean", "ci_lo", "ci_hi"),
            "smooth": tri(out["by_storm"].loc[(smooth, sid)], "mean", "ci_lo", "ci_hi"),
            "ratio": tri(c, "ratio", "ratio_lo", "ratio_hi"),
            "diff": tri(c, "diff_m", "diff_lo", "diff_hi"),
            "adj": tri(c, "diff_adj_m", "adj_lo", "adj_hi"),
            "quiet": tri(c, "diff_quiet_m", "quiet_lo", "quiet_hi"),
            "alt_ratio": (alt["ratio"].min(), alt["ratio"].max()),
            "label": out["labels"][sid]}
    pooled = out["pooled"]
    lags = out["lags"]["mean"]
    high = out["high_summary"]
    used = out["calm"][0]["storm_id"].nunique()
    return {"storms": storms, "verdict": out["verdict"], "objects": out["n_objects"],
            "all": tri(pooled.loc[("MSIS 2.1", "all storms")], "mean", "ci_lo", "ci_hi"),
            "without": tri(pooled.loc[("MSIS 2.1", "without May/Oct 2024")], "mean", "ci_lo", "ci_hi"),
            "all_s3": tri(pooled.loc[(smooth, "all storms")], "mean", "ci_lo", "ci_hi"),
            "best_lag": int(lags.idxmax()), "lag_gain": lags.max() - lags.loc[0],
            "calm_storms": used, "n_storms": len(out["by_storm"].loc["MSIS 2.1"]),
            "high_quiet": (high["diff_quiet_m"].abs().max(), high["quiet_lo"].min(), high["quiet_hi"].max()),
            "sens_ratio": {e: out["tables"][e]["storm"]["ratio"].round(2).tolist() for e in extreme_check.EXTRA_DAYS}}


def fig_extreme_check(out):
    """Three panels: days 0..+3 residual change vs smoothing, running calm-to-calm model
    error, calm-to-calm difference for the key storms and the moderate / strong classes."""
    names = out["names"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    plots.plot_extreme_smoothing(out["by_storm"], names, ax=axes[0])
    plots.plot_calm_cumulative(out["curves"], names, ax=axes[1])
    calm = out["tables"][0]
    rows = pd.concat([calm["storm"].set_index("storm"),
                      calm["class"].loc[["moderate", "strong"]].rename(index=lambda c: f"all {c}")])
    labels = [i if not i.startswith("all") else f"{i} ({int(r.n_storms)} storms)" for i, r in rows.iterrows()]
    plots.plot_calm_summary(rows, labels, ax=axes[2], title="Calm-to-calm loss, obs - MSIS 2.1 (95% CI)")
    axes[0].set_title("A. Days 0..+3 residual vs smoothing", loc="left", fontsize=12)
    axes[1].set_title("B. Running obs - pred from the calm day before", loc="left", fontsize=12)
    axes[2].set_title("C. Calm day before to calm day after", loc="left", fontsize=12)
    fig.suptitle(f"Extreme-storm check, {out['n_objects']} passive objects. Verdict: {out['verdict']}",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return fig, extreme_numbers(out)


# ---------------------------------------------------------------------------
# 9. Vulnerability maps (src/vulnerability_map.py, question 2)
# ---------------------------------------------------------------------------

PER_DRAG_OUTCOME = "Y_total" + vm.PER_DRAG
PROFILE_ALTS = (620, 750)      # rows compared in the per-drag altitude pattern (km)


def per_drag_profile(grid, min_cells=2):
    """Per-drag effect by altitude row: mean over the supported cells (n_near >= MIN_NEAR)
    of every inclination. Rows with fewer than min_cells supported cells are left out, so
    a single-cell row (the grid's lowest and highest altitudes) never sets the pattern."""
    ok = grid[grid["n_near"] >= vm.MIN_NEAR]
    # n_cells counts distinct drag-sensitivity cells: n_near ignores inclination, so one
    # supported cell repeats at every inclination and must not count three times
    prof = ok.groupby("alt_km").agg(effect=("effect", "mean"), n_cells=("drag_sensitivity", "nunique"))
    return prof[prof["n_cells"] >= min_cells]


def profile_ratio(prof, alts=PROFILE_ALTS):
    """Effect at the supported rows nearest alts[0] and alts[1], and their ratio."""
    pick = [prof.index[np.abs(prof.index - a).argmin()] for a in alts]
    lo, hi = prof.loc[pick[0], "effect"], prof.loc[pick[1], "effect"]
    return {"alt_lo": pick[0], "alt_hi": pick[1], "effect_lo": lo, "effect_hi": hi, "ratio": lo / hi}


def q2_numbers(dml_dir=DML):
    """Importances, seed stability, per-drag altitude pattern and cross-check, from the CSVs."""
    d = Path(dml_dir)
    out = {}
    for y in ("Y_total", PER_DRAG_OUTCOME):
        imp = pd.read_csv(d / f"vulnerability_importance_{y}.csv", index_col=0)["importance"]
        st = pd.read_csv(d / f"vulnerability_stats_{y}.csv").iloc[0]
        out[y] = {"importance": imp.to_dict(), "min_corr": st["min_corr"], "mean_corr": st["mean_corr"],
                  "n_seeds": int(st["n_stability_seeds"]), "average": st["average_effect"],
                  "n_objects": int(st["n_objects"])}
    grid = pd.read_csv(d / f"vulnerability_grid_{PER_DRAG_OUTCOME}.csv")
    out["profile"] = profile_ratio(per_drag_profile(grid))
    out["cross_corr"] = pd.read_csv(d / "vulnerability_per_drag_cross_check_summary.csv").iloc[0]["corr_supported_cells"]
    return out


def fig_vulnerability(dml_dir=DML):
    """Three panels: A per-object effect (Y_total) by altitude and drag sensitivity,
    B the Y_total grid map, C the per-drag grid map (both at the median inclination,
    grey = fewer than MIN_NEAR objects nearby)."""
    d = Path(dml_dir)
    objs = pd.read_csv(d / "vulnerability_objects_Y_total.csv")
    grids = {y: pd.read_csv(d / f"vulnerability_grid_{y}.csv") for y in ("Y_total", PER_DRAG_OUTCOME)}
    cmap = matplotlib.colormaps["RdBu"].copy()
    cmap.set_bad("lightgrey")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    vmax = np.abs(objs["effect"]).max()
    sc = axes[0].scatter(objs["drag_sensitivity"], objs["alt_km"], c=objs["effect"], cmap=cmap, vmin=-vmax,
                         vmax=vmax, s=22, edgecolor="k", lw=0.3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("drag sensitivity (m$^2$/kg)")
    axes[0].set_ylabel("altitude (km)")
    fig.colorbar(sc, ax=axes[0], label="Y_total: m/day per nT of ap")
    axes[0].set_title(f"A. Vulnerability per object ({len(objs)} objects)", loc="left", fontsize=12)
    plots._clean_axes(axes[0])

    for ax, (y, g), title, unit in zip(
            axes[1:], grids.items(), ("B. Y_total map", "C. Per unit drag sensitivity"),
            ("m/day per nT", "(m/day per nT) per m$^2$/kg")):
        incs = sorted(g["inc_deg"].unique())
        inc = incs[len(incs) // 2]
        shown = g.assign(effect=g["effect"].where(g["n_near"] >= vm.MIN_NEAR))
        v = np.nanmax(np.abs(shown["effect"]))
        im = vm.draw_map(ax, shown, inc, v, cmap)
        ax.set_ylabel("altitude (km)")
        ax.set_title(f"{title} (inclination {inc:.0f} deg)", loc="left", fontsize=12)
        fig.colorbar(im, ax=ax, label=unit)
    fig.suptitle("Storm vulnerability (causal forest, Y_total); more negative = hit harder; "
                 f"grey = fewer than {vm.MIN_NEAR} objects nearby", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return fig, q2_numbers(dml_dir)


# ---------------------------------------------------------------------------
# Track D numbers for the headline and the refutations block (results/dml)
# ---------------------------------------------------------------------------

def _tri(row, est="coef"):
    return (float(row[est]), float(row["ci_lo"]), float(row["ci_hi"]))


def dml_numbers(dml_dir=DML):
    """DML main estimates, DML by storm class, lag sweeps and refutations, from the CSVs."""
    d = Path(dml_dir)
    out = {"main": {}, "refute": {}}
    for y in ("Y_total", "Y_residual"):
        out["main"][y] = _tri(pd.read_csv(d / f"compare_{y}.csv", index_col=0).loc["dml"])
        ha = pd.read_csv(d / f"refute_high_alt_{y}.csv").iloc[0]
        rc = pd.read_csv(d / f"refute_random_cc_{y}.csv").iloc[0]
        pl = pd.read_csv(d / f"refute_placebo_{y}.csv")
        wo = pd.read_csv(d / f"refute_without_{y}.csv", index_col=0)
        lags = pd.read_csv(d / f"refute_lags_without_{y}.csv", index_col=[0, 1])
        drop = [i for i in wo.index if i != "all storms"][0]
        p = placebo_p(pl, wo.loc["all storms", "coef"])
        out["refute"][y] = {
            "high_alt": _tri(ha), "band": ha["band"], "n_high": int(ha["n_objects"]),
            "equivalent": None if pd.isna(ha["equivalent"]) else str(ha["equivalent"]) == "True",
            "rcc_shift_se": rc["shift_in_se"], "placebo_p": p, "n_shifts": len(pl),
            "placebo_larger": int(round(p * (len(pl) + 1))) - 1,
            "placebo_mean": pl["coef"].mean(), "placebo_sd": pl["coef"].std(),
            "all": _tri(wo.loc["all storms"]), "without": _tri(wo.loc[drop]), "dropped": drop.split(" ", 1)[1],
            "lags_all": {int(k): _tri(r) for k, r in lags.loc["all storms"].iterrows()},
            "lags_without": {int(k): _tri(r) for k, r in lags.loc[drop].iterrows()}}
    cls = pd.read_csv(d / "storm_class_dml_Y_residual.csv", index_col=0)
    out["class"] = {c: _tri(r) + (int(r["n_storms"]),) for c, r in cls.iterrows()}
    out["class_big_p"] = pd.read_csv(d / "storm_class_dml_big_vs_rest_Y_residual.csv").iloc[0]["p_exact"]
    return out


def offset_numbers(causal_dir=CAUSAL):
    """Model-error-corrected offset: all storms, ducking / recovery parts, without May / Oct
    2024, and without the storms whose pre-storm quiet ap was disturbed
    (offset_model.noisy_quiet_storms)."""
    m = pd.read_csv(Path(causal_dir) / "offset_model.csv", index_col=[0, 1, 2])
    c = m.xs("corrected", level="prediction")
    runs = c.index.get_level_values("storms").unique()
    tri = lambda r: (float(r["est"]), float(r["ci_lo"]), float(r["ci_hi"]))     # noqa: E731
    total = c.loc[("all storms", "offset_total")]
    out = {"total": tri(total), "duck": tri(c.loc[("all storms", "duck_part")]),
           "rec": tri(c.loc[("all storms", "rec_part")]),
           "n_storms": int(total["n_storms"]), "n_objects": int(total["n_objects"])}
    without = [r for r in runs if r.startswith("without")][0]
    out["without"] = tri(c.loc[(without, "offset_total")])
    quiet = [r for r in runs if r.startswith("quiet ap")]
    if quiet:
        out["quiet"] = tri(c.loc[(quiet[0], "offset_total")])
        out["quiet_dropped"] = len(re.search(r"without ([\d,]+)", quiet[0]).group(1).split(","))
        out["quiet_n_storms"] = int(c.loc[(quiet[0], "offset_total"), "n_storms"])
    return out


# ---------------------------------------------------------------------------
# Summary and driver
# ---------------------------------------------------------------------------

def _ci(t, unit="", digits=1):
    return f"{t[0]:+.{digits}f}{unit} (95% CI {t[1]:+.{digits}f} to {t[2]:+.{digits}f})"


def _per_nt(t):
    return f"{t[0]:+.3f} (95% CI {t[1]:+.3f} to {t[2]:+.3f})"


def _excludes_zero(t):
    return t[1] > 0 or t[2] < 0


def lag_text(rf):
    """The 12-24 h model-error lag: with all storms and without May / Oct 2024."""
    a, w = rf["lags_all"], rf["lags_without"]
    both = lambda d: f"12 h {_per_nt(d[12])}, 24 h {_per_nt(d[24])}"      # noqa: E731
    gone = not any(_excludes_zero(w[k]) and w[k][0] > 0 for k in (12, 24))
    verdict = ("so it only appears with May + Oct 2024 and is not a general result"
               if gone else "it survives without May + Oct 2024")
    return (f"- The 12-24 h lag effect in the model error (less decay than MSIS predicts the day after high ap, "
            f"i.e. MSIS keeps the storm atmosphere too long): all storms {both(a)}; without May + Oct 2024 "
            f"(storms {rf['dropped']}) {both(w)}; {verdict}.")


def headline_text(n):
    """The Headline section: Q1 (total effect and model error), Q2 (vulnerability), Q3 (offset)."""
    x = n["extreme"]
    may, oct_, dec = (x["storms"][k] for k in ("May 2024", "Oct 2024", "Dec 2024"))
    r = lambda t: f"{t[0]:.2f} (95% CI {t[1]:.2f} to {t[2]:.2f})"     # noqa: E731
    m = lambda t: f"{t[0]:+.0f} m (95% CI {t[1]:+.0f} to {t[2]:+.0f})"  # noqa: E731
    d = n["dml"]
    tot = d["main"]["Y_total"]
    mod, strong = d["class"]["moderate"], d["class"]["strong"]
    excl = ("CIs exclude 0" if _excludes_zero(mod) and _excludes_zero(strong)
            else "not both CIs exclude 0")
    q2, prof = n["vuln"], n["vuln"]["profile"]
    o = n["offset_c"]
    quiet = (f" Without the {o['quiet_dropped']} storms whose pre-storm quiet ap was disturbed "
             f"({o['quiet_n_storms']} storms left): {r(o['quiet'])}." if "quiet" in o else "")
    return [
        "## Headline",
        "",
        "### Q1. Extra decay from storms and model error",
        "",
        f"- Storms cause extra decay beyond what a quiet-ap atmosphere would give: total storm effect, "
        f"days 0-2, {_ci(n['event']['Y_total'], ' m/day')} (event study); DML {_per_nt(tot)} m/day per nT of "
        f"interval-mean ap, i.e. {100 * tot[0]:+.0f} m/day per 100 nT.",
        f"- In ordinary storms MSIS gets the storm-window total about right but under-predicts decay at peak ap "
        f"(DML by class: moderate ({mod[3]} storms) {_per_nt(mod)} and strong ({strong[3]} storms) {_per_nt(strong)} "
        f"m/day per nT, {excl}); in severe storms its errors are storm-specific (Oct 2024 overshoot, May 2024 "
        f"timing, Dec 2024 under-prediction). Storm-window total: event-study model error days 0-2 "
        f"{_ci(n['event']['Y_residual'], ' m/day')}; days 0..+3 change across all {x['n_storms']} storms "
        f"{_ci(x['all'], ' m/day')}, {_ci(x['without'], ' m/day')} without May/Oct 2024.",
        f"- Extreme storms ({x['objects']} objects, fig 8), verdict: **{x['verdict']}**.",
        f"  - Oct 2024 ({oct_['class']}): a real model overshoot. The days 0..+3 over-prediction survives 3-day "
        f"smoothing ({_ci(oct_['smooth'], ' m/day')}) and MSIS also predicts more altitude loss than observed from a "
        f"calm day before to a calm day after: observed / predicted {r(oct_['ratio'])}, {m(oct_['quiet'])} after "
        f"removing the model's local quiet error, in every altitude band (ratio "
        f"{oct_['alt_ratio'][0]:.2f}-{oct_['alt_ratio'][1]:.2f}).",
        f"  - May 2024 ({may['class']}): a timing error, not an overshoot of the total. The days 0..+3 excess survives "
        f"smoothing ({_ci(may['smooth'], ' m/day')}), but the calm-to-calm loss is not over-predicted once the "
        f"model's own error is removed: {m(may['quiet'])} against its +-30-day quiet error, {m(may['adj'])} against "
        f"its pre-storm error (raw observed / predicted {may['ratio'][0]:.2f}; MSIS already over-predicted the decay "
        f"before the storm).",
        f"  - Dec 2024 ({dec['class']}): the opposite, MSIS under-predicts the loss (observed / predicted "
        f"{r(dec['ratio'])}).",
        lag_text(d["refute"]["Y_residual"]),
        "",
        "### Q2. Who is hit hardest",
        "",
        f"- Storm vulnerability = drag sensitivity x an altitude factor; per unit drag the effect falls about "
        f"{prof['ratio']:.0f}x from ~{round(prof['alt_lo'], -1):.0f} to ~{round(prof['alt_hi'], -1):.0f} km "
        f"({prof['effect_lo']:.1f} to {prof['effect_hi']:.1f} (m/day per nT) per m^2/kg, mean of the supported "
        f"cells of each altitude row; fig 9).",
        "",
        "### Q3. Operator offset (Starlink)",
        "",
        f"- Starlinks offset essentially all of the predicted storm loss (model-error corrected "
        f"{o['total'][0]:.2f}, 95% CI {o['total'][1]:.2f}-{o['total'][2]:.2f}; {o['n_objects']} satellites, "
        f"{o['n_storms']} storms), almost all of it during the storm (ducking part {r(o['duck'])}, recovery part "
        f"{o['rec'][0]:+.2f}, 95% CI {o['rec'][1]:+.2f} to {o['rec'][2]:+.2f}, i.e. ~0).{quiet} Without May / Oct "
        f"2024: {r(o['without'])}.",
        "",
    ]


def q2_text(q):
    """Question 2 lines: importances, seed stability, per-drag altitude pattern, cross-check."""
    imp = lambda d: ", ".join(f"{k} {v:.3f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))   # noqa: E731
    t, pd_, prof = q["Y_total"], q[PER_DRAG_OUTCOME], q["profile"]
    return [
        "## Question 2: vulnerability map (fig 9)",
        "",
        f"- `Y_total` forest ({t['n_objects']} objects): importance {imp(t['importance'])}; "
        f"{t['n_seeds']}-seed map correlation min {t['min_corr']:.3f} (mean {t['mean_corr']:.3f}); forest average "
        f"{t['average']:+.3f} m/day per nT (unweighted row average, not the DML estimate).",
        f"- Per unit drag sensitivity (`{PER_DRAG_OUTCOME}`): importance {imp(pd_['importance'])}; "
        f"{pd_['n_seeds']}-seed map correlation min {pd_['min_corr']:.3f} (mean {pd_['mean_corr']:.3f}). Once size is "
        f"taken out, altitude carries the map: the storm effect is proportional to drag sensitivity.",
        f"- Per-drag altitude pattern: {prof['effect_lo']:.1f} at {prof['alt_lo']:.0f} km to {prof['effect_hi']:.1f} "
        f"at {prof['alt_hi']:.0f} km ((m/day per nT) per m^2/kg), about {prof['ratio']:.1f}x; read for the pattern, "
        f"not single cells. Cross-check (Y_total map / drag sensitivity vs the per-drag forest, supported cells): "
        f"correlation {q['cross_corr']:.2f}.",
        "",
    ]


def refutations_text(d):
    """Refutations block: high-altitude equivalence, random common cause, shuffled storm
    timing, with / without May + Oct 2024, and the lag result without them."""
    lines = ["## Refutations (DML, m/day per nT)", ""]
    for y in ("Y_total", "Y_residual"):
        rf = d["refute"][y]
        if rf["equivalent"] is None:
            eq = "no equivalence test (main estimate covers 0)"
        else:
            eq = f"equivalence band +-{rf['band']:.3f}: {'PASS' if rf['equivalent'] else 'FAIL'}"
        lines += [
            f"- `{y}` (main {_per_nt(rf['all'])}):",
            f"  - High-altitude control ({rf['n_high']} objects): {_per_nt(rf['high_alt'])}; {eq}.",
            f"  - Random common cause: estimate moves {rf['rcc_shift_se']:+.2f} SE.",
            f"  - Shuffled storm timing ({rf['n_shifts']} shifts): mean {rf['placebo_mean']:+.3f}, SD "
            f"{rf['placebo_sd']:.3f}; placebo p = {rf['placebo_p']:.3f} ({rf['placebo_larger']} of "
            f"{rf['n_shifts']} as large as the real estimate).",
            f"  - Without May + Oct 2024 (storms {rf['dropped']}): {_per_nt(rf['without'])}.",
        ]
    lines += [f"  - Lag sweep without May + Oct 2024 (`Y_residual`): "
              + ", ".join(f"{k} h {_per_nt(v)}" for k, v in d["refute"]["Y_residual"]["lags_without"].items()) + ".",
              f"- Model error by storm class in DML (`Y_residual`): "
              + ", ".join(f"{c} {_per_nt(v[:3]) if not np.isnan(v[1]) else f'{v[0]:+.3f} (no CI, {v[3]} storm(s))'}"
                          for c, v in d["class"].items())
              + f"; severe + extreme vs the rest, exact permutation p = {d['class_big_p']:.2f}.",
              ""]
    return lines


def summary_text(n):
    cs = n["class"]
    ext, sev = cs.loc["extreme"], cs.loc["severe"]
    f = lambda t: f"{t[0]:+.1f} m/day (95% CI {t[1]:+.1f} to {t[2]:+.1f})"
    o = n["offset"]
    g = lambda t: f"{t[0]:.2f} (95% CI {t[1]:.2f} to {t[2]:.2f})"
    x = n["extreme"]
    may, oct_, dec = (x["storms"][k] for k in ("May 2024", "Oct 2024", "Dec 2024"))
    sens = "; ".join(f"+{e} d {'/'.join(f'{v:.2f}' for v in r)}" for e, r in x["sens_ratio"].items())
    return "\n".join([
        "# Review figure pack",
        "",
        "Made by `python -m src.review_figures` from the analysis table, results/causal and results/dml. "
        "One line per figure with its key number.",
        "",
        *headline_text(n),
        *refutations_text(n["dml"]),
        *q2_text(n["vuln"]),
        "## Figures",
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
        f"({cs.attrs['severe_storms'][0]:+.1f} and {cs.attrs['severe_storms'][1]:+.1f}). Positive = MSIS predicts more decay than observed. "
        f"Caveat (fig 8): on the full cohort the May 2024 value does not vanish with 3-day smoothing "
        f"({may['smooth'][0]:+.1f} m/day), but it is a timing error, not an overshoot of the storm's total loss; "
        f"of the two severe storms only Oct 2024 is a real overshoot, Dec 2024 is an under-prediction.",
        f"6. `fig6_validation.png`: GRACE-FO-1 Oct 2024 TLE vs precise decay rate r = {n['valid']['grace_r']:.2f} "
        f"(Piyush's A7 result; the left panel is cut from his saved figure); fitted vs catalog drag of rocket bodies "
        f"Spearman rho = {n['valid']['rho']:.2f} (n = {n['valid']['n']}).",
        f"7. `fig7_operator_offset.png`: Starlink total offset {g(o['offset_total'])}, of which ducking part "
        f"{g(o['duck_part'])} and recovery part {g(o['rec_part'])}; "
        f"observed loss is smaller than predicted in {n['offset_storms'][0]} of {n['offset_storms'][1]} storms.",
        f"8. `fig8_extreme_check.png`: does the extreme-storm over-prediction survive on {x['objects']} passive "
        f"objects? (A) Days 0..+3 residual change, MSIS 2.1 raw -> smoothed over 3 days: May 2024 "
        f"{may['raw'][0]:+.1f} -> {may['smooth'][0]:+.1f}, Oct 2024 {oct_['raw'][0]:+.1f} -> {oct_['smooth'][0]:+.1f}, "
        f"Dec 2024 {dec['raw'][0]:+.1f} -> {dec['smooth'][0]:+.1f} m/day (all {x['n_storms']} storms "
        f"{x['all'][0]:+.1f} -> {x['all_s3'][0]:+.1f}); best lag {x['best_lag']:+d} h, correlation only "
        f"{x['lag_gain']:.3f} above 0 h. (B) Running observed - predicted loss from the calm day before (dashed: "
        f"the pre-storm error carried on): May keeps rising at its pre-storm rate and gives part back after day 6, "
        f"Oct jumps after the storm and stays up, Dec falls. (C) Calm-to-calm loss, last calm day in days -10..-3 to "
        f"the first calm day after recovery_end (calm = every 3-h ap < 27; {x['calm_storms']} of {x['n_storms']} "
        f"storms have both): observed / predicted May {may['ratio'][0]:.2f}, Oct {oct_['ratio'][0]:.2f}, "
        f"Dec {dec['ratio'][0]:.2f}; minus the +-30-day quiet error May {may['quiet'][0]:+.0f} m, "
        f"Oct {oct_['quiet'][0]:+.0f} m, Dec {dec['quiet'][0]:+.0f} m. With the calm day after pushed later, "
        f"observed / predicted May/Oct/Dec {sens}. High-altitude control (1,040-1,211 km): |difference| at most "
        f"{x['high_quiet'][0]:.0f} m. Verdict (PLAN.md rules): {x['verdict']}.",
        f"9. `fig9_vulnerability.png`: (A) per-object storm effect from the `Y_total` causal forest, (B) the `Y_total` "
        f"map and (C) the map per unit drag sensitivity, both at the median inclination: the effect grows with drag "
        f"sensitivity and, per unit drag, falls from {n['vuln']['profile']['effect_lo']:.1f} at "
        f"{n['vuln']['profile']['alt_lo']:.0f} km to {n['vuln']['profile']['effect_hi']:.1f} at "
        f"{n['vuln']['profile']['alt_hi']:.0f} km.",
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
    tles = pd.read_parquet(extreme_check.TLES_PATH)

    numbers = {}
    for key, name, make in [
        ("may", "fig1_may2024_cohort.png", lambda: fig_may_cohort(table, storms)),
        ("catalog", "fig2_storm_catalog.png", lambda: fig_storm_catalog(sw, storms)),
        ("physics", "fig3_physics_check.png", lambda: fig_physics_check(table, storms, objects)),
        ("event", "fig4_event_study.png", lambda: fig_event_study()),
        ("class", "fig5_model_error_by_class.png", lambda: fig_class_error(table, storms)),
        ("valid", "fig6_validation.png", lambda: fig_validation(table, objects)),
        ("offset", "fig7_operator_offset.png", lambda: fig_offset(storms)),
        ("extreme", "fig8_extreme_check.png",
         lambda: fig_extreme_check(extreme_check.run((table, storms, physics.load_msis_weather(), tles)))),
        ("vuln", "fig9_vulnerability.png", lambda: fig_vulnerability()),
    ]:
        fig, numbers[key] = make()
        if key == "offset":
            numbers["offset_storms"] = numbers[key].pop("storms_smaller")
        print("wrote", _save(fig, name, out_dir))
    numbers["dml"] = dml_numbers()
    numbers["offset_c"] = offset_numbers()
    (out_dir / "SUMMARY.md").write_text(summary_text(numbers), encoding="utf-8")
    print("wrote", out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
