"""Extreme-storm check on the full passive cohort: does "MSIS over-predicts the decay after
the extreme storms (May and Oct 2024)" survive?

    python -m src.extreme_check

Passive main-sample objects of data/processed/analysis_table.parquet only. Steps:
1. Notebook-04 checks at cohort scale (physics.py functions): baseline-relative residual
   (days 0..+3 minus days -10..-3) for all 20 storms, MSIS 2.1 and NRLMSISE-00; the
   prediction smoothed over 1, 2, 3 days (TLE-blur check); lag cross-correlation.
   SEs over objects per storm, over storms when pooling.
2. Blur-free outcome (physics.calm_to_calm): semi-major axis lost from a calm day before
   to a calm day after each storm, observed vs MSIS 2.1, with the onset intervals that
   task A4 left out filled back in (physics.fill_onset_gaps). By storm class, by storm
   for the key storms (May 2024, Oct 2024 and the two severe storms), by altitude band.
3. Verdict with the PLAN.md rules (physics.storm_check_label / extreme_storm_verdict).
High-altitude objects (> 1,000 km) are the negative control of the calm-to-calm loss.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import physics
from src.schema import STORM_CLASSES, high_alt_sample, main_sample

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TLES_PATH = ROOT / "data" / "raw" / "piyush" / "processed" / "cohort_tles_cleaned.parquet"
GAP_CACHE = PROCESSED / "physics_onset_gaps.parquet"
SMOOTH_DAYS = (1, 2, 3)
LAGS_H = range(-24, 25, 3)
EXTRA_DAYS = (0, 2, 3)           # calm day after pushed later (TLE fit windows reach back days)
ALT_EDGES = [-np.inf, 500, 600, 700, np.inf]
ALT_LABELS = ["< 500 km", "500-600 km", "600-700 km", ">= 700 km"]
VARIANTS = {"resid_msis21": "MSIS 2.1", "resid_nrlmsise00": "NRLMSISE-00",
            **{f"resid_msis21_s{d}": f"MSIS 2.1 smoothed {d} d" for d in SMOOTH_DAYS}}


def key_storms(storms):
    """{storm_id: name} for the extreme and severe storms, in time order, e.g. 'May 2024
    (extreme)'; and the verdict pair = the two deepest storms by min Dst (May, Oct 2024)."""
    k = storms[storms["class"].isin(["extreme", "severe"])].sort_values("main_onset")
    names = {r["storm_id"]: f"{r['main_onset']:%b %Y} ({r['class']})" for r in k.to_dict("records")}
    pair = storms.nsmallest(2, "min_dst")["storm_id"].tolist()
    return names, [sid for sid in names if sid in pair]


def load_inputs():
    """(analysis table, storms, MSIS-padded hourly space weather, cleaned cohort TLEs)."""
    from src.storms import load_storms
    return (pd.read_parquet(PROCESSED / "analysis_table.parquet"), load_storms(),
            physics.load_msis_weather(), pd.read_parquet(TLES_PATH))


# ---------------------------------------------------------------------------
# 1. Notebook-04 checks on the cohort
# ---------------------------------------------------------------------------

def clean_rates(table):
    """Main-sample rows without Hampel outliers, as rates, with the smoothed predictions."""
    rates = physics.table_rates(main_sample(table))
    rates = rates[~rates["hampel_outlier"]].reset_index(drop=True)
    for d in SMOOTH_DAYS:
        rates = physics.add_predictions(physics.smooth_driver(rates, "msis21", d), f"msis21_s{d}")
    return rates


def smoothing_check(rates, storms, exclude=()):
    """Baseline-relative residual for every variant (m/day).

    Returns (per_storm, pooled, by_storm): per_storm = {label: storm_baseline_residual},
    pooled = storm_mean over all storms and without `exclude` (storm = cluster),
    by_storm = per_storm_change per (label, storm) (object = cluster).
    """
    per_storm, pooled, by_storm = {}, [], []
    for col, label in VARIANTS.items():
        ps = physics.storm_baseline_residual(rates, storms, col)
        ps[["pre", "post", "change", "se"]] *= 1000
        per_storm[label] = ps
        for sample, excl in (("all storms", ()), ("without May/Oct 2024", tuple(exclude))):
            pooled.append({"outcome": label, "sample": sample, **physics.storm_mean(ps, exclude=excl)})
        by_storm.append(physics.per_storm_change(ps).assign(outcome=label))
    pooled = pd.DataFrame(pooled).set_index(["outcome", "sample"])
    by_storm = pd.concat(by_storm).reset_index().set_index(["outcome", "storm_id"])
    return per_storm, pooled, by_storm


def lag_check(rates, lags=LAGS_H):
    """Mean over objects of corr(observed, prediction shifted by lag) and its SE (object = cluster)."""
    c = physics.lag_correlation(rates, "msis21", lags=lags)
    return pd.DataFrame({"mean": c.mean(axis=1), "se": c.std(axis=1) / np.sqrt(c.shape[1]), "objects": c.shape[1]})


# ---------------------------------------------------------------------------
# 2. Calm-to-calm loss
# ---------------------------------------------------------------------------

def chain_rates(table, tles, sw, sample=main_sample, cache_path=GAP_CACHE):
    """Every interval of the sample's objects (Hampel outliers and climbs included: TLE
    noise cancels in a sum) plus the onset gaps refilled from the TLEs."""
    ids = sample(table)["norad_id"].unique()
    rows = table[(table["is_active"] == 0) & table["norad_id"].isin(ids)]
    rates = physics.table_rates(rows).assign(onset_gap=False)
    B = rows.groupby("norad_id")["drag_sensitivity"].first()
    gaps = physics.onset_gaps(rates)
    filled = physics.fill_onset_gaps(gaps, tles[tles["norad_id"].isin(ids)], sw, B, cache_path=cache_path)
    keep = [c for c in rates.columns if c in filled.columns]
    return pd.concat([rates, filled[keep]], ignore_index=True)


def alt_band(alt_km):
    return pd.cut(alt_km, ALT_EDGES, labels=ALT_LABELS, right=False)


def calm_check(chain, storms, sw, extra_days=EXTRA_DAYS):
    """{extra_days: calm_to_calm rows with alt_band}."""
    return {e: physics.calm_to_calm(chain, storms, sw, extra_days=e).assign(alt_band=lambda d: alt_band(d["alt_km"]))
            for e in extra_days}


def calm_tables(per, names):
    """Summaries of one calm_to_calm run: by class, by key storm, by altitude band (all
    storms, and key storms by band)."""
    key = per[per["storm_id"].isin(list(names))]
    by_storm = physics.calm_summary(key, "storm_id").assign(storm=lambda d: d["storm_id"].map(names))
    return {"class": physics.calm_summary(per, "class").set_index("class").reindex(STORM_CLASSES).dropna(how="all"),
            "storm": by_storm.set_index("storm_id"),
            "alt": physics.calm_summary(per, "alt_band").set_index("alt_band"),
            "storm_alt": physics.calm_summary(key, ["storm_id", "alt_band"]).assign(
                storm=lambda d: d["storm_id"].map(names))}


def cumulative_residual(chain, per, storm_id, storms, grid=np.arange(-10, 15.01, 0.25)):
    """Mean over objects (and SE) of the running sum of observed - predicted loss (m) from
    each object's calm epoch t0, on a grid of days from main_onset (for plots only; NaN
    outside an object's span). Blur moves decay in time: the curve then rises and comes
    back; a real overshoot leaves it up at the calm day after."""
    onset = storms.set_index("storm_id").loc[storm_id, "main_onset"]
    p = per[per["storm_id"] == storm_id].set_index("norad_id")
    curves = []
    for nid, g in chain[chain["norad_id"].isin(p.index)].sort_values("t_start").groupby("norad_id"):
        t0, t1 = p.loc[nid, "t0"], p.loc[nid, "t1"]
        g = g[(g["t_start"] >= t0 - pd.Timedelta(seconds=1)) & (g["t_end"] <= t1 + pd.Timedelta(seconds=1))]
        y = np.r_[0.0, np.cumsum(g["resid_msis21"] * g["dt_days"] * 1000)]
        x = np.r_[(t0 - onset) / pd.Timedelta(days=1), (g["t_end"] - onset) / pd.Timedelta(days=1)]
        curves.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
    c = np.array(curves)
    n = np.isfinite(c).sum(axis=0)
    with warnings.catch_warnings():          # grid points outside every span are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        out = pd.DataFrame({"mean": np.nanmean(c, axis=0), "se": np.nanstd(c, axis=0, ddof=1) / np.sqrt(n),
                            "objects": n}, index=pd.Index(grid, name="event_day"))
    # only where most objects are inside their span; attrs: start day and the pre-storm
    # error slope (m/day) for the dashed baseline in plots.plot_calm_cumulative
    out = out[out["objects"] >= 0.8 * len(p)]
    out.attrs = {"t0_day": float(np.median((p["t0"] - onset) / pd.Timedelta(days=1))),
                 "pre_slope": float(p["pre_resid_m_per_day"].mean()),
                 "quiet_slope": float(p["quiet_resid_m_per_day"].mean())}
    return out


# ---------------------------------------------------------------------------
# 3. Verdict and the whole run
# ---------------------------------------------------------------------------

def verdict(by_storm, calm_storm, names, pair):
    """Per key storm label (physics.storm_check_label, raw vs 3-day smoothed MSIS 2.1 and
    the calm-to-calm loss) and the overall verdict over the pair (May, Oct 2024)."""
    smooth = f"MSIS 2.1 smoothed {max(SMOOTH_DAYS)} d"
    labels = {sid: physics.storm_check_label(by_storm.loc[("MSIS 2.1", sid)], by_storm.loc[(smooth, sid)],
                                             calm_storm.loc[sid]) for sid in names}
    overall = physics.extreme_storm_verdict({names[sid].split(" (")[0]: labels[sid] for sid in pair})
    return labels, overall


def run(inputs=None, log=print):
    """Everything above; returns a dict of frames and the verdict (used by notebooks/09 and
    review_figures.fig_extreme_check)."""
    table, storms, sw, tles = load_inputs() if inputs is None else inputs
    names, pair = key_storms(storms)
    log("smoothing / baseline checks ...")
    rates = clean_rates(table)
    per_storm, pooled, by_storm = smoothing_check(rates, storms, exclude=pair)
    log("lag check ...")
    lags = lag_check(rates)
    log("calm-to-calm loss ...")
    chain = chain_rates(table, tles, sw)
    calm = calm_check(chain, storms, sw)
    tables = {e: calm_tables(per, names) for e, per in calm.items()}
    high_chain = chain_rates(table, tles, sw, sample=high_alt_sample,
                             cache_path=GAP_CACHE.with_name("physics_onset_gaps_high.parquet"))
    high = physics.calm_to_calm(high_chain, storms, sw)
    labels, overall = verdict(by_storm, tables[0]["storm"], names, pair)
    curves = {sid: cumulative_residual(chain, calm[0], sid, storms) for sid in names}
    return {"names": names, "pair": pair, "rates": rates, "per_storm": per_storm, "pooled": pooled,
            "by_storm": by_storm, "lags": lags, "chain": chain, "calm": calm, "tables": tables,
            "high": high, "high_summary": physics.calm_summary(high, "class").set_index("class"),
            "labels": labels, "verdict": overall, "curves": curves, "n_objects": rates["norad_id"].nunique()}


def main():
    pd.set_option("display.width", 200)
    out = run()
    print(out["pooled"].round(2).to_string())
    print(out["tables"][0]["storm"].round(3).to_string())
    for sid, lab in out["labels"].items():
        print(f"{out['names'][sid]}: {lab}")
    print("Verdict:", out["verdict"])


if __name__ == "__main__":
    main()
