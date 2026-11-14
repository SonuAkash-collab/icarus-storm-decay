"""Sensitivity checks on the frozen results: the Hampel filter and the Q3 orbit-change screen.

`python -m src.sensitivity` writes results/review/SENSITIVITY.md and nothing else. No frozen
number, figure or SUMMARY.md changes: every variant is rebuilt in memory from the cached
physics drivers and the saved analysis table.

1. hampel_timing: share of Hampel-flagged rows by event day and storm class, in the
   passive main sample before the tle_ok filter.
2. hampel_variants: event study (days 0-2, Y_total and Y_residual, 95% wild bootstrap CI,
   no placebo dates) with the Hampel filter changed. The TLE endpoint checks stay; only
   the Hampel part of tle_ok changes. Variants:
     frozen          k = 3.5 (rebuilt here; must match the saved table)
     put back        k = 3.5 flags removed from the filter, B and predictions kept as frozen
     no filter       no Hampel anywhere, so B is refitted on every quiet row too
     k = 5           flags at k = 5, B refitted without them
3. screen_sweep: offset_model.run with the orbit-change screen at several thresholds.
   active_rows has no threshold argument and orbit_change_flags binds its default when it is
   defined, so the sweep swaps in an active_rows that passes max_change_km explicitly.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import build_table as bt
from src import event_checks as ec
from src import event_study as es
from src import offset_model as om
from src import physics
from src.schema import EVENT_WINDOW, HIGH_ALT_MIN_KM, MAX_ECC, ROOT

OUT_PATH = ROOT / "results" / "review" / "SENSITIVITY.md"
OUTCOMES = ("Y_total", "Y_residual")
EVENT_BINS = [("days -10..-3", -10, -3), ("days -2..-1", -2, -1), ("day 0", 0, 0),
              ("days 1..2", 1, 2), ("days 3..20", 3, EVENT_WINDOW[1])]
SCREENS_KM = (0.5, 1.0, 2.0)
CLASS_ORDER = ["moderate", "strong", "severe", "extreme"]


# ---------------------------------------------------------------------------
# 1. Hampel removal by timing and storm class
# ---------------------------------------------------------------------------

def pre_tle_sample(table):
    """Passive main-sample rows before the tle_ok filter: same filters as
    schema.main_sample (passive, ecc < 0.01, drag sensitivity, alt < 1000 km) minus tle_ok."""
    return table[(table["is_active"] == 0) & (table["ecc"] < MAX_ECC) & table["drag_sensitivity"].notna()
                 & (table["alt_km"] < HIGH_ALT_MIN_KM)]


def event_bin(event_day):
    """Label of the event-day bin of every row (whole days, as the event study's k);
    rows outside the -10..+20 event windows are 'outside storm windows'."""
    k = np.floor(pd.Series(event_day, dtype=float))
    label = pd.Series("outside storm windows", index=k.index, dtype=object)
    for name, lo, hi in EVENT_BINS:
        label[(k >= lo) & (k <= hi)] = name
    return label


def _share(g):
    return pd.Series({"rows": len(g), "hampel": int(g["hampel_outlier"].sum()),
                      "share_pct": 100 * g["hampel_outlier"].mean()})


def hampel_timing(table):
    """(by_bin, by_class, by_class_bin): rows, flagged rows and share (%) of
    hampel_outlier = 1 in the pre-tle_ok passive main sample."""
    s = pre_tle_sample(table)
    bins = event_bin(s["event_day"])
    order = [b[0] for b in EVENT_BINS] + ["outside storm windows"]
    by_bin = s.groupby(bins).apply(_share, include_groups=False).reindex(order)
    by_bin.loc["all rows"] = _share(s)
    inside = s[bins != "outside storm windows"]
    by_class = inside.groupby(inside["storm_class"].astype(str)).apply(_share, include_groups=False)
    by_class = by_class.reindex([c for c in CLASS_ORDER if c in by_class.index])
    by_class.loc["all storm windows"] = _share(inside)
    by_cb = inside.groupby([inside["storm_class"].astype(str), bins[inside.index]]).apply(_share, include_groups=False)
    return by_bin, by_class, by_cb


# ---------------------------------------------------------------------------
# 2. Event study with the Hampel filter changed
# ---------------------------------------------------------------------------

@contextmanager
def hampel_setting(k):
    """Make build_table.cohort_physics flag Hampel outliers at k robust SDs (k=None: flag
    nothing). cohort_physics calls physics.flag_outliers with the module default, which is
    bound when hampel_flags is defined, so the setting is passed in explicitly."""
    original = physics.flag_outliers

    def flagged(rates, model="msis21", **kwargs):
        if k is None:
            out = rates.copy()
            out["outlier"] = False
            return out
        return original(rates, model, **{**kwargs, "k": k})

    with mock.patch.object(physics, "flag_outliers", flagged):
        yield


def passive_table(drivers, tles, sw, storms, k, keep_flags_off=False):
    """Passive analysis table with Hampel flags at k (None = no filter).
    keep_flags_off: B and predictions as flagged (k), but hampel_outlier set to 0, so the
    filter in tle_ok is undone and the endpoint checks stay."""
    with hampel_setting(k):
        rates, drag = bt.cohort_physics(drivers, storms)
    if keep_flags_off:
        rates = rates.assign(hampel_outlier=False)
    return bt.assemble(rates, drag, sw, storms, tles)


def event_effect(table, storms, outcome, n_boot=999, seed=0):
    """Days 0-2 mean effect with the 95% wild bootstrap CI (event_checks.short_run_effect),
    plus the rows and storms of the fit. No placebo dates."""
    res, report = es.event_study(table, storms, outcome, n_boot=n_boot, seed=seed)
    sr = ec.short_run_effect(res)
    return pd.Series({**sr.to_dict(), "rows": int(res.loc[0, "n_rows"]), "storms_used": int(report["used"].sum())})


def hampel_variants(drivers, tles, sw, storms, n_boot=999, seed=0, log=print):
    """Run every Hampel variant. Returns (effects, tables): effects has one row per
    (variant, outcome); tables holds each variant's passive analysis table."""
    tables = {}
    base = passive_table(drivers, tles, sw, storms, k=physics.HAMPEL_K)
    tables["frozen (k = 3.5)"] = base
    tables["put back (k = 3.5 flags off, B kept)"] = passive_table(drivers, tles, sw, storms, k=physics.HAMPEL_K,
                                                                    keep_flags_off=True)
    tables["no filter (B refitted on all rows)"] = passive_table(drivers, tles, sw, storms, k=None)
    tables["k = 5 (B refitted)"] = passive_table(drivers, tles, sw, storms, k=5.0)
    rows = {}
    for name, t in tables.items():
        for outcome in OUTCOMES:
            if log:
                log(f"event study: {name}, {outcome}")
            rows[(name, outcome)] = event_effect(t, storms, outcome, n_boot, seed)
    effects = pd.DataFrame(rows).T.rename_axis(["variant", "outcome"])
    return effects, tables


def table_matches(a, b, cols=("Y_total", "Y_residual", "hampel_outlier", "tle_ok", "drag_sensitivity")):
    """Largest absolute difference per column between two passive tables of the same rows
    (None if the rows differ): the check that the rebuild reproduces the frozen table."""
    keys = ["norad_id", "t_start"]
    a = a[a["is_active"] == 0].sort_values(keys).reset_index(drop=True)
    b = b[b["is_active"] == 0].sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or not (a[keys].to_numpy() == b[keys].to_numpy()).all():
        return None
    return pd.Series({c: float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) for c in cols})


# ---------------------------------------------------------------------------
# 3. Q3 orbit-change screen
# ---------------------------------------------------------------------------

@contextmanager
def orbit_screen(max_change_km):
    """offset_model.run with the orbit-change screen at max_change_km (passed explicitly)."""
    def active_rows(table):
        ok = (table["is_active"] == 1) & (table["tle_ok"] == 1) & (table["ecc"] < 0.01)
        if "free_drag" in table:
            ok &= table["free_drag"] != 1
        rows = table[ok]
        return rows[om.orbit_change_flags(rows, max_change_km=max_change_km) == 0]

    with mock.patch.object(om, "active_rows", active_rows):
        yield


def screen_sweep(table, storms, sw, screens=SCREENS_KM, n_boot=999, seed=0, log=print):
    """Corrected offset (all storms) per screen threshold: total, ducking part, recovery part
    (est and CI), satellite-storm pairs, storms and satellites. Returns (summary, runs)."""
    rows, runs = {}, {}
    for km in screens:
        if log:
            log(f"offset model: screen {km} km")
        with orbit_screen(km):
            run = om.run(table, storms, n_boot=n_boot, seed=seed, sw=sw)
        runs[km] = run
        s = run["summary"].loc[("all storms", "corrected")]
        row = {"n_pairs": int(s["n_pairs"].iloc[0]), "n_storms": int(s["n_storms"].iloc[0]),
               "n_objects": int(s["n_objects"].iloc[0])}
        for stat in ("offset_total", "duck_part", "rec_part"):
            row.update({f"{stat}": s.loc[stat, "est"], f"{stat}_lo": s.loc[stat, "ci_lo"],
                        f"{stat}_hi": s.loc[stat, "ci_hi"]})
        rows[km] = row
    return pd.DataFrame(rows).T.rename_axis("max_change_km"), runs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _ci(est, lo, hi, d=1):
    return f"{est:+.{d}f} ({lo:+.{d}f} to {hi:+.{d}f})"


def _change(new, base, parent):
    """Relative change of `new` against `base`. Where |base| is under 20% of its parent
    headline (a number that is ~0, so the ratio explodes), the change is measured against
    the parent instead. Returns (percent, what it was measured against)."""
    ref, against = (base, "own baseline") if abs(base) >= 0.2 * abs(parent) else (parent, "parent headline")
    return 100 * (new - base) / abs(ref), against


def headline_table(effects, sweep):
    """The closing table: every headline, its frozen value, the variant value and the change.
    Flags |change| > 20%. Returns a DataFrame (strings) and the list of flagged rows."""
    rows = []
    frozen = "frozen (k = 3.5)"
    y_tot = effects.loc[(frozen, "Y_total"), "est"]
    for variant in [v for v in effects.index.get_level_values(0).unique() if v != frozen]:
        for outcome in OUTCOMES:
            b, v = effects.loc[(frozen, outcome)], effects.loc[(variant, outcome)]
            pct, against = _change(v["est"], b["est"], y_tot if outcome == "Y_residual" else b["est"])
            rows.append({"headline": f"{outcome} days 0-2 (m/day)", "variant": variant,
                         "frozen": _ci(b["est"], b["ci_lo"], b["ci_hi"]), "variant value": _ci(v["est"], v["ci_lo"], v["ci_hi"]),
                         "change %": pct, "measured against": against,
                         "CI includes 0": f"{b['ci_lo'] <= 0 <= b['ci_hi']} -> {v['ci_lo'] <= 0 <= v['ci_hi']}"})
    base = sweep.loc[1.0]
    for km in [k for k in sweep.index if k != 1.0]:
        v = sweep.loc[km]
        for stat, label in (("offset_total", "Q3 corrected total offset"), ("duck_part", "Q3 ducking part"),
                            ("rec_part", "Q3 recovery part")):
            pct, against = _change(v[stat], base[stat], base["offset_total"])
            rows.append({"headline": label, "variant": f"screen {km:g} km",
                         "frozen": _ci(base[stat], base[f"{stat}_lo"], base[f"{stat}_hi"], 2),
                         "variant value": _ci(v[stat], v[f"{stat}_lo"], v[f"{stat}_hi"], 2),
                         "change %": pct, "measured against": against,
                         "CI includes 0": f"{base[f'{stat}_lo'] <= 0 <= base[f'{stat}_hi']} -> "
                                          f"{v[f'{stat}_lo'] <= 0 <= v[f'{stat}_hi']}"})
    out = pd.DataFrame(rows)
    out["flag"] = np.where(out["change %"].abs() > 20, "MOVES > 20%", "")
    return out


def _md(df, floatfmt="{:.1f}"):
    """Plain markdown table of a DataFrame (index kept as the first column)."""
    df = df.reset_index(drop=isinstance(df.index, pd.RangeIndex))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(floatfmt.format(x) if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def reading(timing, effects, sweep):
    """Plain-language lines for the report, with the numbers taken from the results."""
    by_bin, _, by_cb = timing
    frozen = "frozen (k = 3.5)"
    day0 = by_bin.loc["day 0", "share_pct"]
    quiet = by_bin.loc["days -10..-3", "share_pct"]
    top = by_cb["share_pct"].xs("day 0", level=1).idxmax()
    out = [f"- Hampel removal is concentrated at the storm onset: {day0:.1f}% of day-0 rows are flagged against "
           f"{quiet:.1f}% on days -10..-3 and {by_bin.loc['all rows', 'share_pct']:.1f}% overall; the highest day-0 share "
           f"is in {top} storms ({by_cb['share_pct'].xs('day 0', level=1).max():.0f}%)."]
    for variant in [v for v in effects.index.get_level_values(0).unique() if v != frozen]:
        t, r = effects.loc[(variant, "Y_total")], effects.loc[(variant, "Y_residual")]
        out.append(f"- {variant}: Y_total {_ci(t.est, t.ci_lo, t.ci_hi)}, Y_residual {_ci(r.est, r.ci_lo, r.ci_hi)} "
                   f"(frozen {_ci(*effects.loc[(frozen, 'Y_total'), ['est', 'ci_lo', 'ci_hi']])}, "
                   f"{_ci(*effects.loc[(frozen, 'Y_residual'), ['est', 'ci_lo', 'ci_hi']])}).")
    out.append(f"- Q3 offset at 0.5 / 1 / 2 km: {sweep.loc[0.5, 'offset_total']:.2f} / {sweep.loc[1.0, 'offset_total']:.2f} / "
               f"{sweep.loc[2.0, 'offset_total']:.2f}; the screen only changes the pair count "
               f"({int(sweep.loc[0.5, 'n_pairs'])} / {int(sweep.loc[1.0, 'n_pairs'])} / {int(sweep.loc[2.0, 'n_pairs'])}).")
    return out


def write_report(path, timing, effects, repro, sweep, final):
    by_bin, by_class, by_cb = timing
    flagged = final[final["flag"] != ""]
    nice = lambda d: d.astype({"rows": int, "hampel": int}).rename(columns={"hampel": "flagged", "share_pct": "share %"})
    lines = ["# Sensitivity checks", "",
             "Made by `python -m src.sensitivity` from the frozen analysis table and the cached physics drivers. "
             "Nothing frozen changes: variants are rebuilt in memory.", "",
             "## 1. Hampel removal by timing", "",
             "Passive main sample (ecc < 0.01, drag sensitivity, alt < 1000 km) before the `tle_ok` filter; "
             "share of rows with `hampel_outlier = 1`. Event day = whole days from `main_onset` of the linked storm "
             "(-10..+20 window).", "", _md(nice(by_bin).rename_axis("event day")), "", "By storm class (rows inside storm windows):", "",
             _md(nice(by_class)), "", "Class by event day (share %):", "",
             _md(by_cb["share_pct"].unstack().reindex(columns=[b[0] for b in EVENT_BINS])
                 .reindex([c for c in CLASS_ORDER if c in by_cb.index.get_level_values(0)]).rename_axis("storm class")), "",
             "## 2. Event study without the Hampel filter", "",
             "Days 0-2 mean effect, 95% wild cluster bootstrap CI by storm (999 draws, seed 0, same settings as the freeze, "
             "no placebo dates). TLE endpoint checks kept in every variant.", "",
             "Check, rebuilt k = 3.5 table against the saved table (largest absolute difference): "
             + (", ".join(f"{k} {v:.2g}" for k, v in repro.items()) if repro is not None else "rows differ"), "",
             _md(effects.assign(effect=[_ci(r.est, r.ci_lo, r.ci_hi) for r in effects.itertuples()])
                 .astype({"rows": int, "storms_used": int})[["effect", "rows", "storms_used"]]), "",
             "- `put back`: the k = 3.5 flags are not used in the sample, B and predictions stay as frozen.",
             "- `no filter`: no Hampel flags anywhere, so B is also refitted on every quiet row.",
             "- `k = 5`: flags at 5 robust SDs (instead of 3.5), B refitted without them.", "",
             "## 3. Q3 orbit-change screen", "",
             "`offset_model.run` (model-error corrected, all storms, quiet-ap check as in the freeze), "
             "screen = largest net altitude change in a 21-day window that still counts as station keeping. "
             "1 km is the frozen setting. Pairs = satellite-storm pairs.", "",
             _md(sweep.assign(total=[_ci(r.offset_total, r.offset_total_lo, r.offset_total_hi, 2) for r in sweep.itertuples()],
                              ducking=[_ci(r.duck_part, r.duck_part_lo, r.duck_part_hi, 2) for r in sweep.itertuples()],
                              recovery=[_ci(r.rec_part, r.rec_part_lo, r.rec_part_hi, 2) for r in sweep.itertuples()])
                 .astype({"n_pairs": int, "n_storms": int, "n_objects": int})
                 [["total", "ducking", "recovery", "n_pairs", "n_storms", "n_objects"]]), "",
             "## 4. Pairs in the main Q3 run", "",
             f"{int(sweep.loc[1.0, 'n_pairs'])} satellite-storm pairs ({int(sweep.loc[1.0, 'n_objects'])} satellites, "
             f"{int(sweep.loc[1.0, 'n_storms'])} storms), screen 1 km.", "",
             "## Reading", "", *reading(timing, effects, sweep), "",
             "## Summary table", "",
             "Change % = (variant - frozen) / |frozen|. Where the frozen value is under 20% of its parent headline "
             "(Y_residual vs Y_total, the Q3 recovery part vs the total) the ratio explodes, so the change is measured "
             "against the parent headline instead (column `measured against`). Flag: |change| > 20%.", "",
             "**Flags:** " + ("; ".join(f"{r['headline']}, {r['variant']} ({r['change %']:+.0f}%)" for _, r in flagged.iterrows())
                              if len(flagged) else "no headline moves by more than 20%."), "",
             _md(final, "{:+.1f}"), ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--n-boot", type=int, default=999)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    from src.omni_io import load_spaceweather
    from src.schema import load_table
    table = load_table(bt.TABLE_PATH)
    intervals, tles, sw_msis, storms = bt.load_inputs()
    intervals = intervals.rename(columns={"alt_km": "alt_km_tle"})
    drivers = bt.cohort_drivers(tles, intervals, sw_msis, storms)

    timing = hampel_timing(table)
    effects, tables = hampel_variants(drivers, tles, sw_msis, storms, args.n_boot, args.seed)
    repro = table_matches(tables["frozen (k = 3.5)"], table)
    sweep, _ = screen_sweep(table, storms, load_spaceweather(), n_boot=args.n_boot, seed=args.seed)
    final = headline_table(effects, sweep)
    write_report(args.out, timing, effects, repro, sweep, final)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
