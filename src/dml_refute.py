"""Refutation checks for the DML storm effect (src/dml.py), and their real run.

Each check is a way the DML estimate could be fooling us; each should come out as
"no effect" or "no change".

1. high_alt: the same DML on the high-altitude control (schema.high_alt_sample,
   >= 1,000 km). The air there is too thin for a storm effect, but the objects share
   the Sun UV error, the season and storm-time TLE noise, so an effect found there is
   an artefact. Tested as equivalence: the whole 95% interval must sit inside
   (-band, +band), with band = EQUIV_BAND x |main-sample estimate|, fixed before the
   real run. Only meaningful when the main estimate is clearly non-zero (Y_total);
   for Y_residual the interval is reported as it is.
2. random_common_cause: add a column of pure noise as an extra confounder. A sound
   estimate barely moves (reported in standard errors).
3. placebo_shifts: move the whole hourly ap series by a random number of days,
   wrapping around the year on purpose, so storms land on the wrong dates; rebuild
   mean ap per interval and rerun. The estimates should centre on zero, with the real
   one in the tail.
4. without_storms: drop the May and Oct 2024 storms (event_checks.key_storm_ids) and
   rerun, so one superstorm cannot carry the result. Rows go if they are linked to a
   dropped storm or overlap its active window (event_study.contamination_windows).
5. planted_lag (synthetic only): rebuild the synthetic outcome so the storm acts on ap
   from PLANTED_LAG_HOURS earlier; dml.lag_sweep must then peak at that lag. This
   shows the sweep can find a lag, not only confirm "no lag".

Run on the real table (data/processed: analysis_table, storms, spaceweather):

    python -m src.dml_refute

Writes CSVs and refutations.md to results/dml (git-ignored).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import dml
from src.event_checks import key_storm_ids
from src.event_study import contamination_windows
from src.schema import ROOT, load_table

EQUIV_BAND = 0.5          # high-altitude band, share of |main estimate| (decided Oct 31)
PLANTED_LAG_HOURS = 12
OUTCOMES = ["Y_total", "Y_residual"]
OUT_DIR = ROOT / "results" / "dml"


def _summary(r):
    return {k: r[k] for k in ["coef", "se_cr1", "se_boot", "ci_lo", "ci_hi", "n_clusters"]}


def high_alt(table, outcome, main_coef=None, band=EQUIV_BAND, n_boot=999, seed=0, **dml_kw):
    """DML on the high-altitude control; equivalent = interval inside +-band*|main_coef|."""
    df = dml.prepare(table, outcome, sample="high_alt")
    out = _summary(dml.dml(df, outcome, n_boot=n_boot, seed=seed, **dml_kw))
    out["n_rows"], out["n_objects"] = len(df), df["norad_id"].nunique()
    if main_coef is None:
        out["band"], out["equivalent"] = np.nan, None
    else:
        out["band"] = band * abs(main_coef)
        out["equivalent"] = bool(-out["band"] < out["ci_lo"] and out["ci_hi"] < out["band"])
    return out


def random_common_cause(df, outcome, base, n_boot=999, seed=0, **dml_kw):
    """Rerun with an extra confounder of pure noise; shift of the estimate in SEs.

    df: from dml.prepare(table, outcome); base: the dml.dml result without it.
    """
    noisy = df.assign(random_cc=np.random.default_rng(seed).normal(size=len(df)))
    r = dml.dml(noisy, outcome, n_boot=n_boot, seed=seed, features=dml.FEATURES + ["random_cc"], **dml_kw)
    shift = r["coef"] - base["coef"]
    return {"coef_base": base["coef"], "coef_random_cc": r["coef"], "shift": shift,
            "shift_in_se": shift / base["se_boot"]}


def ap_mean_circular(df, hourly, shift_days):
    """Interval-mean ap with the hourly series rolled by shift_days, wrapping around.

    Only for the placebo: storms move to other dates of the same year. The lag sweep
    uses dml.ap_mean_shifted, which never wraps.
    """
    rolled = hourly.assign(ap=np.roll(hourly["ap"].to_numpy(), int(round(24 * shift_days))))
    return dml.ap_mean_shifted(df, rolled, 0)


def placebo_shifts(df, outcome, hourly, n_shifts=20, min_days=30, seed=0, **dml_kw):
    """DML with ap rolled by n_shifts random whole-day shifts of at least min_days.

    Returns a DataFrame (shift_days, coef). df: from dml.prepare(table, outcome).
    """
    span_days = len(hourly) / 24
    shifts = np.random.default_rng(seed).integers(min_days, int(span_days) - min_days, n_shifts)
    rows = []
    for i, d in enumerate(shifts):
        fake = df.assign(ap_mean=ap_mean_circular(df, hourly, d))
        rows.append({"shift_days": int(d), "coef": dml.dml(fake, outcome, n_boot=0, seed=seed + i, **dml_kw)["coef"]})
    return pd.DataFrame(rows)


def drop_storm_rows(df, storms, drop):
    """Rows not linked to a dropped storm and not overlapping its active window."""
    keep = ~df["storm_id"].isin(drop).fillna(False).to_numpy(bool)
    half = pd.to_timedelta(df["dt_days"] / 2, unit="D")
    start, end = df["t_mid"] - half, df["t_mid"] + half
    for _, w0, w1 in contamination_windows(storms[storms["storm_id"].isin(drop)], "active"):
        keep &= ~((start < w1) & (end > w0)).to_numpy()
    return df[keep]


def without_storms(df, outcome, storms, drop, n_boot=999, seed=0, **dml_kw):
    """DML with all storms and without the `drop` storms. df: from dml.prepare."""
    rows = {}
    for name, sub in [("all storms", df), ("without " + ",".join(map(str, drop)), drop_storm_rows(df, storms, drop))]:
        r = _summary(dml.dml(sub, outcome, n_boot=n_boot, seed=seed, **dml_kw))
        r["n_rows"] = len(sub)
        rows[name] = r
    return pd.DataFrame(rows).T


def plant_lag(table, hourly, lag_hours=PLANTED_LAG_HOURS):
    """Synthetic only: the table with its storm effect moved onto ap from lag_hours earlier.

    Y_residual swaps true_effect (same-interval ap) for -true_tau * (lagged mean ap -
    AP_QUIET). Rows without ap at that lag are dropped.
    """
    from src.synthetic import AP_QUIET

    lagged = dml.ap_mean_shifted(table, hourly, lag_hours)
    new_effect = -table["true_tau"] * (lagged - AP_QUIET)
    out = table.assign(Y_residual=table["Y_residual"] - table["true_effect"] + new_effect,
                       true_effect=new_effect)
    return out[~np.isnan(lagged)]


def run(table, storms, hourly, outcomes=None, n_shifts=20, n_boot=999, seed=0, out_dir=OUT_DIR):
    """Every refutation on the real table, per outcome; written to out_dir unless None."""
    out = {}
    for y in outcomes or [o for o in OUTCOMES if o in table]:
        df = dml.prepare(table, y)
        base = dml.dml(df, y, n_boot=n_boot, seed=seed)
        clear = not (base["ci_lo"] <= 0 <= base["ci_hi"])
        out[y] = {
            "base": _summary(base),
            "high_alt": high_alt(table, y, base["coef"] if clear else None, n_boot=n_boot, seed=seed),
            "random_cc": random_common_cause(df, y, base, n_boot=n_boot, seed=seed),
            "placebo": placebo_shifts(df, y, hourly, n_shifts=n_shifts, seed=seed),
            "without": without_storms(df, y, storms, key_storm_ids(storms), n_boot=n_boot, seed=seed),
        }
    if out_dir is not None:
        write(out, Path(out_dir))
    return out


def write(out, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for y, r in out.items():
        pd.DataFrame([r["high_alt"]]).to_csv(out_dir / f"refute_high_alt_{y}.csv", index=False)
        pd.DataFrame([r["random_cc"]]).to_csv(out_dir / f"refute_random_cc_{y}.csv", index=False)
        r["placebo"].to_csv(out_dir / f"refute_placebo_{y}.csv", index=False)
        r["without"].to_csv(out_dir / f"refute_without_{y}.csv")
    (out_dir / "refutations.md").write_text(summary_text(out), encoding="utf-8")


def _ci(r):
    return f"{r['coef']:+.4f} (CI {r['ci_lo']:+.4f} to {r['ci_hi']:+.4f})"


def summary_text(out):
    lines = ["# DML refutations", "", "Effects in m/day per nT of interval-mean ap.", ""]
    for y, r in out.items():
        h, rc, p, w = r["high_alt"], r["random_cc"], r["placebo"], r["without"]
        beyond = (p["coef"].abs() >= abs(r["base"]["coef"])).mean()
        equiv = ("not tested (main estimate covers 0)" if h["equivalent"] is None
                 else f"band +-{h['band']:.4f}: {'PASS' if h['equivalent'] else 'FAIL'}")
        lines += [f"## {y}", "", f"- Main sample DML: {_ci(r['base'])}",
                  f"- High-altitude control ({h['n_objects']} objects, {h['n_rows']:,} rows): {_ci(h)}; equivalence {equiv}",
                  f"- Random common cause: {rc['coef_base']:+.4f} -> {rc['coef_random_cc']:+.4f} "
                  f"({rc['shift_in_se']:+.3f} SE)",
                  f"- Shuffled storm timing ({len(p)} shifts): mean {p['coef'].mean():+.4f}, SD {p['coef'].std():.4f}; "
                  f"share at least as large as the real estimate {beyond:.2f}"]
        for name, row in w.iterrows():
            lines.append(f"- {name}: {_ci(row)}, {int(row['n_rows']):,} rows")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="few draws, for a smoke test")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    from src.omni_io import load_spaceweather
    from src.storms import load_storms
    table, storms, hourly = load_table(), load_storms(), load_spaceweather()
    kw = dict(n_shifts=3, n_boot=49) if args.quick else {}
    print(summary_text(run(table, storms, hourly, seed=args.seed, out_dir=args.out, **kw)))


if __name__ == "__main__":
    main()
