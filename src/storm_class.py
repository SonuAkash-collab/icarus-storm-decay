"""Model error by storm class, with storm-level inference.

Question: does MSIS 2.1 get storms wrong differently by size? The real DML run found
that model error per nT of ap is negative without the May / Oct 2024 storms (objects
fall faster than predicted) and about zero with them, i.e. the biggest storms pull the
other way.

Per storm (review_figures.storm_class_error, reused): the change in Y_residual from
the quiet baseline (days -10..-3) to the storm window (days 0..+3), mean over objects.
Positive = MSIS predicts more decay than observed.

The storm is the unit of inference (20 storms, one value each):
- class_table: mean per class; 95% t interval over storms when a class has >= 3
  storms (moderate 11, strong 6). Severe (2) and extreme (1) get no interval: with one
  or two storms a storm-level interval does not exist, so their storm values are
  listed instead.
- big_vs_rest: mean of the severe + extreme storms minus mean of the others, with an
  exact permutation p-value: every way of choosing that many storms out of all of them
  (C(20, 3) = 1,140), share whose gap is at least as large in absolute value. No
  normal approximation is needed and the 3 big storms cannot be over-counted by their
  many objects.

The same question in DML (dml_by_class): the effect of ap on the outcome (m/day per
nT) separately for the rows of each class's storms. DML's first stage (the two
cross-fitted leftovers, Y_res and T_res, from dml.dml) does not depend on storm class,
so it is fitted once on the whole main sample; only the final slope of Y_res on T_res
(weights dt_days) is taken per class, over the rows linked to that class's storms
(their -10..+20 day windows). Storm-clustered wild bootstrap intervals for classes
with >= 3 storms, none below. dml_big_vs_rest repeats the exact permutation test on
these slopes: for each of the C(20, 3) splits of the storms only the final slope is
recomputed, from per-storm sums of dt*T_res*Y_res and dt*T_res^2.

    python -m src.storm_class      # real table; writes results/dml/storm_class_*.csv
"""

import argparse
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from src import dml
from src.review_figures import storm_class_error
from src.schema import ROOT, STORM_CLASSES, load_table

BIG = ("severe", "extreme")
OUT_DIR = ROOT / "results" / "dml"


def class_table(per_storm):
    """Mean change per class, t interval over storms for classes with >= 3 storms."""
    rows = []
    for cls in STORM_CLASSES:
        g = per_storm[per_storm["storm_class"] == cls]
        if g.empty:
            continue
        n, mean = len(g), g["change"].mean()
        half = stats.t.ppf(0.975, n - 1) * g["change"].std() / np.sqrt(n) if n >= 3 else np.nan
        rows.append({"storm_class": cls, "n_storms": n, "mean": mean, "ci_lo": mean - half, "ci_hi": mean + half,
                     "storm_values": ", ".join(f"{v:+.1f}" for v in g["change"])})
    return pd.DataFrame(rows).set_index("storm_class")


def big_vs_rest(per_storm, big=BIG):
    """Gap between the big-storm mean and the rest, with an exact permutation p-value."""
    values = per_storm["change"].to_numpy(float)
    is_big = per_storm["storm_class"].isin(big).to_numpy()
    k, n = int(is_big.sum()), len(values)
    gap = values[is_big].mean() - values[~is_big].mean()
    total = values.sum()
    gaps = np.array([values[list(c)].mean() - (total - values[list(c)].sum()) / (n - k)
                     for c in combinations(range(n), k)])
    p = float(np.mean(np.abs(gaps) >= abs(gap) - 1e-12))
    return {"gap": gap, "mean_big": values[is_big].mean(), "mean_rest": values[~is_big].mean(),
            "n_big": k, "n_rest": n - k, "n_permutations": comb(n, k), "p_exact": p}


def storm_sums(df, y_res, t_res):
    """Per linked storm: class, sum of dt*T_res*Y_res (a) and dt*T_res^2 (b), rows."""
    w = df["dt_days"].to_numpy()
    d = pd.DataFrame({"storm_id": df["storm_id"].to_numpy(), "storm_class": df["storm_class"].to_numpy(),
                      "a": w * t_res * y_res, "b": w * t_res ** 2})
    d = d[d["storm_id"].notna()]
    return d.groupby("storm_id").agg(storm_class=("storm_class", "first"), a=("a", "sum"), b=("b", "sum"),
                                     n_rows=("a", "size")).reset_index()


def dml_by_class(df, outcome="Y_residual", n_boot=999, seed=0, **dml_kw):
    """DML effect of ap per storm class from one first stage. df: from dml.prepare.

    Returns (class table, per-storm sums). Classes with < 3 storms get no interval.
    """
    r = dml.dml(df, outcome, n_boot=0, seed=seed, **dml_kw)
    y_res, t_res = r["y_res"], r["t_res"]
    sums = storm_sums(df, y_res, t_res)
    rows = []
    for cls in STORM_CLASSES:
        mask = (df["storm_class"] == cls).fillna(False).to_numpy(bool)
        n_storms = int((sums["storm_class"] == cls).sum())
        if not mask.any():
            continue
        w = df["dt_days"].to_numpy()[mask]
        if n_storms >= 3:
            fit = dml.wls_cluster(t_res[mask, None], y_res[mask], w, df["cluster"].to_numpy()[mask], n_boot, seed)
            coef, lo, hi = fit["coef"], fit["ci_lo"], fit["ci_hi"]
        else:
            # one or two storms: the slope only (a clustered error needs more clusters)
            coef = (w * t_res[mask] * y_res[mask]).sum() / (w * t_res[mask] ** 2).sum()
            lo = hi = np.nan
        rows.append({"storm_class": cls, "n_storms": n_storms, "n_rows": int(mask.sum()),
                     "coef": coef, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows).set_index("storm_class")
    out.attrs.update(model_y=r["model_y"], model_t=r["model_t"])
    return out, sums


def dml_big_vs_rest(sums, big=BIG):
    """DML slope of the big storms minus the rest, exact permutation p over storm splits."""
    a, b = sums["a"].to_numpy(), sums["b"].to_numpy()
    is_big = sums["storm_class"].isin(big).to_numpy()
    k, n = int(is_big.sum()), len(a)

    def gap(sel):
        return a[sel].sum() / b[sel].sum() - a[~sel].sum() / b[~sel].sum()

    observed = gap(is_big)
    gaps = []
    for c in combinations(range(n), k):
        sel = np.zeros(n, dtype=bool)
        sel[list(c)] = True
        gaps.append(gap(sel))
    p = float(np.mean(np.abs(gaps) >= abs(observed) - 1e-12))
    return {"gap": observed, "coef_big": a[is_big].sum() / b[is_big].sum(),
            "coef_rest": a[~is_big].sum() / b[~is_big].sum(), "n_big": k, "n_rest": n - k,
            "n_permutations": comb(n, k), "p_exact": p}


def run(table, storms, outcome="Y_residual", out_dir=OUT_DIR, n_boot=999, seed=0):
    per = storm_class_error(table, storms, outcome)
    by_class, sums = dml_by_class(dml.prepare(table, outcome), outcome, n_boot=n_boot, seed=seed)
    out = {"per_storm": per, "classes": class_table(per), "big_vs_rest": big_vs_rest(per),
           "dml_classes": by_class, "dml_big_vs_rest": dml_big_vs_rest(sums)}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        per.to_csv(out_dir / f"storm_class_per_storm_{outcome}.csv", index=False)
        out["classes"].to_csv(out_dir / f"storm_class_{outcome}.csv")
        pd.DataFrame([out["big_vs_rest"]]).to_csv(out_dir / f"storm_class_big_vs_rest_{outcome}.csv", index=False)
        out["dml_classes"].to_csv(out_dir / f"storm_class_dml_{outcome}.csv")
        pd.DataFrame([out["dml_big_vs_rest"]]).to_csv(out_dir / f"storm_class_dml_big_vs_rest_{outcome}.csv", index=False)
    return out


def summary_text(out, outcome="Y_residual"):
    b = out["big_vs_rest"]
    lines = [f"# {outcome} change by storm class (days 0..+3 minus days -10..-3, m/day)", "",
             "| class | storms | mean | 95% CI over storms | storm values |", "|---|---|---|---|---|"]
    for cls, r in out["classes"].iterrows():
        ci = "n/a (< 3 storms)" if np.isnan(r["ci_lo"]) else f"{r['ci_lo']:+.1f} to {r['ci_hi']:+.1f}"
        lines.append(f"| {cls} | {r['n_storms']} | {r['mean']:+.1f} | {ci} | {r['storm_values']} |")
    lines += ["", f"Severe + extreme ({b['n_big']} storms) minus the rest ({b['n_rest']}): {b['gap']:+.1f} m/day "
                  f"({b['mean_big']:+.1f} vs {b['mean_rest']:+.1f}); exact permutation p = {b['p_exact']:.3f} "
                  f"over {b['n_permutations']:,} splits.", ""]
    d, db = out["dml_classes"], out["dml_big_vs_rest"]
    lines += [f"## DML effect of ap by storm class (m/day per nT; helper models {d.attrs['model_y']} / "
              f"{d.attrs['model_t']})", "", "| class | storms | rows | per nT | 95% CI (storm-clustered) |",
              "|---|---|---|---|---|"]
    for cls, r in d.iterrows():
        ci = "n/a (< 3 storms)" if np.isnan(r["ci_lo"]) else f"{r['ci_lo']:+.3f} to {r['ci_hi']:+.3f}"
        lines.append(f"| {cls} | {int(r['n_storms'])} | {int(r['n_rows']):,} | {r['coef']:+.3f} | {ci} |")
    lines += ["", f"Severe + extreme minus the rest: {db['gap']:+.3f} per nT ({db['coef_big']:+.3f} vs "
                  f"{db['coef_rest']:+.3f}); exact permutation p = {db['p_exact']:.3f} over "
                  f"{db['n_permutations']:,} splits.", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outcome", default="Y_residual")
    args = p.parse_args(argv)
    from src.storms import load_storms
    out = run(load_table(), load_storms(), args.outcome)
    print(summary_text(out, args.outcome))


if __name__ == "__main__":
    main()
