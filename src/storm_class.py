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

    python -m src.storm_class      # real table; writes results/dml/storm_class_*.csv
"""

import argparse
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

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


def run(table, storms, outcome="Y_residual", out_dir=OUT_DIR):
    per = storm_class_error(table, storms, outcome)
    out = {"per_storm": per, "classes": class_table(per), "big_vs_rest": big_vs_rest(per)}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        per.to_csv(out_dir / f"storm_class_per_storm_{outcome}.csv", index=False)
        out["classes"].to_csv(out_dir / f"storm_class_{outcome}.csv")
        pd.DataFrame([out["big_vs_rest"]]).to_csv(out_dir / f"storm_class_big_vs_rest_{outcome}.csv", index=False)
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
