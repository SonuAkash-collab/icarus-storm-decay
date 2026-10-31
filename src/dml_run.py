"""Track D real run: double ML (src/dml.py) on one analysis table, for both estimands.

Run on the real table (data/processed/analysis_table.parquet and spaceweather.parquet):

    python -m src.dml_run

or on a synthetic table, to check the pipeline end to end:

    python -m src.dml_run --synthetic --quick

For each outcome the table has (the two estimands of the plan):
    Y_residual  model error: observed minus MSIS 2.1 with the actual ap
    Y_total     total storm effect: observed minus MSIS 2.1 with the storm's quiet ap
it reports
- naive, linear-control and DML estimates side by side (dml.compare), DML also under
  each of the three ap models, and the helper models picked with their held-out R^2;
- the ap lag sweep (0 / 6 / 12 / 24 h; the main lag, 0 h, was fixed in advance), built
  from the hourly space weather.
Effects are in m/day per nT of interval-mean ap; summary.md also gives them per 100 nT
(roughly a strong storm's mean ap over a day).

Writes CSV files and summary.md to the output folder (default results/dml, git-ignored
like results/causal) and returns the results as a dict.
"""

import argparse
from pathlib import Path

import pandas as pd

from src import dml
from src.schema import ROOT, load_table, sample_report

OUTCOMES = ["Y_residual", "Y_total"]
OUT_DIR = ROOT / "results" / "dml"
METHODS = ["naive", "linear", "dml"]


def available_outcomes(table, outcomes=OUTCOMES):
    return [o for o in outcomes if o in table and table[o].notna().any()]


def selection_table(selection, model_y, model_t):
    """Held-out MSE and R^2 of every helper-model candidate, and which one was picked."""
    rows = []
    for part, chosen in [("y", model_y), ("t", model_t)]:
        for name, s in selection[part].items():
            rows.append({"helper": "residual" if part == "y" else "ap", "candidate": name,
                         "mse": s["mse"], "r2": s["r2"], "chosen": name == chosen})
    return pd.DataFrame(rows)


def run(table, hourly, out_dir=OUT_DIR, outcomes=None, n_boot=999, seed=0):
    """DML results for each outcome; written to out_dir unless it is None."""
    out = {"sample_report": sample_report(table), "outcomes": {}}
    for y in outcomes or available_outcomes(table):
        res = dml.compare(table, y, n_boot=n_boot, seed=seed)
        lags = dml.lag_sweep(dml.prepare(table, y), hourly, y, n_boot=n_boot, seed=seed)
        out["outcomes"][y] = {
            "compare": res, "lags": lags,
            "selection": selection_table(res.attrs["selection"], res.attrs["model_y"], res.attrs["model_t"])}
    if out_dir is not None:
        write(out, Path(out_dir))
    return out


def write(out, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out["sample_report"].to_csv(out_dir / "sample_report.csv", index=False)
    for y, r in out["outcomes"].items():
        r["compare"].to_csv(out_dir / f"compare_{y}.csv")
        r["lags"].to_csv(out_dir / f"lag_sweep_{y}.csv")
        r["selection"].to_csv(out_dir / f"selection_{y}.csv", index=False)
    (out_dir / "summary.md").write_text(summary_text(out), encoding="utf-8")


def _fmt(coef, lo, hi, scale=1.0):
    return f"{scale * coef:+.4f} (CI {scale * lo:+.4f} to {scale * hi:+.4f})"


def summary_text(out):
    lines = ["# DML results", "",
             "Effect of interval-mean ap on the outcome, m/day per nT (x100 = per 100 nT).", ""]
    for y, r in out["outcomes"].items():
        res, a = r["compare"], r["compare"].attrs
        lines += [f"## {y}", "",
                  f"Rows {a['n_rows']:,}, clusters {int(res.loc['dml', 'n_clusters'])}; "
                  f"helper models: residual = {a['model_y']}, ap = {a['model_t']}.", "",
                  "| method | per nT | per 100 nT |", "|---|---|---|"]
        for m in res.index:
            row = res.loc[m]
            lines.append(f"| {m} | {_fmt(row.coef, row.ci_lo, row.ci_hi)} | "
                         f"{_fmt(row.coef, row.ci_lo, row.ci_hi, 100)} |")
        lines += ["", f"Share of the naive slope removed by adjustment: {a['adjustment_share']:.2f}", "",
                  "Held-out R^2 of the helper models:", ""]
        for _, s in r["selection"].iterrows():
            lines.append(f"- {s.helper} / {s.candidate}: {s.r2:+.3f}{'  (chosen)' if s.chosen else ''}")
        lags = r["lags"]
        lines += ["", f"ap lag sweep (same {int(lags['n_rows'].iloc[0]):,} rows at every lag, "
                      f"{int(lags['n_dropped'].iloc[0])} dropped; main lag {dml.MAIN_LAG_HOURS} h):", "",
                  "| lag (h) | per nT |", "|---|---|"]
        for lag, row in lags.iterrows():
            lines.append(f"| {lag} | {_fmt(row.coef, row.ci_lo, row.ci_hi)} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--table", help="analysis table (default data/processed/analysis_table.parquet)")
    p.add_argument("--spaceweather", help="hourly space weather (default data/processed/spaceweather.parquet)")
    p.add_argument("--synthetic", action="store_true", help="run on make_synthetic instead")
    p.add_argument("--quick", action="store_true", help="few bootstrap draws, for a smoke test")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.synthetic:
        from src.synthetic import make_synthetic
        table, _, hourly = make_synthetic(seed=args.seed, return_hourly=True)
    else:
        from src.omni_io import load_spaceweather
        table, hourly = load_table(args.table), load_spaceweather(args.spaceweather)
    out = run(table, hourly, args.out, n_boot=49 if args.quick else 999, seed=args.seed)
    print(summary_text(out))


if __name__ == "__main__":
    main()
