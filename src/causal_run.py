"""Every track C result (event study, its checks, offset ratio) on one analysis table.

Run on the real table once it exists (data/processed/analysis_table.parquet and
storms.parquet):

    python -m src.causal_run

or on a synthetic table, to check the pipeline end to end:

    python -m src.causal_run --synthetic --quick

Outcomes (the two estimands of the plan), each run through the event study and its
checks, whichever the table has:
    Y_total     total storm effect: observed minus MSIS 2.1 with the storm's pre-storm
                quiet ap (from B's residual validation)
    Y_residual  model error: observed minus MSIS 2.1 with the actual ap
Each comes with and without the May / Oct 2024 storms and with the (-3, 10)
contamination rule. The offset ratio measures the storm loss on Y_total; without it,
it falls back to Y_residual and says so in the summary, because on real data MSIS
follows most of the storm and that loss can be near zero. A table without active
satellites (the passive cohort before B3) skips the offset ratio and says so.

The high-altitude group (schema.high_alt_sample) gets its own event study as a
negative control: its short-run effect should be about zero.

Writes CSV files and summary.md to the output folder (default results/causal), one
event-study figure per outcome to the figure folder (default results/), and returns
the results as a dict.
"""

import argparse
from pathlib import Path

import pandas as pd

from src import event_checks as ec
from src import event_study as es
from src import offset_ratio as orr
from src.schema import ROOT, high_alt_sample, load_table, sample_report

OUTCOMES = ["Y_total", "Y_residual"]
OFFSET_OUTCOME = "Y_total"
OUT_DIR = ROOT / "results" / "causal"
FIG_DIR = ROOT / "results"
OUTCOME_LABELS = {"Y_total": "Total storm effect (Y_total)",
                  "Y_residual": "MSIS 2.1 model error (Y_residual)"}


def available_outcomes(table, outcomes=OUTCOMES):
    return [o for o in outcomes if o in table and table[o].notna().any()]


def high_alt_control(table, storms, outcome, n_boot=999, seed=0):
    """Short-run effect in the high-altitude group (negative control), or None if it has
    too few rows or storms."""
    high = high_alt_sample(table)
    if high.empty:
        return None
    try:
        res, report = es.event_study(high, storms, outcome, use_main_sample=False, n_boot=n_boot, seed=seed)
    except ValueError:
        return None
    return pd.concat([ec.short_run_effect(res), pd.Series({"storms_used": report["used"].sum(),
                                                           "objects": high["norad_id"].nunique()})])


def offset_part(table, storms, n_boot=999, n_placebo_reps=20, seed=0):
    """Offset ratio with both recovery windows, balance and the placebo spread."""
    outcome = OFFSET_OUTCOME if OFFSET_OUTCOME in available_outcomes(table) else "Y_residual"
    summary, pairs, smd = orr.offset_ratio(table, storms, outcome=outcome, n_boot=n_boot, seed=seed)
    return {"outcome": outcome, "summary": summary, "pairs": pairs, "balance": smd,
            "recovery_sensitivity": orr.recovery_sensitivity(table, storms, outcome=outcome,
                                                             n_boot=n_boot, seed=seed),
            "placebo": orr.placebo_distribution(table, storms, n_reps=n_placebo_reps, seed=seed,
                                                outcome=outcome)}


def run(table, storms, out_dir=OUT_DIR, outcomes=None, n_placebo=500, n_boot=999,
        n_placebo_reps=20, seed=0, fig_dir=FIG_DIR):
    """All track C results on one table; writes them to out_dir and returns a dict."""
    outcomes = available_outcomes(table, outcomes or OUTCOMES)
    if not outcomes:
        raise ValueError(f"The table has none of the outcomes {OUTCOMES}")
    out = {"sample_report": sample_report(table), "event_study": {}, "high_alt": {}}
    for outcome in outcomes:
        out["event_study"][outcome] = ec.run_all(table, storms, outcome, n_placebo, n_boot, seed)
        out["high_alt"][outcome] = high_alt_control(table, storms, outcome, n_boot, seed)
    # The offset ratio needs active satellites (question 3); skip it without them
    has_actives = bool((table["is_active"] == 1).any())
    out["offset"] = offset_part(table, storms, n_boot, n_placebo_reps, seed) if has_actives else None
    write(out, Path(out_dir))
    if fig_dir is not None:
        figures(out, Path(fig_dir))
    return out


def figures(out, fig_dir):
    """event_study_<outcome>.png: coefficients by event day with 95% CIs."""
    import matplotlib.pyplot as plt

    from src.plots import plot_event_study

    fig_dir.mkdir(parents=True, exist_ok=True)
    for outcome, r in out["event_study"].items():
        fig = plot_event_study(r["result"], OUTCOME_LABELS.get(outcome, outcome),
                               out_path=fig_dir / f"event_study_{outcome}.png")
        plt.close(fig)


def write(out, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out["sample_report"].to_csv(out_dir / "sample_report.csv", index=False)
    for outcome, r in out["event_study"].items():
        d = out_dir / outcome
        d.mkdir(exist_ok=True)
        r["result"].to_csv(d / "event_study.csv")
        r["report"].to_csv(d / "clean_controls.csv", index=False)
        r["placebo"].to_csv(d / "placebo_dates.csv", index=False)
        r["with_without"].to_csv(d / "with_without.csv")
        r["contam_sensitivity"].to_csv(d / "contam_sensitivity.csv")
    off = out["offset"]
    if off is None:
        (out_dir / "summary.md").write_text(summary_text(out))
        return
    off["summary"].to_csv(out_dir / "offset_ratio.csv")
    off["pairs"].to_csv(out_dir / "offset_pairs.csv", index=False)
    off["recovery_sensitivity"].to_csv(out_dir / "offset_recovery_sensitivity.csv")
    off["placebo"].to_csv(out_dir / "offset_placebo.csv", index=False)
    (out_dir / "summary.md").write_text(summary_text(out))


def _fmt(row):
    return f"{row['est']:.3f} (95% CI {row['ci_lo']:.3f} to {row['ci_hi']:.3f})"


def summary_text(out):
    """Headline numbers as Markdown, for the paper and the team."""
    lines = ["# Track C results", ""]
    rep = out["sample_report"]
    lines += ["## Sample", "", "| Filter | Rows removed | Share |", "|---|---|---|"]
    lines += [f"| {r.filter} | {r.removed:,} | {r.share:.1%} |" for r in rep.itertuples()]
    for outcome, r in out["event_study"].items():
        sr, ps, pt = r["short_run"], r["placebo_summary"], r["pretrend"]
        lines += ["", f"## Event study: {outcome}", "",
                  f"- {r['summary']}.",
                  f"- Short-run effect (days 0-2, m/day): {_fmt(sr)}.",
                  f"- Placebo dates (main inference): {int(ps['n'])} draws, mean {ps['mean']:.3f}, "
                  f"SD {ps['sd']:.3f}, p = {ps['p_real']:.3f}.",
                  f"- Pre-trend (RMS of days -10..-2 against the placebo draws): p = {pt['p_placebo']:.3f}.",
                  "", "| Run | Short-run effect | 95% CI | Storms used |", "|---|---|---|---|"]
        for name, row in pd.concat([r["with_without"], r["contam_sensitivity"]]).iterrows():
            lines.append(f"| {name} | {row['est']:.3f} | {row['ci_lo']:.3f} to {row['ci_hi']:.3f} "
                         f"| {int(row['storms_used'])} |")
        high = out["high_alt"].get(outcome)
        lines.append("")
        lines.append("- High-altitude negative control: " + (
            f"{_fmt(high)}, {int(high['objects'])} objects, {int(high['storms_used'])} storms."
            if high is not None else "not run (no usable high-altitude rows)."))
    off = out["offset"]
    if off is None:
        lines += ["", "## Offset ratio", "", "Not run: the table has no active satellites."]
        return "\n".join(lines) + "\n"
    s = off["summary"]
    lines += ["", f"## Offset ratio (storm loss on {off['outcome']})", ""]
    if off["outcome"] != OFFSET_OUTCOME:
        lines += [f"**{OFFSET_OUTCOME} is missing, so the loss uses {off['outcome']}. Do not quote these "
                  "numbers for real data.**", ""]
    lines += [f"- {s.attrs['n_storms']} storms, {s.attrs['n_actives']} active satellites, "
              f"{s.attrs['n_pairs']} storm-satellite pairs.",
              f"- Ducking share (headline): {_fmt(s.loc['ducking_share'])}.",
              f"- Recovery share (imprecise): {_fmt(s.loc['recovery_share'])}.",
              f"- Offset ratio: {_fmt(s.loc['offset_ratio'])}.",
              f"- Largest |SMD| after matching: {off['balance'].abs().max():.3f} (bar 0.1).",
              f"- Debris placebo over {len(off['placebo'])} random sets: ducking share mean "
              f"{off['placebo']['ducking_share'].mean():.3f}, SD {off['placebo']['ducking_share'].std():.3f}.",
              "", "| Recovery window | Share | Estimate | 95% CI | Storms |", "|---|---|---|---|---|"]
    for (days, share), row in off["recovery_sensitivity"].iterrows():
        lines.append(f"| {days} days | {share} | {row['est']:.3f} | {row['ci_lo']:.3f} to {row['ci_hi']:.3f} "
                     f"| {int(row['n_storms'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--table", help="analysis table (default data/processed/analysis_table.parquet)")
    p.add_argument("--storms", help="storm catalog (default data/processed/storms.parquet)")
    p.add_argument("--synthetic", action="store_true", help="run on make_synthetic instead")
    p.add_argument("--quick", action="store_true", help="few draws, for a smoke test")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--figures", default=str(FIG_DIR))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.synthetic:
        from src.synthetic import make_synthetic
        table, storms = make_synthetic(n_objects=400, active_share=0.25, noise=1.0, n_high_alt=15, seed=args.seed)
    else:
        from src.storms import load_storms
        table, storms = load_table(args.table), load_storms(args.storms)
    draws = dict(n_placebo=20, n_boot=49, n_placebo_reps=4) if args.quick else {}
    out = run(table, storms, args.out, seed=args.seed, fig_dir=args.figures, **draws)
    print(summary_text(out))


if __name__ == "__main__":
    main()
