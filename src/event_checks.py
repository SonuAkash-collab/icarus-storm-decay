"""Checks that ship with the event study (src/event_study.py).

1. planted_recovery: on synthetic data with a known effect, the event study must
   find it (bias and interval coverage over several seeds).
2. pretrend_test: days -10..-2 jointly zero. Statistic = root mean square of the
   pre-period coefficients; p-value from the placebo draws (share of fake-storm
   runs with a pre-period at least as rough). A Wald test with the bootstrap
   covariance is also reported, but with ~20 storms its 9x9 covariance is noisy and
   overlapping stacks share rows, so it over-rejects on null synthetic data; the
   placebo p-value is the one to quote.
3. placebo_dates: 500 sets of fake storm onsets on quiet days; the short-run
   effect must centre on zero, and the real one should sit in the tail.
4. with_without: the same estimate with and without the May and October 2024
   storms, so one superstorm cannot carry the result.

Summary number used throughout: the short-run effect, the mean of the event-day
coefficients over days 0..2 (m/day relative to day -1).
"""

import numpy as np
import pandas as pd

from src import event_study as es
from src.schema import EVENT_WINDOW, main_sample
from src.synthetic import make_synthetic, planted_event_profile

SHORT_RUN = (0, 2)
PRE_DAYS = (-10, -2)
KEY_MONTHS = [(2024, 5), (2024, 10)]     # May 2024 superstorm, Oct 2024 severe storm
SENSITIVITY_CONTAM = (-3, 10)            # stricter rule: also the 10 days after another onset


def short_run_effect(result, days=SHORT_RUN):
    """Mean coefficient over event days days[0]..days[1], with bootstrap SE and interval."""
    ks = list(range(days[0], days[1] + 1))
    est = result.loc[ks, "coef"].mean()
    draws = result.attrs["draws"]
    if len(draws) < 2:
        return pd.Series({"est": est, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
    d = draws[ks].mean(axis=1) - est
    return pd.Series({"est": est, "se": d.std(ddof=1),
                      "ci_lo": est - np.percentile(d, 97.5), "ci_hi": est - np.percentile(d, 2.5)})


def pre_rms(coefs, days=PRE_DAYS):
    """Root mean square of the coefficients on days[0]..days[1] (Series indexed by k)."""
    return float(np.sqrt((coefs.loc[days[0]:days[1]] ** 2).mean()))


def pretrend_test(result, placebo=None, days=PRE_DAYS):
    """Joint test that the pre-period coefficients are all zero.

    Returns rms (root mean square of the pre coefficients), p_placebo (share of
    placebo draws with rms at least as large; needs `placebo` from placebo_dates),
    wald and p_wald_boot (bootstrap Wald, see module notes), max_abs and n_days.
    """
    ks = [k for k in range(days[0], days[1] + 1) if k in result.index and result.loc[k, "se_cr1"] > 0]
    rms = pre_rms(result["coef"], days)
    p_placebo = np.nan
    if placebo is not None and len(placebo):
        p_placebo = float((placebo["pre_rms"] >= rms).mean())
    b = result.loc[ks, "coef"].to_numpy()
    draws = result.attrs["draws"][ks].to_numpy()
    centred = draws - draws.mean(axis=0)
    V_inv = np.linalg.pinv(np.cov(centred, rowvar=False))
    wald = float(b @ V_inv @ b)
    # Bootstrap null: each draw minus the estimate has the shape of the sampling noise
    boot = np.einsum("ij,jk,ik->i", draws - b, V_inv, draws - b)
    return pd.Series({"rms": rms, "p_placebo": p_placebo, "wald": wald,
                      "p_wald_boot": float((boot >= wald).mean()),
                      "max_abs": float(np.abs(b).max()), "n_days": len(ks)})


def planted_recovery(seeds=range(5), n_objects=150, n_boot=199, **synthetic_kw):
    """Event study on synthetic data for several seeds, against the planted profile.

    Returns one row per (seed, day) for days 0..2: estimate, truth, error, and
    whether the bootstrap interval covers the truth.
    """
    rows = []
    for seed in seeds:
        table, storms = make_synthetic(n_objects=n_objects, seed=seed, **synthetic_kw)
        res, _ = es.event_study(table, storms, n_boot=n_boot, seed=seed)
        truth = planted_event_profile(table)
        for k in range(SHORT_RUN[0], SHORT_RUN[1] + 1):
            t = truth[k] - truth.get(-1, 0.0)
            rows.append({"seed": seed, "k": k, "coef": res.loc[k, "coef"], "truth": t,
                         "error": res.loc[k, "coef"] - t,
                         "covered": res.loc[k, "ci_lo"] <= t <= res.loc[k, "ci_hi"]})
    return pd.DataFrame(rows)


def quiet_days(table, storms, window=EVENT_WINDOW):
    """Candidate placebo onsets: midnights that are not inside any real storm's
    active window and leave a full event window inside the data."""
    lo, hi = window
    t = table["t_mid"]
    days = pd.date_range((t.min() - pd.Timedelta(days=lo)).ceil("D"),
                         (t.max() - pd.Timedelta(days=hi + 1)).floor("D"), freq="D")
    quiet = pd.Series(True, index=days)
    for _, start, end in es.contamination_windows(storms):
        quiet &= ~((days >= start) & (days <= end))
    return days[quiet.to_numpy()]


def placebo_dates(table, storms, n_draws=500, n_storms=None, seed=0, use_main_sample=True, **fit_kw):
    """Short-run effect for n_draws sets of fake storm onsets on quiet days.

    Each draw puts n_storms (default: as many as real storms) fake onsets on random
    quiet days (no two on the same day) and runs the same event study without the
    bootstrap. Only the real storms remove contaminated rows: fake onsets have no
    effect, so they do not contaminate each other. Returns a DataFrame
    with one row per draw (est = short-run effect, pre_rms, n_storms_used).
    """
    if use_main_sample:
        table = main_sample(table)
    rng = np.random.default_rng(seed)
    candidates = quiet_days(table, storms)
    n_storms = n_storms or len(storms)
    rows = []
    for d in range(n_draws):
        onsets = pd.DatetimeIndex(np.sort(rng.choice(candidates, size=n_storms, replace=False)))
        fake = pd.DataFrame({"storm_id": np.arange(1, n_storms + 1), "main_onset": onsets,
                             "recovery_end": onsets + pd.Timedelta(days=1)})
        stacked, report = es.build_stacks(table, fake, others=storms, self_contam=False)
        if report["used"].sum() < 2:
            continue
        res = es.fit(stacked, n_boot=0, **fit_kw)
        rows.append({"draw": d, "est": short_run_effect(res)["est"], "pre_rms": pre_rms(res["coef"]),
                     "n_storms_used": report["used"].sum()})
    return pd.DataFrame(rows)


def placebo_summary(placebo, real_est):
    """Mean and spread of the placebo estimates and the two-sided placebo p-value of the real one."""
    est = placebo["est"]
    return pd.Series({"n": len(est), "mean": est.mean(), "sd": est.std(ddof=1),
                      "p_real": float((est.abs() >= abs(real_est)).mean())})


def key_storm_ids(storms, months=KEY_MONTHS):
    """storm_ids with main_onset in the given (year, month) pairs (May and Oct 2024)."""
    onset = pd.DatetimeIndex(storms["main_onset"])
    pick = np.zeros(len(storms), dtype=bool)
    for year, month in months:
        pick |= (onset.year == year) & (onset.month == month)
    return storms.loc[pick, "storm_id"].tolist()


def with_without(table, storms, drop=None, outcome="Y_residual", n_boot=999, seed=0):
    """Short-run effect with all storms and without the May / Oct 2024 storms."""
    drop = key_storm_ids(storms) if drop is None else drop
    rows = {}
    for name, excl in [("all storms", []), ("without " + ",".join(map(str, drop)), drop)]:
        res, report = es.event_study(table, storms, outcome, exclude_storms=excl, n_boot=n_boot, seed=seed)
        rows[name] = pd.concat([short_run_effect(res), pd.Series({"storms_used": report["used"].sum()})])
    return pd.DataFrame(rows).T


def contam_sensitivity(table, storms, rules=("active", SENSITIVITY_CONTAM), outcome="Y_residual",
                       n_boot=999, seed=0):
    """Short-run effect and storms used under each contamination rule. (-3, 10) also drops
    rows in the 10 days after another storm's onset, where slow post-storm recovery
    could leak into a neighbour's pre-period."""
    rows = {}
    for rule in rules:
        res, report = es.event_study(table, storms, outcome, contam=rule, n_boot=n_boot, seed=seed)
        rows[str(rule)] = pd.concat([short_run_effect(res), pd.Series({"storms_used": report["used"].sum()})])
    return pd.DataFrame(rows).T


def run_all(table, storms, outcome="Y_residual", n_placebo=500, n_boot=999, seed=0):
    """Every check on one table and outcome; returns a dict of results for the notebook / paper."""
    res, report = es.event_study(table, storms, outcome, n_boot=n_boot, seed=seed)
    sr = short_run_effect(res)
    placebo = placebo_dates(table, storms, n_draws=n_placebo, seed=seed, outcome=outcome)
    return {"result": res, "report": report, "summary": es.summarise(report),
            "short_run": sr, "pretrend": pretrend_test(res, placebo),
            "placebo": placebo, "placebo_summary": placebo_summary(placebo, sr["est"]),
            "with_without": with_without(table, storms, outcome=outcome, n_boot=n_boot, seed=seed),
            "contam_sensitivity": contam_sensitivity(table, storms, outcome=outcome, n_boot=n_boot, seed=seed)}
