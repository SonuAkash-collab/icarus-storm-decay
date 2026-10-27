"""Stacked event study of residual decay around storm onsets.

One "stack" per storm: every interval whose midpoint lies within -10..+20 days of
that storm's main_onset, with event_day measured from that onset. The same interval
can sit in several stacks (storms are often < 30 days apart), so each stack keeps
its own copy and its own object fixed effects.

Clean controls: inside the stack of storm s, a row is contaminated if it also lies
inside another storm's contamination window. Contaminated rows are dropped, and a
whole storm is dropped if its day 0 is contaminated or too few clean pre or post
days remain. Both counts are returned in the report. Two contamination windows:
- "active" (default): from 3 days before the other storm's onset (the earlier of
  onset_time and main_onset) to 3 days after its recovery_end, the same storm
  windows physics.py uses to pick quiet intervals. A storm can only disturb the
  orbit while it is active, so this is the window where its effect can leak in.
- (lo, hi): lo..hi days around the other storm's main_onset. (-10, 20), the full
  event window, is the strictest choice; with 20 storms in 13 months it leaves
  almost no clean storm, so we report it but do not use it as the default.

Model, fitted by weighted least squares (weights = dt_days) on the stacked rows:

    Y_residual = alpha[object, storm]
                 + sum_k beta_k * 1[floor(event_day) = k]   (k = -10..20, k != ref)
                 + controls (f107, f107_81d, season sin/cos)
                 + error

alpha absorbs each object's own model bias within each stack, and the controls are
the time-varying confounders from src/dag.py (the object-level ones, altitude,
inclination, local time and drag sensitivity, are constant within a stack and so
are absorbed by alpha). beta_k is the residual decay on day k relative to day `ref`
(default -1, the last day before onset). Negative = more decay than the physics
model predicts.

Inference: only ~20 storms, so errors are clustered by storm_id. We report the
cluster-robust CR1 standard error and a wild cluster bootstrap (Webb six-point
weights, which behave better than Rademacher with few clusters), with basic
(reverse-percentile) intervals from the bootstrap draws.
"""

import numpy as np
import pandas as pd

from src.dag import check_adjustment
from src.schema import EVENT_WINDOW, main_sample

CONTROLS = ["f107", "f107_81d", "season_sin", "season_cos"]
# Controls named as schema columns, for the causal-graph check
DAG_CONTROLS = ["f107", "f107_81d", "day_of_year"]
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _days(a, b):
    return (a - b) / pd.Timedelta(days=1)


ACTIVE_MARGIN = pd.Timedelta(days=3)


def contamination_windows(storms, contam="active"):
    """(storm_id, start, end) of each storm's contamination window."""
    if isinstance(contam, str):
        if contam != "active":
            raise ValueError(f"Unknown contamination rule {contam!r}")
        start = storms[["onset_time", "main_onset"]].min(axis=1) if "onset_time" in storms else storms["main_onset"]
        return list(zip(storms["storm_id"], start - ACTIVE_MARGIN, storms["recovery_end"] + ACTIVE_MARGIN))
    lo, hi = contam
    return list(zip(storms["storm_id"], storms["main_onset"] + pd.Timedelta(days=lo),
                    storms["main_onset"] + pd.Timedelta(days=hi)))


def build_stacks(table, storms, window=EVENT_WINDOW, contam="active",
                 min_pre_days=3, min_post_days=5):
    """Stacked rows with clean controls.

    table: analysis table (already restricted to the rows to use, e.g. main_sample);
    storms: catalog with storm_id, main_onset. Returns (stacked, report):
    stacked has the table columns plus stack (storm_id of the stack) and k (whole
    event day); report has one row per storm with rows kept / dropped and whether
    the storm was used.
    """
    lo, hi = window
    windows = contamination_windows(storms, contam)
    onsets = storms.set_index("storm_id")["main_onset"]
    t_mid = table["t_mid"]
    parts, rows = [], []
    for sid, onset in onsets.items():
        day = _days(t_mid, onset)
        inside = (day >= lo) & (day < hi + 1)
        stack = table[inside].assign(stack=sid, k=np.floor(day[inside]).astype(int))
        stack["k"] = stack["k"].clip(upper=hi)

        dirty = pd.Series(False, index=stack.index)
        for other, start, end in windows:
            if other != sid:
                dirty |= (stack["t_mid"] >= start) & (stack["t_mid"] <= end)
        clean = stack[~dirty]

        pre_days = clean.loc[clean["k"] < 0, "k"].nunique()
        post_days = clean.loc[clean["k"] >= 0, "k"].nunique()
        day0 = (clean["k"] == 0).any()
        used = day0 and pre_days >= min_pre_days and post_days >= min_post_days
        rows.append({"storm_id": sid, "main_onset": onset, "rows": len(stack),
                     "rows_contaminated": int(dirty.sum()), "clean_pre_days": pre_days,
                     "clean_post_days": post_days, "day0_clean": day0, "used": used})
        if used:
            parts.append(clean)

    report = pd.DataFrame(rows)
    stacked = pd.concat(parts, ignore_index=True) if parts else table.iloc[:0].assign(stack=0, k=0)
    return stacked, report


def _design(stacked, ref, window):
    """Within-transformed (object x stack) design matrix, outcome and weights."""
    lo, hi = window
    ks = [k for k in range(lo, hi + 1) if k != ref]
    X = pd.DataFrame({f"d{k}": (stacked["k"] == k).astype(float) for k in ks}, index=stacked.index)
    doy = stacked["day_of_year"].to_numpy()
    controls = pd.DataFrame({
        "f107": stacked["f107"], "f107_81d": stacked["f107_81d"],
        "season_sin": np.sin(2 * np.pi * doy / 365.25),
        "season_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=stacked.index)
    X = pd.concat([X, controls], axis=1)
    return X, ks


def _within(values, groups, w):
    """Subtract the weighted group mean (fixed effects) from each column."""
    v = pd.DataFrame(values)
    sw = pd.Series(w).groupby(groups).transform("sum").to_numpy()
    means = (v.mul(w, axis=0)).groupby(groups).transform("sum").to_numpy() / sw[:, None]
    return v.to_numpy() - means


def fit(stacked, outcome="Y_residual", ref=-1, window=EVENT_WINDOW, n_boot=999, seed=0):
    """WLS with object x stack fixed effects; CR1 and wild cluster bootstrap by storm.

    Returns a DataFrame indexed by event day k (ref row = 0 by construction) with
    coef, se_cr1, se_boot, ci_lo, ci_hi (95% basic bootstrap interval), n_rows, n_storms;
    plus the control coefficients in .attrs["controls"] and the bootstrap draws in
    .attrs["draws"] (n_boot x days) for joint tests.
    """
    check_adjustment(DAG_CONTROLS)
    stacked = stacked[stacked[outcome].notna()]
    X, ks = _design(stacked, ref, window)
    groups = stacked["norad_id"].astype(str).to_numpy() + "_" + stacked["stack"].astype(str).to_numpy()
    w = stacked["dt_days"].to_numpy()
    Xw = _within(X.to_numpy(), groups, w)
    yw = _within(stacked[outcome].to_numpy()[:, None], groups, w)[:, 0]

    sw = np.sqrt(w)
    A = Xw * sw[:, None]
    bread = np.linalg.pinv(A.T @ A)
    P = bread @ A.T                       # beta = P @ (sqrt(w) * y)
    beta = P @ (yw * sw)
    resid = (yw - Xw @ beta) * sw
    fitted = A @ beta

    cl = stacked["stack"].to_numpy()
    uniq, cl_idx = np.unique(cl, return_inverse=True)
    G, n, p = len(uniq), len(yw), Xw.shape[1]
    # CR1: sandwich with the usual small-sample factor
    scores = np.zeros((G, p))
    np.add.at(scores, cl_idx, A * resid[:, None])
    cr1 = G / (G - 1) * (n - 1) / (n - p) * bread @ (scores.T @ scores) @ bread
    se_cr1 = np.sqrt(np.diag(cr1))

    # Wild cluster bootstrap: flip whole-storm residual blocks with Webb weights
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, p))
    for b in range(n_boot):
        v = rng.choice(WEBB, size=G)[cl_idx]
        draws[b] = P @ (fitted + resid * v)
    se_boot = draws.std(axis=0, ddof=1)
    lo_q, hi_q = np.percentile(draws - beta, [97.5, 2.5], axis=0)

    nk = len(ks)
    per_k = stacked.groupby("k").agg(n_rows=("k", "size"), n_storms=("stack", "nunique"))
    out = pd.DataFrame({"coef": beta[:nk], "se_cr1": se_cr1[:nk], "se_boot": se_boot[:nk],
                        "ci_lo": beta[:nk] - lo_q[:nk], "ci_hi": beta[:nk] - hi_q[:nk]},
                       index=pd.Index(ks, name="k"))
    out.loc[ref] = [0.0, 0.0, 0.0, 0.0, 0.0]
    out = out.sort_index().join(per_k)
    out.attrs["controls"] = pd.Series(beta[nk:], index=X.columns[nk:])
    out.attrs["draws"] = pd.DataFrame(draws[:, :nk], columns=ks)
    out.attrs["n_clusters"] = G
    return out


def event_study(table, storms, outcome="Y_residual", use_main_sample=True, exclude_storms=(),
                ref=-1, window=EVENT_WINDOW, contam="active", n_boot=999, seed=0, **stack_kw):
    """Full pipeline: sample -> stacks with clean controls -> fit.

    exclude_storms: storm_ids to leave out (e.g. May and Oct 2024 for the robustness
    run). Returns (result, report); see fit and build_stacks.
    """
    if use_main_sample:
        table = main_sample(table)
    storms = storms[~storms["storm_id"].isin(exclude_storms)]
    stacked, report = build_stacks(table, storms, window, contam, **stack_kw)
    if report["used"].sum() < 2:
        raise ValueError(f"Only {report['used'].sum()} clean storms left with contam={contam!r}")
    result = fit(stacked, outcome, ref, window, n_boot, seed)
    result.attrs["report"] = report
    return result, report


def summarise(report):
    """One-line text summary of the clean-control filter for logs and the paper."""
    used = report["used"].sum()
    return (f"{used} of {len(report)} storms used ({len(report) - used} dropped as contaminated); "
            f"{report['rows_contaminated'].sum()} of {report['rows'].sum()} stacked rows "
            f"dropped because they fall inside another storm's window")
