"""Operator offset ratio: how much of a storm's orbit loss active satellites undo.

For every storm and every active satellite:
1. Twins: the 1-3 passive objects closest on altitude, inclination and log drag
   sensitivity (each standardised by its spread in the passive pool, measured over
   the pre-storm window), within a caliper. Twins come from schema.main_sample.
2. Windows, relative to the storm:
       pre       days -10..-3 before main_onset (baseline)
       ducking   main_onset -> recovery_end (operators turn edge-on)
       recovery  recovery_end -> recovery_end + 10 days (operators climb back)
   Rate in a window = time-weighted mean Y_residual, sum(Y * dt) / sum(dt) (m/day).
3. Difference in differences, which removes each active satellite's own routine
   station keeping and any fixed difference to its twins:
       gap(window) = [active(window) - active(pre)] - [twins(window) - twins(pre)]
   Storm loss the active would have had without an operator = what its twins lost:
       loss = [twins(ducking) - twins(pre)] * ducking length       (m, negative)
       duck_gain = gap(ducking) * ducking length                   (m)
       recovery_gain = gap(recovery) * recovery length             (m)
4. Shares, as ratios of sums over all (storm, active) pairs so big storms count more:
       ducking share  = sum(duck_gain) / -sum(loss)
       recovery share = sum(recovery_gain) / -sum(loss)
       offset ratio   = ducking share + recovery share
   0 = operators undo nothing, 1 = they undo the whole storm loss.
Intervals: bootstrap over storms (resample whole storms with replacement).
Checks: covariate balance after matching (standardised mean difference < 0.1) and
the debris-vs-debris placebo (passive objects posing as actives give ~0).

Two things are removed before the windows are formed:
- Neighbouring storms: rows inside another storm's active window or its recovery
  window (3 days before its onset to RECOVERY_DAYS after its recovery_end) are left
  out, so another storm's loss or the operators' climb after it does not leak in.
- Sun UV: the twins' loss compares the storm days with the pre-storm days, and
  F10.7 changes between them (storms come with high F10.7, the confounding of the
  causal graph). adjust_controls removes the F10.7, F10.7 81-day and season part
  of Y_residual, fitted on quiet rows of passive objects with object fixed effects.
  The active-minus-twin gap does not need it (same days on both sides).

Outcome: Y_residual (model error) by default. On real data MSIS run with the actual
ap already captures much of the storm, so the twins' loss in Y_residual can be near
zero or positive. The real run therefore passes outcome= the total-effect column
(observed minus the model run with pre-storm quiet ap), so the loss is the storm's
whole orbit loss. Every function below reads the column named Y_residual;
offset_ratio and debris_placebo copy `outcome` into it first.

Precision: the ducking share is the headline. The recovery share has a long window
and few clean storms, so it is reported as imprecise, with a 5-day recovery window
as a sensitivity run (recovery_sensitivity).
"""

import numpy as np
import pandas as pd

from src.event_study import ACTIVE_MARGIN
from src.schema import MAX_ECC, main_sample

PRE = (-10, -3)               # days before main_onset
RECOVERY_DAYS = 10
SENSITIVITY_RECOVERY_DAYS = 5
MATCH_VARS = ["alt_km", "inc_deg", "log_drag"]
N_TWINS = 3
CALIPER = 0.5                 # max standardised distance to a twin


def contamination_windows(storms, recovery_days=RECOVERY_DAYS):
    """(storm_id, start, end): a storm's active window plus the operators' recovery window."""
    start = storms[["onset_time", "main_onset"]].min(axis=1) if "onset_time" in storms else storms["main_onset"]
    end = storms["recovery_end"] + pd.Timedelta(days=recovery_days) + ACTIVE_MARGIN
    return list(zip(storms["storm_id"], start - ACTIVE_MARGIN, end))


def season(doy):
    return np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)


def adjust_controls(table, storms):
    """Y_residual minus its F10.7 / F10.7 81-day / season part.

    Coefficients from a within-object regression on quiet rows (outside every storm's
    contamination window) of passive objects in the main sample. Returns
    (adjusted table, coefficients).
    """
    def design(df):
        s, c = season(df["day_of_year"].to_numpy())
        return np.column_stack([df["f107"], df["f107_81d"], s, c])

    quiet = main_sample(table)
    for _, start, end in contamination_windows(storms):
        quiet = quiet[~((quiet["t_mid"] >= start) & (quiet["t_mid"] <= end))]
    g = quiet["norad_id"].to_numpy()
    X = design(quiet)
    X = X - pd.DataFrame(X).groupby(g).transform("mean").to_numpy()
    y = quiet["Y_residual"].to_numpy()
    y = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    adjusted = table.assign(Y_residual=table["Y_residual"] - design(table) @ coef)
    return adjusted, pd.Series(coef, index=["f107", "f107_81d", "season_sin", "season_cos"])


def _clean_actives(table, max_ecc=MAX_ECC):
    """Active satellites that can be matched: near-circular, clean TLEs, known drag."""
    return table[(table["is_active"] == 1) & (table["ecc"] < max_ecc) & (table["tle_ok"] == 1)
                 & table["drag_sensitivity"].notna()]


def windows_for(storm, recovery_days=RECOVERY_DAYS):
    """(name, start, end) of the three windows of one storm (dict with main_onset, recovery_end)."""
    onset, rec = storm["main_onset"], storm["recovery_end"]
    return [("pre", onset + pd.Timedelta(days=PRE[0]), onset + pd.Timedelta(days=PRE[1])),
            ("ducking", onset, rec),
            ("recovery", rec, rec + pd.Timedelta(days=recovery_days))]


def window_rates(rows, storm, min_cover=0.5, recovery_days=RECOVERY_DAYS):
    """Per object: time-weighted mean Y_residual and covered days in each window.
    Objects covering less than min_cover of any window are left out."""
    out = []
    windows = windows_for(storm, recovery_days)
    for name, start, end in windows:
        w = rows[(rows["t_mid"] >= start) & (rows["t_mid"] < end)]
        g = w.assign(yw=w["Y_residual"] * w["dt_days"]).groupby("norad_id")
        out.append(pd.DataFrame({f"{name}": g["yw"].sum() / g["dt_days"].sum(),
                                 f"{name}_days": g["dt_days"].sum()}))
    rates = pd.concat(out, axis=1)
    lengths = {name: (end - start) / pd.Timedelta(days=1) for name, start, end in windows}
    ok = np.all([rates[f"{n}_days"] >= min_cover * lengths[n] for n in lengths], axis=0)
    return rates[ok], lengths


def object_features(rows, storm):
    """Matching variables per object, from its pre-storm window."""
    _, start, end = windows_for(storm)[0]
    w = rows[(rows["t_mid"] >= start) & (rows["t_mid"] < end)]
    f = w.groupby("norad_id")[["alt_km", "inc_deg", "drag_sensitivity"]].mean()
    f["log_drag"] = np.log(f.pop("drag_sensitivity"))
    return f


def match(actives, pool, n_twins=N_TWINS, caliper=CALIPER):
    """Nearest passive twins for each active. actives, pool: feature frames (index norad_id).

    Distance = Euclidean on the features divided by their passive-pool SD.
    Returns a long frame (active, twin, distance).
    """
    scale = pool[MATCH_VARS].std().replace(0, 1)
    A = (actives[MATCH_VARS] / scale).to_numpy()
    P = (pool[MATCH_VARS] / scale).to_numpy()
    rows = []
    for i, a in enumerate(A):
        d = np.sqrt(((P - a) ** 2).sum(axis=1))
        for j in np.argsort(d)[:n_twins]:
            if d[j] <= caliper:
                rows.append({"active": actives.index[i], "twin": pool.index[j], "distance": d[j]})
    return pd.DataFrame(rows, columns=["active", "twin", "distance"])


def matched_features(features_a, features_p, pairs):
    """Per active: its matching variables and the mean of its twins' (for balance)."""
    twin_mean = features_p.loc[pairs["twin"]].assign(active=pairs["active"].to_numpy()).groupby("active").mean()
    act = features_a.loc[twin_mean.index, MATCH_VARS]
    return pd.concat([act.add_suffix("_active"), twin_mean[MATCH_VARS].add_suffix("_twins")], axis=1)


def balance(matched):
    """Standardised mean difference per matching variable over all (storm, active) pairs:
    mean(active - twin mean) / SD of the variable among actives and twins together.
    The plan's bar is |SMD| < 0.1."""
    rows = {}
    for v in MATCH_VARS:
        a, t = matched[f"{v}_active"], matched[f"{v}_twins"]
        sd = np.sqrt((a.var() + t.var()) / 2)
        rows[v] = (a - t).mean() / sd if sd > 0 else 0.0
    return pd.Series(rows, name="smd")


def storm_pairs(table, storms, actives=None, passives=None, n_twins=N_TWINS, caliper=CALIPER,
                min_cover=0.5, recovery_days=RECOVERY_DAYS):
    """One row per (storm, active) with loss, duck_gain, recovery_gain (m) and the twins used.

    actives / passives: tables of candidate rows (default: clean active satellites and
    schema.main_sample), already adjusted with adjust_controls. Returns
    (pairs, matched) where matched holds each pair's matching variables (see balance).
    """
    actives = _clean_actives(table) if actives is None else actives
    passives = main_sample(table) if passives is None else passives
    win = contamination_windows(storms, recovery_days)
    out, bal = [], []
    for s in storms.to_dict("records"):
        def clean(rows):
            dirty = pd.Series(False, index=rows.index)
            for other, start, end in win:
                if other != s["storm_id"]:
                    dirty |= (rows["t_mid"] >= start) & (rows["t_mid"] <= end)
            return rows[~dirty]
        a_rows, p_rows = clean(actives), clean(passives)
        a_rates, lengths = window_rates(a_rows, s, min_cover, recovery_days)
        p_rates, _ = window_rates(p_rows, s, min_cover, recovery_days)
        fa = object_features(a_rows, s).reindex(a_rates.index).dropna()
        fp = object_features(p_rows, s).reindex(p_rates.index).dropna()
        if fa.empty or len(fp) < n_twins:
            continue
        pairs = match(fa, fp, n_twins, caliper)
        if pairs.empty:
            continue
        twins = p_rates.loc[pairs["twin"]].assign(active=pairs["active"].to_numpy()).groupby("active").mean()
        act = a_rates.loc[twins.index]
        gap = {w: (act[w] - act["pre"]) - (twins[w] - twins["pre"]) for w in ("ducking", "recovery")}
        out.append(pd.DataFrame({
            "storm_id": s["storm_id"], "active": twins.index,
            "n_twins": pairs.groupby("active").size().reindex(twins.index).to_numpy(),
            "loss": (twins["ducking"] - twins["pre"]) * lengths["ducking"],
            "duck_gain": gap["ducking"] * lengths["ducking"],
            "recovery_gain": gap["recovery"] * lengths["recovery"],
        }))
        bal.append(matched_features(fa, fp, pairs).assign(storm_id=s["storm_id"]))
    pairs = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["storm_id", "active", "n_twins", "loss", "duck_gain", "recovery_gain"])
    return pairs, (pd.concat(bal) if bal else pd.DataFrame())


def shares(pairs):
    """Ducking share, recovery share and total offset ratio (ratios of sums)."""
    lost = -pairs["loss"].sum()
    duck = pairs["duck_gain"].sum() / lost
    rec = pairs["recovery_gain"].sum() / lost
    return pd.Series({"ducking_share": duck, "recovery_share": rec, "offset_ratio": duck + rec})


def bootstrap(pairs, n_boot=999, seed=0):
    """Shares with 95% percentile intervals from resampling whole storms."""
    rng = np.random.default_rng(seed)
    by_storm = {sid: g for sid, g in pairs.groupby("storm_id")}
    ids = list(by_storm)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        draws.append(shares(pd.concat([by_storm[i] for i in pick])))
    draws = pd.DataFrame(draws)
    est = shares(pairs)
    return pd.DataFrame({"est": est, "ci_lo": draws.quantile(0.025), "ci_hi": draws.quantile(0.975),
                         "se": draws.std(ddof=1)})


def offset_ratio(table, storms, outcome="Y_residual", n_boot=999, seed=0, **kw):
    """Full pipeline on the analysis table. outcome: column measuring the orbit loss
    (see the module notes). Returns (summary, pairs, smd) where smd is the balance of
    the matching variables (see balance)."""
    table, coef = adjust_controls(table.assign(Y_residual=table[outcome]), storms)
    pairs, matched = storm_pairs(table, storms, **kw)
    if pairs["storm_id"].nunique() < 2:
        raise ValueError("Fewer than 2 storms with matched pairs")
    summary = bootstrap(pairs, n_boot, seed)
    summary.attrs["n_pairs"] = len(pairs)
    summary.attrs["n_storms"] = pairs["storm_id"].nunique()
    summary.attrs["n_actives"] = pairs["active"].nunique()
    summary.attrs["controls"] = coef
    summary.attrs["outcome"] = outcome
    return summary, pairs, balance(matched)


def recovery_sensitivity(table, storms, days=(RECOVERY_DAYS, SENSITIVITY_RECOVERY_DAYS), n_boot=999,
                         seed=0, **kw):
    """Shares with each recovery window length, stacked (index: recovery_days, share).
    A shorter window keeps more clean storms but only counts the climb-back in it."""
    out = {}
    for d in days:
        summary, _, _ = offset_ratio(table, storms, n_boot=n_boot, seed=seed, recovery_days=d, **kw)
        out[d] = summary.assign(n_storms=summary.attrs["n_storms"])
    return pd.concat(out, names=["recovery_days", "share"])


def placebo_distribution(table, storms, n_reps=20, seed=0, **kw):
    """Debris placebo repeated with n_reps different random sets of fake actives.

    One placebo run has only a handful of large storms behind it, and fake actives
    share twins, so its storm bootstrap interval is too narrow. The spread across
    repetitions is the honest yardstick: the placebo mean should be ~0, and the
    real offset ratio should sit well outside the placebo spread.
    Returns one row per repetition with the three shares.
    """
    rows = []
    for r in range(n_reps):
        summary, _, _ = debris_placebo(table, storms, n_boot=0, seed=seed + r, **kw)
        rows.append(summary["est"].rename(r))
    return pd.DataFrame(rows)


def debris_placebo(table, storms, n_fake=None, outcome="Y_residual", n_boot=199, seed=0, **kw):
    """Debris-vs-debris placebo: random passive objects pose as actives, matched to the
    other passives. They cannot manoeuvre, so the offset ratio should be ~0."""
    rng = np.random.default_rng(seed)
    table, _ = adjust_controls(table.assign(Y_residual=table[outcome]), storms)
    passive = main_sample(table)
    ids = passive["norad_id"].unique()
    n_fake = n_fake or max(10, len(ids) // 5)
    fake_ids = rng.choice(ids, size=n_fake, replace=False)
    fake = passive[passive["norad_id"].isin(fake_ids)]
    pool = passive[~passive["norad_id"].isin(fake_ids)]
    pairs, matched = storm_pairs(table, storms, actives=fake, passives=pool, **kw)
    summary = bootstrap(pairs, n_boot, seed) if n_boot else shares(pairs).rename("est").to_frame()
    summary.attrs["n_pairs"] = len(pairs)
    return summary, pairs, balance(matched)
