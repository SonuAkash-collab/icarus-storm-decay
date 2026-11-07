"""Model-based operator offset ratio (question 3) without passive twins.

The twin design (src/offset_ratio.py) cannot run on real data: no passive object flies
near the Starlink shells. Here each active satellite is its own control, and the physics
model says how much altitude the storm should have cost it.

For every active satellite (rows not flagged free_drag) and every storm, two windows:
    ducking   main_onset -> recovery_end
    recovery  recovery_end -> recovery_end + RECOVERY_DAYS
and the pre-storm baseline days -10..-3 (PRE).

    predicted storm loss P_w = sum over the window of (pred - pred_quiet) * dt   (m, < 0)
        pred = MSIS 2.1 with actual ap, pred_quiet = MSIS 2.1 with the storm's quiet ap,
        both with the satellite's catalog-scaled B; pred - pred_quiet = Y_total - Y_residual.
    observed storm loss  O_w = sum over the window of (obs - baseline) * dt      (m)
        baseline = the satellite's dt-weighted mean observed decay over PRE, so routine
        station keeping cancels.
    offset ratio (window w)  = 1 - sum(O_w) / sum(P_w)
    total offset ratio       = 1 - sum(O_duck + O_rec) / sum(P_duck + P_rec)
Sums run over all (storm, satellite) pairs, so big storms count more. 0 = the satellite
lost what the model says the storm costs (no operator response), 1 = it lost nothing.
The recovery-window ratio divides by a small P (the storm is mostly over), so it is
noisy; duck_part / rec_part split the total additively instead:
    part_w = sum(P_w - O_w) / sum(P_duck + P_rec),   total = duck_part + rec_part.

All 20 storms are used (no clean-window rule), so windows of nearby storms overlap. Two
rules keep a storm from being scored with another storm's rows (own_rows, default on):
- the ducking and recovery windows end at the next storm's main_onset, so no main phase
  counts twice (without this the May 2 2024 recovery window held most of the May 10
  superstorm a second time);
- the predicted loss of a row uses the quiet ap of the storm it is linked to
  (schema.link_storms: nearest main_onset). Rows in the windows linked to another storm
  (the last days before the next storm) get S = 0: the next storm's quiet ap is the
  wrong counterfactual for them, and these late-recovery days carry little storm loss.
A window shortened to nothing contributes 0; coverage is checked against the shortened
length. The baseline window keeps every row. The storm-clustered bootstrap resamples
whole storms, which is optimistic when storms sit close together.

Quiet ap check: the counterfactual holds ap at the median over days -10..-3, which is
not quiet when an earlier storm falls there. With the hourly space weather, run() adds
a run without the storms whose quiet ap is more than QUIET_AP_RATIO x the median ap of
all storm-free hours (noisy_quiet_storms).
Intervals: wild cluster bootstrap by storm (Webb six-point weights, as in
src/event_study.py) on the ratio: per storm O_s* = R P_s + v_s (O_s - R P_s), with the
same weight v_s for every window of that storm.

Model-error correction: P assumes MSIS 2.1 gets the storm right. The passive cohort
measures its error: delta = mean over passive main-sample objects and storms of
[Y_residual(window) - Y_residual(pre)] (m/day), by storm class and 100 km altitude band
(class only if the band has fewer than MIN_PASSIVE objects). Corrected P_w =
P_w + delta_w * covered days. Negative delta = MSIS under-predicts the storm decay.

Placebo: Starlink rows flagged free_drag (failed / deorbiting, no operator holding
altitude) through the same method should give ~0. Their decay is 0.2-0.5 of the
catalog-B rate, so the placebo rescales B by each satellite's own quiet fit on those
rows (as for passive objects); with the catalog B it is also reported, which shows how
much a wrong B scale moves the ratio. The main estimate depends on the catalog B
(offset = 1 - O / (k P) if the true B is k times it): b_scale_sensitivity.
"""

import numpy as np
import pandas as pd

from src.event_checks import key_storm_ids
from src.event_study import WEBB
from src.schema import main_sample

PRE = (-10, -3)                 # baseline days before main_onset
RECOVERY_DAYS = 10
MIN_COVER = 0.5                 # share of each window an object must cover
ALT_BANDS = (400, 500, 600, 700, 800)
MIN_PASSIVE = 5                 # fewer passive objects in a band -> class-only correction
CHANGE_DAYS = 21                # orbit-change screen: window length ...
MAX_CHANGE_KM = 1.0             # ... and the largest net altitude change still counted as holding
MAX_PLACEBO_B_SCALE = 1.5       # placebo objects decay at most this x the catalog free-drag rate
QUIET_AP_RATIO = 1.5            # quiet ap above this x the storm-free median -> not quiet
WINDOWS = ("duck", "rec")


def windows_for(storm, recovery_days=RECOVERY_DAYS):
    """pre / duck / rec windows (start, end). If the storm has a next_onset (storm_pairs
    sets it), duck and rec end there at the latest."""
    onset, rec = storm["main_onset"], storm["recovery_end"]
    rec_end = rec + pd.Timedelta(days=recovery_days)
    nxt = storm.get("next_onset")
    if nxt is not None and not pd.isna(nxt):
        rec, rec_end = min(rec, nxt), min(rec_end, nxt)
    return {"pre": (onset + pd.Timedelta(days=PRE[0]), onset + pd.Timedelta(days=PRE[1])),
            "duck": (onset, rec),
            "rec": (rec, rec_end)}


def _in(rows, start, end):
    return rows[(rows["t_mid"] >= start) & (rows["t_mid"] < end)]


def alt_band(alt_km, bands=ALT_BANDS):
    """Altitude band label "400-500" etc. (None outside the bands)."""
    alt = np.asarray(alt_km, dtype=float)
    i = np.digitize(alt, bands)
    return np.array([f"{bands[k - 1]}-{bands[k]}" if 0 < k < len(bands) else None for k in i], dtype=object)


def storm_windows(rows, storm, recovery_days=RECOVERY_DAYS, min_cover=MIN_COVER, b_scale=None, own_rows=True):
    """One row per object covering every window: baseline, O_w, P_w and covered days.

    rows: analysis-table rows with obs_decay, Y_total, Y_residual (m/day). b_scale:
    optional Series by norad_id multiplying B (the predicted storm loss scales with B).
    own_rows: rows linked to another storm (storm_id, if the rows have it) get no
    predicted storm loss (S = 0), see the module docstring.
    """
    win = windows_for(storm, recovery_days)
    S = rows["Y_total"] - rows["Y_residual"]
    if own_rows and "storm_id" in rows:
        other = (rows["storm_id"] != storm["storm_id"]).fillna(True).to_numpy(dtype=bool)
        S = S.where(~other, 0.0)
    if b_scale is not None:
        S = S * rows["norad_id"].map(b_scale).fillna(1.0)
    rows = rows.assign(S=S, obs_dt=rows["obs_decay"] * rows["dt_days"], S_dt=S * rows["dt_days"])
    pre = _in(rows, *win["pre"]).groupby("norad_id")
    out = pd.DataFrame({"baseline": pre["obs_dt"].sum() / pre["dt_days"].sum(), "pre_days": pre["dt_days"].sum()})
    ok = out["pre_days"] >= min_cover * (PRE[1] - PRE[0])
    for w in WINDOWS:
        start, end = win[w]
        g = _in(rows, start, end).groupby("norad_id")
        days = g["dt_days"].sum().reindex(out.index).fillna(0)
        out[f"days_{w}"] = days
        out[f"P_{w}"] = g["S_dt"].sum().reindex(out.index).fillna(0.0)
        out[f"O_{w}"] = g["obs_dt"].sum().reindex(out.index).fillna(0.0) - out["baseline"] * days
        ok &= days >= min_cover * (end - start) / pd.Timedelta(days=1)
    out = out[ok].dropna()
    alt = rows.groupby("norad_id")["alt_km"].mean().reindex(out.index)
    return out.assign(storm_id=storm["storm_id"], storm_class=storm["class"], alt_km=alt,
                      band=alt_band(alt)).rename_axis("norad_id").reset_index()


def storm_pairs(rows, storms, recovery_days=RECOVERY_DAYS, min_cover=MIN_COVER, b_scale=None, own_rows=True):
    """storm_windows for every storm, stacked (one row per storm and object). With
    own_rows, each storm's windows end at the next storm's main_onset."""
    storms = storms.sort_values("main_onset")
    if own_rows:
        storms = storms.assign(next_onset=storms["main_onset"].shift(-1))
    parts = [storm_windows(rows, s, recovery_days, min_cover, b_scale, own_rows) for s in storms.to_dict("records")]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def model_error(table, storms, recovery_days=RECOVERY_DAYS, min_cover=MIN_COVER):
    """Passive MSIS 2.1 error by storm class, altitude band and window (m/day).

    For every passive main-sample object and storm: dt-weighted mean Y_residual in the
    window minus in PRE; then the mean over objects and storms. Returns a frame with
    storm_class, band (NaN row = all bands), window, delta, n_objects, n_storms.
    """
    rows = main_sample(table)
    out = []
    for s in storms.to_dict("records"):
        win = windows_for(s, recovery_days)
        r = {}
        for w in ("pre",) + WINDOWS:
            g = _in(rows, *win[w]).assign(yw=lambda d: d["Y_residual"] * d["dt_days"]).groupby("norad_id")
            days = g["dt_days"].sum()
            length = (win[w][1] - win[w][0]) / pd.Timedelta(days=1)
            r[w] = (g["yw"].sum() / days)[days >= min_cover * length]
        both = pd.concat(r, axis=1).dropna()
        if both.empty:
            continue
        alt = rows.groupby("norad_id")["alt_km"].mean().reindex(both.index)
        for w in WINDOWS:
            out.append(pd.DataFrame({"storm_id": s["storm_id"], "storm_class": s["class"], "window": w,
                                     "norad_id": both.index, "band": alt_band(alt),
                                     "d": (both[w] - both["pre"]).to_numpy()}))
    per = pd.concat(out, ignore_index=True)
    by_band = per.groupby(["storm_class", "band", "window"]).agg(
        delta=("d", "mean"), n_objects=("norad_id", "nunique"), n_storms=("storm_id", "nunique")).reset_index()
    by_class = per.groupby(["storm_class", "window"]).agg(
        delta=("d", "mean"), n_objects=("norad_id", "nunique"), n_storms=("storm_id", "nunique")).reset_index()
    return pd.concat([by_band, by_class.assign(band=pd.NA)], ignore_index=True)


def add_correction(pairs, errors, min_passive=MIN_PASSIVE):
    """Columns delta_<w> (m/day, band-specific if it has min_passive objects, else class
    only) and Pc_<w> = P_<w> + delta_<w> * days_<w> (corrected predicted loss)."""
    pairs = pairs.copy()
    for w in WINDOWS:
        e = errors[errors["window"] == w]
        band = e[e["band"].notna() & (e["n_objects"] >= min_passive)].set_index(["storm_class", "band"])["delta"]
        cls = e[e["band"].isna()].set_index("storm_class")["delta"]
        key = pd.MultiIndex.from_arrays([pairs["storm_class"], pairs["band"]])
        d = pd.Series(band.reindex(key).to_numpy(), index=pairs.index)
        pairs[f"delta_{w}"] = d.fillna(pairs["storm_class"].map(cls)).fillna(0.0)
        pairs[f"Pc_{w}"] = pairs[f"P_{w}"] + pairs[f"delta_{w}"] * pairs[f"days_{w}"]
    return pairs


def _stats(O, P):
    """O, P: arrays (n, 2) of observed / predicted loss in the duck and rec windows."""
    so, sp = O.sum(axis=0), P.sum(axis=0)
    ptot = sp.sum()
    return {"offset_duck": 1 - so[0] / sp[0], "offset_rec": 1 - so[1] / sp[1],
            "offset_total": 1 - so.sum() / ptot,
            "duck_part": (sp[0] - so[0]) / ptot, "rec_part": (sp[1] - so[1]) / ptot}


def offset(pairs, pred="P", n_boot=999, seed=0):
    """Offset ratios (est, ci_lo, ci_hi, se) with a storm-clustered wild bootstrap.

    pred: "P" (model) or "Pc" (model-error corrected). The per-storm sums are perturbed
    around each ratio's own fit: O_s* = R P_s + v_s (O_s - R P_s), one Webb weight per
    storm shared by both windows, P fixed.
    """
    per = pairs.groupby("storm_id")[[f"O_{w}" for w in WINDOWS] + [f"{pred}_{w}" for w in WINDOWS]].sum()
    O = per[[f"O_{w}" for w in WINDOWS]].to_numpy()
    P = per[[f"{pred}_{w}" for w in WINDOWS]].to_numpy()
    est = _stats(O, P)
    # Fit per window (R_w = sum O_w / sum P_w): residuals around it are what gets flipped
    R = O.sum(axis=0) / P.sum(axis=0)
    resid = O - P * R
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        v = rng.choice(WEBB, size=len(per))[:, None]
        draws.append(_stats(P * R + resid * v, P))
    out = pd.DataFrame({"est": pd.Series(est)})
    if n_boot > 1:
        d = pd.DataFrame(draws)
        out["ci_lo"], out["ci_hi"], out["se"] = d.quantile(0.025), d.quantile(0.975), d.std(ddof=1)
    out.attrs.update(n_storms=len(per), n_pairs=len(pairs), n_objects=pairs["norad_id"].nunique(),
                     P_total_m=float(P.sum()), O_total_m=float(O.sum()))
    return out


def noisy_quiet_storms(storms, sw, ratio=QUIET_AP_RATIO):
    """Storm ids whose quiet ap (physics.quiet_ap: median over days -10..-3) is more than
    `ratio` x the median hourly ap outside every storm window (physics.storm_windows):
    an earlier storm sits in the quiet window, so the counterfactual is not quiet.
    sw: hourly space weather indexed by time (omni_io.load_spaceweather)."""
    from src.physics import quiet_ap
    from src.physics import storm_windows as storm_spans
    t = sw.index
    calm = np.ones(len(sw), dtype=bool)
    for start, end in storm_spans(storms):
        calm &= ~((t >= start) & (t < end))
    ref = sw.loc[calm, "ap"].median()
    q = quiet_ap(sw, storms)
    return sorted(int(k) for k in q[q > ratio * ref].index)


def orbit_change_flags(rows, days=CHANGE_DAYS, max_change_km=MAX_CHANGE_KM):
    """1 for rows inside a `days`-day window whose net altitude change (sum obs * dt)
    exceeds max_change_km either way: the operator is moving the satellite (orbit raise,
    shell change, deorbit), not holding it. A storm costs a few hundred metres for a few
    days and is climbed back, so it stays below the bar. Same index as rows."""
    flags = pd.Series(0, index=rows.index, dtype="int64")
    span = pd.Timedelta(days=days).to_timedelta64()
    for _, g in rows.sort_values("t_start").groupby("norad_id"):
        t0, t1 = g["t_start"].to_numpy(), g["t_end"].to_numpy()
        a = np.cumsum(np.r_[0, (g["obs_decay"] * g["dt_days"]).to_numpy()]) / 1000
        bad = np.zeros(len(g), dtype=bool)
        for i in range(len(g)):
            j = np.searchsorted(t1, t0[i] + span, side="left")
            if j >= len(g):
                break
            if abs(a[j + 1] - a[i]) > max_change_km:
                bad[i:j + 1] = True
        flags.loc[g.index] = bad.astype("int64")
    return flags


def active_rows(table):
    """Active satellites holding altitude: clean TLEs, near-circular, not flagged free_drag,
    and outside any orbit change (orbit_change_flags)."""
    ok = (table["is_active"] == 1) & (table["tle_ok"] == 1) & (table["ecc"] < 0.01)
    if "free_drag" in table:
        ok &= table["free_drag"] != 1
    rows = table[ok]
    return rows[orbit_change_flags(rows) == 0]


def placebo_rows(table, storms, max_b_scale=MAX_PLACEBO_B_SCALE):
    """Starlink rows flagged free_drag of satellites decaying by drag: no operator holds
    their altitude. Satellites whose own quiet decay is more than max_b_scale x the
    catalog free-drag rate are being thrust down (end-of-life lowering, shell changes)
    and are left out. Returns (rows, own_b_scale of the kept objects)."""
    names = table.get("object_name", pd.Series(pd.NA, index=table.index)).astype("string")
    rows = table[(table["is_active"] == 1) & (table["free_drag"] == 1) & (table["tle_ok"] == 1)
                 & names.str.upper().str.startswith("STARLINK").fillna(False)]
    scale = own_b_scale(rows, storms)
    keep = scale[(scale > 0) & (scale <= max_b_scale)]
    return rows[rows["norad_id"].isin(keep.index)], keep


def own_b_scale(rows, storms):
    """Fitted / catalog B per object from its own quiet rows (outside every storm's
    -3..recovery_end+3 window): sum(obs * dt) / sum(pred * dt), as physics.fit_drag_sensitivity."""
    from src.physics import storm_windows as quiet_cut
    quiet = pd.Series(True, index=rows.index)
    t0 = rows["t_mid"] - pd.to_timedelta(rows["dt_days"] / 2, unit="D")
    t1 = rows["t_mid"] + pd.to_timedelta(rows["dt_days"] / 2, unit="D")
    for start, end in quiet_cut(storms):
        quiet &= ~((t1 > start) & (t0 < end))
    q = rows[quiet]
    obs = (q["obs_decay"] * q["dt_days"]).groupby(q["norad_id"]).sum()
    pred = (q["pred_decay"] * q["dt_days"]).groupby(q["norad_id"]).sum()
    return (obs / pred).rename("b_scale")


def b_scale_sensitivity(pairs, scales=(0.75, 1.25)):
    """Point offsets if the true B were k x the catalog-scaled B (P scales with B)."""
    rows = {}
    for k in scales:
        p = pairs.assign(**{f"P_{w}": pairs[f"P_{w}"] * k for w in WINDOWS})
        rows[k] = pd.Series(_stats(p[[f"O_{w}" for w in WINDOWS]].to_numpy(),
                                   p[[f"P_{w}" for w in WINDOWS]].to_numpy()))
    return pd.DataFrame(rows).rename_axis(columns="b_scale")


def run(table, storms, n_boot=999, seed=0, recovery_days=RECOVERY_DAYS, sw=None):
    """Every model-based offset result. Returns a dict:
    pairs (with the correction columns), errors (model_error), summary (stacked:
    run x statistic with est / ci_lo / ci_hi / se), placebo (own B and catalog B),
    b_sensitivity, placebo_b_scale (own fitted / catalog B of the placebo objects),
    noisy_quiet (storm ids left out of the quiet-ap run; empty without sw).
    Runs: all storms and without May / Oct 2024, uncorrected and corrected; with sw
    (hourly space weather) also without the storms whose quiet ap is not quiet.
    """
    errors = model_error(table, storms, recovery_days)
    pairs = add_correction(storm_pairs(active_rows(table), storms, recovery_days), errors)
    if pairs.empty or pairs["storm_id"].nunique() < 2:
        raise ValueError("Fewer than 2 storms with active satellites covering every window")
    drop = key_storm_ids(storms)
    without = pairs[~pairs["storm_id"].isin(drop)]
    noisy = noisy_quiet_storms(storms, sw) if sw is not None else []
    variants = [("all storms", pairs), (f"without {','.join(map(str, drop))}", without)]
    if noisy:
        variants.append((f"quiet ap (without {','.join(map(str, noisy))})", pairs[~pairs["storm_id"].isin(noisy)]))
    runs = {}
    for name, p in variants:
        for pred, label in (("P", "uncorrected"), ("Pc", "corrected")):
            r = offset(p, pred, n_boot, seed)
            runs[(name, label)] = r.assign(n_storms=r.attrs["n_storms"], n_objects=r.attrs["n_objects"],
                                           n_pairs=r.attrs["n_pairs"])
    summary = pd.concat(runs, names=["storms", "prediction", "statistic"])

    prow, scale = placebo_rows(table, storms)
    placebo = {}
    for label, bs in (("own B", scale), ("catalog B", None)):
        pp = storm_pairs(prow, storms, recovery_days, b_scale=bs)
        if len(pp) and pp["storm_id"].nunique() >= 2:
            pp = add_correction(pp, errors)
            for pred, plabel in (("P", "uncorrected"), ("Pc", "corrected")):
                r = offset(pp, pred, n_boot, seed)
                placebo[(label, plabel)] = r.assign(n_storms=r.attrs["n_storms"], n_objects=r.attrs["n_objects"],
                                                    n_pairs=r.attrs["n_pairs"])
    placebo = pd.concat(placebo, names=["b", "prediction", "statistic"]) if placebo else pd.DataFrame()
    return {"pairs": pairs, "errors": errors, "summary": summary, "placebo": placebo,
            "b_sensitivity": b_scale_sensitivity(pairs), "placebo_b_scale": scale, "noisy_quiet": noisy}
