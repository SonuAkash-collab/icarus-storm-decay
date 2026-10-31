"""Real analysis table: Piyush's cohort intervals -> physics model -> schema.py table.

One command rebuilds everything (the slow physics part is cached in chunks):

    python -m src.build_table

Inputs (all under data/, git-ignored):
    data/processed/orbit_intervals_a2_a4.parquet          decay intervals (A2-A4)
    data/raw/piyush/processed/cohort_tles_cleaned.parquet  cleaned TLEs with line1/line2
    data/processed/spaceweather_msis.parquet               hourly OMNI, padded 3 days
    data/processed/storms.parquet                          storm catalog

Steps:
1. Density drivers per interval for NRLMSISE-00 and MSIS 2.1 (physics.density_drivers)
   and the MSIS 2.1 driver with the storm's quiet ap (physics.quiet_ap_drivers). This is
   the slow part (~45 min for 200k intervals), so it runs CHUNK_OBJECTS objects at a
   time and each chunk is saved to data/processed/physics_chunks/. A crash or Ctrl-C
   loses at most one chunk: rerunning picks up where it stopped.
2. Roles (physics.label_intervals): quiet 'fit' / 'holdout' intervals, 'storm' near storms.
3. Hampel filter on the MSIS 2.1 residual (physics.flag_outliers). Outliers stay in the
   table with tle_ok = 0, so the share removed can be reported.
   Climbs (climb_flag = 1, a passive object's a went up between two TLEs) are KEPT:
   a climb is TLE noise, and dropping only the upward noise makes decay look too fast.
   build_table(drop_climbs=True) gives the version without them, as a check.
4. Drag sensitivity B per object and model on the clean quiet 'fit' intervals
   (physics.fit_drag_sensitivity), with a block bootstrap by calendar week for its
   error bar (drag_bootstrap). Objects with too few fit intervals or a wide error bar
   are flagged drag_unstable = 1 (rule in drag_bootstrap and docs/PLAN.md).
   High-altitude objects (>= 1,000 km): Y_log_ratio = NaN as in src/simulate.py. Their
   quiet fit turned out tight on real data (relative SE ~4%), so they keep it
   (drag_source = "fit"); only an unstable one would fall back to the median B of the
   main cohort (drag_source = "cohort_median", until the B3 catalog exists).
5. Predictions, residuals and the total storm effect (physics.add_predictions,
   physics.add_total_effect), storm link (schema.link_storms), ap integral and F10.7,
   then schema.validate.

6. Active satellites (question 3 only, never pooled with the passive rows): when
   data/processed/orbit_intervals_actives.parquet exists (python -m src.active_cohort)
   and objects.parquet covers them (objects.build_objects), their intervals go through
   the same drivers (cached in data/processed/physics_chunks_actives/) with a catalog
   drag sensitivity, B = 2.2 x A/m scaled by the fitted / catalog ratio of rocket
   bodies at the same altitude (drag_source = "catalog_scaled", catalog_scale), and
   are appended with is_active = 1. Station keeping spoils a quiet fit, and the Hampel
   filter would delete the burns, so neither is used. free_drag_flags marks the rows
   where a satellite keeps losing altitude for 3 weeks (failed, deorbiting or moving
   to a lower shell): free_drag = 1, left out of the offset ratio.

Extra columns beyond schema.py: cohort_group, climb_flag, hampel_outlier, role,
drag_source, drag_unstable, drag_se, obs_decay / pred_decay (m/day, MSIS 2.1),
alt_km_tle (Piyush's a - R_earth; alt_km is the mean geodetic altitude of the SGP4
samples, as in physics.density_drivers); with actives also free_drag, first_flag,
object_name, drag_scale (NaN / 0 for passive rows).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import physics
from src.schema import HIGH_ALT_MIN_KM, ROOT, link_storms, save_table, validate
from src.simulate import ap_integral

PROCESSED_DIR = ROOT / "data" / "processed"
INTERVALS_PATH = PROCESSED_DIR / "orbit_intervals_a2_a4.parquet"
TLES_PATH = ROOT / "data" / "raw" / "piyush" / "processed" / "cohort_tles_cleaned.parquet"
CHUNK_DIR = PROCESSED_DIR / "physics_chunks"
TABLE_PATH = PROCESSED_DIR / "analysis_table.parquet"
CHUNK_OBJECTS = 10              # objects per cached chunk (~7,500 intervals, ~1.5 min)
MAIN_MODEL = "msis21"

# Drag sensitivity bootstrap and the "unstable" rule (fixed before looking at results)
N_BOOT = 500
BOOT_BLOCK = "7D"               # resample whole calendar weeks of fit intervals
MIN_FIT_INTERVALS = 30          # fewer quiet fit intervals than this -> unstable
MAX_REL_SE = 0.25               # bootstrap SE / B above this -> unstable

# Active satellites (src/active_cohort.py, src/objects.py)
ACTIVE_INTERVALS_PATH = PROCESSED_DIR / "orbit_intervals_actives.parquet"
ACTIVE_TLES_PATH = PROCESSED_DIR / "active_tles_cleaned.parquet"
OBJECTS_PATH = PROCESSED_DIR / "objects.parquet"
ACTIVE_CHUNK_DIR = PROCESSED_DIR / "physics_chunks_actives"
ACTIVE_DRAG_NAME = "active_drag.parquet"   # per active object: B, scale, free-drag flag
SCALE_WINDOW_KM = 50            # rocket bodies within +-50 km give the fitted / catalog ratio
SCALE_MIN_BODIES = 5
FREE_DRAG_DAYS = 21             # sustained decay this long -> not station keeping
FREE_DRAG_RATIO = (0.1, 2.0)    # observed / free-drag loss: "drag" in this range, "lowering" above
FREE_DRAG_MIN_LOSS_KM = 1.0     # and the window must lose at least this much altitude


# ---------------------------------------------------------------------------
# 1. Inputs and the cached physics drivers
# ---------------------------------------------------------------------------

def load_inputs(intervals_path=INTERVALS_PATH, tles_path=TLES_PATH):
    """(intervals, tles, hourly space weather padded for MSIS, storms)."""
    from src.storms import load_storms
    intervals = pd.read_parquet(intervals_path)
    tles = pd.read_parquet(tles_path)
    return intervals, tles, physics.load_msis_weather(), load_storms()


def check_no_bridges(intervals, storms):
    """Raise ValueError if any interval spans a storm's main_onset (task A4 rule).

    Never interpolate across an onset: such an interval would average the decay
    before and after day 0. Piyush's build_decay_intervals already omits them.
    """
    bridge = np.zeros(len(intervals), dtype=bool)
    for onset in storms["main_onset"]:
        bridge |= ((intervals["t_start"] < onset) & (intervals["t_end"] > onset)).to_numpy()
    if bridge.any():
        raise ValueError(f"{bridge.sum()} intervals span a main_onset; rebuild them with "
                         "orbit.build_decay_intervals")


def cohort_drivers(tles, intervals, sw, storms, chunk_dir=CHUNK_DIR, chunk_objects=CHUNK_OBJECTS,
                   log=print):
    """Density drivers for every interval, run and cached CHUNK_OBJECTS objects at a time.

    Returns `intervals` (sorted by norad_id, t_start) with the physics columns added:
    alt_km (mean SGP4 sample altitude), driver_nrlmsise00, driver_msis21 (actual ap),
    driver_quiet_msis21 and quiet_ap (storm's quiet ap), storm_id of the quiet-ap link.
    rho_* columns are dropped: air density is the mediator and never goes in the table.
    """
    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    intervals = intervals.sort_values(["norad_id", "t_start"]).reset_index(drop=True)
    ids = np.sort(intervals["norad_id"].unique())
    parts = []
    for k in range(0, len(ids), chunk_objects):
        chunk_ids = ids[k:k + chunk_objects]
        iv = intervals[intervals["norad_id"].isin(chunk_ids)].reset_index(drop=True)
        tl = tles[tles["norad_id"].isin(chunk_ids)]
        name = f"chunk_{k // chunk_objects:03d}"
        done = (chunk_dir / f"{name}_drivers.parquet").exists() and (chunk_dir / f"{name}_quietap.parquet").exists()
        if log:
            log(f"{name}: {len(chunk_ids)} objects, {len(iv):,} intervals" + (" (cached)" if done else ""))
        # Both functions reuse their cache file when it holds exactly these intervals
        drv = physics.density_drivers(tl, iv, sw, cache_path=chunk_dir / f"{name}_drivers.parquet")
        qap = physics.quiet_ap_drivers(tl, iv, sw, storms, cache_path=chunk_dir / f"{name}_quietap.parquet")
        drv = drv.drop(columns=[c for c in drv if c.startswith("rho_")])
        parts.append(drv.assign(quiet_ap=qap["quiet_ap"].to_numpy(),
                                driver_quiet_msis21=qap["driver_quiet_msis21"].to_numpy()))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Drag sensitivity with a block-bootstrap error bar
# ---------------------------------------------------------------------------

def drag_bootstrap(rates, model=MAIN_MODEL, n_boot=N_BOOT, block=BOOT_BLOCK, seed=0,
                   min_fit=MIN_FIT_INTERVALS, max_rel_se=MAX_REL_SE):
    """B per object (physics.fit_drag_sensitivity) with a block bootstrap by week.

    B = sum(observed * dt) / sum(driver * dt) over the 'fit' intervals. The bootstrap
    redraws whole calendar weeks of fit intervals with replacement (neighbouring
    intervals share TLEs, so single rows are not independent) and recomputes B.
    Returns one row per object: B, se, ci_lo, ci_hi (2.5 / 97.5 percentiles), n_fit,
    n_weeks and unstable = 1 if n_fit < min_fit, se / B > max_rel_se or ci_lo <= 0.
    """
    rng = np.random.default_rng(seed)
    B = physics.fit_drag_sensitivity(rates, model)
    fit = rates[rates["role"] == "fit"]
    week = fit["t_mid"].dt.floor(block)
    sums = pd.DataFrame({"obs": fit["da_dt_km_per_day"] * fit["dt_days"],
                         "drv": fit[f"driver_{model}"] * fit["dt_days"]}).groupby([fit["norad_id"], week]).sum()
    rows = []
    for norad_id, g in sums.groupby(level=0):
        n_weeks = len(g)
        # Each bootstrap draw = how many times each week is picked
        counts = rng.multinomial(n_weeks, np.full(n_weeks, 1 / n_weeks), size=n_boot)
        draws = (counts @ g["obs"].to_numpy()) / (counts @ g["drv"].to_numpy())
        rows.append({"norad_id": norad_id, "B": B[norad_id], "se": draws.std(ddof=1),
                     "ci_lo": np.percentile(draws, 2.5), "ci_hi": np.percentile(draws, 97.5),
                     "n_weeks": n_weeks})
    out = pd.DataFrame(rows).set_index("norad_id")
    out["n_fit"] = fit.groupby("norad_id").size().reindex(out.index).fillna(0).astype(int)
    out["unstable"] = ((out["n_fit"] < min_fit) | (out["se"] / out["B"] > max_rel_se)
                       | (out["ci_lo"] <= 0)).astype("int64")
    return out


# ---------------------------------------------------------------------------
# 3. Physics on the cohort: roles, Hampel filter, B, predictions
# ---------------------------------------------------------------------------

def cohort_physics(drivers, storms, drop_climbs=False, seed=0):
    """Roles, outlier flags, drag sensitivity and predictions for the whole cohort.

    drivers: output of cohort_drivers. Returns (rates, drag): rates with role,
    hampel_outlier, B_<model>, pred_/resid_/log_ratio_<model>, pred_quiet_msis21,
    total_msis21; drag = drag_bootstrap of MSIS 2.1 (index norad_id) plus drag_source.
    Hampel outliers get no say in B but keep their predictions.
    """
    rates = drivers if not drop_climbs else drivers[drivers["climb_flag"] == 0]
    rates = physics.label_intervals(rates.reset_index(drop=True), storms)
    high_ids = rates.loc[rates["cohort_group"] == "high_altitude", "norad_id"].unique()

    # First pass B on everything, then the Hampel filter on the MSIS 2.1 residual
    rates = physics.flag_outliers(physics.add_predictions(rates, MAIN_MODEL), MAIN_MODEL)
    rates = rates.rename(columns={"outlier": "hampel_outlier"})
    clean = rates[~rates["hampel_outlier"]]

    drag = drag_bootstrap(clean, MAIN_MODEL, seed=seed)
    # High-altitude objects fall back to the cohort median only when their fit is unstable
    # (on real data their fits are tight, unlike in src/simulate.py's noisier TLEs)
    replace = drag.index.isin(high_ids) & (drag["unstable"] == 1)
    drag["drag_source"] = np.where(replace, "cohort_median", "fit")
    fallback = drag.loc[~drag.index.isin(high_ids), "B"].median()
    for model in physics.MODELS:
        B = physics.fit_drag_sensitivity(clean, model)
        B[B.index.isin(drag.index[replace])] = fallback
        rates = physics.add_predictions(rates, model, B)
    drag.loc[replace, ["B", "se", "ci_lo", "ci_hi"]] = [fallback, np.nan, np.nan, np.nan]
    rates = physics.add_total_effect(rates, MAIN_MODEL)
    return rates, drag


# ---------------------------------------------------------------------------
# 4. Gate: held-out quiet residual ~0
# ---------------------------------------------------------------------------

def holdout_gate(rates, model=MAIN_MODEL, groups=("main",)):
    """Held-out quiet residual (m/day), pooled and per object, Hampel outliers removed.

    Returns (pooled, per_object). pooled: dt-weighted mean, SE from weekly blocks
    (physics.block_se, all objects of a week together: cluster by time block) and t.
    per_object: physics.holdout_residual in m/day with t = mean / weekly-block SE.
    """
    h = rates[(rates["role"] == "holdout") & ~rates["hampel_outlier"] & rates["cohort_group"].isin(groups)]
    r = h[f"resid_{model}"] * 1000
    w = h["dt_days"]
    pooled_mean = (r * w).sum() / w.sum()
    se = physics.block_se(r, h["t_mid"])
    pooled = pd.Series({"mean": pooled_mean, "se_block": se, "t": pooled_mean / se,
                        "unweighted_mean": r.mean(), "intervals": len(h), "objects": h["norad_id"].nunique(),
                        "mean_obs": (h["da_dt_km_per_day"] * 1000 * w).sum() / w.sum()})
    per = physics.holdout_residual(h, model)
    per[["mean_resid", "se_naive", "se_block"]] *= 1000
    per["t"] = per["mean_resid"] / per["se_block"]
    return pooled, per


# ---------------------------------------------------------------------------
# 5. The schema table
# ---------------------------------------------------------------------------

def assemble(rates, drag, sw, storms, tles, is_active=0):
    """Schema-format table (validated) from cohort_physics output (or active_physics with
    is_active=1: Y_log_ratio is then NaN, station keeping makes it meaningless)."""
    rates = rates.sort_values(["norad_id", "t_mid"]).reset_index(drop=True)
    high = (rates["cohort_group"] == "high_altitude").to_numpy() | bool(is_active)
    # Endpoint check from the cleaned TLEs: Piyush's tle_ok also fails climbs, which we keep
    ok = tles.set_index(["norad_id", "epoch"])["tle_ok"]
    ok0 = ok.reindex(pd.MultiIndex.from_frame(rates[["norad_id", "t_start"]].rename(columns={"t_start": "epoch"})))
    ok1 = ok.reindex(pd.MultiIndex.from_frame(rates[["norad_id", "t_end"]].rename(columns={"t_end": "epoch"})))
    endpoints_ok = (ok0.to_numpy() == 1) & (ok1.to_numpy() == 1)
    t_mid = pd.DatetimeIndex(rates["t_mid"])
    hourly = sw.reindex(t_mid.floor("h"))
    d = drag.reindex(rates["norad_id"])
    df = pd.DataFrame({
        "norad_id": rates["norad_id"].to_numpy(),
        "t_mid": t_mid,
        "dt_days": rates["dt_days"].to_numpy(),
        "alt_km": rates["alt_km"].to_numpy(),
        "inc_deg": rates["inc_deg"].to_numpy(),
        "local_solar_time": rates["local_solar_time"].to_numpy(),
        "day_of_year": rates["day_of_year"].to_numpy(),
        "ecc": rates["ecc"].to_numpy(),
        "tle_ok": (endpoints_ok & ~rates["hampel_outlier"].to_numpy()).astype("int64"),
        "is_active": int(is_active),
        "Y_residual": rates[f"resid_{MAIN_MODEL}"].to_numpy() * 1000,
        "Y_residual_nrl": rates["resid_nrlmsise00"].to_numpy() * 1000,
        "Y_log_ratio": np.where(high, np.nan, rates[f"log_ratio_{MAIN_MODEL}"].to_numpy()),
        "T_ap_int": ap_integral(sw, rates["t_end"]) - ap_integral(sw, rates["t_start"]),
        "f107": hourly["f107"].to_numpy(),
        "f107_81d": hourly["f107_81d"].to_numpy(),
        "drag_sensitivity": rates[f"B_{MAIN_MODEL}"].to_numpy(),
        "Y_total": rates[f"total_{MAIN_MODEL}"].to_numpy() * 1000,
        "cohort_group": rates["cohort_group"].to_numpy(),
        "climb_flag": rates["climb_flag"].to_numpy(),
        "hampel_outlier": rates["hampel_outlier"].astype("int64").to_numpy(),
        "role": rates["role"].to_numpy(),
        "drag_source": d["drag_source"].to_numpy(),
        "drag_unstable": d["unstable"].to_numpy().astype("int64"),
        "drag_se": d["se"].to_numpy(),
        "obs_decay": rates["da_dt_km_per_day"].to_numpy() * 1000,
        "pred_decay": rates[f"pred_{MAIN_MODEL}"].to_numpy() * 1000,
        "alt_km_tle": rates["alt_km_tle"].to_numpy(),
        "t_start": rates["t_start"].to_numpy(),
        "t_end": rates["t_end"].to_numpy(),
    })
    df = pd.concat([df, link_storms(rates["t_start"], rates["t_end"], storms)], axis=1)
    return validate(df)


def build_table(drop_climbs=False, path=TABLE_PATH, inputs=None, chunk_dir=CHUNK_DIR, save=True, log=print,
                actives=True, active_inputs=None, active_chunk_dir=ACTIVE_CHUNK_DIR):
    """Rebuild the analysis table end to end. Returns (table, drag, gate) where gate =
    holdout_gate(...) = (pooled, per_object). Saves to `path` when save=True.

    inputs: (intervals, tles, sw, storms) to skip load_inputs (tests pass small frames).
    actives: append the active satellites (question 3, section 6). Their inputs are
    read from disk only when inputs is None and the files exist; tests pass
    active_inputs = (intervals, tles, objects) instead.
    """
    intervals, tles, sw, storms = load_inputs() if inputs is None else inputs
    intervals = intervals.rename(columns={"alt_km": "alt_km_tle"})
    check_no_bridges(intervals, storms)
    drivers = cohort_drivers(tles, intervals, sw, storms, chunk_dir=chunk_dir, log=log)
    rates, drag = cohort_physics(drivers, storms, drop_climbs=drop_climbs)
    gate = holdout_gate(rates)
    table = assemble(rates, drag, sw, storms, tles)
    if actives and active_inputs is None and inputs is None and ACTIVE_INTERVALS_PATH.exists():
        active_inputs = load_active_inputs()
    if actives and active_inputs is not None:
        a_iv, a_tles, objects = active_inputs
        active, active_drag = active_table(a_iv, a_tles, objects, table, sw, storms,
                                           chunk_dir=active_chunk_dir, log=log)
        table = append_actives(table, active)
        if save:
            active_drag.to_parquet(Path(path).parent / ACTIVE_DRAG_NAME)
    if save:
        save_table(table, path)
    return table, drag, gate


# ---------------------------------------------------------------------------
# 6. Active satellites (question 3, offset ratio only)
# ---------------------------------------------------------------------------

def load_active_inputs(intervals_path=ACTIVE_INTERVALS_PATH, tles_path=ACTIVE_TLES_PATH,
                       objects_path=OBJECTS_PATH):
    """(active intervals, their cleaned TLEs, objects table) from src/active_cohort.py and
    src/objects.py. Only intervals of objects the objects table calls active are kept."""
    from src.objects import load_objects
    objects = load_objects(objects_path)
    intervals = pd.read_parquet(intervals_path)
    intervals = intervals[intervals["norad_id"].isin(objects.loc[objects["is_active"] == 1, "norad_id"])]
    return intervals, pd.read_parquet(tles_path), objects


def catalog_scale(passive_table, objects, alt_km, window_km=SCALE_WINDOW_KM, min_bodies=SCALE_MIN_BODIES):
    """Fitted / catalog drag sensitivity of rocket bodies near each altitude in alt_km.

    Rocket bodies of the main group with a stable quiet fit and a catalog A/m. For each
    altitude: median of B_fit / B_catalog over the bodies within +-window_km (mean
    altitude), the window widened in 25 km steps until it holds min_bodies.
    A straight-line fit in altitude does not work: one stage type (39 bodies at
    600-650 km, ratio ~0.5) pulls it far below the ~0.75 seen at 450-550 km.
    Returns a frame indexed like alt_km with scale, n_bodies, window_km.
    """
    rows = passive_table[(passive_table["cohort_group"] == "main") & (passive_table["drag_source"] == "fit")
                         & (passive_table["drag_unstable"] == 0)]
    fit = rows.groupby("norad_id").agg(B=("drag_sensitivity", "first"), alt=("alt_km", "mean"))
    cat = objects.set_index("norad_id")
    fit = fit.join(cat[["object_type", "drag_sensitivity_catalog"]], how="inner")
    fit = fit[(fit["object_type"] == "rocket body") & (fit["drag_sensitivity_catalog"] > 0)]
    ratio = (fit["B"] / fit["drag_sensitivity_catalog"]).to_numpy()
    alt = fit["alt"].to_numpy()
    out = []
    for a in np.atleast_1d(np.asarray(alt_km, dtype=float)):
        w = window_km
        while (np.abs(alt - a) <= w).sum() < min_bodies and w < 1000:
            w += 25
        near = np.abs(alt - a) <= w
        out.append({"scale": np.median(ratio[near]) if near.any() else np.nan,
                    "n_bodies": int(near.sum()), "window_km": w})
    index = alt_km.index if isinstance(alt_km, pd.Series) else None
    return pd.DataFrame(out, index=index)


def free_drag_flags(rows, days=FREE_DRAG_DAYS, ratio=FREE_DRAG_RATIO, min_loss_km=FREE_DRAG_MIN_LOSS_KM):
    """Where active satellites are not station keeping: failed (decaying with drag) or
    being lowered (deorbit, shell change), for at least `days` days in a row.

    rows: intervals with norad_id, t_start, t_end, dt_days, obs_decay and pred_decay
    (m/day, pred = catalog-scaled B times the actual-ap driver).
    Over each window of `days` days starting at an interval: observed altitude change
    (sum obs * dt) vs the free-drag prediction (sum pred * dt). A station-keeping
    satellite holds its altitude (observed / predicted ~0, |ratio| < 0.15 on the real
    Starlinks, < 0.5 km lost per 3 weeks). A window is flagged if it loses more than
    min_loss_km AND at least ratio[0] of the free-drag loss; "drag" up to ratio[1],
    "lowering" above it. The lower bound is well below 1 because deorbiting Starlinks
    decay at only ~0.2-0.5 of the catalog free-drag rate (low-drag attitude).
    Storms last a few days and the operators climb back, so a storm alone does not
    trigger it (tests/test_actives.py).

    Returns (flags, per_object). flags: 0/1 per row of `rows` (same index), 1 inside a
    flagged window; these rows are left out of the offset ratio, so a satellite that
    resumes station keeping after a shell change keeps its good periods. A window that
    is at least half free drag counts, so flags can start up to ~days / 2 early.
    per_object: free_drag (1 = any flagged row), first_flag, flagged_days, flagged_share
    and kind (most common window kind).
    """
    flags = pd.Series(0, index=rows.index, dtype="int64")
    out = []
    for norad_id, g in rows.sort_values("t_start").groupby("norad_id"):
        t0 = g["t_start"].to_numpy()
        t1 = g["t_end"].to_numpy()
        obs = np.cumsum(np.r_[0, (g["obs_decay"] * g["dt_days"]).to_numpy()]) / 1000
        pred = np.cumsum(np.r_[0, (g["pred_decay"] * g["dt_days"]).to_numpy()]) / 1000
        span = pd.Timedelta(days=days).to_timedelta64()
        flagged = np.zeros(len(g), dtype=bool)
        kinds = []
        for i in range(len(g)):
            j = np.searchsorted(t1, t0[i] + span, side="left")
            if j >= len(g):
                break
            d_obs, d_pred = obs[j + 1] - obs[i], pred[j + 1] - pred[i]
            if d_obs > -min_loss_km or d_pred >= 0:
                continue
            r = d_obs / d_pred
            if r >= ratio[0]:
                flagged[i:j + 1] = True
                kinds.append("drag" if r <= ratio[1] else "lowering")
        flags.loc[g.index] = flagged.astype("int64")
        dt = g["dt_days"].to_numpy()
        out.append({"norad_id": norad_id, "free_drag": int(flagged.any()),
                    "first_flag": pd.Timestamp(t0[flagged.argmax()]) if flagged.any() else pd.NaT,
                    "flagged_days": float(dt[flagged].sum()), "flagged_share": float(dt[flagged].sum() / dt.sum()),
                    "kind": (max(set(kinds), key=kinds.count) if kinds else None)})
    per_object = pd.DataFrame(out, columns=["norad_id", "free_drag", "first_flag", "flagged_days",
                                            "flagged_share", "kind"])
    return flags, per_object


def active_physics(drivers, objects, passive_table, storms):
    """Predictions for active satellites with a catalog drag sensitivity.

    B cannot be fitted from quiet decay (station keeping), so B = drag_sensitivity_catalog
    (2.2 x A/m, DISCOSweb) x catalog_scale at the object's mean altitude: the fitted /
    catalog ratio of rocket bodies at that altitude. The same B for both models.
    No Hampel filter: it would remove the station-keeping burns, which are the signal.
    Returns (rates, drag) like cohort_physics; drag has B, scale, n_bodies,
    drag_source = "catalog_scaled", unstable = 0, se = NaN.
    """
    rates = physics.label_intervals(drivers.reset_index(drop=True), storms)
    cat = objects.set_index("norad_id")["drag_sensitivity_catalog"]
    alt = rates.groupby("norad_id")["alt_km"].mean()
    drag = catalog_scale(passive_table, objects, alt)
    drag["B_catalog"] = cat.reindex(drag.index)
    drag["B"] = drag["B_catalog"] * drag["scale"]
    if drag["B"].isna().any():
        raise ValueError(f"No catalog A/m for active objects {drag.index[drag['B'].isna()].tolist()}")
    drag["alt_km"] = alt
    drag["drag_source"] = "catalog_scaled"
    drag["unstable"] = 0
    drag["se"] = np.nan
    for model in physics.MODELS:
        rates = physics.add_predictions(rates, model, drag["B"])
    rates = physics.add_total_effect(rates, MAIN_MODEL)
    rates["hampel_outlier"] = False
    rates["climb_flag"] = 0
    rates["cohort_group"] = "active"
    return rates, drag


def active_table(intervals, tles, objects, passive_table, sw, storms, chunk_dir=ACTIVE_CHUNK_DIR, log=print):
    """Schema rows (is_active = 1) for the active satellites, with extra columns
    object_name, drag_scale, free_drag (1 = row inside a window flagged by
    free_drag_flags: left out of the offset ratio) and first_flag (per object). Returns (rows, drag): drag = one row per
    object with B, B_catalog, scale, n_bodies, window_km and the free_drag_flags columns."""
    intervals = intervals.rename(columns={"alt_km": "alt_km_tle"})
    check_no_bridges(intervals, storms)
    if "climb_flag" not in intervals:
        intervals = intervals.assign(climb_flag=0)
    drivers = cohort_drivers(tles, intervals, sw, storms, chunk_dir=chunk_dir, log=log)
    rates, drag = active_physics(drivers, objects, passive_table, storms)
    df = assemble(rates, drag, sw, storms, tles, is_active=1)
    row_flags, flags = free_drag_flags(df)
    flags = flags.set_index("norad_id")
    names = objects.set_index("norad_id")["object_name"]
    df["object_name"] = df["norad_id"].map(names).astype("string")
    df["drag_scale"] = df["norad_id"].map(drag["scale"])
    df["free_drag"] = row_flags
    df["first_flag"] = df["norad_id"].map(flags["first_flag"])
    return df, drag.join(flags)


def append_actives(passive, active):
    """Passive and active rows in one validated table (passives get free_drag = 0)."""
    return validate(pd.concat([passive.assign(free_drag=0), active], ignore_index=True))


# ---------------------------------------------------------------------------
# 7. Cohort summaries (notebooks/05)
# ---------------------------------------------------------------------------

def _wmean(df, col):
    return (df[col] * df["dt_days"]).sum() / df["dt_days"].sum()


def storm_window_residual(table, storm_id, days=(0, 3), outcomes=("Y_residual", "Y_total")):
    """Per object: dt-weighted mean of each outcome over event days `days` of one storm,
    with mean altitude, observed and predicted decay (m/day). Main sample rows only."""
    from src.schema import main_sample
    rows = main_sample(table)
    rows = rows[(rows["storm_id"] == storm_id) & rows["event_day"].between(*days)]
    cols = list(outcomes) + ["obs_decay", "pred_decay", "alt_km"]
    return pd.DataFrame({c: rows.groupby("norad_id")[[c, "dt_days"]].apply(_wmean, c) for c in cols})


def event_day_profile(table, outcome="Y_residual", exclude=()):
    """Mean outcome by whole event day (floor of event_day), storm = cluster.

    Main sample rows, dt-weighted mean per (storm, day) first, then the mean over
    storms and its SE across storms. exclude: storm_ids to leave out (e.g. May/Oct 2024).
    """
    from src.schema import main_sample
    rows = main_sample(table)
    rows = rows[rows["storm_id"].notna() & ~rows["storm_id"].isin(list(exclude))]
    rows = rows.assign(day=np.floor(rows["event_day"]).astype(int))
    per = rows.groupby(["storm_id", "day"])[[outcome, "dt_days"]].apply(_wmean, outcome).rename("y").reset_index()
    g = per.groupby("day")["y"]
    return pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count()), "storms": g.count()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--drivers-only", action="store_true", help="only run / resume the cached physics chunks")
    p.add_argument("--drop-climbs", action="store_true", help="check version without climb intervals")
    p.add_argument("--no-actives", action="store_true", help="passive cohort only")
    p.add_argument("--out", default=str(TABLE_PATH))
    args = p.parse_args(argv)
    if args.drivers_only:
        intervals, tles, sw, storms = load_inputs()
        cohort_drivers(tles, intervals.rename(columns={"alt_km": "alt_km_tle"}), sw, storms)
        return
    table, drag, (pooled, per) = build_table(drop_climbs=args.drop_climbs, path=args.out,
                                             actives=not args.no_actives)
    print(f"{len(table):,} rows, {table['norad_id'].nunique()} objects -> {args.out}")
    active = table[table["is_active"] == 1]
    if len(active):
        print(f"Active satellites: {active['norad_id'].nunique()} objects, {len(active):,} rows; "
              f"free_drag rows (left out of the offset ratio): {int(active['free_drag'].sum()):,} "
              f"in {active.loc[active['free_drag'] == 1, 'norad_id'].nunique()} objects")
    print("Held-out quiet residual (main cohort, m/day):")
    print(pooled.to_string())
    print(f"Per-object |t| > 2: {(per['t'].abs() > 2).sum()} of {len(per)}")
    print(f"Unstable drag sensitivity: {int(drag['unstable'].sum())} objects")


if __name__ == "__main__":
    main()
