"""Physics simulator: fake orbits that decay through MSIS 2.1 air with a storm boost of
known size, run through the physics pipeline into the analysis table.

Where src/synthetic.py writes Y_residual directly, here the residual comes out of the
same steps as on real data, so a test can check that the pipeline finds what was planted:

1. Space weather: the real hourly table and storm catalog if given; otherwise a
   synthetic hourly series (ap, f107, f107_81d) from src/synthetic.py with storms
   planted at given dates (default: one every STORM_GAP_DAYS days).
2. Objects: passive, near-circular, 400-800 km (plus n_high_alt at 1,000-1,500 km),
   random inclination and local solar time, true drag sensitivity B (m^2/kg).
   The orbital plane drifts in local time with the J2 node precession.
3. Truth: the semi-major axis follows physics.py's driver formula
       da/dt = -B * rho * (1 + boost(event_day)) * sqrt(mu * a)
   rho from pymsis MSIS 2.1 at the object's position along a circular orbit, every
   STEP_MINUTES (a Riemann sum over the orbit). f107, f107a and the ap history are
   passed explicitly (physics.msis_inputs), so pymsis needs no download. Density
   altitude is updated once a day (CHUNK_HOURS); the error from that is < 0.5%.
   boost(event_day) multiplies the true density after each storm's main_onset; the
   default rises to `boost` on days 0..2 and fades to zero by day 5. A negative boost
   (> -1) mimics MSIS over-predicting the storm. Effects of overlapping storms add up.
4. Fake TLEs: irregular epochs (synthetic.intervals: ~1 per day, ~2 per day from
   main_onset to recovery_end, and one at every main_onset so no interval bridges
   it, as in task A4). Gaussian noise on a, sigma_a_m metres on quiet days and
   storm_noise times that from main_onset to recovery_end. smooth_days > 0 replaces
   the true a by its mean over the smooth_days before the epoch (TLE fits use the
   tracking data of the previous days), which blurs the storm in time.
5. Pipeline: orbit.decay_rates -> density driver per interval (physics.sample_times,
   physics.msis_inputs, physics.density on the circular-orbit track, driver as in
   physics.density_drivers) -> physics.label_intervals -> quiet fit of B and
   prediction (physics.add_predictions) for MSIS 2.1 and NRLMSISE-00. High-altitude
   objects use a catalog B (true B with CATALOG_SCATTER log-normal error), marked
   drag_source = "catalog", and Y_log_ratio NaN (as planned for B4-B5 at scale).

Output: the schema.py table (validated), sign convention resid = observed - predicted
(negative = more decay than predicted), plus these extra columns:
    true_effect            planted storm part of the decay (m/day), the boost term alone
    true_decay             true da/dt over the interval (m/day)
    obs_decay              da/dt from the fake TLEs (m/day)
    true_drag_sensitivity  B used to make the orbit (m^2/kg)
    drag_source            "fit" (quiet-time fit) or "catalog"
    role                   physics.label_intervals role: fit / holdout / storm
Air density is never a column: it is the mediator and only builds the baseline.
"""

import numpy as np
import pandas as pd

from src import physics, synthetic
from src.orbit import decay_rates
from src.schema import HIGH_ALT_MIN_KM, link_storms, main_sample, validate

START = pd.Timestamp("2024-04-01")
N_DAYS = 150
STORM_GAP_DAYS = 25                  # default planted storms: one every 25 days ...
FIRST_STORM_DAY = 15                 # ... starting 15 days after START
STORM_CLASS = "strong"               # class of the default planted storms
R_EARTH_KM = 6378.137
J2 = 1.08263e-3
SUN_RATE = 2 * np.pi / (365.2422 * 86400)    # rad/s, mean motion of the Sun
STEP_MINUTES = 20                    # truth time step
CHUNK_HOURS = 24                     # density altitude refreshed this often
SW_PAD = pd.Timedelta(days=3)        # MSIS needs 57 h of ap history
MIN_GAP_DAYS = 0.2                   # shortest interval next to an inserted main_onset epoch
REENTRY_KM = 250.0                   # TLEs stop below this true altitude
SIGMA_A_M = 20.0                     # quiet TLE noise on a (m)
STORM_NOISE = 3.0                    # noise factor from main_onset to recovery_end
CATALOG_SCATTER = 0.1                # log-normal error of the catalog B
MODEL = "msis21"                     # truth and main outcome


# ---------------------------------------------------------------------------
# 1. Space weather
# ---------------------------------------------------------------------------

def planted_storms(dates, storm_class=STORM_CLASS):
    """Storm catalog (columns of storms.storm_catalog used downstream) with main_onset at `dates`."""
    onsets = pd.DatetimeIndex(sorted(pd.to_datetime(dates)))
    classes = [storm_class] * len(onsets)
    return pd.DataFrame({
        "storm_id": np.arange(1, len(onsets) + 1),
        "onset_time": onsets,
        "main_onset": onsets,
        "peak_time": onsets + pd.Timedelta(hours=6),
        "class": classes,
        "recovery_end": onsets + pd.to_timedelta([synthetic.RECOVERY[c] for c in classes], unit="D"),
    })


def default_storm_dates(start=START, n_days=N_DAYS):
    """One storm every STORM_GAP_DAYS days, the last one with 20 days left after it."""
    days = np.arange(FIRST_STORM_DAY, n_days - 20 + 1, STORM_GAP_DAYS)
    return start + pd.to_timedelta(days, unit="D") + pd.Timedelta(hours=6)


def synthetic_spaceweather(start, end, storms, rng):
    """Hourly ap, f107, f107_81d (omni_io.load_spaceweather format) from SW_PAD before
    start to end, with one ap spike per storm (src/synthetic.py)."""
    days = pd.date_range((start - SW_PAD).floor("D"), end.ceil("D"), freq="D")
    flux = synthetic.solar_flux(days, rng)
    return synthetic.hourly_spaceweather(flux, synthetic.hourly_ap(flux, storms, rng))


def ap_integral(sw, t):
    """Integral of hourly ap (nT*day) from the first hour of sw up to times t."""
    hours = ((sw.index - sw.index[0]) / pd.Timedelta(hours=1)).to_numpy()
    cum = np.concatenate([[0], np.cumsum(sw["ap"].to_numpy() / 24)])
    grid = np.concatenate([hours, [hours[-1] + 1]])
    return np.interp((pd.DatetimeIndex(t) - sw.index[0]) / pd.Timedelta(hours=1), grid, cum)


# ---------------------------------------------------------------------------
# 2. Objects and their circular-orbit track
# ---------------------------------------------------------------------------

def make_objects(n_objects, n_high_alt, rng):
    """Passive near-circular objects: synthetic.make_objects / make_high_alt_objects with
    ecc below 0.01, plus orbital phase u0 (rad) and node local-time drift (h/day)."""
    low = synthetic.make_objects(n_objects, 0.0, rng)
    high = synthetic.make_high_alt_objects(n_high_alt, 90000 + n_objects, rng)
    obj = pd.concat([low, high], ignore_index=True).drop(columns=["offset", "phase"])
    obj["ecc"] = np.clip(obj["ecc"], 0, 0.009)
    obj["u0"] = rng.uniform(0, 2 * np.pi, len(obj))
    a_m = (R_EARTH_KM + obj["alt_km"].to_numpy()) * 1000
    n = np.sqrt(physics.MU_SI / a_m**3)
    node_rate = -1.5 * n * J2 * (R_EARTH_KM * 1000 / a_m) ** 2 * np.cos(np.radians(obj["inc_deg"]))
    obj["mean_motion"] = n                                            # rad/s
    obj["lst_drift"] = (node_rate - SUN_RATE) * 86400 * 12 / np.pi    # h/day
    return obj.rename(columns={"drag_sensitivity": "true_B"})


def local_time_of_node(obj, t, start):
    """Local solar time of the ascending node (h, 0-24) at times t (arrays matching obj)."""
    days = (pd.DatetimeIndex(t) - start) / pd.Timedelta(days=1)
    return np.mod(obj["local_solar_time"] + obj["lst_drift"] * np.asarray(days), 24)


def track(obj, t, start):
    """Latitude, longitude (deg) along a circular orbit; obj is a dict of arrays matching t."""
    t = pd.DatetimeIndex(t)
    secs = ((t - start) / pd.Timedelta(seconds=1)).to_numpy()
    u = obj["u0"] + obj["mean_motion"] * secs
    inc = np.radians(obj["inc_deg"])
    lat = np.degrees(np.arcsin(np.sin(inc) * np.sin(u)))
    dlon = np.arctan2(np.cos(inc) * np.sin(u), np.cos(u))          # east of the node
    lst = local_time_of_node(obj, t, start) + dlon * 12 / np.pi
    ut = ((t - t.floor("D")) / pd.Timedelta(hours=1)).to_numpy()
    lon = np.mod((lst - ut) * 15 + 180, 360) - 180
    return lat, lon


def _repeat(obj, n):
    """Object columns as arrays, each object repeated n times (object-major order)."""
    return {c: np.repeat(obj[c].to_numpy(), n) for c in ["u0", "mean_motion", "inc_deg",
                                                         "local_solar_time", "lst_drift"]}


# ---------------------------------------------------------------------------
# 3. Truth
# ---------------------------------------------------------------------------

def default_boost(event_day, peak=0.5):
    """Density boost: `peak` on days 0..2 after main_onset, linear fade to 0 at day 5."""
    d = np.asarray(event_day, float)
    shape = np.where((d >= 0) & (d <= 2), 1.0, np.clip((5 - d) / 3, 0, 1) * (d > 2))
    return peak * shape


def boost_at(times, storms, boost):
    """Sum over storms of boost(days since main_onset) at `times`."""
    t = pd.DatetimeIndex(times)
    out = np.zeros(len(t))
    for onset in storms["main_onset"]:
        out += boost((t - onset) / pd.Timedelta(days=1))
    if (out <= -1).any():
        raise ValueError("boost must stay above -1 (density cannot go negative)")
    return out


def true_orbits(obj, sw, storms, start, end, boost, step_minutes=STEP_MINUTES,
                chunk_hours=CHUNK_HOURS):
    """True semi-major axis (m) and cumulative boost part of the change in a (m).

    Returns (grid, a, a_boost): grid times (n_grid,), a and a_boost (n_obj, n_grid).
    Each step uses rho at the mid-step position; the altitude for rho and a in
    sqrt(mu a) are refreshed every chunk_hours.
    """
    step = pd.Timedelta(minutes=step_minutes)
    grid = pd.date_range(start, end, freq=step)
    mids = grid[:-1] + step / 2
    f107, f107a, aps = physics.msis_inputs(sw, mids)
    gain = boost_at(mids, storms, boost)
    dt = step / pd.Timedelta(seconds=1)
    n_obj, per_chunk = len(obj), int(pd.Timedelta(hours=chunk_hours) / step)
    a = np.empty((n_obj, len(grid)))
    a_boost = np.zeros((n_obj, len(grid)))
    a[:, 0] = (R_EARTH_KM + obj["alt_km"].to_numpy()) * 1000
    B = obj["true_B"].to_numpy()[:, None]
    for s in range(0, len(mids), per_chunk):
        sl = slice(s, min(s + per_chunk, len(mids)))
        n = sl.stop - sl.start
        o = _repeat(obj, n)
        t = np.tile(mids[sl], n_obj)
        lat, lon = track(o, t, start)
        # Objects below REENTRY_KM stop decaying (their TLEs are dropped later)
        alive = a[:, s] / 1000 - R_EARTH_KM > REENTRY_KM
        alt = np.repeat(np.maximum(a[:, s] / 1000 - R_EARTH_KM, REENTRY_KM), n)
        rho = physics.density(t, lat, lon, alt, np.tile(f107[sl], n_obj), np.tile(f107a[sl], n_obj),
                              np.tile(aps[sl], (n_obj, 1)), physics.MODELS[MODEL]).reshape(n_obj, n)
        rate = -B * rho * np.sqrt(physics.MU_SI * a[:, [s]]) * alive[:, None]   # m/s, no boost
        a[:, s + 1:sl.stop + 1] = a[:, [s]] + np.cumsum(rate * (1 + gain[sl]) * dt, axis=1)
        a_boost[:, s + 1:sl.stop + 1] = a_boost[:, [s]] + np.cumsum(rate * gain[sl] * dt, axis=1)
    return grid, a, a_boost


# ---------------------------------------------------------------------------
# 4. Fake TLEs
# ---------------------------------------------------------------------------

def storm_noise_factor(times, storms, factor):
    """`factor` from main_onset to recovery_end of any storm, else 1."""
    t = pd.DatetimeIndex(times)
    inside = np.zeros(len(t), dtype=bool)
    for s in storms.to_dict("records"):
        inside |= (t >= s["main_onset"]) & (t <= s["recovery_end"])
    return np.where(inside, factor, 1.0)


def fake_tles(obj, grid, a, storms, rng, start, end, sigma_a_m=SIGMA_A_M,
              storm_noise=STORM_NOISE, smooth_days=0.0):
    """TLE-like rows (norad_id, epoch, a_km, true_a_km) at irregular epochs per object."""
    g_s = ((grid - start) / pd.Timedelta(seconds=1)).to_numpy()
    rows = []
    for i, norad_id in enumerate(obj["norad_id"]):
        t0, t1 = synthetic.intervals(storms, rng, start, end)
        epochs = t0.append(t1[-1:])
        # Regular epochs too close to an inserted main_onset epoch would make
        # near-zero intervals (noise / dt blows up): drop them, keep the onset
        near = np.zeros(len(epochs), dtype=bool)
        for onset in storms["main_onset"]:
            gap = np.abs((epochs - onset) / pd.Timedelta(days=1))
            near |= (gap > 0) & (gap < MIN_GAP_DAYS)
        epochs = epochs[~near]
        e_s = ((epochs - start) / pd.Timedelta(seconds=1)).to_numpy()
        true_a = np.interp(e_s, g_s, a[i])
        if smooth_days > 0:
            # Mean of a over the smooth_days before the epoch (from the cumulative integral)
            cum = np.concatenate([[0], np.cumsum(np.diff(g_s) * (a[i, 1:] + a[i, :-1]) / 2)])
            lo = np.maximum(e_s - smooth_days * 86400, 0)
            width = e_s - lo
            seen = np.where(width > 0, (np.interp(e_s, g_s, cum) - np.interp(lo, g_s, cum))
                            / np.where(width > 0, width, 1), true_a)
        else:
            seen = true_a
        sigma = sigma_a_m * storm_noise_factor(epochs, storms, storm_noise)
        keep = true_a / 1000 - R_EARTH_KM > REENTRY_KM
        rows.append(pd.DataFrame({"norad_id": norad_id, "epoch": epochs,
                                  "a_km": (seen + rng.normal(0, 1, len(epochs)) * sigma) / 1000,
                                  "true_a_km": true_a / 1000})[keep])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Pipeline
# ---------------------------------------------------------------------------

def circular_drivers(rates, obj, sw, start, n=physics.N_SAMPLES):
    """physics.density_drivers for circular orbits: same samples, density and driver,
    with positions from `track` instead of SGP4 and a interpolated between the TLEs.

    Adds alt_km (mean sample altitude) and driver_<model> (km/day per m^2/kg, negative).
    """
    rates = rates.reset_index(drop=True)
    times = pd.DatetimeIndex(np.concatenate([physics.sample_times(r.t_start, r.t_end, n)
                                             for r in rates.itertuples(index=False)]))
    frac = np.tile((np.arange(n) + 0.5) / n, len(rates))
    a_start = rates["a_start_km"].to_numpy()
    a_km = np.repeat(a_start, n) + np.repeat(rates["da_km"].to_numpy(), n) * frac
    o = obj.set_index("norad_id").loc[rates["norad_id"]].reset_index()
    lat, lon = track(_repeat(o, n), times, start)
    alt = a_km - R_EARTH_KM
    f107, f107a, aps = physics.msis_inputs(sw, times)

    out = rates.copy()
    out["alt_km"] = alt.reshape(-1, n).mean(axis=1)
    a_m = 1000 * (a_start + rates["da_km"].to_numpy() / 2)
    for name, version in physics.MODELS.items():
        rho = physics.density(times, lat, lon, alt, f107, f107a, aps, version).reshape(-1, n).mean(axis=1)
        out[f"driver_{name}"] = -rho * np.sqrt(physics.MU_SI * a_m) * physics.M_PER_S_TO_KM_PER_DAY
    return out


def _interval_true(tles, rates, grid, a_boost, obj, start):
    """True decay and planted boost part (m/day) per interval."""
    g_s = ((grid - start) / pd.Timedelta(seconds=1)).to_numpy()
    row = pd.Series(np.arange(len(obj)), index=obj["norad_id"]).loc[rates["norad_id"]].to_numpy()
    true_a = tles.set_index(["norad_id", "epoch"])["true_a_km"]
    a0 = true_a.reindex(pd.MultiIndex.from_frame(rates[["norad_id", "t_start"]])).to_numpy()
    a1 = true_a.reindex(pd.MultiIndex.from_frame(rates[["norad_id", "t_end"]])).to_numpy()
    s0 = ((pd.DatetimeIndex(rates["t_start"]) - start) / pd.Timedelta(seconds=1)).to_numpy()
    s1 = ((pd.DatetimeIndex(rates["t_end"]) - start) / pd.Timedelta(seconds=1)).to_numpy()
    b0 = np.array([np.interp(s, g_s, a_boost[r]) for s, r in zip(s0, row)])
    b1 = np.array([np.interp(s, g_s, a_boost[r]) for s, r in zip(s1, row)])
    dt = rates["dt_days"].to_numpy()
    return 1000 * (a1 - a0) / dt, (b1 - b0) / dt


def simulate(n_objects=30, n_high_alt=3, storm_dates=None, spaceweather=None, storms=None,
             start=START, n_days=N_DAYS, boost=0.5, sigma_a_m=SIGMA_A_M,
             storm_noise=STORM_NOISE, smooth_days=0.0, step_minutes=STEP_MINUTES,
             return_hourly=False, seed=0):
    """Simulated analysis table in the schema format, and the storm catalog used.

    spaceweather, storms: real hourly table (ap, f107, f107_81d indexed by time,
        covering 3 days before start to the end) and storms.storm_catalog frame; if
        None, both are synthetic, with storms at storm_dates (default_storm_dates).
    boost: peak density boost for default_boost (0 = no storm effect, < 0 = MSIS
        over-predicts), or a callable boost(event_day) -> fraction.
    sigma_a_m, storm_noise: TLE noise on a (m) on quiet days, and its factor in storms.
    smooth_days: TLE blur (mean of a over this many days before the epoch), 0 = off.
    Returns (table, storms), or (table, storms, hourly) with return_hourly=True;
    settings in table.attrs["truth"].
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=n_days)
    if storms is None:
        storms = planted_storms(default_storm_dates(start, n_days) if storm_dates is None else storm_dates)
    storms = storms[(storms["main_onset"] >= start) & (storms["main_onset"] < end)].reset_index(drop=True)
    # Own generator for the space weather, so passing the same frames back gives the same table
    sw = (synthetic_spaceweather(start, end, storms, np.random.default_rng([seed, 1]))
          if spaceweather is None else spaceweather)
    boost_fn = boost if callable(boost) else (lambda d: default_boost(d, boost))

    obj = make_objects(n_objects, n_high_alt, rng)
    grid, a, a_boost = true_orbits(obj, sw, storms, start, end, boost_fn, step_minutes)
    tles = fake_tles(obj, grid, a, storms, rng, start, end, sigma_a_m, storm_noise, smooth_days)

    rates = decay_rates(tles)
    a_start = tles.set_index(["norad_id", "epoch"])["a_km"]
    rates["a_start_km"] = a_start.reindex(pd.MultiIndex.from_frame(rates[["norad_id", "t_start"]])).to_numpy()
    rates = physics.label_intervals(circular_drivers(rates, obj, sw, start), storms)

    catalog = obj.set_index("norad_id")["true_B"] * rng.lognormal(0, CATALOG_SCATTER, len(obj))
    high_ids = obj.loc[obj["alt_km"] >= HIGH_ALT_MIN_KM, "norad_id"]
    for model in physics.MODELS:
        B = physics.fit_drag_sensitivity(rates, model).reindex(obj["norad_id"])
        B[high_ids] = catalog[high_ids]
        rates = physics.add_predictions(rates, model, B)
    rates["true_decay"], rates["true_effect"] = _interval_true(tles, rates, grid, a_boost, obj, start)

    o = obj.set_index("norad_id").loc[rates["norad_id"]].reset_index()
    t_mid = pd.DatetimeIndex(rates["t_mid"])
    daily = sw.reindex(t_mid.floor("h"))
    high = rates["norad_id"].isin(high_ids).to_numpy()
    df = pd.DataFrame({
        "norad_id": rates["norad_id"].to_numpy(),
        "t_mid": t_mid,
        "dt_days": rates["dt_days"].to_numpy(),
        "alt_km": rates["alt_km"].to_numpy(),
        "inc_deg": o["inc_deg"].to_numpy(),
        "local_solar_time": np.asarray(local_time_of_node(o, t_mid, start)),
        "day_of_year": t_mid.dayofyear,
        "ecc": o["ecc"].to_numpy(),
        "tle_ok": 1,
        "is_active": 0,
        "Y_residual": rates[f"resid_{MODEL}"].to_numpy() * 1000,
        "Y_residual_nrl": rates["resid_nrlmsise00"].to_numpy() * 1000,
        "Y_log_ratio": np.where(high, np.nan, rates[f"log_ratio_{MODEL}"].to_numpy()),
        "T_ap_int": ap_integral(sw, rates["t_end"]) - ap_integral(sw, rates["t_start"]),
        "f107": daily["f107"].to_numpy(),
        "f107_81d": daily["f107_81d"].to_numpy(),
        "drag_sensitivity": rates[f"B_{MODEL}"].to_numpy(),
        "true_effect": rates["true_effect"].to_numpy(),
        "true_decay": rates["true_decay"].to_numpy(),
        "obs_decay": rates["da_dt_km_per_day"].to_numpy() * 1000,
        "true_drag_sensitivity": o["true_B"].to_numpy(),
        "drag_source": np.where(high, "catalog", "fit"),
        "role": rates["role"].to_numpy(),
    })
    df = pd.concat([df, link_storms(rates["t_start"], rates["t_end"], storms)], axis=1)
    table = validate(df.sort_values(["norad_id", "t_mid"]).reset_index(drop=True))
    table.attrs["truth"] = {"boost": boost if not callable(boost) else "callable",
                            "sigma_a_m": sigma_a_m, "storm_noise": storm_noise,
                            "smooth_days": smooth_days, "n_high_alt": n_high_alt, "seed": seed}
    if return_hourly:
        return table, storms, sw
    return table, storms


def planted_profile(table, storms, ref=-1, **stack_kw):
    """Planted effect (m/day) by event day k, relative to day `ref`, on the same stacked
    main-sample rows (dt-weighted) the event study fits: the curve it should find."""
    from src.event_study import build_stacks
    stacked, _ = build_stacks(main_sample(table), storms, **stack_kw)
    w = stacked["dt_days"]
    prof = (stacked["true_effect"] * w).groupby(stacked["k"]).sum() / w.groupby(stacked["k"]).sum()
    return prof - prof.get(ref, 0.0)
