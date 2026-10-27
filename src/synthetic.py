"""Fake analysis table with a known storm effect, for testing the causal methods.

The table has the exact columns of src/schema.py, plus the planted truth:
    true_tau          storm effect of this object (m/day per nT of ap above quiet), >= 0
    true_effect       storm part of Y_residual (m/day) = -true_tau * (mean ap - AP_QUIET)
    true_manoeuvre    operator storm response in Y_residual (m/day), 0 for passive objects
    true_station_keeping  routine operator boosts in Y_residual (m/day, constant per
                      active satellite), 0 for passive objects
    true_artifact     TLE-fit part of Y_residual (m/day), 0 for clean intervals
A method passes if it recovers these from the schema columns alone.

How the fake world works:
- Sun UV: f107_81d follows a slow solar-cycle decline plus a ~150-day wave, f107
  adds the 27-day solar rotation and day-to-day noise.
- Storms: main_onset days are drawn with probability rising with F10.7 (and a little
  near the equinoxes), so storms come with high Sun UV: this is the confounding.
  Each storm has a class (moderate/strong/severe/extreme in the real 11/6/2/1 mix),
  an ap spike decaying over hours, and a recovery_end. A real storm catalog
  (storms.storm_catalog) can be passed instead.
- Hourly ap = quiet level (slightly higher at high F10.7) + storm spikes.
- Objects: altitude 400-800 km, mixed inclinations, random local solar time, log-normal
  drag sensitivity, a few active satellites, eccentricity mostly below 0.01 with a
  tail above it. Intervals are ~1 day long, ~0.5 day during storms (TLEs are
  refitted more often then), and never bridge a main_onset (as in task A4).
- TLE artifacts (the reviewer's point: TLEs are fitted mean elements, not altitudes):
  * oval orbits (ecc >= 0.01) get a slow fake altitude swing, ECC_SWING * ecc m/day,
    with the ~60-day perigee cycle;
  * some intervals are bad fits, more often during storms, with a fake drop
    (mostly negative, so it looks like extra storm decay); they get tle_ok = 0.
  schema.main_sample removes both; using every row biases the storm effect.
- Residual (m/day) = object offset + UV model error (gamma * (f107 - 150))
                     + season wave + true_effect + true_manoeuvre
                     + true_station_keeping + noise.
  true_tau shrinks with altitude (thinner air higher up) and grows with drag
  sensitivity, so the effect differs by altitude as the vulnerability map expects.
- Active satellites: during the storm (main_onset to recovery_end) operators cut
  duck_share of the storm effect; in the next RECOVERY_DAYS they win back
  recover_share of what was still lost. NO_DRAG_FIT of them have no
  drag_sensitivity (manoeuvres spoil the quiet-time fit), and every one does
  routine station keeping (a constant boost, STATION_KEEPING m/day) that has
  nothing to do with storms. Total offset ratio = duck_share +
  (1 - duck_share) * recover_share.
- High-altitude control (n_high_alt > 0): extra passive, near-circular objects at
  1,000-1,500 km. The air there is so thin that their planted storm effect is about
  zero, but they share the Sun UV error, season wave and TLE bad fits, so a method
  should find no storm effect for them. They come from their own random generator,
  so adding them leaves the rest of the table unchanged.
- return_hourly=True also returns the hourly space weather (ap, f107, f107_81d,
  indexed by time) in the format of omni_io.load_spaceweather(), for methods that
  need ap lags.
"""

import numpy as np
import pandas as pd

from src.schema import STORM_CLASSES, link_storms, main_sample, validate

START = pd.Timestamp("2024-04-01")
END = pd.Timestamp("2025-04-01")

AP_QUIET = 7.0                       # nT, typical quiet ap
CLASS_MIX = [11, 6, 2, 1]            # moderate, strong, severe, extreme (real catalog)
PEAK_AP = {"moderate": 60, "strong": 130, "severe": 250, "extreme": 400}      # nT
RECOVERY = {"moderate": 1.5, "strong": 2.5, "severe": 3.5, "extreme": 4.5}    # days
SPIKE_HOURS = 18.0                   # e-folding time of the ap spike
RECOVERY_DAYS = 10                   # length of the operator recovery window
ECC_SWING = 2000.0                   # m/day of fake swing per unit eccentricity
APSIDAL_DAYS = 60.0                  # period of the perigee swing
BAD_FIT_QUIET = 0.02                 # share of bad TLE fits on quiet days ...
BAD_FIT_STORM = 0.10                 # ... and inside storm windows
BAD_FIT_JUMP = (-40.0, 20.0)         # mean and sd of the fake jump (m/day)
STATION_KEEPING = (0.5, 3.0)         # m/day of routine boosts per active satellite
NO_DRAG_FIT = 0.2                    # share of active satellites without drag_sensitivity
HIGH_ALT_KM = (1000.0, 1500.0)       # altitude range of the high-altitude control group
INCLINATIONS = [51.6, 53.0, 65.0, 72.0, 82.0, 87.0, 98.0]


def solar_flux(days, rng):
    """Daily f107 and f107_81d (sfu) for `days` (DatetimeIndex)."""
    t = (days - START).days.to_numpy()
    # Slow solar-cycle drift plus a ~5-month wave, so it is not just another season wave
    f107_81d = 185 - 30 * t / 365 + 20 * np.sin(2 * np.pi * t / 150)
    noise = np.zeros(len(t))
    for i in range(1, len(t)):                       # AR(1) day-to-day noise
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 8)
    f107 = f107_81d + 25 * np.sin(2 * np.pi * t / 27) + noise
    return pd.DataFrame({"f107": np.clip(f107, 70, None), "f107_81d": f107_81d}, index=days)


def make_storms(flux, n_storms=20, rng=None, min_gap_days=7):
    """Fake storm catalog with the columns of storms.storm_catalog that the analysis uses.

    Onset day probability ~ (f107 - 60)^4 * (1 + 0.3 * equinox wave).
    """
    rng = np.random.default_rng(rng)
    days = flux.index[10:-25]                        # keep the full event window inside
    doy = days.dayofyear.to_numpy()
    weight = (flux.loc[days, "f107"].to_numpy() - 60) ** 4 * (1 + 0.3 * np.cos(4 * np.pi * (doy - 80) / 365))
    weight /= weight.sum()

    chosen = []
    for d in rng.choice(len(days), size=len(days), replace=False, p=weight):
        if all(abs(d - c) >= min_gap_days for c in chosen):
            chosen.append(d)
        if len(chosen) == n_storms:
            break
    onsets = days[sorted(chosen)] + pd.to_timedelta(rng.integers(0, 24, len(chosen)), unit="h")
    classes = rng.choice(STORM_CLASSES, size=len(chosen), p=np.array(CLASS_MIX) / sum(CLASS_MIX))
    recovery = onsets + pd.to_timedelta([RECOVERY[c] for c in classes], unit="D")
    return pd.DataFrame({
        "storm_id": np.arange(1, len(chosen) + 1),
        "onset_time": onsets,
        "main_onset": onsets,
        "peak_time": onsets + pd.Timedelta(hours=6),
        "class": classes,
        "recovery_end": recovery,
    })


def hourly_ap(flux, storms, rng):
    """Hourly ap (nT): quiet level rising with F10.7, plus one decaying spike per storm."""
    hours = pd.date_range(flux.index[0], flux.index[-1] + pd.Timedelta(hours=23), freq="h")
    f107 = flux["f107"].reindex(hours.floor("D")).to_numpy()
    ap = (AP_QUIET + 0.03 * (f107 - 150)) * rng.lognormal(0, 0.3, len(hours))
    for s in storms.to_dict("records"):
        since = (hours - s["main_onset"]).total_seconds().to_numpy() / 3600
        ap += np.where(since >= 0, PEAK_AP[s["class"]] * np.exp(-since / SPIKE_HOURS), 0)
    return pd.Series(np.clip(ap, 0, None), index=hours)


def make_objects(n_objects, active_share, rng):
    """Object table: norad_id, alt_km, inc_deg, local_solar_time, drag_sensitivity, is_active,
    ecc, plus the hidden per-object model bias (offset) and perigee phase."""
    inc = rng.choice(INCLINATIONS, n_objects) + rng.normal(0, 0.3, n_objects)
    return pd.DataFrame({
        "norad_id": 90000 + np.arange(n_objects),
        "alt_km": rng.uniform(400, 800, n_objects),
        "inc_deg": np.clip(inc, 0, 180),
        "local_solar_time": rng.uniform(0, 24, n_objects),
        "drag_sensitivity": rng.lognormal(np.log(0.01), 0.5, n_objects),
        "is_active": (rng.random(n_objects) < active_share).astype("int64"),
        "ecc": np.clip(rng.lognormal(np.log(0.003), 0.9, n_objects), 0, 0.2),
        "phase": rng.uniform(0, 2 * np.pi, n_objects),     # perigee phase at START
        "offset": rng.normal(0, 2.0, n_objects),          # object-level model bias (m/day)
    })


def make_high_alt_objects(n_objects, first_id, rng):
    """Passive, near-circular (ecc < 0.01) objects at HIGH_ALT_KM, same columns as make_objects."""
    inc = rng.choice(INCLINATIONS, n_objects) + rng.normal(0, 0.3, n_objects)
    return pd.DataFrame({
        "norad_id": first_id + np.arange(n_objects),
        "alt_km": rng.uniform(*HIGH_ALT_KM, n_objects),
        "inc_deg": np.clip(inc, 0, 180),
        "local_solar_time": rng.uniform(0, 24, n_objects),
        "drag_sensitivity": rng.lognormal(np.log(0.01), 0.5, n_objects),
        "is_active": np.zeros(n_objects, dtype="int64"),
        "ecc": np.clip(rng.lognormal(np.log(0.002), 0.5, n_objects), 0, 0.009),
        "phase": rng.uniform(0, 2 * np.pi, n_objects),
        "offset": rng.normal(0, 2.0, n_objects),
    })


def hourly_spaceweather(flux, ap):
    """Hourly ap, f107, f107_81d indexed by time, like omni_io.load_spaceweather()."""
    daily = flux.reindex(ap.index.floor("D"))
    out = pd.DataFrame({"ap": ap.to_numpy(), "f107": daily["f107"].to_numpy(),
                        "f107_81d": daily["f107_81d"].to_numpy()}, index=ap.index)
    out.index.name = "time"
    return out


def storm_effect_per_ap(alt_km, drag_sensitivity, tau0, scale_km):
    """Planted effect (m/day per nT): tau0 at 400 km and B = 0.01 m^2/kg, falling with altitude."""
    return tau0 * np.exp(-(np.asarray(alt_km) - 400) / scale_km) * (np.asarray(drag_sensitivity) / 0.01)


def intervals(storms, rng, start=START, end=END):
    """Interval boundaries for one object: ~1 day steps (~0.5 day from main_onset to
    recovery_end), and every main_onset is a boundary."""
    span = (end - start) / pd.Timedelta(days=1)
    onset = ((storms["main_onset"] - start) / pd.Timedelta(days=1)).to_numpy()
    recovered = ((storms["recovery_end"] - start) / pd.Timedelta(days=1)).to_numpy()
    t, edges = 0.0, [0.0]
    while t < span:
        busy = ((t >= onset) & (t < recovered)).any()
        t += rng.uniform(0.25, 0.75) if busy else rng.uniform(0.5, 1.5)
        edges.append(min(t, span))
    edges = (start + pd.to_timedelta(edges, unit="D")).round("s")
    edges = edges.append(pd.DatetimeIndex(storms["main_onset"])).sort_values().unique()
    return edges[:-1], edges[1:]


def make_synthetic(n_objects=200, n_storms=20, storms=None, active_share=0.15,
                   tau0=0.05, scale_km=150.0, gamma=-0.12, noise=3.0,
                   duck_share=0.3, recover_share=0.5, n_high_alt=0, return_hourly=False, seed=0):
    """Fake analysis table in the schema format, and the storm catalog used.

    tau0: storm effect at 400 km (m/day per nT of ap); scale_km: how fast it fades
    with altitude; gamma: UV model error (m/day per sfu; the confounding strength);
    noise: residual noise (m/day for a 1-day interval); duck_share, recover_share:
    operator offset of active satellites; n_high_alt: number of extra passive objects
    at 1,000-1,500 km (the high-altitude control, 10-20 is a realistic count).
    Returns (table, storms), or (table, storms, hourly) with return_hourly=True; the
    planted settings are in table.attrs["truth"].
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D")
    flux = solar_flux(days, rng)
    if storms is None:
        storms = make_storms(flux, n_storms, rng)
    storms = storms[(storms["main_onset"] >= START) & (storms["main_onset"] < END)].reset_index(drop=True)
    ap = hourly_ap(flux, storms, rng)
    # Integral of ap (nT*day) from START up to any time: cumulative hourly sum, interpolated
    hours_since = ((ap.index - START).total_seconds() / 3600).to_numpy()
    cum = np.concatenate([[0], np.cumsum(ap.to_numpy() / 24)])
    grid = np.concatenate([hours_since, [hours_since[-1] + 1]])

    def ap_integral(t):
        return np.interp((t - START).total_seconds().to_numpy() / 3600, grid, cum)

    objects = make_objects(n_objects, active_share, rng)
    # Routine station keeping of active satellites (own generator, so adding it did
    # not change the other random draws of a seed)
    sk = np.random.default_rng([seed, 1]).uniform(*STATION_KEEPING, n_objects)
    objects["station_keeping"] = np.where(objects["is_active"] == 1, sk, 0.0)
    # High-altitude control group: own generator for every draw, so the rest is unchanged
    hrng = np.random.default_rng([seed, 2])
    high = make_high_alt_objects(n_high_alt, 90000 + n_objects, hrng).assign(station_keeping=0.0)
    rows = []
    for obj in objects.to_dict("records"):
        t0, t1 = intervals(storms, rng)
        rows.append(pd.DataFrame({"norad_id": obj["norad_id"], "t_start": t0, "t_end": t1}))
    for obj in high.to_dict("records"):
        t0, t1 = intervals(storms, hrng)
        rows.append(pd.DataFrame({"norad_id": obj["norad_id"], "t_start": t0, "t_end": t1}))
    df = pd.concat(rows, ignore_index=True).merge(pd.concat([objects, high], ignore_index=True),
                                                  on="norad_id")
    # Row-level random draws: main objects from rng (in the original order), the
    # high-altitude rows (last in df) from hrng
    n_main = int((df["norad_id"] < 90000 + n_objects).sum())

    def draw(method, *args):
        main = getattr(rng, method)(*args, n_main)
        return np.concatenate([main, getattr(hrng, method)(*args, len(df) - n_main)])

    df["dt_days"] = (df["t_end"] - df["t_start"]) / pd.Timedelta(days=1)
    df["t_mid"] = df["t_start"] + (df["t_end"] - df["t_start"]) / 2
    df["day_of_year"] = df["t_mid"].dt.dayofyear
    df["T_ap_int"] = ap_integral(pd.DatetimeIndex(df["t_end"])) - ap_integral(pd.DatetimeIndex(df["t_start"]))
    daily = flux.reindex(df["t_mid"].dt.floor("D"))
    df["f107"] = daily["f107"].to_numpy()
    df["f107_81d"] = daily["f107_81d"].to_numpy()
    df = pd.concat([df, link_storms(df["t_start"], df["t_end"], storms)], axis=1)

    # Planted storm effect
    df["true_tau"] = storm_effect_per_ap(df["alt_km"], df["drag_sensitivity"], tau0, scale_km)
    df["true_effect"] = -df["true_tau"] * (df["T_ap_int"] / df["dt_days"] - AP_QUIET)

    # Operators: duck during the storm, recover in the following days
    df["true_station_keeping"] = df["station_keeping"]
    df["true_manoeuvre"] = 0.0
    active = df["is_active"] == 1
    for s in storms.to_dict("records"):
        during = active & (df["t_mid"] >= s["main_onset"]) & (df["t_mid"] < s["recovery_end"])
        df.loc[during, "true_manoeuvre"] = -duck_share * df.loc[during, "true_effect"]
        after = (active & (df["t_mid"] >= s["recovery_end"])
                 & (df["t_mid"] < s["recovery_end"] + pd.Timedelta(days=RECOVERY_DAYS)))
        # Loss left after ducking (m, negative), won back evenly over the recovery window
        left = ((df.loc[during, "true_effect"] + df.loc[during, "true_manoeuvre"]) * df.loc[during, "dt_days"])
        lost = left.groupby(df.loc[during, "norad_id"]).sum()
        df.loc[after, "true_manoeuvre"] += (-recover_share * df.loc[after, "norad_id"].map(lost).fillna(0)
                                            / RECOVERY_DAYS)

    # TLE artifacts: perigee swing on oval orbits, bad fits (more in storms)
    days_in = (df["t_mid"] - START) / pd.Timedelta(days=1)
    swing = np.where(df["ecc"] >= 0.01,
                     ECC_SWING * df["ecc"] * np.sin(2 * np.pi * days_in / APSIDAL_DAYS + df["phase"]), 0)
    bad = draw("random") < np.where(df["T_storm"] == 1, BAD_FIT_STORM, BAD_FIT_QUIET)
    jump = np.where(bad, draw("normal", *BAD_FIT_JUMP), 0)
    df["tle_ok"] = (~bad).astype("int64")
    df["true_artifact"] = swing + jump

    season = 1.0 * np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["Y_residual"] = (df["offset"] + gamma * (df["f107"] - 150) + season
                        + df["true_effect"] + df["true_manoeuvre"] + df["true_artifact"]
                        + df["true_station_keeping"]
                        + draw("normal", 0, noise) / np.sqrt(df["dt_days"]))

    # NRLMSISE-00 version: same truth, a slightly different model error (it runs hotter
    # at high Sun UV) and its own fit noise
    df["Y_residual_nrl"] = (df["Y_residual"] + 0.02 * (df["f107"] - 150)
                            + draw("normal", 0, 0.5 * noise) / np.sqrt(df["dt_days"]))

    # Physics-model prediction (m/day, negative) only to build the log ratio
    predicted = -60 * np.exp(-(df["alt_km"] - 400) / 60) * (df["drag_sensitivity"] / 0.01) * df["f107"] / 150
    observed = predicted + df["Y_residual"]
    df["Y_log_ratio"] = np.log((observed / predicted).where(observed < 0))
    # Predicted decay is ~0 at high altitude, so the ratio means nothing there
    df.loc[df["alt_km"] >= HIGH_ALT_KM[0], "Y_log_ratio"] = np.nan

    # Some active satellites have no usable quiet-time fit (manoeuvres), as in real data
    no_fit = objects.loc[(objects["is_active"] == 1) & (rng.random(n_objects) < NO_DRAG_FIT), "norad_id"]
    df.loc[df["norad_id"].isin(no_fit), "drag_sensitivity"] = np.nan

    df = df.drop(columns=["t_start", "t_end", "offset", "phase", "station_keeping"]).sort_values(["norad_id", "t_mid"])
    table = validate(df.reset_index(drop=True))
    table.attrs["truth"] = {"tau0": tau0, "scale_km": scale_km, "gamma": gamma,
                            "duck_share": duck_share, "recover_share": recover_share,
                            "recovery_share": (1 - duck_share) * recover_share,
                            "offset_ratio": duck_share + (1 - duck_share) * recover_share,
                            "ap_quiet": AP_QUIET, "n_high_alt": n_high_alt, "seed": seed}
    if return_hourly:
        return table, storms, hourly_spaceweather(flux, ap)
    return table, storms


def planted_event_profile(table):
    """Mean planted effect (m/day) by whole event_day in the main sample, the curve an
    event study should find."""
    t = main_sample(table)
    t = t[t["event_day"].notna()]
    return t.groupby(np.floor(t["event_day"]).astype(int))["true_effect"].mean()
