"""Physics-model prediction of orbital decay, drag sensitivity, and residual decay.

For a near-circular orbit, air drag shrinks the semi-major axis at the rate
    da/dt = -B * rho * sqrt(mu * a),       B = Cd * A / m  (m^2/kg, "drag sensitivity")
rho is the neutral mass density along the orbit. We take rho from the empirical
atmosphere models in pymsis: NRLMSISE-00 (version=0) and MSIS 2.1 (version=2.1),
driven by our own OMNI space weather (F10.7, 81-day F10.7, 3-hour ap).

Steps per decay interval (two consecutive TLEs of one object, see src/orbit.py):
1. Propagate the orbit with SGP4 at N_SAMPLES evenly spaced times and convert to
   geodetic latitude, longitude, altitude (skyfield). The first half of the samples
   is propagated from the TLE at the start, the second half from the TLE at the end,
   so no sample is far from a TLE epoch.
2. Density rho at every sample from both models.
3. Density driver = -mean(rho * sqrt(mu * a)), in km/day per unit B (m^2/kg).
   The minus sign makes the driver negative like the observed decay, so B > 0 and
   predicted da/dt = B * driver.
4. B per object and model = sum(observed da/dt * dt) / sum(driver * dt) over quiet
   intervals (no storm nearby), i.e. total observed decay / total driver. Every other
   quiet interval is used to estimate B ("fit"), the rest is kept for validation
   ("holdout"). Run remove_outliers first so bad TLE pairs do not pull B.
5. residual = observed - predicted (negative = more decay than the model predicts),
   log_ratio = log(observed / predicted) (> 0 = more decay than predicted).

Air density is a mediator of the storm effect, so it is only used here to build
the physics baseline, never as a control variable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymsis
from skyfield.api import EarthSatellite, load, wgs84

from src.omni_io import PERIOD_END, PERIOD_START, build_spaceweather
from src.orbit import MU_WGS72

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

N_SAMPLES = 24                      # sample times per decay interval
MODELS = {"nrlmsise00": 0, "msis21": 2.1}   # our name -> pymsis version
MU_SI = MU_WGS72 * 1e9              # m^3/s^2
M_PER_S_TO_KM_PER_DAY = 86400 / 1000
QUIET_MARGIN = pd.Timedelta(days=3)

# Built-in time scale data, so skyfield never needs to download anything
_TS = load.timescale(builtin=True)


# ---------------------------------------------------------------------------
# 1. Orbit samples: time, latitude, longitude, altitude
# ---------------------------------------------------------------------------

def sample_times(t_start, t_end, n=N_SAMPLES):
    """n evenly spaced times inside [t_start, t_end], at the centres of n equal bins."""
    frac = (np.arange(n) + 0.5) / n
    return pd.Timestamp(t_start) + (pd.Timestamp(t_end) - pd.Timestamp(t_start)) * frac


def geodetic_track(line1, line2, times):
    """Propagate one TLE with SGP4 to `times` (naive UTC) -> (lat_deg, lon_deg, alt_km).

    skyfield runs SGP4 and rotates the TEME position to the Earth-fixed frame,
    then to WGS84 latitude, longitude and height.
    """
    sat = EarthSatellite(line1, line2, ts=_TS)
    times = pd.DatetimeIndex(times).tz_localize("UTC")
    pos = sat.at(_TS.from_datetimes(list(times.to_pydatetime())))
    lat, lon = wgs84.latlon_of(pos)
    alt = wgs84.height_of(pos)
    return lat.degrees, lon.degrees, alt.km


# ---------------------------------------------------------------------------
# 2. Space-weather inputs and density
# ---------------------------------------------------------------------------

def load_msis_weather(path=None, force=False):
    """Hourly OMNI table padded 3 days before the study period (cached).

    MSIS needs ap up to 57 h and F10.7 one day before each time, so intervals in the
    first days of April need data from late March.
    """
    if path is None:
        path = PROCESSED_DIR / "spaceweather_msis.parquet"
    if force or not Path(path).exists():
        build_spaceweather(start=PERIOD_START - pd.Timedelta(days=3), end=PERIOD_END, out_path=path)
    return pd.read_parquet(path).set_index("time")


def msis_inputs(sw, times):
    """F10.7, 81-day F10.7 and the 7-element ap array that MSIS wants, from our OMNI table.

    sw: hourly table indexed by time with ap (3-hour ap repeated each hour),
        f107 (daily) and f107_81d.
    Returns f107 (n,), f107a (n,), aps (n, 7) where (pymsis docs):
        f107  = daily F10.7 of the previous day
        f107a = 81-day F10.7 mean centred on the day
        aps   = [daily Ap, ap now, ap 3 h before, 6 h before, 9 h before,
                 mean of the eight 3-h ap from 12 to 33 h before,
                 mean of the eight 3-h ap from 36 to 57 h before]
    """
    times = pd.DatetimeIndex(times)
    ap3 = sw["ap"].resample("3h").first()            # one value per 3-hour block
    daily_ap = sw["ap"].resample("D").mean()          # mean of the 8 blocks of the UT day
    daily_f107 = sw["f107"].resample("D").first()
    daily_f107a = sw["f107_81d"].resample("D").first()

    day = times.floor("D")
    block = times.floor("3h")

    def ap_at(hours_before):
        return ap3.reindex(block - pd.Timedelta(hours=hours_before)).to_numpy()

    aps = np.column_stack([
        daily_ap.reindex(day).to_numpy(),
        ap_at(0), ap_at(3), ap_at(6), ap_at(9),
        np.mean([ap_at(h) for h in range(12, 34, 3)], axis=0),   # 12, 15, ..., 33 h
        np.mean([ap_at(h) for h in range(36, 58, 3)], axis=0),   # 36, 39, ..., 57 h
    ])
    f107 = daily_f107.reindex(day - pd.Timedelta(days=1)).to_numpy()
    f107a = daily_f107a.reindex(day).to_numpy()
    if np.isnan(aps).any() or np.isnan(f107).any() or np.isnan(f107a).any():
        raise ValueError("Space-weather table does not cover these times (needs 57 h of history)")
    return f107, f107a, aps


def density(times, lats, lons, alts_km, f107, f107a, aps, version):
    """Neutral mass density (kg/m^3) from pymsis at matching points (fly-through mode).

    geomagnetic_activity=-1 switches MSIS to storm-time mode, which uses the full
    3-hour ap history instead of only the daily Ap.
    """
    dates = pd.DatetimeIndex(times).to_numpy().astype("datetime64[ns]")
    out = pymsis.calculate(dates, lons, lats, alts_km, f107, f107a, aps,
                           version=version, geomagnetic_activity=-1)
    return out[:, pymsis.Variable.MASS_DENSITY]


# ---------------------------------------------------------------------------
# 3. Density driver per decay interval
# ---------------------------------------------------------------------------

def _interval_samples(tles, rates, n=N_SAMPLES):
    """All sample points of all intervals: one row per (interval, sample)."""
    lines = tles.set_index(["norad_id", "epoch"])[["line1", "line2"]]
    rows = []
    for i, r in enumerate(rates.itertuples(index=False)):
        times = sample_times(r.t_start, r.t_end, n)
        half = n // 2
        lat, lon, alt = np.empty(n), np.empty(n), np.empty(n)
        # First half from the start TLE, second half from the end TLE
        for sl, epoch in ((slice(0, half), r.t_start), (slice(half, n), r.t_end)):
            l1, l2 = lines.loc[(r.norad_id, pd.Timestamp(epoch))]
            lat[sl], lon[sl], alt[sl] = geodetic_track(l1, l2, times[sl])
        rows.append(pd.DataFrame({"interval": i, "time": times, "lat": lat, "lon": lon, "alt_km": alt}))
    return pd.concat(rows, ignore_index=True)


def density_drivers(tles, rates, sw, n=N_SAMPLES, cache_path=None, force=False):
    """Add mean altitude, mean density and the density driver per model to `rates`.

    tles: cleaned TLE frame with line1/line2 (src/orbit.py); rates: decay_rates(tles);
    sw: hourly space weather covering the period plus 57 h before (load_msis_weather()).
    New columns: alt_km (mean sample altitude), rho_<model> (mean density, kg/m^3),
    driver_<model> (km/day per m^2/kg, negative).
    Results are cached as Parquet (default data/processed/physics_drivers.parquet);
    the cache is used only if it has exactly the same intervals.
    """
    if cache_path is None:
        cache_path = PROCESSED_DIR / "physics_drivers.parquet"
    keys = ["norad_id", "t_start", "t_end"]
    if Path(cache_path).exists() and not force:
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(rates) and cached[keys].equals(rates[keys].reset_index(drop=True)):
            return cached

    rates = rates.reset_index(drop=True)
    samples = _interval_samples(tles, rates, n)
    f107, f107a, aps = msis_inputs(sw, samples["time"])

    out = rates.copy()
    grouped = samples.groupby("interval")
    out["alt_km"] = grouped["alt_km"].mean().to_numpy()
    # a for sqrt(mu a): mean of the Brouwer a at the two ends, in metres
    a_m = 1000 * (tles.set_index(["norad_id", "epoch"])["a_km"]
                  .reindex(pd.MultiIndex.from_frame(rates[["norad_id", "t_start"]])).to_numpy()
                  + rates["da_km"].to_numpy() / 2)
    for name, version in MODELS.items():
        samples[f"rho_{name}"] = density(samples["time"], samples["lat"], samples["lon"],
                                         samples["alt_km"], f107, f107a, aps, version)
        rho_mean = samples.groupby("interval")[f"rho_{name}"].mean().to_numpy()
        out[f"rho_{name}"] = rho_mean
        # rho * sqrt(mu a) is in m/s per (m^2/kg); convert to km/day
        out[f"driver_{name}"] = -rho_mean * np.sqrt(MU_SI * a_m) * M_PER_S_TO_KM_PER_DAY

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return out


# ---------------------------------------------------------------------------
# 4. Quiet intervals, drag sensitivity B, prediction and residual
# ---------------------------------------------------------------------------

def storm_windows(storms, margin=QUIET_MARGIN):
    """(start, end) per storm: 3 days before the storm to 3 days after recovery_end.

    The window starts at the earlier of onset_time and main_onset, so a weaker dip
    merged into the same event (e.g. Oct 6 2024 before the Oct 10 main phase) is
    not counted as quiet.
    """
    start = storms[["onset_time", "main_onset"]].min(axis=1) - margin
    end = storms["recovery_end"] + margin
    return list(zip(start, end))


def label_intervals(rates, storms, margin=QUIET_MARGIN):
    """Column `role`: 'fit' / 'holdout' for quiet intervals (alternating in time per
    object), 'storm' for intervals that overlap any storm window."""
    rates = rates.copy()
    quiet = np.ones(len(rates), dtype=bool)
    for start, end in storm_windows(storms, margin):
        quiet &= ~((rates["t_end"] > start) & (rates["t_start"] < end)).to_numpy()
    rates["role"] = "storm"
    for _, g in rates[quiet].sort_values("t_start").groupby("norad_id"):
        rates.loc[g.index, "role"] = np.where(np.arange(len(g)) % 2 == 0, "fit", "holdout")
    return rates


def fit_drag_sensitivity(rates, model):
    """B per object (m^2/kg) over the 'fit' intervals, dt-weighted ratio of sums:
    B = sum(observed * dt) / sum(driver * dt).

    The earlier median of observed / driver weighted every interval the same, short
    noisy ones included, and came out ~3% low for NORAD 41341: its held-out quiet
    residual was -4.6 m/day (t = -2.2). The ratio of sums makes the mean residual on
    the fit intervals exactly zero and the held-out one ~0 (notebooks/04).
    """
    fit = rates[rates["role"] == "fit"]
    obs = (fit["da_dt_km_per_day"] * fit["dt_days"]).groupby(fit["norad_id"]).sum()
    drv = (fit[f"driver_{model}"] * fit["dt_days"]).groupby(fit["norad_id"]).sum()
    return obs / drv


def add_predictions(rates, model, B=None):
    """Add B_<model>, pred_<model>, resid_<model>, log_ratio_<model> (B fitted if not given)."""
    rates = rates.copy()
    if B is None:
        B = fit_drag_sensitivity(rates, model)
    rates[f"B_{model}"] = rates["norad_id"].map(B)
    pred = rates[f"B_{model}"] * rates[f"driver_{model}"]
    obs = rates["da_dt_km_per_day"]
    rates[f"pred_{model}"] = pred
    rates[f"resid_{model}"] = obs - pred
    # log ratio only defined while the object is observed to decay
    rates[f"log_ratio_{model}"] = np.log((obs / pred).where(obs < 0))
    return rates


def run_physics(tles, rates, sw, storms, **kwargs):
    """Drivers + labels + predictions for every model in one call."""
    out = label_intervals(density_drivers(tles, rates, sw, **kwargs), storms)
    for model in MODELS:
        out = add_predictions(out, model)
    return out


# ---------------------------------------------------------------------------
# 5. Summaries
# ---------------------------------------------------------------------------

def mean_se(x):
    """Mean and naive standard error (std / sqrt(n)) of a Series, NaNs dropped.

    Consecutive TLE intervals are not independent, so read the SE as a lower bound.
    """
    x = pd.Series(x).dropna()
    return x.mean(), x.std() / np.sqrt(len(x)), len(x)


def block_se(values, times, block="7D"):
    """Standard error of the mean from block means (default: calendar weeks).

    Residuals of neighbouring intervals are correlated (the model bias drifts slowly
    with season and solar activity), so the naive SE is too small. Averaging within
    weekly blocks first and using the scatter of the block means is closer to the
    'cluster by time block' rule.
    """
    blocks = pd.Series(np.asarray(values), index=pd.DatetimeIndex(times)).dropna()
    means = blocks.groupby(blocks.index.floor(block)).mean()
    return means.std() / np.sqrt(len(means))


def holdout_residual(rates, model):
    """Mean residual (km/day) on the held-out quiet intervals per object, with the
    naive SE and the weekly-block SE (use the block SE for inference)."""
    rows = []
    for norad_id, g in rates[rates["role"] == "holdout"].groupby("norad_id"):
        m, se, n = mean_se(g[f"resid_{model}"])
        rows.append({"norad_id": norad_id, "model": model, "mean_resid": m,
                     "se_naive": se, "se_block": block_se(g[f"resid_{model}"], g["t_mid"]), "n": n})
    return pd.DataFrame(rows)


def window_residual(rates, model, t0, days=3):
    """Decay in [t0, t0 + days] per object: time-weighted mean observed, predicted, residual.

    Uses intervals whose midpoint lies inside the window. Time-weighted mean =
    sum(x * dt) / sum(dt). SE from the scatter of the interval residuals (lower bound).
    """
    t0 = pd.Timestamp(t0)
    win = rates[(rates["t_mid"] >= t0) & (rates["t_mid"] <= t0 + pd.Timedelta(days=days))]
    rows = []
    for norad_id, g in win.groupby("norad_id"):
        w = g["dt_days"] / g["dt_days"].sum()
        obs = (w * g["da_dt_km_per_day"]).sum()
        pred = (w * g[f"pred_{model}"]).sum()
        rows.append({"norad_id": norad_id, "model": model, "obs": obs, "pred": pred,
                     "resid": obs - pred, "se": mean_se(g[f"resid_{model}"])[1],
                     "obs_over_pred": obs / pred, "n": len(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Residual validation: outlier filter, quiet-ap prediction, TLE-blur check
# ---------------------------------------------------------------------------

# Hampel settings: with ~2.5 intervals per day a 5-day window holds ~12 points. Shorter
# windows make the MAD itself noisy: 3 days and k = 3 flag ~4% of pure normal noise,
# 5 days and k = 3.5 about 1% (tests/test_physics.py, notebooks/04).
HAMPEL_WINDOW = "5D"     # centred rolling window (time-based, TLE spacing is irregular)
HAMPEL_K = 3.5           # flag if more than k robust SDs from the rolling median
MAD_TO_SD = 1.4826       # MAD * 1.4826 = SD for normal noise
QUIET_AP_DAYS = (-10, -3)     # days around main_onset: storm's quiet ap and baseline window
POST_DAYS = (0, 3)            # post-storm window of the baseline-relative residual


def hampel_flags(values, times, window=HAMPEL_WINDOW, k=HAMPEL_K, min_periods=5):
    """Hampel filter: True where a value is an outlier against its rolling median.

    For every point, take the points within +/- window/2 in time, their median m and
    MAD (median absolute deviation). The point is an outlier if
    |x - m| > k * 1.4826 * MAD. Points with fewer than min_periods neighbours are kept.
    """
    s = pd.Series(np.asarray(values, float), index=pd.DatetimeIndex(times))
    order = np.argsort(s.index.to_numpy(), kind="stable")
    s = s.iloc[order]
    roll = s.rolling(window, center=True, min_periods=min_periods)
    med = roll.median()
    mad = roll.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    flags = ((s - med).abs() > k * MAD_TO_SD * mad).to_numpy()
    out = np.zeros(len(s), dtype=bool)
    out[order] = flags
    return out


def flag_outliers(rates, model="msis21", **kwargs):
    """Column `outlier`: Hampel filter on the residual of `model`, per object.

    The filter runs on the residual (observed - predicted), not on the observed decay:
    the observed decay jumps at every storm onset and a rolling median would flag the
    real storm response, while the residual mostly jumps for bad TLE pairs.
    """
    rates = rates.copy()
    rates["outlier"] = False
    for _, g in rates.groupby("norad_id"):
        rates.loc[g.index, "outlier"] = hampel_flags(g[f"resid_{model}"], g["t_mid"], **kwargs)
    return rates


def remove_outliers(rates, model="msis21", **kwargs):
    """Drop Hampel outliers and refit B and the predictions of every model without them.

    Returns (clean rates, report); report gives the intervals removed per object and
    role (fit / holdout / storm) and the share removed.
    """
    flagged = flag_outliers(rates, model, **kwargs)
    report = (flagged.groupby(["norad_id", "role"])["outlier"].agg(["sum", "size", "mean"])
              .rename(columns={"sum": "removed", "size": "intervals", "mean": "share"}).reset_index())
    clean = flagged[~flagged["outlier"]].drop(columns="outlier")
    for name in MODELS:
        clean = add_predictions(clean, name)
    return clean, report


def quiet_ap(sw, storms, days=QUIET_AP_DAYS):
    """Quiet ap per storm (Series by storm_id): median hourly ap over days -10..-3 of main_onset.

    ap is a global index, so one value per storm.
    """
    lo, hi = (pd.Timedelta(days=d) for d in days)
    return pd.Series({s["storm_id"]: sw.loc[s["main_onset"] + lo:s["main_onset"] + hi, "ap"].median()
                      for s in storms.to_dict("records")}, name="quiet_ap")


def quiet_ap_drivers(tles, rates, sw, storms, models=("msis21",), n=N_SAMPLES, cache_path=None, force=False):
    """Density driver with the storm's quiet ap instead of the actual ap (actual F10.7 kept).

    Each interval is linked to a storm as in schema.link_storms (event window -10..+20
    days around main_onset, nearest onset wins). For linked intervals all 7 ap inputs
    of MSIS are set to that storm's quiet_ap; intervals outside every event window keep
    the actual ap. New columns: storm_id, quiet_ap, driver_quiet_<model> (km/day per
    m^2/kg). Cached like density_drivers (default data/processed/physics_drivers_quietap.parquet).
    """
    from src.schema import link_storms

    if cache_path is None:
        cache_path = PROCESSED_DIR / "physics_drivers_quietap.parquet"
    keys = ["norad_id", "t_start", "t_end"]
    cols = ["storm_id", "quiet_ap"] + [f"driver_quiet_{m}" for m in models]
    rates = rates.reset_index(drop=True)
    if Path(cache_path).exists() and not force:
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(rates) and cached[keys].equals(rates[keys]) and set(cols) <= set(cached):
            return rates.assign(**{c: cached[c].to_numpy() for c in cols})

    out = rates.copy()
    out["storm_id"] = link_storms(rates["t_start"], rates["t_end"], storms)["storm_id"].to_numpy()
    out["quiet_ap"] = out["storm_id"].map(quiet_ap(sw, storms)).astype(float)

    samples = _interval_samples(tles, rates, n)
    f107, f107a, aps = msis_inputs(sw, samples["time"])
    ap_q = out["quiet_ap"].to_numpy()[samples["interval"].to_numpy()]
    linked = ~np.isnan(ap_q)
    aps[linked] = ap_q[linked, None]
    # a for sqrt(mu a), as in density_drivers
    a_m = 1000 * (tles.set_index(["norad_id", "epoch"])["a_km"]
                  .reindex(pd.MultiIndex.from_frame(rates[["norad_id", "t_start"]])).to_numpy()
                  + rates["da_km"].to_numpy() / 2)
    for name in models:
        rho = density(samples["time"], samples["lat"], samples["lon"], samples["alt_km"],
                      f107, f107a, aps, MODELS[name])
        rho_mean = pd.Series(rho).groupby(samples["interval"]).mean().to_numpy()
        out[f"driver_quiet_{name}"] = -rho_mean * np.sqrt(MU_SI * a_m) * M_PER_S_TO_KM_PER_DAY

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    out[keys + cols].to_parquet(cache_path, index=False)
    return out


def add_total_effect(rates, model="msis21"):
    """Add pred_quiet_<model> and total_<model> = observed - B * driver_quiet (km/day).

    total_<model> is the total storm effect: negative = extra decay beyond what a
    quiet-ap atmosphere with the same Sun UV would cause. Uses B_<model>, so run
    add_predictions first. schema's Y_total = total_msis21 * 1000 (m/day).
    """
    rates = rates.copy()
    rates[f"pred_quiet_{model}"] = rates[f"B_{model}"] * rates[f"driver_quiet_{model}"]
    rates[f"total_{model}"] = rates["da_dt_km_per_day"] - rates[f"pred_quiet_{model}"]
    return rates


def smooth_driver(rates, model, days):
    """Column driver_<model>_s<days>: the density driver averaged over a centred window.

    A TLE is fitted to several days of tracking, so the observed decay is a blurred
    version of the true decay. Smoothing the prediction by the same amount asks
    whether a residual is only blur. Time-weighted (by dt_days) rolling mean, per object.
    Use it with add_predictions(rates, f"{model}_s{days}").
    """
    rates = rates.copy()
    name = f"driver_{model}_s{days:g}"
    rates[name] = np.nan
    for _, g in rates.sort_values("t_mid").groupby("norad_id"):
        x = pd.DataFrame({"xw": (g[f"driver_{model}"] * g["dt_days"]).to_numpy(),
                          "w": g["dt_days"].to_numpy()}, index=pd.DatetimeIndex(g["t_mid"]))
        roll = x.rolling(pd.Timedelta(days=days), center=True, min_periods=1).sum()
        rates.loc[g.index, name] = (roll["xw"] / roll["w"]).to_numpy()
    return rates


def shift_driver(rates, model, hours):
    """Column driver_<model>_lag<hours>: the driver from `hours` earlier (per object,
    linear interpolation between interval midpoints; NaN beyond the ends).
    Positive lag = the observed decay responds later than the model."""
    rates = rates.copy()
    name = f"driver_{model}_lag{hours:g}"
    rates[name] = np.nan
    for _, g in rates.sort_values("t_mid").groupby("norad_id"):
        t = g["t_mid"].astype("int64").to_numpy() / 3.6e12          # hours
        rates.loc[g.index, name] = np.interp(t - hours, t, g[f"driver_{model}"].to_numpy(),
                                             left=np.nan, right=np.nan)
    return rates


def lag_correlation(rates, model, lags=range(-72, 73, 6)):
    """Correlation of the observed decay with the driver shifted by each lag (hours).
    Returns a frame: one row per lag, one column per object."""
    out = {}
    for lag in lags:
        x = shift_driver(rates, model, lag)[f"driver_{model}_lag{lag:g}"]
        out[lag] = {nid: g["da_dt_km_per_day"].corr(x[g.index]) for nid, g in rates.groupby("norad_id")}
    return pd.DataFrame(out).T.rename_axis("lag_h")


def _weighted_mean(x, w):
    ok = x.notna()
    return (x[ok] * w[ok]).sum() / w[ok].sum() if ok.any() else np.nan


def storm_baseline_residual(rates, storms, col, pre=QUIET_AP_DAYS, post=POST_DAYS):
    """Baseline-relative residual per object and storm (same units as `col`).

    change = time-weighted mean of col over days `post` (0..+3 of main_onset) minus
    the same over days `pre` (-10..-3). Positive change in resid_<model> = the model
    over-predicts decay after the storm, relative to its own pre-storm error. Uses
    intervals whose midpoint is in the window and that do not bridge main_onset. For
    resid_ columns also gives obs_over_pred in the post window. se is a rough SE of
    the change from the interval scatter (lower bound: neighbours are correlated).
    """
    rows = []
    for s in storms.to_dict("records"):
        day = (rates["t_mid"] - s["main_onset"]).dt.total_seconds() / 86400
        bridge = (rates["t_start"] < s["main_onset"]) & (rates["t_end"] > s["main_onset"])
        for norad_id, g in rates[~bridge].groupby("norad_id"):
            d = day[g.index]
            a, b = g[d.between(*pre)], g[d.between(*post)]
            row = {"storm_id": s["storm_id"], "norad_id": norad_id, "class": s["class"],
                   "min_dst": s["min_dst"], "pre": _weighted_mean(a[col], a["dt_days"]),
                   "post": _weighted_mean(b[col], b["dt_days"]), "n_pre": len(a), "n_post": len(b),
                   # rough SE of the change from the interval scatter (a lower bound)
                   "se": np.sqrt(a[col].var() / len(a) + b[col].var() / len(b)) if len(a) and len(b) else np.nan}
            if col.startswith("resid_"):
                pred = col.replace("resid_", "pred_")
                row["obs_over_pred"] = (_weighted_mean(b["da_dt_km_per_day"], b["dt_days"])
                                        / _weighted_mean(b[pred], b["dt_days"]))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["change"] = out["post"] - out["pre"]
    return out


def storm_mean(per_storm, value="change", exclude=()):
    """Mean over storms of `value` (objects averaged within each storm first), with the
    SE across storms (storm = cluster) and a t-based 95% interval (n - 1 df)."""
    from scipy import stats

    by_storm = per_storm[~per_storm["storm_id"].isin(exclude)].groupby("storm_id")[value].mean().dropna()
    n = len(by_storm)
    se = by_storm.std() / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    return pd.Series({"mean": by_storm.mean(), "se": se, "ci_lo": by_storm.mean() - half,
                      "ci_hi": by_storm.mean() + half, "storms": n,
                      "positive": int((by_storm > 0).sum())})


def post_storm_verdict(raw_all, raw_without, smooth_all, extremes):
    """Apply the PLAN.md decision rules to the post-storm over-prediction.

    raw_all, raw_without, smooth_all: storm_mean results (raw prediction with all 20
    storms, raw without May/Oct 2024, most-smoothed prediction with all storms).
    extremes: one row per (extreme storm, object) with change and se of the
    most-smoothed prediction. "Holds" = 95% interval above zero / change > 2 SE.
    Returns (label, explanation).
    """
    holds = lambda s: s["ci_lo"] > 0     # noqa: E731
    survive = extremes["change"] > 2 * extremes["se"]
    if holds(raw_without):
        if holds(smooth_all):
            return "models recover too slowly after storms", "holds without May/Oct and after smoothing"
        return ("TLE blur", "holds without May/Oct but not after smoothing: switch the outcome to "
                "total altitude lost from a calm day before to a calm day after the storm")
    if holds(raw_all):
        return "extreme-storm driven", "holds across all storms only because of May/Oct 2024"
    return ("extreme-storm driven",
            f"no over-prediction across the {int(raw_all['storms'])} storms (95% interval "
            f"{raw_all['ci_lo']:.1f} to {raw_all['ci_hi']:.1f}); seen only in the extreme storms, "
            f"where {int(survive.sum())} of {len(extremes)} storm-object pairs survive the strongest smoothing")


# ---------------------------------------------------------------------------
# 7. Extreme-storm check on the cohort: table rows as rates, onset gaps, calm-to-calm loss
# ---------------------------------------------------------------------------

QUIET_DAY_AP = 27            # calm UT day: every 3-hour ap of the day below this (Kp below 4)
CALM_AFTER_MAX_DAYS = 7      # look this many days past recovery_end for the calm day after
MIN_COVERAGE = 0.95          # share of the calm-to-calm span the intervals must cover
CALM_BOOT = 2000             # bootstrap draws in calm_summary
CALM_QUIET_DAYS = 30         # quiet-interval baseline of calm_to_calm: days -30..+30 of main_onset


def table_rates(table):
    """Analysis-table rows (m/day) as a `rates` frame (km/day) for the functions above.

    Columns: norad_id, t_start, t_end (t_mid -+ dt/2 if the table has none), t_mid,
    dt_days, da_dt_km_per_day and, for msis21 and nrlmsise00, pred_/resid_<model>,
    B_<model> = 1 and driver_<model> = the prediction itself. The table stores no density
    driver, but B is constant per object, so smoothing or shifting the prediction is the
    same as smoothing the driver, and add_predictions on a smoothed driver refits a factor
    ~1 on the 'fit' intervals (notebook 04 refits B the same way). role, alt_km and
    hampel_outlier are kept when present.
    """
    t = table.sort_values(["norad_id", "t_mid"]).reset_index(drop=True)
    half = pd.to_timedelta(t["dt_days"] / 2, unit="D")
    obs = t["obs_decay"] / 1000
    resid = {"msis21": t["Y_residual"] / 1000, "nrlmsise00": t["Y_residual_nrl"] / 1000}
    out = pd.DataFrame({
        "norad_id": t["norad_id"],
        "t_start": t["t_start"] if "t_start" in t else t["t_mid"] - half,
        "t_end": t["t_end"] if "t_end" in t else t["t_mid"] + half,
        "t_mid": t["t_mid"], "dt_days": t["dt_days"], "da_dt_km_per_day": obs})
    for model, r in resid.items():
        out[f"pred_{model}"] = obs - r
        out[f"resid_{model}"] = r
        out[f"driver_{model}"] = out[f"pred_{model}"]
        out[f"B_{model}"] = 1.0
    for col in ("role", "alt_km", "hampel_outlier"):
        if col in t:
            out[col] = t[col]
    if "hampel_outlier" in out:
        out["hampel_outlier"] = out["hampel_outlier"].astype(bool)
    return out


def onset_gaps(rates, tol=pd.Timedelta(seconds=1)):
    """Holes in each object's chain of intervals (t_end of one is not t_start of the next).

    On the cohort these are the intervals task A4 left out because they bridge a
    main_onset. The calm-to-calm loss must cover them, or the decay inside them would be
    missing. Holes shorter than tol are rounding (t_start/t_end rebuilt from t_mid).
    Returns norad_id, t_start, t_end of each hole.
    """
    r = rates.sort_values(["norad_id", "t_start"])
    nxt = r.groupby("norad_id")["t_start"].shift(-1)
    hole = nxt.notna() & (nxt - r["t_end"] > tol)
    return pd.DataFrame({"norad_id": r.loc[hole, "norad_id"].to_numpy(),
                         "t_start": r.loc[hole, "t_end"].to_numpy(),
                         "t_end": nxt[hole].to_numpy()})


def fill_onset_gaps(gaps, tles, sw, B, model="msis21", cache_path=None):
    """Rates rows for the onset gaps: observed decay from the two TLEs at the ends,
    prediction = B * density driver over the gap (density_drivers, cached).

    These rows are only for the calm-to-calm loss (they bridge main_onset, so they never
    enter the baseline-relative residual). B: drag sensitivity per norad_id (m^2/kg).
    """
    if cache_path is None:
        cache_path = PROCESSED_DIR / "physics_onset_gaps.parquet"
    a = tles.set_index(["norad_id", "epoch"])["a_km"]
    g = gaps.copy()
    a0 = a.reindex(pd.MultiIndex.from_frame(g[["norad_id", "t_start"]].rename(columns={"t_start": "epoch"})))
    a1 = a.reindex(pd.MultiIndex.from_frame(g[["norad_id", "t_end"]].rename(columns={"t_end": "epoch"})))
    g["da_km"] = a1.to_numpy() - a0.to_numpy()
    g["dt_days"] = (g["t_end"] - g["t_start"]).dt.total_seconds() / 86400
    g["t_mid"] = g["t_start"] + (g["t_end"] - g["t_start"]) / 2
    g = g.dropna(subset=["da_km"]).reset_index(drop=True)
    g = density_drivers(tles, g, sw, cache_path=cache_path)
    g["da_dt_km_per_day"] = g["da_km"] / g["dt_days"]
    g[f"pred_{model}"] = g["norad_id"].map(B) * g[f"driver_{model}"]
    g[f"resid_{model}"] = g["da_dt_km_per_day"] - g[f"pred_{model}"]
    g["onset_gap"] = True
    return g.drop(columns=[c for c in g if c.startswith("rho_")])


def calm_days(sw, quiet_ap_max=QUIET_DAY_AP):
    """UT days on which every 3-hour ap stays below quiet_ap_max (index: day, value: bool)."""
    ap = sw["ap"] if isinstance(sw.index, pd.DatetimeIndex) else sw.set_index("time")["ap"]
    return ap.resample("D").max() < quiet_ap_max


def calm_spans(storms, calm, pre=QUIET_AP_DAYS, max_after_days=CALM_AFTER_MAX_DAYS, extra_days=0):
    """Per storm_id: candidate calm days before (latest first) and after (earliest first).

    Before: days pre (-10..-3) of main_onset that end before the storm starts (earlier of
    onset_time and main_onset) and start after the previous storm's recovery_end.
    After: calm days from recovery_end + extra_days, at most max_after_days later, that
    end before the next storm starts. Overlapping storms can leave either list empty.
    """
    s = storms.sort_values("main_onset").reset_index(drop=True)
    begin = s[["onset_time", "main_onset"]].min(axis=1)
    days = calm.index[calm.to_numpy()]
    one = pd.Timedelta(days=1)
    out = {}
    for k, r in s.iterrows():
        prev_end = s.loc[k - 1, "recovery_end"] if k > 0 else pd.Timestamp.min
        next_begin = begin[k + 1] if k + 1 < len(s) else pd.Timestamp.max
        d0 = r["main_onset"].normalize()
        before = [d for d in days if d0 + pre[0] * one <= d <= d0 + pre[1] * one
                  and d + one <= begin[k] and d >= prev_end][::-1]
        lo = r["recovery_end"] + extra_days * one
        after = [d for d in days if lo <= d <= lo + max_after_days * one and d + one <= next_begin]
        out[r["storm_id"]] = (before, after)
    return out


def _epoch_on(epochs, days, pick):
    """First day of `days` holding an epoch, and its last (pick=max) or first (min) epoch."""
    one = np.timedelta64(1, "D")
    for d in days:
        on = epochs[(epochs >= np.datetime64(d)) & (epochs < np.datetime64(d) + one)]
        if len(on):
            return pick(on)
    return None


def calm_to_calm(rates, storms, sw, model="msis21", quiet_ap_max=QUIET_DAY_AP, pre=QUIET_AP_DAYS,
                 max_after_days=CALM_AFTER_MAX_DAYS, extra_days=0, min_coverage=MIN_COVERAGE,
                 quiet_days=CALM_QUIET_DAYS):
    """Blur-free storm outcome: semi-major axis lost from a calm day before to a calm day after.

    A TLE spreads the storm's decay over the following days, but it cannot change the
    total: summed over a chain of intervals, da/dt * dt telescopes to a(end) - a(start).
    So, per object and storm, from the object's last TLE epoch on the last calm day in
    days -10..-3 (t0) to its first epoch on the first calm day after recovery_end (t1):
        obs_m  = sum of observed da/dt * dt   (= a(t1) - a(t0), metres, < 0 = loss)
        pred_m = sum of predicted da/dt * dt  (pred_<model>, same intervals)
        diff_m = obs_m - pred_m               (> 0: less loss than the model predicts)
        diff_adj_m = diff_m - span_days * pre_resid, where pre_resid (m/day) is the
            object's dt-weighted mean residual over days -10..-3, after the previous
            storm's recovery_end and before this storm starts (the baseline idea of
            storm_baseline_residual, kept free of the previous storm)
        diff_quiet_m = diff_m - span_days * quiet_resid, quiet_resid = the object's mean
            residual over its quiet intervals (role fit / holdout) within quiet_days of
            main_onset: a steadier baseline than the few days before the storm
        obs_over_pred = obs_m / pred_m        (< 1: the model over-predicts the loss)
    Calm day = every 3-hour ap below quiet_ap_max (calm_days, calm_spans). rates must hold
    every interval of the chain, Hampel outliers and the onset gaps included
    (fill_onset_gaps), since TLE noise cancels in the sum; pairs whose intervals cover less
    than min_coverage of the span are dropped. extra_days pushes the calm day after later
    (sensitivity: a TLE fitted on the first calm day still uses some storm-time tracking).
    Returns one row per (storm, object); alt_km = dt-weighted mean over the span.
    """
    spans = calm_spans(storms, calm_days(sw, quiet_ap_max), pre, max_after_days, extra_days)
    info = storms.set_index("storm_id")
    begin = info[["onset_time", "main_onset"]].min(axis=1)
    order = storms.sort_values("main_onset")
    # previous storm's recovery_end (a year before the first storm for the first one)
    first = order["main_onset"].min() - pd.Timedelta(days=365)
    prev_end = pd.Series(order["recovery_end"].shift(1).fillna(first).to_numpy(), index=order["storm_id"])
    rows = []
    for norad_id, g in rates.sort_values("t_start").groupby("norad_id"):
        ts, te = g["t_start"].to_numpy(), g["t_end"].to_numpy()
        tm = g["t_mid"].to_numpy()
        epochs = np.unique(np.concatenate([ts, te]))
        dt = g["dt_days"].to_numpy()
        obs_dt = (g["da_dt_km_per_day"] * g["dt_days"]).to_numpy()
        pred_dt = (g[f"pred_{model}"] * g["dt_days"]).to_numpy()
        resid = g[f"resid_{model}"].to_numpy()
        alt = g["alt_km"].to_numpy() if "alt_km" in g else np.full(len(g), np.nan)
        quiet = g["role"].isin(["fit", "holdout"]).to_numpy() if "role" in g else np.zeros(len(g), bool)
        # rows allowed in the pre-storm baseline: not an onset gap, not a Hampel outlier
        base = np.ones(len(g), bool)
        for flag in ("onset_gap", "hampel_outlier"):
            if flag in g:
                base &= ~g[flag].astype(bool).to_numpy()
        for sid, (before, after) in spans.items():
            t0, t1 = _epoch_on(epochs, before, np.max), _epoch_on(epochs, after, np.min)
            if t0 is None or t1 is None:
                continue
            # 1 s slack: t_start / t_end rebuilt from t_mid differ from the epochs by rounding
            inside = (ts >= t0 - np.timedelta64(1, "s")) & (te <= t1 + np.timedelta64(1, "s"))
            span = (t1 - t0) / np.timedelta64(1, "D")
            coverage = dt[inside].sum() / span
            if coverage < min_coverage:
                continue
            day = (tm - np.datetime64(info.loc[sid, "main_onset"])) / np.timedelta64(1, "D")
            p = (base & (day >= pre[0]) & (day <= pre[1]) & (te <= np.datetime64(begin[sid]))
                 & (ts >= np.datetime64(prev_end[sid])))
            pre_resid = 1000 * np.sum(resid[p] * dt[p]) / dt[p].sum() if p.any() else np.nan
            q = base & quiet & (np.abs(day) <= quiet_days)
            quiet_resid = 1000 * np.sum(resid[q] * dt[q]) / dt[q].sum() if q.any() else np.nan
            obs, pred = 1000 * obs_dt[inside].sum(), 1000 * pred_dt[inside].sum()
            rows.append({"storm_id": sid, "norad_id": norad_id, "class": info.loc[sid, "class"],
                         "min_dst": info["min_dst"].get(sid, np.nan) if "min_dst" in info else np.nan,
                         "t0": pd.Timestamp(t0), "t1": pd.Timestamp(t1),
                         "span_days": span, "coverage": coverage, "obs_m": obs, "pred_m": pred,
                         "diff_m": obs - pred, "pre_resid_m_per_day": pre_resid,
                         "diff_adj_m": obs - pred - span * pre_resid, "quiet_resid_m_per_day": quiet_resid,
                         "diff_quiet_m": obs - pred - span * quiet_resid, "obs_over_pred": obs / pred,
                         "alt_km": np.sum(alt[inside] * dt[inside]) / dt[inside].sum()})
    return pd.DataFrame(rows)


CALM_MEANS = {"diff": "diff_m", "adj": "diff_adj_m", "quiet": "diff_quiet_m"}   # summary name -> column


def _calm_stats(obs, pred, means, storm, idx):
    """[ratio = sum obs / sum pred] + each column of `means` averaged over objects, then
    storms. storm: integer codes 0..k-1 (NaN values are left out of the means)."""
    out = [obs[idx].sum() / pred[idx].sum()]
    for v in means.values():
        x, st = v[idx], storm[idx]
        ok = np.isfinite(x)
        n = np.bincount(st[ok], minlength=storm.max() + 1)
        tot = np.bincount(st[ok], weights=x[ok], minlength=storm.max() + 1)
        out.append(np.mean(tot[n > 0] / n[n > 0]) if (n > 0).any() else np.nan)
    return out


def calm_summary(per, by="class", n_boot=CALM_BOOT, seed=0):
    """Observed / predicted loss and the difference per group of calm_to_calm rows.

    ratio = sum of observed loss / sum of predicted loss; diff_m, diff_adj_m and
    diff_quiet_m = mean over storms of the per-storm mean over objects (metres), with
    95% bounds ratio_lo/hi, diff_lo/hi, adj_lo/hi, quiet_lo/hi. 95% intervals from a cluster
    bootstrap: whole storms when the group has >= 3 storms (storm = cluster), otherwise
    objects resampled within each storm (object = cluster; with one or two storms an
    interval over storms means nothing). by: column name or list (e.g. "storm_id",
    "alt_band", ["class", "alt_band"]).
    """
    rng = np.random.default_rng(seed)
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, g in per.dropna(subset=["obs_m", "pred_m"]).groupby(by, observed=True):
        g = g.reset_index(drop=True)
        ids, storm = np.unique(g["storm_id"].to_numpy(), return_inverse=True)
        obs, pred = g["obs_m"].to_numpy(), g["pred_m"].to_numpy()
        means = {k: g[c].to_numpy(float) for k, c in CALM_MEANS.items() if c in g}
        members = {s: np.flatnonzero(storm == s) for s in range(len(ids))}
        if len(ids) >= 3:
            basis = "storms"
            draw = lambda: np.concatenate([members[s] for s in rng.choice(len(ids), len(ids))])   # noqa: E731
        else:
            basis = "objects"
            draw = lambda: np.concatenate([rng.choice(m, len(m)) for m in members.values()])  # noqa: E731
        point = _calm_stats(obs, pred, means, storm, np.arange(len(g)))
        boot = np.array([_calm_stats(obs, pred, means, storm, draw()) for _ in range(n_boot)])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        key = key if isinstance(key, tuple) else (key,)
        row = {**dict(zip(by, key)), "obs_m": g["obs_m"].mean(), "pred_m": g["pred_m"].mean(),
               "ratio": point[0], "ratio_lo": lo[0], "ratio_hi": hi[0]}
        for j, k in enumerate(means, start=1):
            row.update({CALM_MEANS[k]: point[j], f"{k}_lo": lo[j], f"{k}_hi": hi[j]})
        rows.append({**row, "span_days": g["span_days"].mean(), "n_storms": len(ids),
                     "n_objects": g["norad_id"].nunique(), "n_pairs": len(g), "ci_basis": basis})
    return pd.DataFrame(rows)


def per_storm_change(per_storm, value="change"):
    """Mean over objects of a storm_baseline_residual column per storm, SE over objects
    (object = cluster) and the 95% normal interval."""
    g = per_storm.dropna(subset=[value]).groupby("storm_id")[value]
    out = pd.DataFrame({"mean": g.mean(), "se": g.std() / np.sqrt(g.count()), "n_objects": g.count()})
    out["ci_lo"], out["ci_hi"] = out["mean"] - 1.96 * out["se"], out["mean"] + 1.96 * out["se"]
    return out


def storm_check_label(raw, smooth, calm):
    """Label one storm from three 95% intervals: raw and smooth = per_storm_change rows
    (ci_lo) of the residual change with the raw and the most-smoothed prediction, calm =
    calm_summary row (adj_lo) of the calm-to-calm loss.

    PLAN.md rules: the over-prediction is a real model overshoot only if it survives the
    smoothing AND shows in the calm-to-calm loss (observed loss smaller than predicted
    relative to the object's own model error: diff_adj_m > 0 and, when present,
    diff_quiet_m > 0); neither = TLE blur; one = mixed.
    """
    if not raw["ci_lo"] > 0:
        return "no over-prediction"
    # the calm-to-calm loss must be smaller than predicted under both baselines
    smooth_ok = smooth["ci_lo"] > 0
    calm_ok = calm["adj_lo"] > 0 and calm.get("quiet_lo", np.inf) > 0
    if smooth_ok and calm_ok:
        return "real model overshoot"
    if not smooth_ok and not calm_ok:
        return "TLE blur"
    return "mixed: " + ("survives smoothing, not in calm-to-calm" if smooth_ok
                        else "in calm-to-calm, gone after smoothing")


def extreme_storm_verdict(labels):
    """Overall verdict from storm_check_label of the extreme storms, {name: label}
    (May and Oct 2024): "real model overshoot in extreme storms", "TLE blur", or
    "mixed (...)" naming each storm's label."""
    vals = set(labels.values())
    if vals == {"real model overshoot"}:
        return "real model overshoot in extreme storms"
    if vals <= {"TLE blur", "no over-prediction"}:
        return "TLE blur"
    return "mixed (" + "; ".join(f"{k}: {v.removeprefix('mixed: ')}" for k, v in labels.items()) + ")"
