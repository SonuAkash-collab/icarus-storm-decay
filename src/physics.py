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
4. B per object and model = median(observed da/dt / driver) over quiet intervals
   (no storm nearby). Every other quiet interval is used to estimate B ("fit"),
   the rest is kept for validation ("holdout").
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
    """B per object (m^2/kg) = median of observed / driver over the 'fit' intervals."""
    fit = rates[rates["role"] == "fit"]
    return (fit["da_dt_km_per_day"] / fit[f"driver_{model}"]).groupby(fit["norad_id"]).median()


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
