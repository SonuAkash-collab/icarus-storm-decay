"""Tests for the physics-model prediction (src/physics.py).

The density tests use fixed space-weather inputs. The drag-sensitivity tests use the
two rocket bodies (NORAD 10114, 41341) for Apr 2024 - Apr 2025: their TLEs are read
from data/raw/ (downloaded from Space-Track on first run if .env has credentials).
Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import physics  # noqa: E402
from src.orbit import decay_rates, drop_duplicate_epochs, tles_to_frame  # noqa: E402
from src.spacetrack_io import RAW_DIR, download_tle_history  # noqa: E402
from src.storms import storm_catalog  # noqa: E402

TLE_FILE = "tle_rocketbodies_2024_2025.json"
ROCKET_BODIES = [10114, 41341]

# Two LAGEOS-1 TLEs (~5,900 km): there is essentially no air there
LAGEOS_TLES = [
    {"TLE_LINE1": "1 08820U 76039A   24117.27600289  .00000019  00000-0  00000-0 0  9992",
     "TLE_LINE2": "2 08820 109.8466 271.4858 0044336 110.0668  29.7177  6.38664884863206"},
    {"TLE_LINE1": "1  8820U 76039A   24118.47313858  .00000023  00000-0  00000+0 0  9992",
     "TLE_LINE2": "2  8820 109.8469 271.8960 0044337 109.8087 262.1672  6.38664880863691"},
]


def msis_at(alts, ap, version, f107=150.0):
    """Density at one place and time for a list of altitudes, with constant ap."""
    n = len(alts)
    times = pd.DatetimeIndex(["2024-06-01 12:00"] * n)
    aps = np.full((n, 7), float(ap))
    return physics.density(times, np.zeros(n), np.zeros(n), np.asarray(alts, float),
                           np.full(n, f107), np.full(n, f107), aps, version)


# --- density model -----------------------------------------------------------

@pytest.mark.parametrize("version", physics.MODELS.values())
def test_density_falls_with_altitude(version):
    rho = msis_at([300, 400, 500, 600, 800], ap=10, version=version)
    assert np.all(np.diff(rho) < 0)
    assert 1e-13 < rho[2] < 1e-11      # ~500 km is of order 1e-12 kg/m^3


@pytest.mark.parametrize("version", physics.MODELS.values())
def test_density_rises_with_ap(version):
    quiet = msis_at([500], ap=4, version=version)[0]
    storm = msis_at([500], ap=200, version=version)[0]
    assert storm > 1.5 * quiet


def test_msis_inputs_pick_the_right_hours():
    # Synthetic table: ap = hours since the start, F10.7 = days since the start
    hours = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    k = np.arange(len(hours))
    sw = pd.DataFrame({"ap": (k // 3) * 3.0, "f107": (k // 24).astype(float),
                       "f107_81d": 100.0 + k // 24}, index=hours)
    t = pd.Timestamp("2024-01-05 10:30")      # inside the 09:00 block, day 4
    f107, f107a, aps = physics.msis_inputs(sw, [t])
    assert f107[0] == 3                       # previous day
    assert f107a[0] == 104
    assert aps[0, 0] == np.mean(np.arange(96, 120, 3))         # daily Ap of day 4
    assert list(aps[0, 1:5]) == [105, 102, 99, 96]             # now, -3 h, -6 h, -9 h
    assert aps[0, 5] == np.mean([105 - h for h in range(12, 34, 3)])
    assert aps[0, 6] == np.mean([105 - h for h in range(36, 58, 3)])


def test_lageos_predicted_decay_is_essentially_zero(tmp_path):
    tles = tles_to_frame(LAGEOS_TLES)
    rates = decay_rates(tles)
    sw = physics.load_msis_weather()
    out = physics.density_drivers(tles, rates, sw, cache_path=tmp_path / "lageos.parquet")
    assert out["alt_km"].iloc[0] > 5000
    # Even with a large drag sensitivity (0.05 m^2/kg, more than the rocket bodies)
    # the predicted decay is at least 10x below the ~0.1 m/day noise floor of the TLEs
    for model in physics.MODELS:
        pred_m_per_day = 0.05 * out[f"driver_{model}"].iloc[0] * 1000
        assert abs(pred_m_per_day) < 0.01


# --- drag sensitivity and residual on the two rocket bodies --------------------

@pytest.fixture(scope="module")
def rocket_bodies(tmp_path_factory):
    has_file = (RAW_DIR / TLE_FILE).exists()
    has_creds = (Path(__file__).resolve().parents[1] / ".env").exists()
    if not (has_file or has_creds):
        pytest.skip("No cached TLEs and no Space-Track credentials")
    records = download_tle_history(ROCKET_BODIES, "2024-04-01", "2025-04-30", out_name=TLE_FILE)
    tles = drop_duplicate_epochs(tles_to_frame(records))
    sw = physics.load_msis_weather()
    storms = storm_catalog(sw.loc[physics.PERIOD_START:, "dst"])
    return physics.run_physics(tles, decay_rates(tles), sw, storms)


def test_quiet_split_alternates(rocket_bodies):
    counts = rocket_bodies["role"].value_counts()
    assert counts["storm"] > 0
    assert abs(counts["fit"] - counts["holdout"]) <= len(ROCKET_BODIES)


@pytest.mark.parametrize("model", physics.MODELS)
def test_drag_sensitivity_is_positive(rocket_bodies, model):
    B = physics.fit_drag_sensitivity(rocket_bodies, model)
    assert set(B.index) == set(ROCKET_BODIES)
    # Rocket bodies: Cd*A/m of order 0.001 - 0.1 m^2/kg
    assert (B > 1e-3).all() and (B < 0.1).all()


@pytest.mark.parametrize("model", physics.MODELS)
def test_holdout_residual_is_near_zero(rocket_bodies, model):
    res = physics.holdout_residual(rocket_bodies, model)
    for _, r in res.iterrows():
        # Within 2 weekly-block standard errors (residuals are autocorrelated)
        assert abs(r["mean_resid"]) < 2 * r["se_block"], r.to_dict()


# --- residual validation: Hampel outlier filter --------------------------------

def noisy_series(n=300, seed=0):
    """Residual-like series: smooth wave + small noise, irregular times ~0.4 day apart."""
    rng = np.random.default_rng(seed)
    times = pd.Timestamp("2024-06-01") + pd.to_timedelta(np.cumsum(rng.uniform(0.2, 0.6, n)), unit="D")
    values = 0.005 * np.sin(np.arange(n) / 20) + rng.normal(0, 0.002, n)
    return times, values


def test_hampel_flags_planted_spikes():
    times, values = noisy_series()
    spikes = [40, 120, 250]
    values[spikes] += np.array([0.05, -0.08, 0.06])       # 25-40 noise SDs
    flags = physics.hampel_flags(values, times)
    assert flags[spikes].all()
    assert flags.mean() < 0.03                             # few false alarms on normal noise


def test_hampel_keeps_a_step_and_handles_unsorted_input():
    times, values = noisy_series()
    values[150:] -= 0.03                                   # level shift (a real change, not an outlier)
    flags = physics.hampel_flags(values, times)
    assert flags[145:155].sum() <= 2
    order = np.random.default_rng(1).permutation(len(values))
    shuffled = physics.hampel_flags(values[order], times[order])
    assert (shuffled == flags[order]).all()


def test_remove_outliers_reports_share_and_refits():
    times, values = noisy_series()
    rates = pd.DataFrame({"norad_id": 1, "t_mid": times, "t_start": times, "t_end": times,
                          "dt_days": 0.4, "role": np.where(np.arange(len(times)) % 2, "holdout", "fit")})
    for model in physics.MODELS:
        rates[f"driver_{model}"] = -10.0
    rates["da_dt_km_per_day"] = -0.1 + values
    rates.loc[[30, 31 + 100], "da_dt_km_per_day"] += 0.2   # one bad pair in fit, one in holdout
    rates = physics.add_predictions(physics.add_predictions(rates, "nrlmsise00"), "msis21")
    clean, report = physics.remove_outliers(rates)
    assert len(clean) == len(rates) - report["removed"].sum()
    assert report["removed"].sum() >= 2
    assert np.isclose(report["share"].mul(report["intervals"]).sum(), report["removed"].sum())
    assert clean["resid_msis21"].abs().max() < 0.05


# --- residual validation: Y_total and the storm summaries ----------------------

def test_quiet_ap_is_median_of_days_minus10_to_minus3():
    hours = pd.date_range("2024-05-01", "2024-05-31", freq="h")
    sw = pd.DataFrame({"ap": 5.0}, index=hours)
    onset = pd.Timestamp("2024-05-20")
    sw.loc[onset - pd.Timedelta(days=2):, "ap"] = 300.0            # storm starts after day -3
    sw.loc[onset - pd.Timedelta(days=9):onset - pd.Timedelta(days=8), "ap"] = 80.0   # short burst
    storms = pd.DataFrame({"storm_id": [7], "main_onset": [onset]})
    assert physics.quiet_ap(sw, storms).loc[7] == 5.0


def test_total_effect_uses_quiet_driver():
    rates = pd.DataFrame({"da_dt_km_per_day": [-0.20, -0.10], "B_msis21": [0.01, 0.01],
                          "driver_quiet_msis21": [-10.0, -10.0]})
    out = physics.add_total_effect(rates)
    assert np.allclose(out["pred_quiet_msis21"], -0.1)
    assert np.allclose(out["total_msis21"], [-0.1, 0.0])


def test_y_total_is_optional_in_schema():
    from src import schema
    from src.synthetic import make_synthetic

    table, _ = make_synthetic(n_objects=10, seed=0)
    assert "Y_total" not in table
    schema.validate(table)                                          # still valid without it
    with_total = schema.validate(table.assign(Y_total=table["Y_residual"].astype("float32")))
    assert with_total["Y_total"].dtype == "float64"
    assert "Y_total" in schema.OUTCOMES
    desc = schema.describe().set_index("column")
    assert desc.loc["Y_total", "unit"] == "m/day" and not desc.loc["Y_total", "required"]


def test_storm_baseline_residual_recovers_planted_shift():
    onset = pd.Timestamp("2024-06-15")
    t = onset + pd.to_timedelta(np.arange(-12, 6, 0.5) + 0.25, unit="D")
    resid = np.where(t >= onset, 0.02, 0.005)                       # +0.015 km/day after onset
    rates = pd.DataFrame({"norad_id": 1, "t_mid": t, "t_start": t - pd.Timedelta(hours=6),
                          "t_end": t + pd.Timedelta(hours=6), "dt_days": 0.5, "y": resid})
    storms = pd.DataFrame({"storm_id": [1], "main_onset": [onset], "class": ["strong"], "min_dst": [-150.0]})
    out = physics.storm_baseline_residual(rates, storms, "y")
    assert np.isclose(out["change"].iloc[0], 0.015)
    assert physics.storm_mean(out)["storms"] == 1


def test_smoothing_keeps_a_constant_driver():
    t = pd.Timestamp("2024-06-01") + pd.to_timedelta(np.arange(20) * 0.4, unit="D")
    rates = pd.DataFrame({"norad_id": 1, "t_mid": t, "dt_days": 0.4, "driver_msis21": -3.0})
    assert np.allclose(physics.smooth_driver(rates, "msis21", 2)["driver_msis21_s2"], -3.0)
    shifted = physics.shift_driver(rates.assign(driver_msis21=np.arange(20.0)), "msis21", 0.4 * 24)
    assert np.isnan(shifted["driver_msis21_lag9.6"].iloc[0])
    assert np.allclose(shifted["driver_msis21_lag9.6"].iloc[1:], np.arange(19.0))


def test_y_total_more_negative_than_residual_in_may_storm(rocket_bodies, tmp_path):
    """Real data: in the May 2024 storm (ap far above quiet) the quiet-ap prediction is
    smaller, so Y_total shows more extra decay than Y_residual."""
    records = download_tle_history(ROCKET_BODIES, "2024-04-01", "2025-04-30", out_name=TLE_FILE)
    tles = drop_duplicate_epochs(tles_to_frame(records))
    rates = decay_rates(tles)
    sw = physics.load_msis_weather()
    storms = storm_catalog(sw.loc[physics.PERIOD_START:, "dst"])
    q = physics.quiet_ap_drivers(tles, rates, sw, storms)
    out = physics.add_total_effect(rocket_bodies.merge(
        q[["norad_id", "t_start", "storm_id", "quiet_ap", "driver_quiet_msis21"]], on=["norad_id", "t_start"]))
    may = storms.loc[storms["min_dst"].idxmin()]
    post = out[(out["t_mid"] >= may["main_onset"]) & (out["t_mid"] <= may["main_onset"] + pd.Timedelta(days=3))]
    assert (post["storm_id"] == may["storm_id"]).all()
    assert post["total_msis21"].mean() < post["resid_msis21"].mean() - 0.02    # > 20 m/day more
    # Outside every event window the quiet prediction is the actual one
    free = out[out["storm_id"].isna()]
    assert np.allclose(free["total_msis21"], free["resid_msis21"])
