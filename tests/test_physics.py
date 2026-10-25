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
