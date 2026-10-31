"""Tests for the active satellites in the analysis table (src/build_table.py, section 6).

Catalog drag scaling, the free-drag (failed / deorbiting) flag, the full build with a
small fake cohort of passive objects plus station-keeping satellites, and the offset
ratio leaving flagged satellites out. No data files, no downloads.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import build_table as bt  # noqa: E402
from src import offset_ratio as orr  # noqa: E402
from src import schema  # noqa: E402
from src.orbit import EARTH_EQUATORIAL_RADIUS_KM, build_decay_intervals, clean_tle_history, tle_to_row  # noqa: E402
from test_build_table import N_DAYS, START, fake_cohort, tle_lines  # noqa: E402


# ---------------------------------------------------------------------------
# catalog_scale
# ---------------------------------------------------------------------------

def passive_rows(alts, ratios, catalog=0.02):
    """Fake passive table (one row per rocket body) and objects table with known ratios."""
    ids = np.arange(1, len(alts) + 1)
    table = pd.DataFrame({"norad_id": ids, "alt_km": alts, "drag_sensitivity": np.asarray(ratios) * catalog,
                          "cohort_group": "main", "drag_source": "fit", "drag_unstable": 0})
    objects = pd.DataFrame({"norad_id": ids, "object_type": "rocket body", "drag_sensitivity_catalog": catalog})
    return table, objects


def test_catalog_scale_uses_bodies_at_the_same_altitude():
    table, objects = passive_rows([450, 470, 480, 500, 520, 620, 630, 640, 650, 660],
                                  [0.7, 0.72, 0.75, 0.78, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5])
    out = bt.catalog_scale(table, objects, pd.Series([480.0, 640.0], index=[7, 8]))
    assert out.loc[7, "scale"] == pytest.approx(0.75)
    assert out.loc[8, "scale"] == pytest.approx(0.5)
    assert (out["window_km"] == bt.SCALE_WINDOW_KM).all()


def test_catalog_scale_widens_a_sparse_window():
    table, objects = passive_rows([450, 700, 710, 720, 730], [0.7, 1.0, 1.0, 1.0, 1.0])
    out = bt.catalog_scale(table, objects, [450.0], min_bodies=3)
    assert out.loc[0, "n_bodies"] >= 3 and out.loc[0, "window_km"] > bt.SCALE_WINDOW_KM


def test_catalog_scale_ignores_debris_and_unstable_fits():
    table, objects = passive_rows([500] * 6, [0.8, 0.8, 0.8, 0.8, 5.0, 5.0])
    objects.loc[4, "object_type"] = "debris"
    table.loc[5, "drag_unstable"] = 1
    assert bt.catalog_scale(table, objects, [500.0], min_bodies=1).loc[0, "scale"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# free_drag_flags
# ---------------------------------------------------------------------------

def decay_rows(obs_fn, pred=-60.0, days=120, norad_id=1):
    """Daily intervals with observed decay obs_fn(day) and constant predicted decay (m/day)."""
    d = np.arange(days, dtype=float)
    t0 = START + pd.to_timedelta(d, unit="D")
    return pd.DataFrame({"norad_id": norad_id, "t_start": t0, "t_end": t0 + pd.Timedelta(days=1),
                         "dt_days": 1.0, "obs_decay": [obs_fn(x) for x in d], "pred_decay": pred})


def station_keeping(day):
    """Decays with drag for 3 days, then a burn puts the lost altitude back."""
    return 120.0 if day % 4 == 3 else -40.0


def test_station_keeper_is_not_flagged():
    assert bt.free_drag_flags(decay_rows(station_keeping))[1]["free_drag"].iat[0] == 0


def test_storm_drop_is_not_flagged():
    # Three days of fast decay during a storm, then the operator climbs back
    rows = decay_rows(lambda d: -300.0 if 40 <= d < 43 else (100.0 if 43 <= d < 52 else station_keeping(d)))
    assert bt.free_drag_flags(rows)[1]["free_drag"].iat[0] == 0


def test_failed_satellite_is_flagged_from_failure():
    rows = decay_rows(lambda d: -60.0 * (1 + 0.1 * np.sin(d)) if d >= 50 else station_keeping(d))
    flags, per_object = bt.free_drag_flags(rows)
    out = per_object.iloc[0]
    assert out["free_drag"] == 1 and out["kind"] == "drag"
    # Never resumes station keeping: flagged from the failure to the end of the window list
    assert flags[rows["t_start"] >= START + pd.Timedelta(days=50)].iloc[:-bt.FREE_DRAG_DAYS].all()
    # A window that is at least half free drag already counts, so the first flagged
    # window may start up to ~days / 2 before the failure, never after it
    assert 50 - bt.FREE_DRAG_DAYS / 2 - 1 <= (out["first_flag"] - START).days <= 50


def test_lowering_satellite_is_flagged():
    out = bt.free_drag_flags(decay_rows(lambda d: -250.0 if d >= 60 else station_keeping(d)))[1].iloc[0]
    assert out["free_drag"] == 1 and out["kind"] == "lowering"


def test_shell_change_keeps_later_station_keeping():
    # Lowered by ~6 km over days 30-60, then station keeping at the new shell
    rows = decay_rows(lambda d: -200.0 if 30 <= d < 60 else station_keeping(d))
    flags, per_object = bt.free_drag_flags(rows)
    assert per_object["free_drag"].iat[0] == 1
    day = (rows["t_start"] - START).dt.days
    assert flags[(day >= 35) & (day < 60)].all()
    # Windows that hold part of the drop are flagged whole, so the flag spreads at most
    # FREE_DRAG_DAYS either side of the lowering; station keeping after that is kept
    assert not flags[day > 60 + bt.FREE_DRAG_DAYS].any() and not flags[day < 30 - bt.FREE_DRAG_DAYS].any()
    assert (day > 60 + bt.FREE_DRAG_DAYS).sum() > 0


def test_slow_steady_deorbit_is_flagged():
    # Losing altitude at only 0.25 of the free-drag rate (low-drag attitude), for months
    rows = decay_rows(lambda d: -60.0 * (1 + 0.5 * np.sin(d)), pred=-240.0)
    assert bt.free_drag_flags(rows)[1]["free_drag"].iat[0] == 1


def test_short_free_drag_spell_is_not_flagged():
    rows = decay_rows(lambda d: -60.0 if 50 <= d < 60 else station_keeping(d))
    assert bt.free_drag_flags(rows)[1]["free_drag"].iat[0] == 0


# ---------------------------------------------------------------------------
# Full build with active satellites
# ---------------------------------------------------------------------------

def fake_actives(storms, seed=1):
    """(intervals, tles, objects) for two station-keeping satellites at 520 km with burns
    every ~3 days, in the format of build_table.load_active_inputs. Objects also lists the
    fake passive cohort as rocket bodies with a catalog B, for catalog_scale."""
    rng = np.random.default_rng(seed)
    rows = []
    for norad_id, inc in ((95001, 53.0), (95002, 43.0)):
        days = np.sort(rng.uniform(0, N_DAYS, int(N_DAYS * 2)))
        for d in days:
            a = EARTH_EQUATORIAL_RADIUS_KM + 520.0 - 0.05 * (d % 3.0) + rng.normal(0, 0.005)
            l1, l2 = tle_lines(norad_id, START + pd.Timedelta(days=float(d)), a, inc, 0.0)
            rows.append({**tle_to_row(l1, l2), "line1": l1, "line2": l2})
    tles, _ = clean_tle_history(pd.DataFrame(rows), passive=False)
    intervals, _ = build_decay_intervals(tles, storms, passive=False)
    objects = pd.DataFrame({
        "norad_id": [95001, 95002, 90001, 90002, 90003, 90004],
        "object_name": ["STARLINK-1", "STARLINK-2", "R/B A", "R/B B", "R/B C", "R/B D"],
        "object_type": ["payload", "payload"] + ["rocket body"] * 4,
        "drag_sensitivity_catalog": [0.1, 0.1, 0.02, 0.02, 0.02, 0.02],
        "is_active": [1, 1, 0, 0, 0, 0],
    })
    return intervals, tles, objects


@pytest.fixture(scope="module")
def cohort():
    return fake_cohort()


@pytest.fixture(scope="module")
def built(cohort, tmp_path_factory):
    actives = fake_actives(cohort[3])
    table, _, _ = bt.build_table(inputs=cohort, chunk_dir=tmp_path_factory.mktemp("chunks"), save=False,
                                 log=None, active_inputs=actives,
                                 active_chunk_dir=tmp_path_factory.mktemp("active_chunks"))
    return table, actives


def test_actives_appended_and_valid(built):
    table, (intervals, _, _) = built
    assert schema.problems(table) == []
    active = table[table["is_active"] == 1]
    assert set(active["norad_id"]) == {95001, 95002}
    assert len(active) == len(intervals)
    assert (active["drag_source"] == "catalog_scaled").all()
    assert active["Y_log_ratio"].isna().all()
    assert (active["hampel_outlier"] == 0).all() and (table.loc[table["is_active"] == 0, "free_drag"] == 0).all()
    # Actives never enter the passive samples
    assert not schema.main_sample(table)["is_active"].any()


def test_active_drag_is_scaled_catalog(built, cohort):
    table, (_, _, objects) = built
    active = table[table["is_active"] == 1]
    passive = table[table["is_active"] == 0]
    scale = bt.catalog_scale(passive, objects, active.groupby("norad_id")["alt_km"].mean())["scale"]
    B = active.groupby("norad_id")["drag_sensitivity"].first()
    np.testing.assert_allclose(B, 0.1 * scale.reindex(B.index))
    assert (active["drag_scale"] > 0).all()


def test_active_outcomes(built):
    active = built[0][built[0]["is_active"] == 1]
    outside = active["storm_id"].isna()
    np.testing.assert_allclose(active.loc[outside, "Y_total"], active.loc[outside, "Y_residual"])
    np.testing.assert_allclose(active["Y_residual"], active["obs_decay"] - active["pred_decay"])
    # Station keeping: altitude held, so it is not flagged as free drag
    assert (active["free_drag"] == 0).all()


def test_passive_rows_unchanged_by_actives(built, cohort, tmp_path):
    passive_only, _, _ = bt.build_table(inputs=cohort, chunk_dir=tmp_path, save=False, log=None)
    with_actives = built[0][built[0]["is_active"] == 0].drop(
        columns=["free_drag", "first_flag", "object_name", "drag_scale"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(with_actives, passive_only.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Offset ratio leaves flagged satellites out
# ---------------------------------------------------------------------------

def test_clean_actives_drops_free_drag():
    table = pd.DataFrame({"is_active": [1, 1, 0], "ecc": 0.001, "tle_ok": 1, "drag_sensitivity": 0.1,
                          "free_drag": [0, 1, 0], "norad_id": [1, 2, 3]})
    assert orr._clean_actives(table)["norad_id"].tolist() == [1]
    assert orr._clean_actives(table.drop(columns="free_drag"))["norad_id"].tolist() == [1, 2]
