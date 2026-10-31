"""Tests for the model-based operator offset ratio (src/offset_model.py).

A small fake table with known storm loss and planted operator responses: daily
intervals, a predicted storm loss S on the ducking days, active satellites that undo a
known share of it, passive objects that measure the model error, and drag-only
satellites for the placebo. No data files, no downloads.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import offset_model as om  # noqa: E402

START = pd.Timestamp("2024-04-01")
N_DAYS = 330
S_DUCK = -60.0          # predicted extra decay on the ducking days (m/day)
QUIET = -30.0           # predicted quiet decay (m/day)


def fake_storms(n=8, gap=40, duck_days=2):
    onset = START + pd.to_timedelta(20 + gap * np.arange(n) + 0.5, unit="D")
    return pd.DataFrame({"storm_id": np.arange(1, n + 1), "main_onset": onset, "onset_time": onset,
                         "recovery_end": onset + pd.Timedelta(days=duck_days), "class": "strong"})


def fake_table(storms, duck=0.0, recover=0.0, model_error=1.0, n_active=6, n_passive=8, n_drag=0,
               routine=25.0, noise=0.0, seed=0):
    """Rows for active satellites (undo `duck` of the storm loss during the storm and win
    back `recover` of what is left over the recovery window), passive objects and
    drag-only satellites (free_drag rows). The real storm loss is model_error x S."""
    rng = np.random.default_rng(seed)
    t0 = START + pd.to_timedelta(np.arange(N_DAYS), unit="D")
    rows = []
    kinds = [("active", i) for i in range(n_active)] + [("passive", i) for i in range(n_passive)] \
        + [("drag", i) for i in range(n_drag)]
    for k, (kind, i) in enumerate(kinds):
        t_mid = t0 + pd.Timedelta(hours=12)
        S = np.zeros(N_DAYS)
        gain = np.zeros(N_DAYS)
        for s in storms.to_dict("records"):
            in_duck = (t_mid >= s["main_onset"]) & (t_mid < s["recovery_end"])
            S[in_duck] = S_DUCK
            lost = (1 - duck) * model_error * S_DUCK * in_duck.sum()          # m left after ducking
            in_rec = (t_mid >= s["recovery_end"]) & (t_mid < s["recovery_end"] + pd.Timedelta(days=om.RECOVERY_DAYS))
            gain[in_rec] = -recover * lost / in_rec.sum()
        real = QUIET + model_error * S                                          # what drag really does
        if kind == "active":
            obs = real + routine + duck * (-model_error * S) + gain
        else:
            obs = real
        obs = obs + rng.normal(0, noise, N_DAYS)
        pred = QUIET + S
        rows.append(pd.DataFrame({
            "norad_id": 1000 + k, "t_start": t0, "t_end": t0 + pd.Timedelta(days=1), "t_mid": t_mid,
            "dt_days": 1.0, "obs_decay": obs, "pred_decay": pred, "Y_residual": obs - pred,
            "Y_total": obs - QUIET, "alt_km": 520.0 if kind != "passive" else 540.0, "ecc": 0.001,
            "tle_ok": 1, "is_active": int(kind != "passive"), "free_drag": int(kind == "drag"),
            "drag_sensitivity": 0.05 if kind == "passive" else 0.08,
            "object_name": f"STARLINK-{k}" if kind != "passive" else f"DEB {k}"}))
    return pd.concat(rows, ignore_index=True)


def est(result, stat):
    return result.loc[stat, "est"]


def test_alt_band_labels():
    assert list(om.alt_band([450.0, 500.0, 799.0, 900.0])) == ["400-500", "500-600", "700-800", None]


def test_no_operator_gives_zero():
    storms = fake_storms()
    table = fake_table(storms, routine=0.0)
    table.loc[table["is_active"] == 1, "obs_decay"] = table["pred_decay"]       # actives just decay
    table = table.assign(Y_residual=table["obs_decay"] - table["pred_decay"],
                         Y_total=table["obs_decay"] - QUIET)
    r = om.offset(om.storm_pairs(om.active_rows(table), storms), n_boot=0)
    assert est(r, "offset_duck") == pytest.approx(0, abs=1e-9)
    assert est(r, "offset_total") == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("duck,recover", [(0.6, 0.0), (0.3, 0.5), (1.0, 0.0)])
def test_planted_shares_recovered(duck, recover):
    storms = fake_storms()
    r = om.offset(om.storm_pairs(om.active_rows(fake_table(storms, duck, recover)), storms), n_boot=0)
    assert est(r, "offset_duck") == pytest.approx(duck, abs=1e-6)
    assert est(r, "offset_total") == pytest.approx(duck + (1 - duck) * recover, abs=1e-6)
    assert est(r, "duck_part") + est(r, "rec_part") == pytest.approx(est(r, "offset_total"))


def test_routine_station_keeping_cancels():
    storms = fake_storms()
    a = om.offset(om.storm_pairs(om.active_rows(fake_table(storms, 0.4, routine=0.0)), storms), n_boot=0)
    b = om.offset(om.storm_pairs(om.active_rows(fake_table(storms, 0.4, routine=40.0)), storms), n_boot=0)
    pd.testing.assert_series_equal(a["est"], b["est"])


def test_model_error_correction():
    # The storm really costs 1.5 x what the model predicts; passives show it in Y_residual
    storms = fake_storms()
    table = fake_table(storms, duck=0.5, model_error=1.5)
    pairs = om.add_correction(om.storm_pairs(om.active_rows(table), storms), om.model_error(table, storms))
    raw, corr = om.offset(pairs, "P", n_boot=0), om.offset(pairs, "Pc", n_boot=0)
    assert pairs["delta_duck"].iloc[0] == pytest.approx(0.5 * S_DUCK)
    assert est(corr, "offset_duck") == pytest.approx(0.5, abs=1e-6)
    assert abs(est(raw, "offset_duck") - 0.5) > 0.2


def test_bootstrap_interval_covers_truth():
    storms = fake_storms(n=8)
    pairs = om.storm_pairs(om.active_rows(fake_table(storms, 0.5, noise=15.0, seed=3)), storms)
    r = om.offset(pairs, n_boot=499, seed=1)
    assert r.loc["offset_duck", "ci_lo"] < 0.5 < r.loc["offset_duck", "ci_hi"]
    assert r.attrs["n_storms"] == 8 and r.loc["offset_duck", "se"] > 0


def test_orbit_change_screen():
    storms = fake_storms()
    table = fake_table(storms, 0.5, n_active=2, n_passive=0)
    # Satellite 1000 is raised by 3 km over days 100-110
    raise_ = (table["norad_id"] == 1000) & ((table["t_start"] - START).dt.days.between(100, 109))
    table.loc[raise_, "obs_decay"] += 300.0
    flags = om.orbit_change_flags(table[table["is_active"] == 1])
    day = (table["t_start"] - START).dt.days
    assert flags[raise_].all() and not flags[table["norad_id"] == 1001].any()
    assert not flags[(table["norad_id"] == 1000) & (day > 110 + om.CHANGE_DAYS)].any()
    assert len(om.active_rows(table)) < (table["is_active"] == 1).sum()


def test_placebo_drag_only_satellites_give_zero():
    storms = fake_storms()
    table = fake_table(storms, 0.7, n_drag=4)
    rows, scale = om.placebo_rows(table, storms)
    assert set(rows["norad_id"]) == set(table.loc[table["free_drag"] == 1, "norad_id"])
    assert np.allclose(scale, 1.0)
    r = om.offset(om.storm_pairs(rows, storms, b_scale=scale), n_boot=0)
    assert est(r, "offset_duck") == pytest.approx(0, abs=1e-9)
    assert est(r, "offset_total") == pytest.approx(0, abs=1e-9)


def test_placebo_rescales_b_and_drops_thrust_lowering():
    storms = fake_storms()
    table = fake_table(storms, n_drag=3)
    drag = table["free_drag"] == 1
    ids = sorted(table.loc[drag, "norad_id"].unique())
    # Satellite ids[0] decays at half the catalog rate (low-drag attitude), ids[1] is thrust down
    for nid, k in ((ids[0], 0.5), (ids[1], 4.0)):
        m = table["norad_id"] == nid
        table.loc[m, "obs_decay"] = k * table.loc[m, "pred_decay"]
    table = table.assign(Y_residual=table["obs_decay"] - table["pred_decay"],
                         Y_total=table["obs_decay"] - QUIET)
    rows, scale = om.placebo_rows(table, storms)
    assert ids[1] not in set(rows["norad_id"]) and scale[ids[0]] == pytest.approx(0.5)
    own = om.offset(om.storm_pairs(rows, storms, b_scale=scale), n_boot=0)
    cat = om.offset(om.storm_pairs(rows, storms), n_boot=0)
    assert est(own, "offset_duck") == pytest.approx(0, abs=1e-9)
    assert est(cat, "offset_duck") > 0.1                     # catalog B too big -> fake offset


def test_run_with_and_without_key_storms():
    storms = fake_storms()
    out = om.run(fake_table(storms, 0.5, recover=0.4, n_drag=2), storms, n_boot=19)
    s = out["summary"]
    assert set(s.index.get_level_values("prediction")) == {"uncorrected", "corrected"}
    assert s.index.get_level_values("storms").nunique() == 2
    assert s.loc[("all storms", "uncorrected", "offset_total"), "est"] == pytest.approx(0.7, abs=1e-6)
    assert not out["placebo"].empty and set(out["b_sensitivity"].columns) == {0.75, 1.25}
