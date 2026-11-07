"""Tests for the calm-to-calm storm outcome and the extreme-storm verdict (src/physics.py,
section 7). The recovery tests run the physics simulator (src/simulate.py) with a known
negative boost: the true density is lower than MSIS after each storm, so MSIS
over-predicts the storm's altitude loss by a known amount.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import physics  # noqa: E402
from src.simulate import planted_storms, simulate  # noqa: E402

SIM = {"n_objects": 12, "n_high_alt": 0, "n_days": 150, "return_hourly": True, "seed": 1}
BOOST = -0.4          # true density 40% below MSIS on days 0..2 after each storm
SLACK = pd.Timedelta(seconds=1)


def planted_loss(table, per):
    """Planted storm part of the loss (m) over each calm-to-calm span, from true_effect."""
    half = pd.to_timedelta(table["dt_days"] / 2, unit="D")
    ts, te = table["t_mid"] - half, table["t_mid"] + half
    part = table["true_effect"] * table["dt_days"]
    return np.array([part[(table["norad_id"] == r.norad_id) & (ts >= r.t0 - SLACK) & (te <= r.t1 + SLACK)].sum()
                     for r in per.itertuples()])


def run(boost, smooth_days=0.0, extra_days=0):
    table, storms, sw = simulate(boost=boost, smooth_days=smooth_days, **SIM)
    per = physics.calm_to_calm(physics.table_rates(table), storms, sw, extra_days=extra_days)
    return table, per.assign(planted_m=planted_loss(table, per)).set_index(["storm_id", "norad_id"])


@pytest.fixture(scope="module")
def sims():
    """Same seed with and without the boost: same objects and TLE noise, so the boost
    is the only difference between the two runs (common random numbers)."""
    return {"boost": run(BOOST), "null": run(0.0)}


def test_calm_to_calm_recovers_planted_loss(sims):
    (_, boost), (_, null) = sims["boost"], sims["null"]
    assert len(boost) == 12 * 5 and (boost["coverage"] > 0.999).all()
    assert (boost["planted_m"] > 0).all()                        # MSIS over-predicts the loss
    found = boost["diff_m"] - null.loc[boost.index, "diff_m"]
    assert np.allclose(found, boost["planted_m"], atol=1.0)      # to the metre, pair by pair
    s = physics.calm_summary(boost.reset_index(), "class", n_boot=300).iloc[0]
    assert s["ratio"] < 1 and s["adj_lo"] > 0 and s["quiet_lo"] > 0
    assert abs(s["diff_quiet_m"] - boost["planted_m"].mean()) < 0.25 * boost["planted_m"].mean()


def test_calm_to_calm_null_is_zero(sims):
    _, null = sims["null"]
    s = physics.calm_summary(null.reset_index(), "class", n_boot=300).iloc[0]
    assert s["adj_lo"] < 0 < s["adj_hi"] and s["quiet_lo"] < 0 < s["quiet_hi"]
    assert 0.95 < s["ratio"] < 1.05


def test_calm_to_calm_is_blur_free_with_a_late_calm_day():
    """TLE blur (a = mean over the 2 days before the epoch) moves decay in time but not
    the total, once the TLE at the end no longer averages storm-time orbit."""
    _, boost = run(BOOST, smooth_days=2, extra_days=3)
    _, null = run(0.0, smooth_days=2, extra_days=3)
    found = boost["diff_m"] - null.loc[boost.index, "diff_m"]
    assert abs(found.mean() - boost["planted_m"].mean()) < 0.05 * boost["planted_m"].mean()
    assert (found - boost["planted_m"]).abs().max() < 5.0


def test_onset_gaps_and_rounding():
    t = pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04"])
    rates = pd.DataFrame({"norad_id": 1, "t_start": t[[0, 2]], "t_end": [t[1], t[3]]})
    gaps = physics.onset_gaps(rates)
    assert len(gaps) == 1 and gaps.iloc[0]["t_start"] == t[1] and gaps.iloc[0]["t_end"] == t[2]
    tiny = rates.assign(t_start=[t[0], t[1] + pd.Timedelta(microseconds=3)], t_end=[t[1], t[3]])
    assert physics.onset_gaps(tiny).empty


def test_calm_spans_respect_neighbouring_storms():
    storms = planted_storms(["2024-05-10 06:00", "2024-05-16 06:00"])
    days = pd.date_range("2024-04-25", "2024-06-10", freq="D")
    calm = pd.Series(True, index=days)
    spans = physics.calm_spans(storms, calm)
    before1, after1 = spans[1]
    before2, after2 = spans[2]
    assert before1[0] == pd.Timestamp("2024-05-07")           # day -3, latest first
    assert all(d + pd.Timedelta(days=1) <= storms.loc[1, "onset_time"] for d in after1)
    assert all(d >= storms.loc[0, "recovery_end"] for d in before2)
    calm[pd.Timestamp("2024-05-07")] = False                  # a stormy day is skipped
    assert physics.calm_spans(storms, calm)[1][0][0] == pd.Timestamp("2024-05-06")


def test_storm_labels_and_verdict():
    ci = lambda lo: pd.Series({"ci_lo": lo})                               # noqa: E731
    calm = lambda adj, quiet: pd.Series({"adj_lo": adj, "quiet_lo": quiet})  # noqa: E731
    assert physics.storm_check_label(ci(1), ci(1), calm(5, 5)) == "real model overshoot"
    assert physics.storm_check_label(ci(1), ci(-1), calm(-5, 5)) == "TLE blur"
    assert physics.storm_check_label(ci(1), ci(1), calm(-5, 5)).startswith("mixed: survives smoothing")
    assert physics.storm_check_label(ci(1), ci(-1), calm(5, 5)).startswith("mixed: in calm-to-calm")
    assert physics.storm_check_label(ci(-1), ci(1), calm(5, 5)) == "no over-prediction"
    real = "real model overshoot"
    assert physics.extreme_storm_verdict({"May": real, "Oct": real}) == "real model overshoot in extreme storms"
    assert physics.extreme_storm_verdict({"May": "TLE blur", "Oct": "TLE blur"}) == "TLE blur"
    mixed = physics.extreme_storm_verdict({"May": "TLE blur", "Oct": real})
    assert mixed.startswith("mixed (") and "Oct: real" in mixed


def test_calm_summary_cluster_basis():
    rng = np.random.default_rng(0)
    rows = [{"storm_id": s, "norad_id": o, "class": "moderate" if s < 4 else "extreme",
             "obs_m": -100 + rng.normal(), "pred_m": -110.0, "span_days": 8.0}
            for s in range(1, 5) for o in range(20)]
    per = pd.DataFrame(rows).assign(diff_m=lambda d: d["obs_m"] - d["pred_m"])
    per["diff_adj_m"] = per["diff_quiet_m"] = per["diff_m"]
    s = physics.calm_summary(per, "class", n_boot=200).set_index("class")
    assert s.loc["moderate", "ci_basis"] == "storms" and s.loc["extreme", "ci_basis"] == "objects"
    assert np.isclose(s.loc["extreme", "ratio"], per.loc[per["storm_id"] == 4, "obs_m"].sum() / (-110 * 20))
    assert s.loc["moderate", "diff_lo"] < 10 < s.loc["moderate", "diff_hi"]
