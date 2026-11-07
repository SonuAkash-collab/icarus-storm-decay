"""Tests for model error by storm class (src/storm_class.py).

Run with:  pytest
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import storm_class as sc  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402


def per_storm(big_value, rest_values):
    classes = ["extreme", "severe", "severe"] + ["moderate"] * 11 + ["strong"] * (len(rest_values) - 11)
    return pd.DataFrame({"storm_id": range(1, 4 + len(rest_values)), "storm_class": classes,
                         "change": [big_value] * 3 + list(rest_values)})


def test_permutation_p_is_exact_when_big_storms_stand_out():
    out = sc.big_vs_rest(per_storm(10.0, [0.0] * 17))
    assert out["gap"] == pytest.approx(10.0)
    assert out["n_permutations"] == comb(20, 3) == 1140
    assert out["p_exact"] == pytest.approx(1 / 1140)


def test_permutation_p_is_one_without_a_difference():
    assert sc.big_vs_rest(per_storm(0.0, [0.0] * 17))["p_exact"] == pytest.approx(1.0)


def test_class_table_intervals_only_with_three_storms():
    rng = np.random.default_rng(0)
    per = per_storm(5.0, list(rng.normal(size=17)))
    tab = sc.class_table(per)
    assert tab.loc["extreme", "n_storms"] == 1 and np.isnan(tab.loc["extreme", "ci_lo"])
    assert tab.loc["severe", "n_storms"] == 2 and np.isnan(tab.loc["severe", "ci_hi"])
    assert tab.loc["severe", "storm_values"] == "+5.0, +5.0"
    m = per.loc[per["storm_class"] == "moderate", "change"]
    half = stats.t.ppf(0.975, 10) * m.std() / np.sqrt(11)
    assert tab.loc["moderate", "ci_lo"] == pytest.approx(m.mean() - half)


def test_bigger_storms_show_more_planted_decay():
    # the planted storm effect grows with ap, so class means must order by storm size
    # (scored on true_effect: in the noisy Y_residual the small moderate effect is buried)
    table, storms = make_synthetic(n_objects=120, seed=0)
    out = sc.run(table.assign(hampel_outlier=0), storms, "true_effect", out_dir=None)
    classes = out["classes"]
    assert set(out["per_storm"]["storm_class"]) <= set(classes.index)
    assert classes.loc["severe", "mean"] < classes.loc["strong", "mean"] < classes.loc["moderate", "mean"] < 0
    assert classes.loc["moderate", "ci_hi"] < 0
    assert out["big_vs_rest"]["gap"] < 0 and out["big_vs_rest"]["p_exact"] < 0.05


def test_dml_permutation_is_exact_when_big_storms_have_a_steeper_slope():
    # per-storm sums: big storms have slope a/b = 2, the rest 0
    classes = ["extreme", "severe", "severe"] + ["moderate"] * 11 + ["strong"] * 6
    sums = pd.DataFrame({"storm_id": range(1, 21), "storm_class": classes,
                         "a": [2.0] * 3 + [0.0] * 17, "b": [1.0] * 20})
    out = sc.dml_big_vs_rest(sums)
    assert out["gap"] == pytest.approx(2.0) and out["coef_rest"] == pytest.approx(0.0)
    assert out["p_exact"] == pytest.approx(1 / comb(20, 3))


def test_dml_by_class_recovers_the_same_planted_slope_in_every_class():
    # the synthetic effect is linear in ap, so its per-nT size does not depend on the class
    from src import dml

    table, _ = make_synthetic(n_objects=200, seed=0)
    df = dml.prepare(table)
    by_class, sums = sc.dml_by_class(df, n_boot=99, model_y="xgb", model_t="linear")
    planted = -df["true_tau"].mean()
    assert set(by_class.index) == set(df["storm_class"].dropna())
    assert (by_class["coef"] / planted).between(0.75, 1.25).all()
    assert len(sums) == df["storm_id"].nunique()
    assert sc.dml_big_vs_rest(sums)["p_exact"] > 0.05


def test_dml_by_class_handles_a_one_storm_class():
    # the real catalog has a single extreme storm: slope only, no interval, no crash
    from src import dml

    table, storms = make_synthetic(n_objects=80, seed=0)
    one = int(storms["storm_id"].iloc[0])
    table = table.assign(storm_class=table["storm_class"].where(table["storm_id"] != one, "extreme"))
    by_class, _ = sc.dml_by_class(dml.prepare(table), n_boot=19, model_y="xgb", model_t="linear")
    assert by_class.loc["extreme", "n_storms"] == 1
    assert np.isfinite(by_class.loc["extreme", "coef"]) and np.isnan(by_class.loc["extreme", "ci_lo"])
