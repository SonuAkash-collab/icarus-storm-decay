"""Tests for the track D runner (src/dml_run.py), on synthetic data.

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import dml  # noqa: E402
from src import dml_run  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def fake():
    table, _, hourly = make_synthetic(n_objects=80, seed=0, return_hourly=True)
    # stand-in for the second estimand, so both outcomes go through the runner
    return table.assign(Y_total=table["Y_residual"] + table["true_effect"]), hourly


@pytest.fixture(scope="module")
def out(fake, tmp_path_factory):
    return dml_run.run(*fake, out_dir=tmp_path_factory.mktemp("dml"), n_boot=19)


def test_runs_both_outcomes(out):
    assert list(out["outcomes"]) == ["Y_residual", "Y_total"]


def test_lag_sweep_uses_each_outcome(fake, out):
    # the 0 h row must be DML on that outcome (same rows as the sweep), not on Y_residual
    table, hourly = fake
    for y in ["Y_residual", "Y_total"]:
        df = dml.prepare(table, y)
        ok = ~np.isnan(dml.ap_mean_shifted(df, hourly, max(dml.LAGS_HOURS)))
        expected = dml.dml(df[ok], y, n_boot=0)["coef"]
        assert out["outcomes"][y]["lags"].loc[0, "coef"] == pytest.approx(expected)
    lags = [out["outcomes"][y]["lags"]["coef"].astype(float) for y in ["Y_residual", "Y_total"]]
    assert not lags[0].equals(lags[1])


def test_skips_missing_outcome(fake):
    assert dml_run.available_outcomes(fake[0].drop(columns="Y_total")) == ["Y_residual"]


def test_reports_methods_lags_and_selection(out):
    r = out["outcomes"]["Y_residual"]
    assert set(dml_run.METHODS) <= set(r["compare"].index)
    assert list(r["lags"].index) == list(dml.LAGS_HOURS)
    sel = r["selection"]
    assert sel.groupby("helper")["chosen"].sum().to_dict() == {"ap": 1, "residual": 1}
    assert sel.loc[sel.chosen & (sel.helper == "ap"), "candidate"].item() == r["compare"].attrs["model_t"]


def test_writes_files(fake, tmp_path):
    dml_run.run(*fake, out_dir=tmp_path, outcomes=["Y_residual"], n_boot=9)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"summary.md", "sample_report.csv", "compare_Y_residual.csv",
            "lag_sweep_Y_residual.csv", "selection_Y_residual.csv"} <= names
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Y_residual" in text and "lag sweep" in text
