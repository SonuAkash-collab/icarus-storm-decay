"""Tests for the track C runner (src/causal_run.py), on synthetic data.

Run with:  pytest
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import causal_run as cr  # noqa: E402
from src import event_checks as ec  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")
QUICK = dict(n_placebo=5, n_boot=19, n_placebo_reps=2)


@pytest.fixture(scope="module")
def fake():
    return make_synthetic(n_objects=250, active_share=0.25, noise=1.0, n_high_alt=10, seed=0)


def test_checks_follow_the_outcome(fake):
    table, storms = fake
    renamed = table.assign(Y_total=table["Y_residual"], Y_residual=0.0)
    a = ec.run_all(table, storms, n_placebo=3, n_boot=9)
    b = ec.run_all(renamed, storms, "Y_total", n_placebo=3, n_boot=9)
    for key in ("result", "placebo", "with_without", "contam_sensitivity"):
        pd.testing.assert_frame_equal(a[key], b[key])


def test_run_without_y_total_falls_back_and_says_so(fake, tmp_path):
    out = cr.run(*fake, out_dir=tmp_path, **QUICK)
    assert list(out["event_study"]) == ["Y_residual"]
    assert out["offset"]["outcome"] == "Y_residual"
    text = (tmp_path / "summary.md").read_text()
    assert "Y_total is missing" in text and "## Event study: Y_residual" in text
    for name in ("sample_report.csv", "offset_ratio.csv", "offset_recovery_sensitivity.csv",
                 "Y_residual/event_study.csv", "Y_residual/placebo_dates.csv"):
        assert (tmp_path / name).exists()
    high = out["high_alt"]["Y_residual"]
    assert high["objects"] == 10 and high["ci_lo"] < 0 < high["ci_hi"]


def test_run_with_both_estimands(fake, tmp_path):
    table, storms = fake
    out = cr.run(table.assign(Y_total=table["Y_residual"]), storms, out_dir=tmp_path, **QUICK)
    assert list(out["event_study"]) == ["Y_total", "Y_residual"]
    assert out["offset"]["outcome"] == "Y_total"
    assert "missing" not in (tmp_path / "summary.md").read_text()


def test_passive_only_table_skips_offset_ratio(fake, tmp_path):
    table, storms = fake
    out = cr.run(table[table["is_active"] == 0], storms, out_dir=tmp_path, **QUICK)
    assert out["offset"] is None
    assert "Not run: the table has no active satellites" in (tmp_path / "summary.md").read_text()
    assert not (tmp_path / "offset_ratio.csv").exists()


def test_no_outcome_raises(fake, tmp_path):
    table, storms = fake
    with pytest.raises(ValueError, match="none of the outcomes"):
        cr.run(table.drop(columns="Y_residual"), storms, out_dir=tmp_path, **QUICK)
