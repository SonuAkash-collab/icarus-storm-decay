import numpy as np
import pandas as pd

from src import review_figures as rf


def test_class_summary_uses_storm_ci_only_with_three_storms():
    per = pd.DataFrame({
        "storm_class": ["moderate"] * 4 + ["extreme"],
        "change": [1.0, 2.0, 3.0, 2.0, 10.0],
        "se_objects": [0.5] * 4 + [1.0], "n_objects": [10] * 5})
    cs = rf.class_summary(per)
    assert cs.loc["moderate", "ci_basis"] == "storms"
    assert np.isclose(cs.loc["moderate", "mean"], 2.0)
    assert cs.loc["extreme", "ci_basis"] == "objects"
    assert np.isclose(cs.loc["extreme", "ci_hi"] - cs.loc["extreme", "mean"], 1.96)


def test_parse_short_run(tmp_path):
    text = ("## Event study: Y_total\n\n- Short-run effect (days 0-2, m/day): -14.064 (95% CI -19.751 to -8.401).\n\n"
            "## Event study: Y_residual\n\n- Short-run effect (days 0-2, m/day): 0.153 (95% CI -3.162 to 3.189).\n")
    path = tmp_path / "summary.md"
    path.write_text(text)
    out = rf.parse_short_run(path)
    assert out["Y_total"] == (-14.064, -19.751, -8.401)
    assert out["Y_residual"] == (0.153, -3.162, 3.189)


def test_per_drag_profile_skips_single_cell_rows():
    # three altitude rows; the top row has only one supported cell and must be left out
    grid = pd.DataFrame({
        "inc_deg": [50.0] * 6, "alt_km": [620, 620, 690, 690, 750, 750],
        "drag_sensitivity": [0.01, 0.02] * 3, "effect": [-12, -12, -7, -7, -4, -99],
        "n_near": [5, 5, 5, 5, 5, 0]})
    prof = rf.per_drag_profile(grid)
    assert list(prof.index) == [620, 690]
    out = rf.profile_ratio(prof, alts=(620, 750))
    assert out["alt_lo"] == 620 and out["alt_hi"] == 690      # nearest supported row to 750
    assert np.isclose(out["ratio"], 12 / 7)
    # one supported cell repeated at three inclinations is still one cell
    rep = pd.concat([grid.assign(inc_deg=i) for i in (40.0, 60.0, 80.0)])
    assert list(rf.per_drag_profile(rep).index) == [620, 690]


def test_offset_numbers_reads_corrected_runs(tmp_path):
    rows = []
    for storms, n in [("all storms", 20), ("without 3,4,11", 17), ("quiet ap (without 10,15,16,17,19,20)", 14)]:
        for pred, shift in [("uncorrected", 0.0), ("corrected", 0.1)]:
            for stat, est in [("offset_total", 1.0), ("duck_part", 1.1), ("rec_part", -0.1)]:
                rows.append({"storms": storms, "prediction": pred, "statistic": stat, "est": est + shift,
                             "ci_lo": est - 0.2, "ci_hi": est + 0.2, "n_storms": n, "n_objects": 21})
    pd.DataFrame(rows).to_csv(tmp_path / "offset_model.csv", index=False)
    out = rf.offset_numbers(tmp_path)
    assert np.isclose(out["total"][0], 1.1) and np.isclose(out["rec"][0], 0.0)
    assert out["quiet_dropped"] == 6 and out["quiet_n_storms"] == 14
    assert out["n_storms"] == 20 and out["n_objects"] == 21


def test_lag_text_says_when_lag_needs_the_big_storms():
    ci = lambda c, lo, hi: (c, lo, hi)     # noqa: E731
    rf_ = {"dropped": "3,4,11",
           "lags_all": {0: ci(-0.05, -0.1, 0.0), 6: ci(0, -0.1, 0.1), 12: ci(0.08, 0.02, 0.14), 24: ci(0.11, 0.04, 0.19)},
           "lags_without": {0: ci(-0.15, -0.2, -0.1), 6: ci(-0.1, -0.2, 0.0), 12: ci(0, -0.1, 0.1), 24: ci(0.05, -0.02, 0.12)}}
    assert "not a general result" in rf.lag_text(rf_)
    rf_["lags_without"][24] = ci(0.1, 0.05, 0.15)
    assert "survives" in rf.lag_text(rf_)
