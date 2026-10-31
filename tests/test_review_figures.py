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
