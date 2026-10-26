"""Tests for the causal graph (src/dag.py). Run with:  pytest"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import dag  # noqa: E402
from src.schema import COLUMNS  # noqa: E402


def test_graph_is_acyclic_and_complete():
    assert dag.is_acyclic()
    used = {n for edge in dag.EDGES for n in edge}
    assert used == set(dag.NODES)


def test_adjustment_set_uses_schema_columns():
    assert set(dag.ADJUST) <= set(COLUMNS)


def test_plan_adjustment_set_is_valid():
    assert dag.satisfies_backdoor(dag.ADJUST)
    assert dag.check_adjustment(dag.ADJUST) == dag.ADJUST


def test_no_adjustment_is_confounded():
    assert not dag.satisfies_backdoor([])


def test_dropping_sun_uv_opens_backdoor():
    assert not dag.satisfies_backdoor([c for c in dag.ADJUST if c != "f107"])


@pytest.mark.parametrize("bad", dag.NEVER_ADJUST)
def test_mediators_and_descendants_are_rejected(bad):
    assert bad in dag.descendants(dag.TREATMENT)
    assert not dag.satisfies_backdoor(dag.ADJUST + [bad])
    with pytest.raises(ValueError, match="Never adjust"):
        dag.check_adjustment(dag.ADJUST + [bad])


def test_d_separation_basics():
    # Chain blocked by its middle node, collider opened by conditioning on it
    edges = [("a", "b"), ("b", "c"), ("x", "k"), ("y", "k")]
    assert not dag.d_separated("a", "c", [], edges)
    assert dag.d_separated("a", "c", ["b"], edges)
    assert dag.d_separated("x", "y", [], edges)
    assert not dag.d_separated("x", "y", ["k"], edges)


def test_dot_output():
    dot = dag.to_dot()
    assert dot.startswith("digraph") and "T_ap_int -> air_density;" in dot


def test_tle_artifacts_are_filtered_not_adjusted():
    # The artifact path starts at the storm, so it is not a back-door path ...
    assert "tle_artifact" in dag.descendants(dag.TREATMENT)
    # ... and eccentricity only reaches the outcome through artifacts
    assert dag.children("ecc") == {"tle_artifact"}
