"""Tests for cocapn_dreamer.scenario."""

import pytest
from cocapn_dreamer.scenario import Scenario


class TestScenarioCreation:
    def test_default_scenario(self):
        s = Scenario(state={"x": 1})
        assert s.state == {"x": 1}
        assert s.probability == 1.0
        assert s.score == 0.0
        assert s.children == []
        assert s.parent is None
        assert s.label == ""
        assert len(s.id) == 12

    def test_scenario_with_all_fields(self):
        s = Scenario(
            state={"a": 10},
            probability=0.75,
            label="test",
            score=5.0,
            metadata={"foo": "bar"},
        )
        assert s.probability == 0.75
        assert s.label == "test"
        assert s.score == 5.0
        assert s.metadata["foo"] == "bar"


class TestScenarioTree:
    def test_add_child(self):
        parent = Scenario(state={"x": 1})
        child = Scenario(state={"x": 2}, probability=0.8)
        parent.add_child(child)
        assert child in parent.children
        assert child.parent is parent

    def test_branch(self):
        root = Scenario(state={"x": 1}, probability=1.0)
        child = root.branch(state={"x": 2}, probability=0.5, label="up")
        assert child.state == {"x": 2}
        assert child.parent is root
        assert child in root.children
        assert child.label == "up"
        assert child.probability == pytest.approx(0.5)

    def test_branch_capped_probability(self):
        """Child probability should be capped at parent probability."""
        root = Scenario(state={"x": 1}, probability=0.5)
        child = root.branch(state={"x": 2}, probability=0.9)
        assert child.probability == pytest.approx(0.45)

    def test_depth(self):
        root = Scenario(state={"x": 0})
        c1 = root.branch(state={"x": 1})
        c2 = c1.branch(state={"x": 2})
        c3 = c2.branch(state={"x": 3})
        assert root.depth == 0
        assert c1.depth == 1
        assert c2.depth == 2
        assert c3.depth == 3

    def test_root_property(self):
        root = Scenario(state={"x": 0})
        c1 = root.branch(state={"x": 1})
        c2 = c1.branch(state={"x": 2})
        assert c2.root is root
        assert root.root is root

    def test_path(self):
        root = Scenario(state={"x": 0})
        c1 = root.branch(state={"x": 1})
        c2 = c1.branch(state={"x": 2})
        path = c2.path
        assert len(path) == 3
        assert path[0] is root
        assert path[1] is c1
        assert path[2] is c2

    def test_leaves(self):
        root = Scenario(state={"x": 0})
        c1 = root.branch(state={"x": 1})
        c2 = root.branch(state={"x": 2})
        c1.branch(state={"x": 3})
        leaves = root.leaves
        assert len(leaves) == 2
        assert all(l.is_leaf() for l in leaves)

    def test_all_scenarios(self):
        root = Scenario(state={"x": 0})
        root.branch(state={"x": 1})
        root.branch(state={"x": 2})
        assert len(root.all_scenarios()) == 3

    def test_is_leaf(self):
        root = Scenario(state={"x": 0})
        assert root.is_leaf()
        root.branch(state={"x": 1})
        assert not root.is_leaf()

    def test_repr(self):
        s = Scenario(state={"x": 1}, label="test")
        r = repr(s)
        assert "Scenario" in r
        assert "test" in r
