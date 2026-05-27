"""Tests for cocapn_dreamer.evaluator."""

import pytest
from cocapn_dreamer.evaluator import Evaluator
from cocapn_dreamer.scenario import Scenario


def _build_tree() -> Scenario:
    root = Scenario(state={"users": 0, "cost": 100}, probability=1.0)
    root.branch(state={"users": 50, "cost": 120}, probability=0.6, label="moderate")
    root.branch(state={"users": 200, "cost": 300}, probability=0.3, label="viral")
    root.branch(state={"users": 5, "cost": 90}, probability=0.1, label="flop")
    return root


class TestEvaluatorBasic:
    def test_score_with_criteria(self):
        ev = Evaluator(criteria={"users": 1.0, "cost": -0.1})
        s = Scenario(state={"users": 100, "cost": 50}, probability=1.0)
        score = ev.score(s)
        assert score == pytest.approx(95.0)

    def test_score_probability_weighted(self):
        ev = Evaluator(criteria={"users": 1.0})
        s1 = Scenario(state={"users": 100}, probability=1.0)
        s2 = Scenario(state={"users": 100}, probability=0.5)
        assert ev.score(s1) > ev.score(s2)

    def test_score_tree(self):
        root = _build_tree()
        ev = Evaluator(criteria={"users": 1.0})
        ev.score_tree(root)
        for s in root.all_scenarios():
            assert s.score != 0 or s.state.get("users", 0) == 0

    def test_rank(self):
        root = _build_tree()
        ev = Evaluator(criteria={"users": 1.0, "cost": -0.5})
        ranked = ev.rank(root)
        assert ranked[0].score >= ranked[-1].score

    def test_best(self):
        root = _build_tree()
        ev = Evaluator(criteria={"users": 1.0, "cost": -0.1})
        best = ev.best(root)
        assert best.score > 0

    def test_best_leaf(self):
        root = _build_tree()
        ev = Evaluator(criteria={"users": 1.0, "cost": -0.1})
        best = ev.best_leaf(root)
        assert best is not None
        assert best.is_leaf()

    def test_compare(self):
        ev = Evaluator(criteria={"x": 1.0})
        a = Scenario(state={"x": 10}, probability=1.0)
        b = Scenario(state={"x": 5}, probability=1.0)
        assert ev.compare(a, b) == 1
        assert ev.compare(b, a) == -1

    def test_rank_with_limit(self):
        root = _build_tree()
        ev = Evaluator(criteria={"users": 1.0})
        ranked = ev.rank(root, limit=2)
        assert len(ranked) == 2

    def test_rank_from_list(self):
        scenarios = [
            Scenario(state={"x": 10}, probability=1.0),
            Scenario(state={"x": 5}, probability=1.0),
            Scenario(state={"x": 20}, probability=1.0),
        ]
        ev = Evaluator(criteria={"x": 1.0})
        ranked = ev.rank(scenarios)
        assert ranked[0].state["x"] == 20
        assert ranked[-1].state["x"] == 5


class TestEvaluatorCustom:
    def test_custom_scorer(self):
        def scorer(state):
            return state.get("happiness", 0) * 2

        ev = Evaluator(custom_scorer=scorer)
        s = Scenario(state={"happiness": 10}, probability=1.0)
        assert ev.score(s) == pytest.approx(20.0)

    def test_penalty_keys(self):
        ev = Evaluator(criteria={"value": 1.0}, penalty_keys=["risk"], penalty_weight=2.0)
        s = Scenario(state={"value": 100, "risk": 10}, probability=1.0)
        score = ev.score(s)
        assert score == pytest.approx(80.0)
