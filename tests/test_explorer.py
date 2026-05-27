"""Tests for cocapn_dreamer.explorer."""

import pytest
from cocapn_dreamer.explorer import Explorer
from cocapn_dreamer.scenario import Scenario


def _build_tree() -> Scenario:
    """Build a simple test tree:
         root
        /    \
       a      b
      / \    / \
     d   e  f   g
    """
    root = Scenario(state={"val": 0}, label="root")
    a = root.branch(state={"val": 10}, probability=0.6, label="a")
    b = root.branch(state={"val": 20}, probability=0.4, label="b")
    a.branch(state={"val": 100}, probability=0.3, label="d")
    a.branch(state={"val": 101}, probability=0.3, label="e")
    b.branch(state={"val": 200}, probability=0.2, label="f")
    b.branch(state={"val": 201}, probability=0.2, label="g")
    return root


class TestExplorerDFS:
    def test_dfs_visits_all(self):
        root = _build_tree()
        ex = Explorer(strategy="dfs")
        visited = ex.explore(root)
        assert len(visited) == 7

    def test_dfs_root_first(self):
        root = _build_tree()
        ex = Explorer(strategy="dfs")
        visited = ex.explore(root)
        assert visited[0] is root


class TestExplorerBFS:
    def test_bfs_visits_all(self):
        root = _build_tree()
        ex = Explorer(strategy="bfs")
        visited = ex.explore(root)
        assert len(visited) == 7

    def test_bfs_order(self):
        root = _build_tree()
        ex = Explorer(strategy="bfs")
        visited = ex.explore(root)
        assert visited[0] is root
        # Next level should be a and b
        level1 = {visited[1].label, visited[2].label}
        assert level1 == {"a", "b"}


class TestExplorerMCTS:
    def test_mcts_returns_nodes(self):
        root = _build_tree()
        ex = Explorer(strategy="mcts", max_visits=20)
        visited = ex.explore(root)
        assert len(visited) == 20

    def test_mcts_sets_metadata(self):
        root = _build_tree()
        ex = Explorer(strategy="mcts", max_visits=20)
        ex.explore(root)
        for s in root.all_scenarios():
            assert "mcts_visits" in s.metadata
            assert "mcts_avg_score" in s.metadata


class TestExplorerHelpers:
    def test_find_best_path(self):
        root = _build_tree()
        # Set scores manually
        root.score = 0
        for s in root.all_scenarios():
            if s.label == "a":
                s.score = 10
            elif s.label == "b":
                s.score = 20
            elif s.label == "f":
                s.score = 200
            elif s.label == "g":
                s.score = 100
            elif s.label == "d":
                s.score = 50
            elif s.label == "e":
                s.score = 5

        ex = Explorer()
        path = ex.find_best_path(root)
        labels = [s.label for s in path]
        assert labels == ["root", "b", "f"]

    def test_find_by_probability(self):
        root = _build_tree()
        ex = Explorer()
        high_prob = ex.find_by_probability(root, threshold=0.3)
        # root(1.0), a(0.6), b(0.4); children get prob*parent so d,e=0.18, f,g=0.08
        assert len(high_prob) == 3

    def test_invalid_strategy(self):
        root = Scenario(state={"x": 1})
        ex = Explorer(strategy="invalid")
        with pytest.raises(ValueError):
            ex.explore(root)
