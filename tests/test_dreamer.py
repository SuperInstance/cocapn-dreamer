"""Tests for cocapn_dreamer.dreamer."""

import pytest
from cocapn_dreamer.dreamer import Dreamer
from cocapn_dreamer.scenario import Scenario


class TestDreamerBasic:
    def test_default_dream_produces_tree(self):
        d = Dreamer(seed=42)
        root = d.dream({"x": 10, "y": 5})
        assert isinstance(root, Scenario)
        assert root.state == {"x": 10, "y": 5}
        assert len(root.children) > 0

    def test_dream_respects_depth(self):
        d = Dreamer(seed=42, max_depth=2, branches=1)
        root = d.dream({"val": 100})
        # depth 0 = root, then 2 more levels
        max_depth = max(s.depth for s in root.all_scenarios())
        assert max_depth <= 2

    def test_dream_respects_branches(self):
        d = Dreamer(seed=42, max_depth=1, branches=3)
        root = d.dream({"val": 100})
        assert len(root.children) <= 3

    def test_dream_with_custom_generator(self):
        def gen(state):
            x = state.get("x", 0)
            return [
                ({"x": x + 1}, 0.6, "increment"),
                ({"x": x - 1}, 0.4, "decrement"),
            ]

        d = Dreamer(generator=gen, max_depth=2, branches=2, seed=0)
        root = d.dream({"x": 0})
        leaves = root.leaves
        # All leaves should have integer x values
        for leaf in leaves:
            assert isinstance(leaf.state["x"], int)

    def test_dream_generator_exception_handled(self):
        def bad_gen(state):
            raise ValueError("boom")

        d = Dreamer(generator=bad_gen, max_depth=2)
        root = d.dream({"x": 1})
        assert len(root.children) == 0

    def test_redream_multiple_trees(self):
        d = Dreamer(seed=42, max_depth=1, branches=2)
        trees = d.redream({"x": 10}, n=3)
        assert len(trees) == 3
        # Each tree should be different (different seeds)
        ids = {t.id for t in trees}
        assert len(ids) == 3

    def test_dream_seed_reproducible(self):
        d1 = Dreamer(seed=99, max_depth=2, branches=2)
        d2 = Dreamer(seed=99, max_depth=2, branches=2)
        r1 = d1.dream({"x": 5})
        r2 = d2.dream({"x": 5})
        # Same seed should produce same structure
        assert len(r1.all_scenarios()) == len(r2.all_scenarios())

    def test_dream_state_isolation(self):
        """Dreaming shouldn't mutate the input state."""
        original = {"x": 10}
        d = Dreamer(seed=42)
        d.dream(original, depth=1)
        assert original == {"x": 10}
