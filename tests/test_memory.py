"""Tests for cocapn_dreamer.memory."""

import json
import os
import tempfile

import pytest
from cocapn_dreamer.evaluator import Evaluator
from cocapn_dreamer.memory import DreamMemory, DreamRecord
from cocapn_dreamer.scenario import Scenario


def _make_scored_tree() -> Scenario:
    root = Scenario(state={"x": 0}, probability=1.0, label="root")
    root.branch(state={"x": 10}, probability=0.7, label="good")
    root.branch(state={"x": 2}, probability=0.3, label="ok")
    ev = Evaluator(criteria={"x": 1.0})
    ev.score_tree(root)
    return root


class TestDreamMemoryBasic:
    def test_record_dream(self):
        mem = DreamMemory()
        root = _make_scored_tree()
        rec = mem.record(root, name="test_dream")
        assert rec.name == "test_dream"
        assert rec.scenario_count == 3
        assert rec.best_score > 0

    def test_record_with_labels(self):
        mem = DreamMemory()
        root = _make_scored_tree()
        rec = mem.record(root, name="labeled", labels=["experiment", "v1"])
        assert "experiment" in rec.labels

    def test_recall_by_name(self):
        mem = DreamMemory()
        root = _make_scored_tree()
        mem.record(root, name="alpha")
        mem.record(root, name="beta run")
        results = mem.recall(name="alpha")
        assert len(results) == 1
        assert results[0].name == "alpha"

    def test_recall_by_labels(self):
        mem = DreamMemory()
        root = _make_scored_tree()
        mem.record(root, name="a", labels=["exp1"])
        mem.record(root, name="b", labels=["exp2"])
        mem.record(root, name="c", labels=["exp1", "exp2"])
        results = mem.recall(labels=["exp1"])
        assert len(results) == 2

    def test_recall_best(self):
        mem = DreamMemory()
        root = _make_scored_tree()
        mem.record(root, name="first")
        # Create a better tree
        root2 = Scenario(state={"x": 0}, probability=1.0)
        root2.branch(state={"x": 1000}, probability=0.5)
        ev = Evaluator(criteria={"x": 1.0})
        ev.score_tree(root2)
        mem.record(root2, name="better")
        best = mem.recall_best()
        assert best.name == "better"

    def test_stats(self):
        mem = DreamMemory()
        root = _make_scored_tree()
        mem.record(root, name="a")
        mem.record(root, name="b")
        stats = mem.stats()
        assert stats["total_dreams"] == 2
        assert stats["total_scenarios_explored"] == 6

    def test_stats_empty(self):
        mem = DreamMemory()
        stats = mem.stats()
        assert stats["total_dreams"] == 0

    def test_max_records_eviction(self):
        mem = DreamMemory(max_records=3)
        for i in range(5):
            root = _make_scored_tree()
            mem.record(root, name=f"dream_{i}")
        assert len(mem.records) == 3
        # Oldest should be evicted
        names = [r.name for r in mem.records]
        assert "dream_0" not in names

    def test_clear(self):
        mem = DreamMemory()
        root = _make_scored_tree()
        mem.record(root, name="test")
        mem.clear()
        assert len(mem.records) == 0


class TestDreamMemoryPersistence:
    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mem = DreamMemory(persist_path=path)
            root = _make_scored_tree()
            mem.record(root, name="persistent")

            # Create new memory instance loading from same file
            mem2 = DreamMemory(persist_path=path)
            assert len(mem2.records) == 1
            assert mem2.records[0].name == "persistent"
        finally:
            os.unlink(path)

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "memory.json")
            mem = DreamMemory(persist_path=path)
            root = _make_scored_tree()
            mem.record(root, name="nested")
            assert os.path.exists(path)

    def test_load_missing_file(self):
        mem = DreamMemory(persist_path="/tmp/nonexistent_dream_memory.json")
        assert len(mem.records) == 0
