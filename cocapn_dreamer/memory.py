"""DreamMemory — store and recall past dream explorations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from cocapn_dreamer.scenario import Scenario


@dataclass
class DreamRecord:
    """A stored record of a single dream session.

    Attributes:
        id: Unique identifier for this record.
        timestamp: Unix timestamp when the dream was recorded.
        name: Human-readable name for the dream session.
        initial_state: The starting state that was dreamt from.
        scenario_count: Total number of scenarios generated.
        best_score: Best scenario score found.
        best_state: State of the best-scoring scenario.
        labels: Tags/categories for this dream.
    """

    id: str
    timestamp: float
    name: str = ""
    initial_state: dict[str, Any] = field(default_factory=dict)
    scenario_count: int = 0
    best_score: float = 0.0
    best_state: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)


@dataclass
class DreamMemory:
    """Stores and recalls past dream sessions.

    Can persist to a JSON file for cross-session memory.

    Attributes:
        records: Stored dream records.
        max_records: Maximum records to keep (oldest evicted first).
        persist_path: Optional file path to save/load records.
    """

    records: list[DreamRecord] = field(default_factory=list)
    max_records: int = 100
    persist_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.persist_path:
            self.load()

    def record(
        self,
        root: Scenario,
        name: str = "",
        labels: Optional[list[str]] = None,
    ) -> DreamRecord:
        """Record a dream session from a scenario tree root."""
        all_scenarios = root.all_scenarios()
        best = max(all_scenarios, key=lambda s: s.score) if all_scenarios else root

        rec = DreamRecord(
            id=root.id,
            timestamp=time.time(),
            name=name,
            initial_state=root.state,
            scenario_count=len(all_scenarios),
            best_score=best.score,
            best_state=best.state,
            labels=labels or [],
        )

        self.records.append(rec)

        # Evict oldest if over limit
        while len(self.records) > self.max_records:
            self.records.pop(0)

        if self.persist_path:
            self.save()

        return rec

    def recall(self, name: Optional[str] = None, labels: Optional[list[str]] = None, limit: int = 10) -> list[DreamRecord]:
        """Recall past dream records, optionally filtered by name/labels."""
        results = self.records

        if name:
            results = [r for r in results if name.lower() in r.name.lower()]

        if labels:
            label_set = set(labels)
            results = [r for r in results if label_set & set(r.labels)]

        return list(reversed(results[-limit:]))

    def recall_best(self) -> Optional[DreamRecord]:
        """Return the record with the highest best_score."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.best_score)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics over all stored records."""
        if not self.records:
            return {"total_dreams": 0}
        scores = [r.best_score for r in self.records]
        return {
            "total_dreams": len(self.records),
            "avg_best_score": sum(scores) / len(scores),
            "max_best_score": max(scores),
            "min_best_score": min(scores),
            "total_scenarios_explored": sum(r.scenario_count for r in self.records),
        }

    def save(self) -> None:
        """Persist records to a JSON file."""
        if not self.persist_path:
            return
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for r in self.records:
            data.append({
                "id": r.id,
                "timestamp": r.timestamp,
                "name": r.name,
                "initial_state": r.initial_state,
                "scenario_count": r.scenario_count,
                "best_score": r.best_score,
                "best_state": r.best_state,
                "labels": r.labels,
            })
        path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Load records from a JSON file."""
        if not self.persist_path:
            return
        path = Path(self.persist_path)
        if not path.exists():
            return
        text = path.read_text().strip()
        if not text:
            return
        data = json.loads(text)
        self.records.clear()
        for d in data:
            self.records.append(DreamRecord(
                id=d["id"],
                timestamp=d["timestamp"],
                name=d.get("name", ""),
                initial_state=d.get("initial_state", {}),
                scenario_count=d.get("scenario_count", 0),
                best_score=d.get("best_score", 0.0),
                best_state=d.get("best_state", {}),
                labels=d.get("labels", []),
            ))

    def clear(self) -> None:
        """Clear all records."""
        self.records.clear()
        if self.persist_path:
            self.save()
