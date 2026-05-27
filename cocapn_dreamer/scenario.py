"""Scenario — a possible future state with branching paths and probability estimates."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Scenario:
    """A single speculative scenario node in a tree of possible futures.

    Attributes:
        state: The hypothesized state at this scenario node.
        probability: Estimated probability of reaching this scenario (0.0–1.0).
        parent: Optional parent scenario this branches from.
        children: Child scenarios (branches from this node).
        label: Human-readable label for the scenario.
        score: Evaluated desirability score (set by Evaluator).
        metadata: Arbitrary extra data attached to this scenario.
        id: Unique identifier for this scenario node.
    """

    state: dict[str, Any]
    probability: float = 1.0
    parent: Optional[Scenario] = field(default=None, repr=False)
    children: list[Scenario] = field(default_factory=list)
    label: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def add_child(self, child: Scenario) -> Scenario:
        """Attach a child scenario and return it."""
        child.parent = self
        child.probability = min(child.probability, self.probability)
        self.children.append(child)
        return child

    def branch(self, state: dict[str, Any], probability: float = 1.0, label: str = "") -> Scenario:
        """Create and attach a child scenario with the given state delta."""
        child = Scenario(
            state=state,
            probability=probability * self.probability,
            parent=self,
            label=label,
        )
        self.children.append(child)
        return child

    @property
    def depth(self) -> int:
        """Depth of this node from the root."""
        d = 0
        node: Optional[Scenario] = self
        while node and node.parent:
            d += 1
            node = node.parent
        return d

    @property
    def root(self) -> Scenario:
        """Walk up to the root scenario."""
        node: Scenario = self
        while node.parent:
            node = node.parent
        return node

    @property
    def path(self) -> list[Scenario]:
        """Path from root to this scenario (inclusive)."""
        result: list[Scenario] = []
        node: Optional[Scenario] = self
        while node:
            result.append(node)
            node = node.parent
        result.reverse()
        return result

    @property
    def leaves(self) -> list[Scenario]:
        """All leaf descendants (scenarios with no children)."""
        if not self.children:
            return [self]
        result: list[Scenario] = []
        for child in self.children:
            result.extend(child.leaves)
        return result

    def all_scenarios(self) -> list[Scenario]:
        """Return all scenarios in this subtree (self + all descendants)."""
        result = [self]
        for child in self.children:
            result.extend(child.all_scenarios())
        return result

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        lbl = f" {self.label!r}" if self.label else ""
        return f"Scenario({self.state}, p={self.probability:.3f}, score={self.score:.2f}{lbl})"
