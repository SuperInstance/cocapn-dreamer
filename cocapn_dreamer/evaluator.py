"""Evaluator — scores and ranks scenarios by desirability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cocapn_dreamer.scenario import Scenario

# Type for a custom scoring function
ScoreFn = Callable[[dict[str, Any]], float]


@dataclass
class Evaluator:
    """Scores scenarios based on weighted criteria or a custom function.

    Attributes:
        criteria: Dict mapping state key names to positive weights.
                 Higher weights mean that key matters more.
                 Only numeric state values are considered.
        custom_scorer: Optional custom scoring function. If provided,
                      overrides criteria-based scoring.
        penalty_keys: State keys that should be penalized (negative weight).
        penalty_weight: Weight multiplier for penalty keys.
    """

    criteria: dict[str, float] = field(default_factory=dict)
    custom_scorer: Optional[ScoreFn] = None
    penalty_keys: list[str] = field(default_factory=list)
    penalty_weight: float = 1.0

    def score(self, scenario: Scenario) -> float:
        """Compute and assign a score to a single scenario."""
        if self.custom_scorer:
            s = self.custom_scorer(scenario.state)
        else:
            s = 0.0
            for key, weight in self.criteria.items():
                val = scenario.state.get(key)
                if val is not None and isinstance(val, (int, float)):
                    s += float(val) * weight
            for key in self.penalty_keys:
                val = scenario.state.get(key)
                if val is not None and isinstance(val, (int, float)):
                    s -= float(val) * self.penalty_weight

        # Weight by probability — more likely scenarios should rank higher
        s *= scenario.probability
        scenario.score = s
        return s

    def score_tree(self, root: Scenario) -> None:
        """Score all scenarios in a tree."""
        for scenario in root.all_scenarios():
            self.score(scenario)

    def rank(self, root_or_list: Scenario | list[Scenario], limit: Optional[int] = None) -> list[Scenario]:
        """Score and rank scenarios, highest score first.

        Args:
            root_or_list: Either a root Scenario (scores whole tree) or a list of Scenarios.
            limit: Max number of results to return.
        """
        if isinstance(root_or_list, Scenario):
            scenarios = root_or_list.all_scenarios()
        else:
            scenarios = list(root_or_list)

        for s in scenarios:
            self.score(s)

        ranked = sorted(scenarios, key=lambda s: s.score, reverse=True)
        if limit:
            ranked = ranked[:limit]
        return ranked

    def best(self, root: Scenario) -> Scenario:
        """Return the single best-scoring scenario in the tree."""
        ranked = self.rank(root, limit=1)
        return ranked[0] if ranked else root

    def best_leaf(self, root: Scenario) -> Optional[Scenario]:
        """Return the best-scoring leaf scenario."""
        ranked = self.rank(root.leaves, limit=1)
        return ranked[0] if ranked else None

    def compare(self, a: Scenario, b: Scenario) -> int:
        """Compare two scenarios: returns 1 if a > b, -1 if a < b, 0 if equal."""
        sa = self.score(a)
        sb = self.score(b)
        if sa > sb:
            return 1
        elif sa < sb:
            return -1
        return 0
