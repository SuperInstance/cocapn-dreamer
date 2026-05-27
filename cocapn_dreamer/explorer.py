"""Explorer — tree search through scenario space (DFS, BFS, MCTS)."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from cocapn_dreamer.scenario import Scenario


@dataclass
class Explorer:
    """Searches through scenario trees using various strategies.

    Attributes:
        strategy: Search strategy — 'dfs', 'bfs', or 'mcts'.
        max_visits: Maximum nodes to visit (for MCTS termination).
        exploration_weight: UCB1 exploration parameter for MCTS.
    """

    strategy: str = "bfs"
    max_visits: int = 100
    exploration_weight: float = 1.414  # sqrt(2)

    def explore(self, root: Scenario) -> list[Scenario]:
        """Explore a scenario tree and return visited scenarios in order."""
        if self.strategy == "dfs":
            return self._dfs(root)
        elif self.strategy == "bfs":
            return self._bfs(root)
        elif self.strategy == "mcts":
            return self._mcts(root)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def _dfs(self, root: Scenario) -> list[Scenario]:
        """Depth-first traversal."""
        visited: list[Scenario] = []
        stack: list[Scenario] = [root]
        while stack:
            node = stack.pop()
            visited.append(node)
            # Push children in reverse so leftmost is visited first
            for child in reversed(node.children):
                stack.append(child)
        return visited

    def _bfs(self, root: Scenario) -> list[Scenario]:
        """Breadth-first traversal."""
        visited: list[Scenario] = []
        queue: deque[Scenario] = deque([root])
        while queue:
            node = queue.popleft()
            visited.append(node)
            for child in node.children:
                queue.append(child)
        return visited

    def _mcts(self, root: Scenario) -> list[Scenario]:
        """Monte Carlo Tree Search over scenario tree.

        Uses UCB1 for selection. Simulates random rollouts and backpropagates scores.
        """
        visits: dict[str, int] = {}
        total_scores: dict[str, float] = {}
        visited_nodes: list[Scenario] = []

        for scenario in root.all_scenarios():
            visits[scenario.id] = 0
            total_scores[scenario.id] = 0.0

        for _ in range(self.max_visits):
            # Selection: walk down using UCB1
            node = self._select_ucb1(root, visits, total_scores)
            visited_nodes.append(node)

            # Simulation: random rollout to a leaf
            leaf = self._rollout(node)

            # Evaluate the leaf
            score = self._evaluate_leaf(leaf)

            # Backpropagate
            self._backpropagate(node, score, visits, total_scores)

        # Store visit counts and average scores in metadata
        for scenario in root.all_scenarios():
            v = visits.get(scenario.id, 0)
            scenario.metadata["mcts_visits"] = v
            scenario.metadata["mcts_avg_score"] = total_scores.get(scenario.id, 0.0) / max(v, 1)

        return visited_nodes

    def _select_ucb1(
        self, root: Scenario, visits: dict[str, int], total_scores: dict[str, float]
    ) -> Scenario:
        """Select a node using UCB1, expanding unvisited children first."""
        node = root
        while node.children:
            # Pick unvisited child if any
            unvisited = [c for c in node.children if visits.get(c.id, 0) == 0]
            if unvisited:
                return random.choice(unvisited)

            # Otherwise use UCB1
            best_child = None
            best_value = float("-inf")
            parent_visits = visits.get(node.id, 1)

            for child in node.children:
                child_visits = visits.get(child.id, 1)
                avg = total_scores.get(child.id, 0.0) / child_visits
                ucb = avg + self.exploration_weight * math.sqrt(
                    math.log(parent_visits) / child_visits
                )
                if ucb > best_value:
                    best_value = ucb
                    best_child = child

            if best_child is None:
                return node
            node = best_child

        return node

    def _rollout(self, node: Scenario) -> Scenario:
        """Random walk down to a leaf."""
        current = node
        while current.children:
            current = random.choice(current.children)
        return current

    def _evaluate_leaf(self, leaf: Scenario) -> float:
        """Evaluate a leaf — simple heuristic based on state values."""
        score = 0.0
        for key, val in leaf.state.items():
            if isinstance(val, (int, float)):
                score += float(val)
        return score * leaf.probability

    def _backpropagate(
        self,
        node: Scenario,
        score: float,
        visits: dict[str, int],
        total_scores: dict[str, float],
    ) -> None:
        """Backpropagate score up to root."""
        current: Optional[Scenario] = node
        while current:
            visits[current.id] = visits.get(current.id, 0) + 1
            total_scores[current.id] = total_scores.get(current.id, 0.0) + score
            current = current.parent

    def find_best_path(self, root: Scenario) -> list[Scenario]:
        """Find the highest-scoring path from root to a leaf.

        Uses greedy selection based on scenario scores.
        """
        path: list[Scenario] = [root]
        current = root
        while current.children:
            best = max(current.children, key=lambda c: c.score)
            path.append(best)
            current = best
        return path

    def find_by_probability(self, root: Scenario, threshold: float = 0.5) -> list[Scenario]:
        """Return all scenarios with probability above a threshold."""
        return [s for s in root.all_scenarios() if s.probability >= threshold]
