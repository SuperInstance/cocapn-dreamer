"""Dreamer — generates speculative scenario trees from an initial state."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cocapn_dreamer.scenario import Scenario

# Type for a generator function: takes current state, returns list of (new_state, probability, label)
GeneratorFn = Callable[[dict[str, Any]], list[tuple[dict[str, Any], float, str]]]


@dataclass
class Dreamer:
    """Generates trees of speculative scenarios by repeatedly applying generator functions.

    A Dreamer explores possible futures by branching from an initial state.
    Each branching step uses a generator function that produces child states
    with associated probabilities.

    Attributes:
        name: Identifier for this dreamer instance.
        generator: The function used to produce child states from a given state.
        max_depth: Maximum tree depth to explore.
        branches: Number of branches to generate per node.
        seed: Optional random seed for reproducibility.
    """

    name: str = "dreamer"
    generator: Optional[GeneratorFn] = None
    max_depth: int = 3
    branches: int = 2
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)

    def dream(
        self,
        initial_state: dict[str, Any],
        depth: Optional[int] = None,
        branches: Optional[int] = None,
        generator: Optional[GeneratorFn] = None,
    ) -> Scenario:
        """Generate a scenario tree from an initial state.

        Returns the root Scenario with children populated.
        """
        depth = depth or self.max_depth
        branches = branches or self.branches
        gen = generator or self.generator or self._default_generator

        root = Scenario(state=copy.deepcopy(initial_state), label="root", probability=1.0)
        self._expand(root, gen, remaining_depth=depth, branches=branches)
        return root

    def _expand(
        self,
        scenario: Scenario,
        generator: GeneratorFn,
        remaining_depth: int,
        branches: int,
    ) -> None:
        """Recursively expand a scenario node."""
        if remaining_depth <= 0:
            return

        try:
            candidates = generator(scenario.state)
        except Exception:
            return

        # Take up to `branches` candidates
        selected = candidates[:branches]
        if not selected:
            return

        for new_state, prob, label in selected:
            child = scenario.branch(
                state=copy.deepcopy(new_state),
                probability=prob,
                label=label,
            )
            self._expand(child, generator, remaining_depth - 1, branches)

    @staticmethod
    def _default_generator(state: dict[str, Any]) -> list[tuple[dict[str, Any], float, str]]:
        """Simple default generator that perturbs numeric values."""
        results: list[tuple[dict[str, Any], float, str]] = []
        new_state = copy.deepcopy(state)
        for key, val in state.items():
            if isinstance(val, (int, float)):
                delta = random.uniform(-0.2, 0.2)
                new_state[key] = round(val * (1 + delta), 4)
        prob = round(random.uniform(0.3, 0.9), 3)
        results.append((new_state, prob, "perturbed"))
        # Second branch: smaller perturbation
        new_state2 = copy.deepcopy(state)
        for key, val in state.items():
            if isinstance(val, (int, float)):
                delta = random.uniform(-0.05, 0.05)
                new_state2[key] = round(val * (1 + delta), 4)
        prob2 = round(random.uniform(0.5, 1.0), 3)
        results.append((new_state2, prob2, "stable"))
        return results

    def redream(
        self,
        root_state: dict[str, Any],
        n: int = 5,
        depth: Optional[int] = None,
        branches: Optional[int] = None,
    ) -> list[Scenario]:
        """Run dream() multiple times with different random seeds, returning all root scenarios."""
        results: list[Scenario] = []
        for i in range(n):
            seed_val = (self.seed or 0) + i + 1
            rng_state = random.getstate()
            random.seed(seed_val)
            tree = self.dream(root_state, depth=depth, branches=branches)
            random.setstate(rng_state)
            results.append(tree)
        return results
