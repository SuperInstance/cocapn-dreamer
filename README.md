# cocapn-dreamer

> Speculative execution and "dreaming" for agents — explore possible futures before committing to action.

Cocapn Dreamer lets agents generate speculative scenarios, explore branching outcome trees, and evaluate which futures are most desirable — all without taking real actions.

## Features

- **Scenario generation** — Create branching possible futures with probability estimates
- **Tree exploration** — Depth-first, breadth-first, and Monte Carlo tree search through scenario space
- **Evaluation** — Score scenarios by custom desirability criteria
- **Dream memory** — Store and recall past explorations to learn from speculative runs
- **Pure Python** — No external dependencies beyond `pytest` for testing

## Quick Start

```python
from cocapn_dreamer import Dreamer, Scenario, Evaluator, Explorer

# Create a dreamer
dreamer = Dreamer(name="planner")

# Define a starting state
state = {"action": "launch_product", "budget": 10000, "users": 0}

# Generate speculative scenarios
tree = dreamer.dream(state, depth=3, branches=2)

# Evaluate them
evaluator = Evaluator(criteria={"users": 1.0, "budget": 0.5})
ranked = evaluator.rank(tree)

for scenario in ranked[:3]:
    print(f"{scenario.probability:.2f} | score={scenario.score:.2f} | {scenario.state}")
```

## Architecture

- **`Scenario`** — A possible future state with probability, parent/children links, and metadata
- **`Dreamer`** — Generates scenario trees from an initial state using configurable generators
- **`Explorer`** — Searches through scenario trees (DFS, BFS, MCTS)
- **`Evaluator`** — Scores and ranks scenarios by desirability
- **`DreamMemory`** — Persists past dream sessions for recall and learning

## License

[MIT](LICENSE) © 2026 Cocapn
