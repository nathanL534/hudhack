"""engine/policies.py — the WEAK and STRONG players.

The learnability signal is the gap between these two on the same env:

    weak  = random_policy()      a no-knowledge agent
    strong = scripted_optimal()  BFS-to-goal, the engine's ground truth

A learnable env is one where strong reliably wins and weak sometimes-but-not-
always wins (p in the middle band). Both are REAL here — no stubs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from engine.gridworld import ACTIONS, GridWorld

# A Policy maps a GridWorld (its current state is on the object) to an action id.
Policy = Callable[[GridWorld], int]


def random_policy(seed: int = 0) -> Policy:
    """WEAK player: picks a uniformly random action each step."""
    rng = np.random.default_rng(seed)

    def act(_env: GridWorld) -> int:
        return int(rng.integers(len(ACTIONS)))

    return act


def scripted_optimal() -> Policy:
    """STRONG player: follows the BFS shortest path to the goal.

    Replans from the agent's current cell each call, so it's robust even if the
    agent is perturbed. Falls back to a random-ish move if no path exists (it
    shouldn't — gridworld guarantees solvability).
    """
    _plan: dict = {"path": None, "idx": 0, "from": None}

    def act(env: GridWorld) -> int:
        # Recompute the plan from the agent's current position.
        if _plan["from"] != env.agent or _plan["path"] is None:
            path = _bfs_from(env, env.agent)
            _plan["path"] = path
            _plan["idx"] = 0
            _plan["from"] = env.agent
        path = _plan["path"]
        if not path:
            return 0  # no move available; engine treats as no-op step
        action = path[_plan["idx"]] if _plan["idx"] < len(path) else path[-1]
        _plan["idx"] += 1
        return action

    return act


def _bfs_from(env: GridWorld, source: tuple[int, int]) -> list[int]:
    """Shortest action list from an arbitrary source to env.goal (or [])."""
    from collections import deque

    goal = env.goal
    visited = {source}
    queue: deque[tuple[tuple[int, int], list[int]]] = deque([(source, [])])
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for a, (dr, dc) in ACTIONS.items():
            nr, nc = r + dr, c + dc
            if env._passable(nr, nc) and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [a]))
    return []
