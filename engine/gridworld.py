"""engine/gridworld.py — a REAL minimal, configurable, deterministic gridworld.

The Teacher's whole job is to emit `params` for this environment. So the
parameter surface here IS the action space of the RL problem. Keep it small but
real:

    params = {
        "size":             int   grid is size x size
        "obstacle_density": float in [0, 1], fraction of free cells that are walls
        "goal_distance":    float in [0, 1], how far (Manhattan, normalized) the
                                  goal is placed from the start
        "difficulty":       float in [0, 1], a master knob that scales ALL axes
    }

`difficulty` is a convenience dial: when the Teacher only wants to push one
number, difficulty scales size, obstacle_density and goal_distance together (see
`effective_params`). Explicit per-axis values, when given, are blended with the
difficulty-scaled defaults.

Determinism: every method takes (or is constructed with) a `seed`. Same params +
same seed => identical grid, identical rollouts. This is non-negotiable — the
learnability signal and the head-to-head eval both depend on it.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np

# Cell encodings on the grid.
EMPTY = 0
WALL = 1
START = 2
GOAL = 3

# Action ids. Order matters: BFS and the scripted policy iterate this.
ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

# Bounds the Teacher's params get clamped to. Keeps grids small + runnable.
MIN_SIZE = 4
MAX_SIZE = 12


class GridParams(TypedDict, total=False):
    size: int
    obstacle_density: float
    goal_distance: float
    difficulty: float


def effective_params(params: GridParams) -> dict:
    """Resolve a (possibly partial) Teacher param dict into concrete axis values.

    `difficulty` provides defaults for any axis the Teacher did not pin. An axis
    that IS pinned overrides the difficulty default. Everything is clamped to a
    runnable range so a bad Teacher can never crash the engine.
    """
    difficulty = float(params.get("difficulty", 0.5))
    difficulty = min(1.0, max(0.0, difficulty))

    # difficulty -> default per-axis values.
    size_default = MIN_SIZE + round(difficulty * (MAX_SIZE - MIN_SIZE))
    density_default = 0.05 + difficulty * 0.30          # 5%..35% walls
    goal_default = 0.30 + difficulty * 0.65             # near .. far

    size = int(params.get("size", size_default))
    size = min(MAX_SIZE, max(MIN_SIZE, size))

    obstacle_density = float(params.get("obstacle_density", density_default))
    obstacle_density = min(0.45, max(0.0, obstacle_density))

    goal_distance = float(params.get("goal_distance", goal_default))
    goal_distance = min(1.0, max(0.1, goal_distance))

    return {
        "size": size,
        "obstacle_density": obstacle_density,
        "goal_distance": goal_distance,
        "difficulty": difficulty,
    }


class GridWorld:
    """A deterministic, fully-observable gridworld.

    Construct with resolved params + a seed, then call reset()/step(). The agent
    moves on a `size x size` grid from START to GOAL, avoiding WALLs. Reward is
    sparse: +1 on reaching the goal, 0 otherwise, with a small step penalty so
    that "score" rewards shorter solutions.
    """

    def __init__(self, params: GridParams, seed: int, max_steps: Optional[int] = None):
        self.params = effective_params(params)
        self.seed = int(seed)
        self.size = self.params["size"]
        self.max_steps = max_steps if max_steps is not None else self.size * self.size * 2
        self._rng = np.random.default_rng(self.seed)
        self.grid: np.ndarray = np.zeros((self.size, self.size), dtype=np.int8)
        self.start: tuple[int, int] = (0, 0)
        self.goal: tuple[int, int] = (self.size - 1, self.size - 1)
        self.agent: tuple[int, int] = (0, 0)
        self.steps_taken = 0
        self._build()

    # --- construction -----------------------------------------------------

    def _build(self) -> None:
        """Lay out start, goal, and walls deterministically from the seed."""
        n = self.size
        self.grid = np.zeros((n, n), dtype=np.int8)

        # Start fixed at top-left for determinism + an easy BFS anchor.
        self.start = (0, 0)

        # Goal placed at a Manhattan distance scaled by goal_distance.
        max_manhattan = 2 * (n - 1)
        target_dist = max(1, int(round(self.params["goal_distance"] * max_manhattan)))
        self.goal = self._place_goal(target_dist)

        # Scatter walls over free cells, never on start/goal.
        free = [
            (r, c)
            for r in range(n)
            for c in range(n)
            if (r, c) != self.start and (r, c) != self.goal
        ]
        n_walls = int(round(self.params["obstacle_density"] * len(free)))
        if n_walls > 0:
            idx = self._rng.choice(len(free), size=n_walls, replace=False)
            for i in idx:
                r, c = free[i]
                self.grid[r, c] = WALL

        self.grid[self.start] = START
        self.grid[self.goal] = GOAL

        # Guarantee solvability: if walls blocked the only path, clear a route.
        # The learnability signal needs the STRONG player to actually be able to
        # win, otherwise strong_score collapses and every env looks unlearnable.
        if self.bfs_path() is None:
            self._carve_path()

    def _place_goal(self, target_dist: int) -> tuple[int, int]:
        n = self.size
        candidates = [
            (r, c)
            for r in range(n)
            for c in range(n)
            if abs(r) + abs(c) == target_dist and (r, c) != self.start
        ]
        if not candidates:
            return (n - 1, n - 1)
        return candidates[int(self._rng.integers(len(candidates)))]

    def _carve_path(self) -> None:
        """Clear walls along a simple L-path start->goal to ensure solvability."""
        (sr, sc), (gr, gc) = self.start, self.goal
        for c in range(min(sc, gc), max(sc, gc) + 1):
            if self.grid[sr, c] == WALL:
                self.grid[sr, c] = EMPTY
        for r in range(min(sr, gr), max(sr, gr) + 1):
            if self.grid[r, gc] == WALL:
                self.grid[r, gc] = EMPTY
        self.grid[self.start] = START
        self.grid[self.goal] = GOAL

    # --- gym-like API -----------------------------------------------------

    def reset(self) -> tuple[int, int]:
        self.agent = self.start
        self.steps_taken = 0
        return self.agent

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        """Apply an action. Returns (state, reward, done).

        Reward: +1.0 for reaching goal, minus a tiny per-step penalty so faster
        solutions score higher. Walls/edges block movement (no-op move).
        """
        dr, dc = ACTIONS[action]
        r, c = self.agent
        nr, nc = r + dr, c + dc

        if self._passable(nr, nc):
            self.agent = (nr, nc)

        self.steps_taken += 1
        done = self.agent == self.goal or self.steps_taken >= self.max_steps

        if self.agent == self.goal:
            reward = 1.0
        else:
            reward = -1.0 / self.max_steps  # small shaping toward shorter paths
        return self.agent, reward, done

    def _passable(self, r: int, c: int) -> bool:
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        return self.grid[r, c] != WALL

    # --- planning helper (used by scripted_optimal) -----------------------

    def bfs_path(self) -> Optional[list[int]]:
        """Return the shortest action sequence start->goal, or None if blocked.

        This is the engine's ground-truth optimal planner; policies.py wraps it.
        """
        from collections import deque

        start, goal = self.start, self.goal
        visited = {start}
        queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start, [])])
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                return path
            for a, (dr, dc) in ACTIONS.items():
                nr, nc = r + dr, c + dc
                if self._passable(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [a]))
        return None


if __name__ == "__main__":
    # Smoke test: build a grid, confirm it's solvable + deterministic.
    p = {"difficulty": 0.5}
    a = GridWorld(p, seed=7)
    b = GridWorld(p, seed=7)
    assert np.array_equal(a.grid, b.grid), "non-deterministic build!"
    path = a.bfs_path()
    print(f"size={a.size} goal={a.goal} bfs_len={None if path is None else len(path)}")
    print(a.grid)
