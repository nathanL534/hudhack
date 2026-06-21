"""engine/games.py — the Game registry.

A "Game" is the thing the Teacher is asked to generate environments FOR. It
pins three things:

  1. a `name`
  2. a `param_schema` — the keys (and ranges) the Teacher is allowed to emit
  3. `to_gridworld_params(params)` — how a Game's params map onto the concrete
     GridWorld param surface in gridworld.py

GAME_1 is the training game. GAME_2 is the HELD-OUT anti-gaming game — a
different mapping so the Teacher can't memorize GAME_1's quirks. If a policy
learned on GAME_1 also makes good GAME_2 envs, the signal is real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Game:
    name: str
    # param_schema: key -> (low, high) allowed range for a Teacher-emitted value.
    param_schema: dict[str, tuple[float, float]]
    # Maps a Game param dict onto gridworld params. Pure, deterministic.
    to_gridworld_params: Callable[[dict], dict]
    description: str = ""
    # Default params a stub Teacher can fall back to.
    defaults: dict = field(default_factory=dict)


def _game1_map(params: dict) -> dict:
    """GAME_1: difficulty is the primary dial; size derived from it."""
    return {
        "difficulty": params.get("difficulty", 0.5),
        "obstacle_density": params.get("obstacle_density", 0.15),
        "goal_distance": params.get("goal_distance", 0.6),
    }


def _game2_map(params: dict) -> dict:
    """GAME_2 (held-out): same knobs, DIFFERENT mapping.

    GAME_2 emphasizes obstacle layout over raw distance, so a Teacher that only
    learned "crank distance on GAME_1" will NOT transfer. This is the anti-gaming
    test surface. TODO(rag26): make GAME_2 a genuinely distinct generator (e.g.
    a maze topology) once GAME_1 signal is locked in.
    """
    return {
        "difficulty": params.get("difficulty", 0.5),
        "obstacle_density": params.get("obstacle_density", 0.30),
        "goal_distance": params.get("goal_distance", 0.4),
    }


GAME_1 = Game(
    name="reach_goal_v1",
    param_schema={
        "difficulty": (0.0, 1.0),
        "obstacle_density": (0.0, 0.45),
        "goal_distance": (0.1, 1.0),
    },
    to_gridworld_params=_game1_map,
    description="Reach the goal on a configurable gridworld (training game).",
    defaults={"difficulty": 0.5, "obstacle_density": 0.15, "goal_distance": 0.6},
)

GAME_2 = Game(
    name="reach_goal_heldout_v1",
    param_schema={
        "difficulty": (0.0, 1.0),
        "obstacle_density": (0.0, 0.45),
        "goal_distance": (0.1, 1.0),
    },
    to_gridworld_params=_game2_map,
    description="Held-out anti-gaming game; distinct mapping from GAME_1.",
    defaults={"difficulty": 0.5, "obstacle_density": 0.30, "goal_distance": 0.4},
)

GAMES = {GAME_1.name: GAME_1, GAME_2.name: GAME_2}
