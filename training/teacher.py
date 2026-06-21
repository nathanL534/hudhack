"""training/teacher.py — the Teacher policy (the thing RFT actually trains).

The Teacher reads a Game's param schema and emits a `params` dict for that Game.
RFT then optimizes the Teacher so the params it emits maximize the learnability
reward (training/reward.py).

I/O contract (pin this — rft_run.py and eval depend on it):

    Teacher.generate(game) -> params: dict
        input:  a Game (engine/games.py) — read game.param_schema for valid keys
                and ranges, and game.defaults for a safe fallback.
        output: a dict whose keys are a subset of game.param_schema, each value
                inside the schema's (low, high) range. Must be JSON-serializable
                (it crosses the Fireworks boundary as plain JSON).

Right now `generate` is a STUB: it samples uniformly-random valid params. That
keeps the whole pipeline runnable on day one. The real Teacher is a prompted
Qwen3-4B served by Fireworks.

TODO(rag26): replace RandomTeacher.generate with a Fireworks call:
  - prompt Qwen3-4B with game.name + game.param_schema, ask for a JSON params obj
  - parse + clamp the JSON to the schema (reuse _clamp_to_schema below)
  - this generate() becomes the `rollout` the RFT RemoteRolloutProcessor scores
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow `python training/teacher.py` from repo root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.games import Game


def _clamp_to_schema(params: dict, game: Game) -> dict:
    """Keep only schema keys, clamp each into its (low, high) range."""
    out = {}
    for key, (low, high) in game.param_schema.items():
        val = float(params.get(key, game.defaults.get(key, (low + high) / 2)))
        out[key] = float(min(high, max(low, val)))
    return out


class Teacher:
    """Base Teacher interface. Subclass and override generate()."""

    def generate(self, game: Game) -> dict:  # pragma: no cover - interface only
        raise NotImplementedError


class RandomTeacher(Teacher):
    """STUB Teacher: emits uniformly-random valid params. Replace with Qwen3-4B."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def generate(self, game: Game) -> dict:
        raw = {
            key: float(self._rng.uniform(low, high))
            for key, (low, high) in game.param_schema.items()
        }
        return _clamp_to_schema(raw, game)


# TODO(rag26): class FireworksTeacher(Teacher): ... wraps Qwen3-4B via fireworks-ai.


def default_teacher(seed: int = 0) -> Teacher:
    """Factory the pipeline imports. Swap to FireworksTeacher when wired."""
    return RandomTeacher(seed=seed)


if __name__ == "__main__":
    from engine.games import GAME_1

    t = default_teacher(seed=1)
    for _ in range(3):
        print(t.generate(GAME_1))
