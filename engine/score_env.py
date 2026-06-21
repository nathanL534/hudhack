"""engine/score_env.py — score a generated environment's learnability.

This is the heart of the reward signal. Given Game params, we run the STRONG
player and the WEAK player over many seeds and return an EnvScore:

    weak_score   = mean success of random_policy
    strong_score = mean success of scripted_optimal
    p            = weak_score   (the solvability dial; see schemas.EnvScore)

We keep TWO implementations:

  * score_env_stub  — returns random values. Lets training/eval run with zero
                      dependence on the engine being correct (useful during the
                      first hours when the engine is half-built).
  * score_env_real  — the real rollout-based scorer.

`score_env` defaults to the REAL one and transparently falls back to the stub if
the engine raises for any reason. Flip CRUCIBLE_FORCE_STUB=1 to force the stub.

TODO(nathan): tune N_ROLLOUTS and the success threshold once prove_signal.py
shows where the median p*(1-p) lands.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow `python engine/score_env.py` from repo root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.games import Game
from engine.gridworld import GridWorld
from engine.policies import Policy, random_policy, scripted_optimal
from schemas import EnvScore

N_ROLLOUTS = 16  # seeds per (weak/strong) evaluation; bump for less variance


def _rollout(env: GridWorld, policy: Policy) -> float:
    """Run one episode; return 1.0 if the goal was reached, else 0.0."""
    env.reset()
    done = False
    success = 0.0
    while not done:
        action = policy(env)
        _state, reward, done = env.step(action)
        if reward >= 1.0:
            success = 1.0
    return success


def score_env_real(params: dict, game: Game, n_rollouts: int = N_ROLLOUTS) -> EnvScore:
    """REAL scorer: roll strong + weak over n_rollouts seeds, measure success."""
    grid_params = game.to_gridworld_params(params)

    strong_hits = 0.0
    weak_hits = 0.0
    for seed in range(n_rollouts):
        # Strong: deterministic BFS — fresh policy per env so its plan resets.
        strong_env = GridWorld(grid_params, seed=seed)
        strong_hits += _rollout(strong_env, scripted_optimal())

        # Weak: random policy, its own seed stream for reproducibility.
        weak_env = GridWorld(grid_params, seed=seed)
        weak_hits += _rollout(weak_env, random_policy(seed=1000 + seed))

    strong_score = strong_hits / n_rollouts
    weak_score = weak_hits / n_rollouts
    return EnvScore(weak_score=weak_score, strong_score=strong_score, p=weak_score)


def score_env_stub(params: dict, game: Game, n_rollouts: int = N_ROLLOUTS) -> EnvScore:
    """STUB scorer: random but plausibly-shaped values. No engine dependency."""
    rng = np.random.default_rng(abs(hash((game.name, tuple(sorted(params.items()))))) % (2**32))
    weak = float(rng.uniform(0.0, 1.0))
    strong = float(rng.uniform(weak, 1.0))  # strong is never worse than weak
    return EnvScore(weak_score=weak, strong_score=strong, p=weak)


def score_env(params: dict, game: Game, n_rollouts: int = N_ROLLOUTS) -> EnvScore:
    """Public entry point. Real scorer by default; stub on failure or env flag."""
    if os.environ.get("CRUCIBLE_FORCE_STUB") == "1":
        return score_env_stub(params, game, n_rollouts)
    try:
        return score_env_real(params, game, n_rollouts)
    except Exception as exc:  # engine still half-built? don't block the pipeline.
        print(f"[score_env] real scorer failed ({exc!r}); falling back to stub")
        return score_env_stub(params, game, n_rollouts)


if __name__ == "__main__":
    from engine.games import GAME_1

    for diff in (0.1, 0.5, 0.9):
        s = score_env({"difficulty": diff}, GAME_1)
        print(f"difficulty={diff}: {s}")
