"""harness/scoring.py — the ONE scoring function (REAL, never faked).

There is exactly ONE place that turns a curriculum into a reward. The reward is
computed INSIDE this function — callers never recompute it. This is the contract
that prevents the hour-1 divergence between the scorer and the reward wiring.

The formula is REAL even in Phase 1. Only the policies it runs are supplied by
the GameAdapter (fake in Phase 1, real engine later). The scorer itself does not
change when the adapter goes real.
"""

from __future__ import annotations

from contracts import CurriculumSpec, RewardMode, ScoringResult
from harness.interfaces import GameAdapter

# Learnable-band threshold: an env is "learnable" only when the weak policy
# succeeds often enough to get signal but not so often the task is trivial, i.e.
# p*(1-p) is in the interesting band. 0.2 is the precommitted gate.
LEARNABLE_BAND_THRESHOLD = 0.2


def _mean_score(bundle: dict, agent_label: str) -> float:
    """Pull a scalar mean score out of a ScoreBundle for one agent label.

    Accepts either {"<label>": {"mean_score": float, ...}} or {"<label>": float}
    so adapters can return a rich bundle without forcing the scorer to know the
    full shape.
    """
    entry = bundle[agent_label]
    if isinstance(entry, dict):
        return float(entry["mean_score"])
    return float(entry)


def score_curriculum(
    spec: CurriculumSpec,
    game: GameAdapter,
    *,
    mode: RewardMode = "gap_proxy",
    rollout_id: str | None = None,
) -> ScoringResult:
    """Score a curriculum and return the full ScoringResult (reward inside).

    Steps:
      1. Build the arenas from the curriculum.
      2. Run the adapter's scripted_expert (STRONG) and random_policy (WEAK).
      3. p = weak_score (solvability proxy).
      4. reward = (strong - weak) if p*(1-p) is in the learnable band else 0.0.

    `rollout_id` is reserved for the future async /init handler (Phase 7/8): it
    will tag this call so concurrent rollouts don't cross-tag rewards. Accepted
    here so the signature is frozen now; not yet used.
    """
    # TODO Phase 7/8: thread rollout_id into per-rollout logging once the async
    # /init service exists. Until then it is accepted and ignored on purpose.
    _ = rollout_id

    if mode == "ppo_improvement":
        # Reserved for the eval-only real-PPO reward. Not implemented in Phase 1.
        raise NotImplementedError(
            "mode='ppo_improvement' is reserved for a later phase; "
            "Phase 1 uses the gap_proxy reward only."
        )

    # 1. Build arenas.
    arenas = game.build(spec)

    # 2. Run the two reference policies through the adapter.
    strong_bundle = game.evaluate(game.scripted_expert(), arenas)
    weak_bundle = game.evaluate(game.random_policy(), arenas)

    strong_score = _mean_score(strong_bundle, "scripted_expert")
    weak_score = _mean_score(weak_bundle, "random_policy")

    # 3. Solvability proxy.
    p = weak_score

    # 4. The learnability reward — computed HERE, inside the scorer.
    in_band = (p * (1.0 - p)) > LEARNABLE_BAND_THRESHOLD
    reward = (strong_score - weak_score) if in_band else 0.0

    return ScoringResult(
        reward=reward,
        weak_score=weak_score,
        strong_score=strong_score,
        p=p,
        curriculum_id=spec.curriculum_id,
        mode=mode,
    )
