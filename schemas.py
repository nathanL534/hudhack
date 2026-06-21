"""schemas.py — THE pinned contract for Crucible.

These three types are the interface boundary between the three halves of the
project. Nothing here should change without both devs agreeing, because every
module reads/writes these shapes:

    engine/  produces EnvScore
    training/ produces StepRecord
    eval/    produces HeadToHead

We use TypedDict (not dataclass) on purpose: these objects get logged to JSON,
passed to plotting, and may eventually cross the Fireworks RFT boundary as
plain dicts. TypedDict gives us editor/type-checker hints with zero runtime
wrapping — a logged dict IS the contract.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class EnvScore(TypedDict):
    """Output of engine/score_env.py for a single generated environment.

    weak_score:   mean reward of the WEAK player (random policy), in [0, 1].
    strong_score: mean reward of the STRONG player (scripted optimal), in [0, 1].
    p:            the "solvability" probability used by the learnability reward.
                  By convention p == weak_score (how often a weak agent stumbles
                  into success). A learnable env has p in the middle band so that
                  p*(1-p) is large AND strong > weak.
    """

    weak_score: float
    strong_score: float
    p: float


class StepRecord(TypedDict):
    """One row of the training log emitted by training/rft_run.py per step.

    teacher_step:  the RFT optimization step index (0-based).
    game1_reward:  the learnability reward the Teacher earned on GAME_1 this step.
    game2_winrate: held-out GAME_2 win-rate this step. This is the ANTI-GAMING
                   signal — if the Teacher is really learning to make learnable
                   envs (not memorizing GAME_1), game2_winrate must co-rise with
                   game1_reward.
    """

    teacher_step: int
    game1_reward: float
    game2_winrate: float


class HeadToHead(TypedDict):
    """One head-to-head match result emitted by eval/headtohead.py.

    seed:   the env seed the two players competed on.
    winner: which player won this seed.
    """

    seed: int
    winner: Literal["A", "B", "draw"]
