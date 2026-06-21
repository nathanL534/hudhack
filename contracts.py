"""contracts.py — THE FROZEN data contracts for Crucible (Phase 1).

This module is the interface boundary between the two halves of the project.
Every component — Teacher, GameAdapter, PlayerTrainer, scorer, runner — reads
and writes ONLY these types. Freeze these first; both devs import them before
writing any other code.

Design rules (from the converged architecture plan):
  * Pydantic models, so every contract is SERIALIZABLE (model_dump_json /
    model_validate_json) and validated at the boundary. A fake that returns a
    bare float or dict is a *different* interface, not a fake.
  * SEED-ADDRESSABLE from day 1: configs and results carry seeds so a run can
    be reproduced and so per-seed scores aggregate cleanly.
  * The reward is computed INSIDE the scorer (see harness/scoring.py), never by
    a caller. ScoringResult carries the already-computed reward.

Nothing here may change without both devs agreeing — every module depends on
these shapes.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Reward modes
# ---------------------------------------------------------------------------

# "gap_proxy"        — the fast, no-Player-training reward used in the hot RFT
#                      loop: strong (scripted/BFS) vs weak (random) score gap.
# "ppo_improvement"  — the real, eval-only reward: a Player's learning gain
#                      after PPO training. Reserved for a later phase; the
#                      contract supports it so the swap is a flag, not a rewrite.
RewardMode = Literal["gap_proxy", "ppo_improvement"]


# ---------------------------------------------------------------------------
# Arena / curriculum specs (the Teacher's output)
# ---------------------------------------------------------------------------

class ArenaSpec(BaseModel):
    """Parameters for a single gridworld arena.

    This is the param schema the Teacher emits — never raw env code, always
    parameters over the configurable gridworld skeleton.
    """

    map_size: int = Field(..., ge=3, le=64, description="Side length of the square grid.")
    doors: int = Field(..., ge=0, description="Number of doors in the arena.")
    keys: int = Field(..., ge=0, description="Number of keys in the arena.")
    hazard_density: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of cells that are hazards."
    )
    difficulty: float = Field(
        ..., ge=0.0, le=1.0, description="Abstract difficulty dial in [0, 1]."
    )


class CurriculumSpec(BaseModel):
    """A full curriculum: a set of arenas for one game.

    `model_json_schema()` exposes the JSON Schema that is read by (a) the
    Teacher prompt, (b) the boundary validator, and (c) the fake — one shared
    schema, no drift.
    """

    game_id: str = Field(..., min_length=1, description="Which game these arenas belong to.")
    curriculum_id: str = Field(
        ..., min_length=1, description="Stable id for this curriculum (seed-addressable)."
    )
    arenas: list[ArenaSpec] = Field(..., min_length=1, description="One or more arena specs.")

    @field_validator("arenas")
    @classmethod
    def _at_least_one_arena(cls, v: list[ArenaSpec]) -> list[ArenaSpec]:
        if not v:
            raise ValueError("CurriculumSpec must contain at least one arena.")
        return v

    def validate(self) -> "CurriculumSpec":
        """Explicit, idempotent re-validation.

        Pydantic already validates on construction; this re-runs validation on
        the current field values and returns a validated instance, so callers
        can write an explicit `spec.validate()` step in the orchestrator flow.
        """
        return self.model_validate(self.model_dump())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class ScoringResult(BaseModel):
    """The output of the ONE scoring function (harness/scoring.py).

    The reward is already computed — the caller never recomputes it.
    """

    reward: float = Field(..., description="The learnability reward (computed inside the scorer).")
    weak_score: float = Field(..., description="Mean score of the weak (random) policy.")
    strong_score: float = Field(..., description="Mean score of the strong (scripted) policy.")
    p: float = Field(..., description="Solvability proxy; by convention p == weak_score.")
    curriculum_id: str = Field(..., description="The curriculum this result scores.")
    mode: RewardMode = Field(..., description="Which reward mode produced this result.")


# ---------------------------------------------------------------------------
# Player / training config
# ---------------------------------------------------------------------------

class TrainingBudget(BaseModel):
    """How much compute a Player gets. Episodes for now; steps reserved."""

    episodes: int = Field(..., ge=1, description="Number of training episodes.")
    steps: Optional[int] = Field(
        default=None, ge=1, description="Optional step budget (used in later phases)."
    )


class PlayerConfig(BaseModel):
    """Configuration for one Player-training experiment."""

    architecture: str = Field(..., min_length=1, description="Player net architecture, e.g. 'mlp'.")
    num_seeds: int = Field(..., ge=1, description="Number of seeds to train (one job each).")
    budget: TrainingBudget = Field(..., description="Per-seed training budget.")
    seed: int = Field(..., ge=0, description="Base seed; per-job seeds derive from this.")
    # Modal lives behind THIS flag, inside PlayerTrainer — never in the runner.
    # local -> Modal is one flag flip.
    modal_parallel: bool = Field(
        default=False, description="If True, fan jobs out to Modal (later phase)."
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class MatchResult(BaseModel):
    """One per-arena / per-seed score row produced by a Player job."""

    seed: int = Field(..., description="The seed this result is for.")
    architecture: str = Field(..., description="The architecture this result is for.")
    curriculum_id: str = Field(..., description="The curriculum trained on.")
    arena_scores: list[float] = Field(
        ..., description="Per-arena scores for this seed (one entry per arena)."
    )
    mean_score: float = Field(..., description="Mean over arena_scores for this seed.")


class ExperimentResult(BaseModel):
    """The top-level result of an ExperimentRunner.run()."""

    per_seed_scores: dict[str, list[float]] = Field(
        ..., description="architecture -> list of per-seed mean scores."
    )
    delta_per_arch: dict[str, float] = Field(
        ..., description="architecture -> improvement delta vs its own baseline."
    )
    n_seeds_completed: int = Field(..., ge=0, description="How many Player jobs finished.")
    p_value: float = Field(..., description="One-sided significance of the result.")
    curricula_ids: list[str] = Field(..., description="Curriculum ids exercised in this experiment.")
    reward: float = Field(..., description="The Teacher reward from the real scorer.")


# ---------------------------------------------------------------------------
# Scoring context (reserve the async/service hook)
# ---------------------------------------------------------------------------

class ScoringContext(BaseModel):
    """Per-call context for the scorer.

    Reserved so the future async `/init` handler can thread a per-rollout id
    through scoring without changing the frozen call shape. With n concurrent
    rollouts, every reward MUST be tagged with its own rollout_id or rewards
    cross-tag silently (A's reward filed under B's id) — a bug that only shows
    up at real batch size, so the field has to exist in the contract from day 1.
    """

    # TODO Phase 7/8: async /init + rollout_id threading. The background task
    # per rollout will populate this and tag every log line with rollout_id.
    rollout_id: Optional[str] = None
