"""harness/fakes.py — Phase 1 fakes.

These are FAKES, not stubs: each returns the FULL frozen contract type. A fake
that returned a bare float or random.random() would be a *different* interface
and the milestone would give false confidence. Each fake is swapped, one at a
time, for its real counterpart (see the README swap map).

The fake reward is meaningful: the fake adapter's scripted_expert scores HIGH
and its random_policy scores LOW, and those scores sit inside the learnable band,
so the REAL score_curriculum formula produces a real, non-zero reward from real
inputs. The milestone therefore exercises the real reward + aggregation, not a
hardcoded number.
"""

from __future__ import annotations

from typing import Any, Optional

from contracts import (
    CurriculumSpec,
    MatchResult,
    PlayerConfig,
)
from harness.interfaces import (
    Arena,
    GameAdapter,
    PlayerTrainer,
    Policy,
    TeacherClient,
    TrainingJob,
)

# Fake reference scores. These are chosen so that:
#   p = weak_score = 0.5  ->  p*(1-p) = 0.25 > 0.2  (in the learnable band)
#   reward = strong - weak = 0.9 - 0.5 = 0.4        (real, non-zero)
_FAKE_STRONG_SCORE = 0.9
_FAKE_WEAK_SCORE = 0.5


class FakeTeacher(TeacherClient):
    """Returns one hardcoded VALID CurriculumSpec (2 arenas)."""

    def generate(self, game_spec: dict, feedback: Optional[dict] = None) -> CurriculumSpec:
        return CurriculumSpec(
            game_id="gridworld",
            curriculum_id="fake-curriculum-001",
            arenas=[
                {"map_size": 9, "doors": 2, "keys": 2, "hazard_density": 0.12, "difficulty": 0.55},
                {"map_size": 11, "doors": 3, "keys": 3, "hazard_density": 0.18, "difficulty": 0.65},
            ],
        )


class _FakeArena:
    """A trivial arena object. Opaque to the orchestrator — it only carries the
    spec it was built from so a real adapter's shape is mirrored."""

    def __init__(self, index: int, spec: dict):
        self.index = index
        self.spec = spec


class _FakePolicy:
    """A trivial policy object carrying the fixed score it yields per arena."""

    def __init__(self, label: str, per_arena_score: float):
        self.label = label
        self.per_arena_score = per_arena_score


class FakeGameAdapter(GameAdapter):
    """Builds trivial arenas; scripted_expert scores high, random scores low."""

    def build(self, spec: CurriculumSpec) -> list[Arena]:
        return [_FakeArena(i, a.model_dump()) for i, a in enumerate(spec.arenas)]

    def scripted_expert(self) -> Policy:
        return _FakePolicy("scripted_expert", _FAKE_STRONG_SCORE)

    def random_policy(self) -> Policy:
        return _FakePolicy("random_policy", _FAKE_WEAK_SCORE)

    def evaluate(self, policy: Policy, arenas: list[Arena]) -> dict:
        # Real ScoreBundle shape: keyed by agent label, with mean + per-arena.
        per_arena = [policy.per_arena_score for _ in arenas]
        mean_score = sum(per_arena) / len(per_arena) if per_arena else 0.0
        return {policy.label: {"mean_score": mean_score, "per_arena": per_arena}}


class _InstantTrainingJob(TrainingJob):
    """A TrainingJob that is already finished — gap-proxy / fake jobs resolve
    instantly, so the runner's submit-then-collect path is identical to Modal's.
    """

    def __init__(self, result: MatchResult):
        self._result = result

    def is_done(self) -> bool:
        return True

    def result(self, timeout: Optional[float] = None) -> MatchResult:
        return self._result


class FakePlayerTrainer(PlayerTrainer):
    """submit() returns an instantly-resolving job with real-shaped per-seed scores."""

    def submit(self, config: PlayerConfig, arenas: list[Arena]) -> TrainingJob:
        if config.modal_parallel:
            # Modal lives behind this flag; no Modal code exists yet in Phase 1.
            raise NotImplementedError(
                "modal_parallel=True is reserved for a later phase; "
                "Phase 1 runs Player jobs locally / instantly."
            )

        n_arenas = max(len(arenas), 1)
        curriculum_id = self._curriculum_id_of(arenas)

        # A deterministic-ish per-seed score: derive from the base seed so the
        # aggregate is reproducible. Real PPO replaces this; the SHAPE is real.
        per_arena = [self._seed_score(config.seed) for _ in range(n_arenas)]
        mean_score = sum(per_arena) / len(per_arena)

        result = MatchResult(
            seed=config.seed,
            architecture=config.architecture,
            curriculum_id=curriculum_id,
            arena_scores=per_arena,
            mean_score=mean_score,
        )
        return _InstantTrainingJob(result)

    @staticmethod
    def _seed_score(seed: int) -> float:
        # Spread per-seed scores deterministically in (0, 1) without randomness.
        return round(0.5 + 0.05 * ((seed % 5) - 2), 4)

    @staticmethod
    def _curriculum_id_of(arenas: list[Any]) -> str:
        for a in arenas:
            spec = getattr(a, "spec", None)
            if isinstance(spec, dict) and "curriculum_id" in spec:
                return spec["curriculum_id"]
        return "fake-curriculum-001"
