"""harness/runner.py — the ExperimentRunner that wires the whole pipeline.

Flow (the orchestrator):
    1. teacher.generate(...) -> CurriculumSpec
    2. spec.validate()                          (boundary validation)
    3. game.build(spec) -> arenas
    4. submit N = num_seeds Player jobs (collect TrainingJob handles),
       then .result() each                      (local & Modal share this path)
    5. score_curriculum(spec, game)             (the ONE real reward)
    6. aggregate -> ExperimentResult

Modal is honored ONLY via PlayerConfig.modal_parallel and is rejected with a
NotImplemented branch here — no Modal code exists in Phase 1. The reward is
always the real formula from harness/scoring.py.
"""

from __future__ import annotations

from contracts import (
    CurriculumSpec,
    ExperimentResult,
    PlayerConfig,
    RewardMode,
)
from harness.interfaces import (
    ExperimentRunner,
    GameAdapter,
    PlayerTrainer,
    TeacherClient,
)
from harness.scoring import score_curriculum


def _one_sided_binomial_p_value(successes: int, n: int, p0: float = 0.5) -> float:
    """One-sided binomial tail: P(X >= successes) under H0: p = p0.

    Placeholder significance for the head-to-head: how surprising is it to see
    this many seed-wins if the curriculum were no better than chance. Pure
    Python (no scipy) so Phase 1 has no heavy deps. Real eval can refine this.
    """
    if n <= 0:
        return 1.0
    successes = max(0, min(successes, n))

    # Binomial coefficient via an integer-exact running product.
    def comb(a: int, b: int) -> int:
        b = min(b, a - b)
        if b < 0:
            return 0
        num = 1
        for i in range(b):
            num = num * (a - i) // (i + 1)
        return num

    tail = 0.0
    for k in range(successes, n + 1):
        tail += comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
    return min(1.0, tail)


class DefaultExperimentRunner(ExperimentRunner):
    """The Phase 1 orchestrator. Same code path for fakes and (later) real."""

    def __init__(
        self,
        trainer: PlayerTrainer,
        *,
        mode: RewardMode = "gap_proxy",
        result_timeout: float = 30.0,
    ):
        self._trainer = trainer
        self._mode = mode
        self._result_timeout = result_timeout

    def run(
        self,
        teacher: TeacherClient,
        game: GameAdapter,
        player_config: PlayerConfig,
    ) -> ExperimentResult:
        if player_config.modal_parallel:
            # The flag is honored here but no Modal dispatch exists yet.
            raise NotImplementedError(
                "PlayerConfig.modal_parallel=True is reserved for a later phase. "
                "Phase 1 runs Player jobs locally via the trainer."
            )

        # 1. Teacher generates a curriculum.
        spec: CurriculumSpec = teacher.generate(game_spec={"game_id": "gridworld"})

        # 2. Validate the curriculum at the boundary.
        spec = spec.validate()

        # 3. Build arenas.
        arenas = game.build(spec)

        # 4. Submit N = num_seeds Player jobs (non-blocking), THEN collect.
        jobs = []
        for i in range(player_config.num_seeds):
            seed_config = player_config.model_copy(
                update={"seed": player_config.seed + i, "num_seeds": 1}
            )
            jobs.append((seed_config, self._trainer.submit(seed_config, arenas)))

        per_seed_scores: dict[str, list[float]] = {}
        n_seeds_completed = 0
        for seed_config, job in jobs:
            match = job.result(timeout=self._result_timeout)
            per_seed_scores.setdefault(seed_config.architecture, []).append(match.mean_score)
            n_seeds_completed += 1

        # 5. The ONE real reward (gap proxy: strong vs weak, inside the scorer).
        scoring = score_curriculum(spec, game, mode=self._mode)

        # 6. Aggregate.
        delta_per_arch = self._compute_deltas(per_seed_scores)

        # Significance over the head-to-head: count per-seed "wins" (a seed whose
        # mean score beats a 0.5 chance baseline) and run a one-sided binomial.
        all_scores = [s for scores in per_seed_scores.values() for s in scores]
        successes = sum(1 for s in all_scores if s > 0.5)
        p_value = _one_sided_binomial_p_value(successes, len(all_scores))

        return ExperimentResult(
            per_seed_scores=per_seed_scores,
            delta_per_arch=delta_per_arch,
            n_seeds_completed=n_seeds_completed,
            p_value=p_value,
            curricula_ids=[spec.curriculum_id],
            reward=scoring.reward,
        )

    @staticmethod
    def _compute_deltas(per_seed_scores: dict[str, list[float]]) -> dict[str, float]:
        """Per-architecture delta vs a 0.5 baseline (each arch vs its own base).

        Phase 1 has no separate baseline run, so the baseline is the 0.5 chance
        line. Later phases compare each architecture against its OWN untrained
        baseline (the strongest anti-gaming evidence).
        """
        deltas: dict[str, float] = {}
        for arch, scores in per_seed_scores.items():
            mean = sum(scores) / len(scores) if scores else 0.0
            deltas[arch] = round(mean - 0.5, 6)
        return deltas
