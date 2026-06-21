"""Stage 4: local/Modal seed fan-out behind one stable interface.

Local mode is production-useful for development.  Modal mode calls a separately
deployed function by app/function name, avoiding import-time Modal side effects
and keeping the core repository testable without Modal credentials.

``PlayerFanout`` is the low-level primitive (seed -> dict).  ``FanoutPlayerTrainer``
bridges it to the frozen ``PlayerTrainer``/``TrainingJob`` ABCs so the
orchestrator's submit-then-collect path is identical for local and Modal — Modal
lives behind ``PlayerConfig.modal_parallel`` INSIDE the trainer, never in the
runner.  ``aggregate_experiment_result`` folds the per-seed ``MatchResult`` rows
into the frozen ``ExperimentResult``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

from contracts import (
    ExperimentResult,
    MatchResult,
    PlayerConfig,
)
from harness.interfaces import Arena, PlayerTrainer, TrainingJob


SeedWorker = Callable[[dict, int], dict]


@dataclass(frozen=True)
class FanoutConfig:
    backend: str = "local"
    max_workers: int = 8
    modal_app: str = "crucible-player"
    modal_function: str = "train_player"


class PlayerFanout:
    def __init__(self, config: FanoutConfig, local_worker: SeedWorker):
        self.config = config
        self._local_worker = local_worker

    def run(self, payload: dict, seeds: list[int]) -> list[dict]:
        if not seeds:
            return []
        if self.config.backend == "local":
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                futures = {
                    seed: pool.submit(self._local_worker, payload, seed)
                    for seed in seeds
                }
                return [futures[seed].result() for seed in seeds]
        if self.config.backend == "modal":
            return self._run_modal(payload, seeds)
        raise ValueError(f"unknown fan-out backend: {self.config.backend}")

    def _run_modal(self, payload: dict, seeds: list[int]) -> list[dict]:
        try:
            import modal
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError("install modal to use backend='modal'") from exc

        fn = modal.Function.from_name(
            self.config.modal_app,
            self.config.modal_function,
        )
        results_by_seed = {
            seed: result
            for seed, result in zip(seeds, fn.map([payload] * len(seeds), seeds))
        }
        return [results_by_seed[seed] for seed in seeds]


# ---------------------------------------------------------------------------
# Contract bridge: PlayerFanout -> PlayerTrainer / TrainingJob / ExperimentResult
# ---------------------------------------------------------------------------


def _row_to_match_result(row: dict, *, architecture: str, curriculum_id: str) -> MatchResult:
    """Turn one fan-out row (seed -> dict) into a frozen MatchResult.

    Accepts either ``arena_scores`` (a list) or a scalar ``mean_score`` /
    ``score`` so the smoke worker and a real PPO worker both fit the contract.
    """
    seed = int(row["seed"])
    arena_scores = row.get("arena_scores")
    if arena_scores is None:
        scalar = float(row.get("mean_score", row.get("score", 0.0)))
        arena_scores = [scalar]
    arena_scores = [float(s) for s in arena_scores]
    mean_score = float(row.get("mean_score", sum(arena_scores) / len(arena_scores)))
    return MatchResult(
        seed=seed,
        architecture=architecture,
        curriculum_id=curriculum_id,
        arena_scores=arena_scores,
        mean_score=mean_score,
    )


class _CompletedJob(TrainingJob):
    """A TrainingJob whose result is already in hand — the fan-out resolves all
    seeds eagerly, so the runner's submit-then-collect path is unchanged."""

    def __init__(self, result: MatchResult):
        self._result = result

    def is_done(self) -> bool:
        return True

    def result(self, timeout: Optional[float] = None) -> MatchResult:
        return self._result


class FanoutPlayerTrainer(PlayerTrainer):
    """A frozen-interface PlayerTrainer backed by PlayerFanout.

    ``submit`` runs ONE seed through the fan-out (the runner submits one job per
    seed) and returns a completed ``TrainingJob``.  Local vs Modal is chosen by
    ``PlayerConfig.modal_parallel`` here, inside the trainer — the runner never
    sees the backend.
    """

    def __init__(self, local_worker: SeedWorker, *, max_workers: int = 8):
        self._local_worker = local_worker
        self._max_workers = max_workers

    def submit(self, config: PlayerConfig, arenas: list[Arena]) -> TrainingJob:
        backend = "modal" if config.modal_parallel else "local"
        fanout = PlayerFanout(
            FanoutConfig(backend=backend, max_workers=self._max_workers),
            local_worker=self._local_worker,
        )
        payload = {"curriculum_id": _curriculum_id_of(arenas), "architecture": config.architecture}
        row = fanout.run(payload, [config.seed])[0]
        result = _row_to_match_result(
            row,
            architecture=config.architecture,
            curriculum_id=payload["curriculum_id"],
        )
        return _CompletedJob(result)


def _curriculum_id_of(arenas: list[Any]) -> str:
    for arena in arenas:
        spec = getattr(arena, "spec", None)
        if isinstance(spec, dict) and "curriculum_id" in spec:
            return spec["curriculum_id"]
    return "fanout-curriculum"


def aggregate_experiment_result(
    matches: list[MatchResult],
    *,
    reward: float,
    curricula_ids: list[str],
    p_value: float = 1.0,
    baseline: float = 0.5,
) -> ExperimentResult:
    """Fold per-seed MatchResults into the frozen ExperimentResult.

    Groups per-seed mean scores by architecture, computes each architecture's
    delta vs ``baseline``, and carries the real Teacher ``reward`` through. The
    reward is NOT recomputed here — it is supplied by the one real scorer.
    """
    per_seed_scores: dict[str, list[float]] = {}
    for match in matches:
        per_seed_scores.setdefault(match.architecture, []).append(match.mean_score)

    delta_per_arch = {
        arch: round((sum(scores) / len(scores) if scores else 0.0) - baseline, 6)
        for arch, scores in per_seed_scores.items()
    }

    return ExperimentResult(
        per_seed_scores=per_seed_scores,
        delta_per_arch=delta_per_arch,
        n_seeds_completed=len(matches),
        p_value=p_value,
        curricula_ids=curricula_ids,
        reward=reward,
    )

