"""Stage 4: local/Modal seed fan-out behind one stable interface.

Local mode is production-useful for development.  Modal mode calls a separately
deployed function by app/function name, avoiding import-time Modal side effects
and keeping the core repository testable without Modal credentials.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


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

