"""harness/interfaces.py — the frozen INTERFACES (ABCs).

These are the seams where fakes get swapped for real implementations, one
component at a time. Right contracts -> parallel independent work -> swap fakes
for real. Each interface here is implemented twice over the project's life: once
as a fake (harness/fakes.py, Phase 1) and once for real (later phases).

Component owners (see README):
  * TeacherClient   -> rag26  (Fireworks Qwen3-4B)
  * GameAdapter     -> Nathan (gridworld engine + scripted/random policies + scoring)
  * PlayerTrainer   -> rag26  (PPO on Modal)
  * ExperimentRunner-> shared orchestrator (harness/runner.py)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from contracts import (
    CurriculumSpec,
    ExperimentResult,
    PlayerConfig,
)

# An "Arena" is whatever the GameAdapter builds from an ArenaSpec. Phase 1 keeps
# it deliberately opaque (Any) — the orchestrator never inspects an arena's
# internals, it only passes them between adapter methods. A concrete game can
# use any object it likes; the contract is "build returns a list, evaluate takes
# that same list".
Arena = Any

# A Policy is similarly opaque to the orchestrator. The GameAdapter owns both
# the policies it produces (scripted_expert / random_policy) and how it runs
# them in evaluate().
Policy = Any


class TeacherClient(ABC):
    """Generates curricula. Fake = hardcoded; real = Fireworks Qwen3-4B."""

    @abstractmethod
    def generate(self, game_spec: dict, feedback: Optional[dict] = None) -> CurriculumSpec:
        """Produce a CurriculumSpec for the given game.

        game_spec: the game's rules / param schema (the real Teacher prompts on
                   this; the fake ignores it).
        feedback:  optional prior-curriculum scores / failure stats (lean v1
                   ignores it; reserved for the refinement loop).
        """
        raise NotImplementedError


class GameAdapter(ABC):
    """Owns the game: building arenas, the two reference policies, and scoring.

    Keeping scripted_expert() and random_policy() on the adapter is deliberate:
    they are game-specific (a BFS expert for a maze is not a BFS expert for a
    fighter), so the orchestrator must never own them. This is also why the ONE
    scorer takes a GameAdapter rather than raw policies — no game-specifics leak
    up into the orchestrator.
    """

    @abstractmethod
    def build(self, spec: CurriculumSpec) -> list[Arena]:
        """Construct the concrete arenas for a curriculum (one per ArenaSpec)."""
        raise NotImplementedError

    @abstractmethod
    def scripted_expert(self) -> Policy:
        """The STRONG reference policy (e.g. BFS-optimal). Game-owned."""
        raise NotImplementedError

    @abstractmethod
    def random_policy(self) -> Policy:
        """The WEAK reference policy (random actions). Game-owned."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, policy: Policy, arenas: list[Arena]) -> dict:
        """Run `policy` over `arenas`, return a ScoreBundle.

        Return shape (the ScoreBundle): a dict keyed by an agent label, e.g.
            {"<agent>": {"mean_score": float, "per_arena": [float, ...]}}
        The scorer reads mean_score; richer keys are allowed but must not be
        required by the orchestrator.
        """
        raise NotImplementedError


class TrainingJob(ABC):
    """A handle to a (possibly remote, possibly instant) Player-training job.

    Non-blocking submit -> handle -> .result(). Gap-proxy / fake jobs resolve
    instantly, so local and Modal share ONE code path: the runner always submits
    then collects, regardless of where the work runs.
    """

    @abstractmethod
    def is_done(self) -> bool:
        """True if the job has finished (result is ready without blocking)."""
        raise NotImplementedError

    @abstractmethod
    def result(self, timeout: Optional[float] = None):
        """Block (up to timeout) for the job's MatchResult. Raises on timeout."""
        raise NotImplementedError


class PlayerTrainer(ABC):
    """Trains Players. Fake = instant per-seed scores; real = PPO on Modal.

    Modal lives behind PlayerConfig.modal_parallel INSIDE this class — never in
    the runner. local -> Modal is one flag flip here.
    """

    @abstractmethod
    def submit(self, config: PlayerConfig, arenas: list[Arena]) -> "TrainingJob":
        """Submit a single training job (NON-blocking). Returns a handle."""
        raise NotImplementedError

    def train(self, config: PlayerConfig, arenas: list[Arena]):
        """Convenience wrapper: submit then block for the result."""
        return self.submit(config, arenas).result()


class ExperimentRunner(ABC):
    """Orchestrates one full experiment over the frozen contracts."""

    @abstractmethod
    def run(
        self,
        teacher: TeacherClient,
        game: GameAdapter,
        player_config: PlayerConfig,
    ) -> ExperimentResult:
        """Validate -> build -> train N seeds -> score (real reward) -> aggregate."""
        raise NotImplementedError
