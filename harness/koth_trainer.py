"""harness/koth_trainer.py — the KotH PlayerTrainer (Student-vs-Student support).

A structural twin of ``harness/ppo_trainer.py`` for King of the Hill. Stage 6's
PRIMARY metric fights two FROZEN Students head-to-head, and the KOTH cross-game
test needs the SAME machinery as the fighter: train a fresh Student on a KOTH
curriculum SET, expose the raw SB3 model so its policy weights can be serialized,
and score the trained Student on its training arenas to fill the contract
``MatchResult``.

It deliberately does NOT touch ``harness/ppo_trainer.py`` (fighter-typed and
frozen) — KotH gets its own trainer with the IDENTICAL SB3 wiring (same net_arch,
n_steps, batch_size, learning rate) so a KOTH Student is the architectural twin of
a fighter Student. The only differences are the env class (``KothEnv``), the arena
type (``KothArena``), and the scoring opponent (``parametric_koth``) — exactly the
"same engine, different game" pattern.

``KothPlayerTrainer.submit(config, arenas).model`` exposes the trained model so
``output/stage6/policy.serialize_policy`` can pull its state_dict. Conforms to the
frozen ``PlayerTrainer`` / ``TrainingJob`` ABCs.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from contracts import MatchResult, PlayerConfig
from games.koth import KothArena, KothEnv, parametric_koth, play_match
from harness.interfaces import Arena, PlayerTrainer, TrainingJob

# Mirror the fighter trainer's budget conversion exactly.
_STEPS_PER_EPISODE = 60
_MIN_TIMESTEPS = 4_000


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _arena_of(handle: Arena) -> KothArena:
    arena = getattr(handle, "arena", None)
    if not isinstance(arena, KothArena):
        raise TypeError(
            "KothPlayerTrainer expects arenas built by KothGameAdapter "
            f"(opaque handles carrying a KothArena); got {type(handle)!r}."
        )
    return arena


def _curriculum_id_of(arenas: list[Arena]) -> str:
    for h in arenas:
        cid = getattr(h, "curriculum_id", "")
        if cid:
            return cid
    return "koth-direct"


class _MultiArenaKothEnv(gym.Env):
    """Gym env that samples a fresh KOTH arena from the training set each reset.

    Mirrors ``harness.ppo_trainer._MultiArenaFighterEnv`` — trains across the WHOLE
    curriculum so the learned policy reflects the full environment population, not
    one arena. Thin wrapper around ``KothEnv``; swaps the arena on reset.
    """

    metadata = {"render_modes": []}

    def __init__(self, arenas: list[KothArena], seed: int):
        super().__init__()
        assert arenas, "need at least one arena to train on"
        self._arenas = arenas
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._inner = KothEnv(
            arenas[0],
            opponent_factory=lambda a: parametric_koth(a, ego=1, seed=seed),
            seed=seed,
        )
        self.action_space = self._inner.action_space
        self.observation_space = self._inner.observation_space
        self.render_mode = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        arena = self._arenas[int(self._rng.integers(len(self._arenas)))]
        ep_seed = int(self._rng.integers(1_000_000))
        self._inner = KothEnv(
            arena,
            opponent_factory=lambda a: parametric_koth(a, ego=1, seed=ep_seed),
            seed=ep_seed,
        )
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        return self._inner.step(action)

    def close(self):
        return None


def _make_policy_from_model(model) -> Callable[[np.ndarray], int]:
    def act(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return act


class _KothTrainingJob(TrainingJob):
    """A synchronously-finished TrainingJob carrying the policy AND raw model."""

    def __init__(self, result: MatchResult, policy: Callable[[np.ndarray], int], model=None):
        self._result = result
        self.policy = policy
        self.model = model

    def is_done(self) -> bool:
        return True

    def result(self, timeout: Optional[float] = None) -> MatchResult:
        return self._result


class KothPlayerTrainer(PlayerTrainer):
    """Real KotH PlayerTrainer: SB3 PPO vs the parametric KotH opponent."""

    def __init__(self, *, eval_seeds: int = 25, verbose: int = 0,
                 net_arch=(64, 64)):
        self._eval_seeds = eval_seeds
        self._verbose = verbose
        self._net_arch = list(net_arch)

    def submit(self, config: PlayerConfig, arenas: list[Arena]) -> TrainingJob:
        if config.modal_parallel:
            raise NotImplementedError(
                "modal_parallel=True is reserved; this trainer runs PPO locally."
            )

        from stable_baselines3 import PPO

        _set_global_seeds(config.seed)
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        koth_arenas = [_arena_of(h) for h in arenas]
        curriculum_id = _curriculum_id_of(arenas)
        total_timesteps = max(_MIN_TIMESTEPS, config.budget.episodes * _STEPS_PER_EPISODE)

        env = _MultiArenaKothEnv(koth_arenas, seed=config.seed)
        model = PPO(
            "MlpPolicy",
            env,
            seed=config.seed,
            verbose=self._verbose,
            policy_kwargs={"net_arch": self._net_arch},
            n_steps=512,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            learning_rate=3e-4,
            device="cpu",
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=False)

        policy = _make_policy_from_model(model)
        arena_scores = [self._winrate(a, policy) for a in koth_arenas]
        mean_score = sum(arena_scores) / len(arena_scores) if arena_scores else 0.0

        result = MatchResult(
            seed=config.seed,
            architecture=config.architecture,
            curriculum_id=curriculum_id,
            arena_scores=arena_scores,
            mean_score=mean_score,
        )
        return _KothTrainingJob(result, policy, model=model)

    def _winrate(self, arena: KothArena, policy: Callable[[np.ndarray], int]) -> float:
        wins = 0
        for s in range(self._eval_seeds):
            opponent = parametric_koth(arena, ego=1, seed=10_000 + s)
            if play_match(arena, policy, opponent, seed=s) == 0:
                wins += 1
        return wins / self._eval_seeds
