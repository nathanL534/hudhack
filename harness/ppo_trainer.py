"""harness/ppo_trainer.py — the real PlayerTrainer: PPO on the Stage-1 fighter.

Conforms to the FROZEN ``PlayerTrainer`` / ``TrainingJob`` interfaces
(harness/interfaces.py). ``submit(config, arenas)`` trains a SMALL MLP Player via
Stable-Baselines3 PPO **against the fixed scripted opponent** (fighter 1), then
returns a ``TrainingJob`` whose ``.result()`` is a ``MatchResult`` carrying the
trained Player's per-arena win-rates.

Design notes:
  * SINGLE-AGENT. The Player is fighter 0; the scripted opponent is baked into
    ``FighterEnv.step``. No self-play, no multi-agent.
  * SHORT budget on purpose — this is a *trend check* (does it learn at all,
    and do arenas differ), not a tuned competitor. ``TrainingBudget.episodes``
    drives total timesteps via a fixed steps/episode estimate.
  * The job resolves synchronously (training runs inside ``submit``) but is still
    handed back as a ``TrainingJob`` so the runner's submit-then-collect path is
    identical to a future Modal/async trainer — exactly the contract's intent.
  * ``train()`` (convenience wrapper = ``submit().result()``) is inherited from
    the ABC; we don't override it.

The trained policy is exposed via ``job.policy`` (a plain ``Callable[[obs], int]``)
so ``prove_ppo_learns.py`` can score it through ``FighterGameAdapter.evaluate``
on HELD-OUT arenas using the very same scoring path as the reference policies.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from contracts import MatchResult, PlayerConfig
from games.fighter import (
    STAGE1_ACTIONS,
    FighterArena,
    FighterEnv,
    parametric_fighter,
    play_match,
)
from harness.interfaces import Arena, PlayerTrainer, TrainingJob

# Rough steps-per-episode estimate to convert the episode budget into PPO
# timesteps. Matches FighterArena.max_steps order of magnitude; kept modest so
# the budget stays in minutes.
_STEPS_PER_EPISODE = 60
# Floor so even a tiny episode budget still gives PPO something to chew on.
_MIN_TIMESTEPS = 4_000


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _arena_of(handle: Arena) -> FighterArena:
    """Pull the concrete FighterArena out of an opaque arena handle."""
    arena = getattr(handle, "arena", None)
    if not isinstance(arena, FighterArena):
        raise TypeError(
            "PPOPlayerTrainer expects arenas built by FighterGameAdapter "
            f"(opaque handles carrying a FighterArena); got {type(handle)!r}."
        )
    return arena


def _curriculum_id_of(arenas: list[Arena]) -> str:
    for h in arenas:
        cid = getattr(h, "curriculum_id", "")
        if cid:
            return cid
    return "fighter-direct"


class _MultiArenaFighterEnv(gym.Env):
    """Gymnasium env that samples a fresh arena from the training set each reset.

    Training across all arenas in the curriculum (rather than a single one)
    makes the learned policy reflect the WHOLE environment config, which is the
    quantity the milestone compares between config A and config B. Thin wrapper
    around ``FighterEnv`` — delegates the Gym API, just swaps the arena on reset.
    Subclasses ``gymnasium.Env`` so SB3 accepts it directly.

    OPPONENT LEAGUE (Stage-6 Student path only, default OFF). When ``opponent_league``
    is ``None`` the opponent is ALWAYS the single difficulty-scaled ``parametric_fighter``
    — exactly the original behavior. When a league is supplied, each reset deterministically
    picks an opponent style from the active league (via the teammate's
    ``select_opponent_id`` / ``make_opponent``, seeded by this env's ``self._rng``) and
    fights the Student against THAT style, so a Stage-6 Student trains against a SPREAD of
    opponent behaviors instead of one parametric fighter (the 48.5%-draw fix). Per-episode
    opponent-id counts are recorded in ``self._opp_counts`` for auditing.
    """

    metadata = {"render_modes": []}

    def __init__(self, arenas: list[FighterArena], seed: int,
                 anti_camping_reward: bool = False,
                 opponent_league: Optional[list] = None,
                 prior_student=None):
        super().__init__()
        assert arenas, "need at least one arena to train on"
        self._arenas = arenas
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._anti_camping_reward = anti_camping_reward
        # League is OPT-IN. ``None`` => single parametric opponent (original path).
        self._opponent_league = opponent_league
        self._prior_student = prior_student
        # Per-opponent-id episode counter (audit which styles were actually used).
        self._opp_counts: dict[str, int] = {}
        self._inner = FighterEnv(
            arenas[0],
            opponent_factory=lambda a: parametric_fighter(a, ego=1, seed=seed),
            seed=seed,
            anti_camping_reward=anti_camping_reward,
        )
        self.action_space = self._inner.action_space
        self.observation_space = self._inner.observation_space
        self.render_mode = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        arena = self._arenas[int(self._rng.integers(len(self._arenas)))]
        # The opponent is the PARAMETRIC one (strength scales with
        # arena.difficulty via epsilon-mixing), seeded per-episode so its random
        # choices are reproducible — the whole run replays for a fixed base seed.
        ep_seed = int(self._rng.integers(1_000_000))
        if self._opponent_league is None:
            # ORIGINAL behavior, byte-identical: single parametric opponent.
            opponent_factory = lambda a: parametric_fighter(a, ego=1, seed=ep_seed)
        else:
            # Stage-6 league path: pick an opponent style deterministically from
            # this env's RNG, build it for THIS arena, and capture it so a fresh
            # FighterEnv re-uses the SAME opponent for the whole episode.
            from games.opponents_league import make_opponent, select_opponent_id

            oid = select_opponent_id(self._opponent_league, self._rng)
            opp = make_opponent(
                oid, arena, ego=1, seed=ep_seed,
                prior_student=self._prior_student,
            )
            self._opp_counts[oid] = self._opp_counts.get(oid, 0) + 1
            opponent_factory = lambda a: opp
        self._inner = FighterEnv(
            arena,
            opponent_factory=opponent_factory,
            seed=ep_seed,
            anti_camping_reward=self._anti_camping_reward,
        )
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        return self._inner.step(action)

    def close(self):
        return None


def _make_policy_from_model(model) -> Callable[[np.ndarray], int]:
    """Wrap a trained SB3 model as a plain ``obs -> action`` policy (deterministic)."""

    def act(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return act


class _PPOTrainingJob(TrainingJob):
    """A TrainingJob that ran synchronously inside submit(); already finished.

    Carries the trained ``policy`` (callable) alongside the contract
    ``MatchResult`` so the milestone can score the Player on held-out arenas.
    """

    def __init__(self, result: MatchResult, policy: Callable[[np.ndarray], int],
                 model=None, opp_counts: Optional[dict] = None):
        self._result = result
        self.policy = policy
        # The trained SB3 model is exposed so callers that need the raw weights
        # (e.g. Stage-6 policy serialization for Student-vs-Student) can pull
        # ``job.model.policy.state_dict()``. ``None`` for jobs that predate this.
        self.model = model
        # Per-opponent-id episode counts when the opponent league was active
        # (Stage-6 Student path); empty dict for the original single-opponent path.
        # Lets the Student artifact audit which opponent styles were actually trained on.
        self.opp_counts = dict(opp_counts) if opp_counts else {}

    def is_done(self) -> bool:
        return True

    def result(self, timeout: Optional[float] = None) -> MatchResult:
        return self._result


class PPOPlayerTrainer(PlayerTrainer):
    """Real PlayerTrainer: trains a small MLP via SB3 PPO vs the scripted opponent."""

    def __init__(self, *, eval_seeds: int = 25, verbose: int = 0,
                 ent_coef: float = 0.0, min_timesteps: int = _MIN_TIMESTEPS,
                 anti_camping_reward: bool = False,
                 opponent_league: Optional[list] = None,
                 prior_student=None, net_arch=(64, 64)):
        self._eval_seeds = eval_seeds
        self._verbose = verbose
        self._ent_coef = ent_coef
        self._min_timesteps = min_timesteps
        self._anti_camping_reward = anti_camping_reward
        self._net_arch = list(net_arch)
        # Stage-6 Student-only opponent league. ``None`` (default) => the original
        # single-parametric-opponent training env, UNCHANGED. The Teacher inner-reward
        # workers construct this trainer without the flag, so they keep the old path.
        self._opponent_league = opponent_league
        self._prior_student = prior_student

    def submit(self, config: PlayerConfig, arenas: list[Arena]) -> TrainingJob:
        if config.modal_parallel:
            # Modal lives behind this flag, INSIDE the trainer (never the runner).
            raise NotImplementedError(
                "modal_parallel=True is reserved for a later phase; this trainer "
                "runs PPO locally."
            )

        # Lazy import so the rest of the harness (and Phase 1) never needs torch.
        from stable_baselines3 import PPO

        _set_global_seeds(config.seed)
        # Keep PPO single-threaded & quiet for reproducibility in CI/minutes-budget.
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        fighter_arenas = [_arena_of(h) for h in arenas]
        curriculum_id = _curriculum_id_of(arenas)
        total_timesteps = max(
            self._min_timesteps, config.budget.episodes * _STEPS_PER_EPISODE
        )

        env = _MultiArenaFighterEnv(
            fighter_arenas,
            seed=config.seed,
            anti_camping_reward=self._anti_camping_reward,
            opponent_league=self._opponent_league,
            prior_student=self._prior_student,
        )
        model = PPO(
            "MlpPolicy",
            env,
            seed=config.seed,
            verbose=self._verbose,
            # Net size is configurable (Stage-6 Students only); default [64,64].
            policy_kwargs={"net_arch": self._net_arch},
            n_steps=512,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=self._ent_coef,
            device="cpu",
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=False)

        policy = _make_policy_from_model(model)

        # Score the trained Player on the TRAINING arenas (win-rate vs scripted)
        # so MatchResult carries a real per-arena signal. The milestone separately
        # scores held-out arenas through the adapter; this is the in-distribution
        # number that fills the contract.
        arena_scores = [
            self._winrate(arena, policy) for arena in fighter_arenas
        ]
        mean_score = sum(arena_scores) / len(arena_scores) if arena_scores else 0.0

        result = MatchResult(
            seed=config.seed,
            architecture=config.architecture,
            curriculum_id=curriculum_id,
            arena_scores=arena_scores,
            mean_score=mean_score,
        )
        return _PPOTrainingJob(
            result, policy, model=model, opp_counts=getattr(env, "_opp_counts", None)
        )

    def _winrate(self, arena: FighterArena, policy: Callable[[np.ndarray], int]) -> float:
        # Score against the SAME parametric opponent the Player trained on, with
        # a per-seed opponent seed so the random mix is reproducible. This is the
        # in-distribution number that fills MatchResult.
        wins = 0
        for s in range(self._eval_seeds):
            opponent = parametric_fighter(arena, ego=1, seed=10_000 + s)
            if play_match(arena, policy, opponent, seed=s) == 0:
                wins += 1
        return wins / self._eval_seeds
