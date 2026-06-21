#!/usr/bin/env python3
"""prove_koth_learns.py — King-of-the-Hill learnability check (the transfer proof).

KotH is the structurally-DIFFERENT held-out game. This script answers the same
prerequisite question for KotH that ``prove_ppo_learns.py`` answers for the
fighter, with the SAME shape of evidence:

  (1) Can a PPO Player actually LEARN King of the Hill? (occupy-the-zone, a
      different objective from ring-out)
  (2) Is the scripted KotH expert clearly stronger than random? (so a low PPO
      score would mean "hard to learn", never "unwinnable")

How it answers them, through the SAME frozen pieces the fighter uses:

  * Build a KotH arena in the learnable band (scripted wins big, random loses).
  * Train a small-MLP PPO Player as player 0 against the PARAMETRIC opponent on a
    ``KothEnv`` — the identical SB3 wiring the fighter trainer uses, just the KotH
    env class. (KotH plugs into PPO exactly like the fighter; that IS the
    drop-in.)
  * Score win-rate vs the parametric opponent on a HELD-OUT arena of the same
    config (unseen seeds), BEFORE training (random-init net) and AFTER training,
    through ``KothGameAdapter.evaluate`` — the same scoring path as the
    references.
  * Assert AFTER > BEFORE (it learned) and scripted > random (winnable).

Exit 0 iff both hold; non-zero otherwise. Numbers printed honestly.

Run:  .venv/bin/python prove_koth_learns.py
Env overrides (for a quick smoke run):  N_SEEDS, PPO_EPISODES, EVAL_SEEDS, DIFFICULTY
"""

from __future__ import annotations

import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from games.koth import KothArena, KothEnv, parametric_koth
from harness.koth_adapter import KothGameAdapter

# --- knobs (env-overridable so a smoke run is fast) ------------------------
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
PPO_EPISODES = int(os.environ.get("PPO_EPISODES", "1200"))  # -> ~72k timesteps
EVAL_SEEDS = int(os.environ.get("EVAL_SEEDS", "40"))
# Difficulty in the learnable band: scripted wins ~1.0 and the untrained net ~0.0,
# so any positive AFTER rate is unambiguous learning, and there is real headroom
# to climb (the contester is only partly engaged here, unlike the top of the dial).
DIFFICULTY = float(os.environ.get("DIFFICULTY", "0.3"))

# AFTER must beat BEFORE by at least this to count as "it learned".
MIN_LEARN_DELTA = float(os.environ.get("MIN_LEARN_DELTA", "0.10"))

# Steps/episode estimate -> PPO timesteps (mirrors ppo_trainer's constants).
_STEPS_PER_EPISODE = 60
_MIN_TIMESTEPS = 4_000

# The arena to learn (the zone dials are KotH-specific; difficulty sets opponent
# strength + zone geometry would normally come from a spec — here we dial it
# directly, same as prove_ppo_learns.py dials FighterArena fields).
ARENA = KothArena(platform_width=12.0, zone_half=1.6, zone_center_frac=0.5, difficulty=DIFFICULTY)


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class _KothTrainEnv:
    """Factory: a fresh KothEnv vs the parametric opponent, seeded per episode.

    KotH trains with the SAME single-arena SB3 wiring the fighter uses; we keep it
    inline here (rather than importing the fighter-specific trainer) because the
    PPO trainer module is frozen and fighter-typed. The construction shape is
    identical — Discrete action space, Box obs, baked-in opponent in step.
    """

    def __init__(self, arena: KothArena, seed: int):
        self.arena = arena
        self.seed = seed


def _make_env(arena: KothArena, seed: int):
    import gymnasium as gym

    class _Wrapped(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._rng = np.random.default_rng(seed)
            self._inner = KothEnv(
                arena,
                opponent_factory=lambda a: parametric_koth(a, ego=1, seed=seed),
                seed=seed,
            )
            self.action_space = self._inner.action_space
            self.observation_space = self._inner.observation_space
            self.render_mode = None

        def reset(self, *, seed=None, options=None):
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

    return _Wrapped()


def _make_policy_from_model(model):
    def act(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return act


def _train_policy(arena: KothArena, seed: int):
    from stable_baselines3 import PPO

    _set_global_seeds(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    total_timesteps = max(_MIN_TIMESTEPS, PPO_EPISODES * _STEPS_PER_EPISODE)

    env = _make_env(arena, seed)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        learning_rate=3e-4,
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return _make_policy_from_model(model)


def _untrained_policy(arena: KothArena, seed: int):
    """Random-init PPO net (no training) wrapped as obs->action — honest BEFORE."""
    from stable_baselines3 import PPO

    env = _make_env(arena, seed)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [64, 64]},
        device="cpu",
    )
    return _make_policy_from_model(model)


def _label_score(bundle: dict) -> float:
    (entry,) = bundle.values()
    return float(entry["mean_score"])


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    print("KotH learnability: does a PPO Player learn King of the Hill?")
    print(f"  ARENA: {ARENA}")
    print(f"  seeds={N_SEEDS}  ppo_episodes={PPO_EPISODES}  eval_seeds={EVAL_SEEDS}  difficulty={DIFFICULTY}")
    print()

    adapter = KothGameAdapter(eval_seeds=EVAL_SEEDS)
    held = adapter.arenas_from_configs([ARENA], curriculum_id="heldout-koth")

    # Winnability check: scripted must clearly beat random, else a low PPO score
    # would mean "unwinnable", not "hard to learn".
    strong = _label_score(adapter.evaluate(adapter.scripted_expert(), held))
    weak = _label_score(adapter.evaluate(adapter.random_policy(), held))
    print(f"  winnability check: scripted={strong:.3f}  random={weak:.3f}")
    if strong <= weak:
        print(
            f"FAIL: KotH arena not winnable in principle (scripted {strong:.3f} <= random {weak:.3f}).",
            file=sys.stderr,
        )
        return 2
    print()

    before_rates: list[float] = []
    after_rates: list[float] = []
    for i in range(N_SEEDS):
        seed = 700 + i
        before_policy = _untrained_policy(ARENA, seed)
        before = _label_score(adapter.evaluate(before_policy, held))
        after_policy = _train_policy(ARENA, seed)
        after = _label_score(adapter.evaluate(after_policy, held))
        before_rates.append(before)
        after_rates.append(after)
        print(f"  seed {seed:>3}: before={before:.3f}  after={after:.3f}")

    before_mean = _mean(before_rates)
    after_mean = _mean(after_rates)
    delta = after_mean - before_mean

    print()
    print("=" * 60)
    print("  KotH MILESTONE — PPO learnability on a HELD-OUT game")
    print("=" * 60)
    print(f"  scripted={strong:.3f}  random={weak:.3f}")
    print(f"  held-out win-rate:  before={before_mean:.3f}  after={after_mean:.3f}  Δ={delta:+.3f}")
    print("=" * 60)

    winnable = strong > weak
    learned = delta >= MIN_LEARN_DELTA and after_mean > before_mean

    print("  Acceptance:")
    print(f"    [{'PASS' if winnable else 'FAIL'}] winnable: scripted({strong:.3f}) > random({weak:.3f})")
    print(
        f"    [{'PASS' if learned else 'FAIL'}] learned: after({after_mean:.3f}) > before({before_mean:.3f}) "
        f"by {delta:+.3f} (need >= +{MIN_LEARN_DELTA:.2f})"
    )

    if winnable and learned:
        print("\n  RESULT: PASS — a PPO Player learns King of the Hill (held-out game).")
        return 0
    print("\n  RESULT: FAIL — see numbers above (reported honestly).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
