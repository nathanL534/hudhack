#!/usr/bin/env python3
"""prove_target_knockback_learns.py — Target-Knockback learnability check.

Target Knockback is Teacher-training game #2. This script answers the same
prerequisite questions for it that ``prove_ppo_learns.py`` answers for the fighter
and ``prove_koth_learns.py`` for KotH, with the SAME shape of evidence:

  (1) Can a PPO Player actually LEARN Target Knockback? (knock the OPPONENT into the
      target zone — a different objective from ring-out and from self-occupancy)
  (2) Is the scripted expert clearly stronger than random? (so a low PPO score would
      mean "hard to learn", never "unwinnable")
  (3) Does difficulty produce DISTINCT learning outcomes — a smooth win-rate slide
      across a difficulty grid (not a 0/1 cliff)?

How it answers them, through the SAME frozen pieces the fighter / KotH use:

  * Build a Target-Knockback arena in the learnable band (scripted wins big, random
    is beatable but not trivial, leaving real headroom to climb).
  * Train a small-MLP PPO Player as player 0 against the PARAMETRIC opponent on a
    ``TargetKnockbackEnv`` — the identical SB3 wiring the fighter / KotH trainers
    use, just the Target-Knockback env class. (It plugs into PPO exactly like the
    others; that IS the drop-in.)
  * Score win-rate vs the parametric opponent on a HELD-OUT arena of the same config
    (unseen seeds), BEFORE training (random-init net) and AFTER training, through
    ``TargetKnockbackGameAdapter.evaluate`` — the same scoring path as the
    references.
  * Print before/after/Δ per seed, assert AFTER > BEFORE (it learned) and
    scripted > random (winnable), and print a difficulty-grid slide.

Exit 0 iff the learnability and winnability gates hold; non-zero otherwise. Numbers
printed honestly.

Run:  .venv/bin/python prove_target_knockback_learns.py
Env overrides (for a quick smoke run): N_SEEDS, PPO_EPISODES, EVAL_SEEDS, DIFFICULTY
"""

from __future__ import annotations

import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from games.target_knockback import (
    TargetKnockbackArena,
    TargetKnockbackEnv,
    parametric_target_knockback,
)
from harness.target_knockback_adapter import TargetKnockbackGameAdapter

# --- knobs (env-overridable so a smoke run is fast) ------------------------
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
PPO_EPISODES = int(os.environ.get("PPO_EPISODES", "2000"))  # -> ~120k timesteps
EVAL_SEEDS = int(os.environ.get("EVAL_SEEDS", "50"))
# Difficulty chosen so the UNTRAINED (random-init) net reliably scores LOW (~0.18:
# the jump_turtle punishes passive / flailing play and self-scoring is suppressed)
# while a SKILLED policy has real headroom to climb (scripted ~0.66). This is the
# learnable band for THIS game: not so easy the untrained net already wins (which a
# higher win-rate band would allow — the untrained-net baseline is high-variance
# there), not so hard nothing learns. Empirically d=0.6 is where the before-baseline
# is tight and low with a clear after-ceiling. See the difficulty slide below.
DIFFICULTY = float(os.environ.get("DIFFICULTY", "0.6"))

# AFTER must beat BEFORE (in the MEAN over seeds) by at least this to count as
# "it learned". Per-seed deltas are printed honestly; the aggregate is the gate.
MIN_LEARN_DELTA = float(os.environ.get("MIN_LEARN_DELTA", "0.05"))

# Steps/episode estimate -> PPO timesteps (mirrors ppo_trainer's constants).
_STEPS_PER_EPISODE = 60
_MIN_TIMESTEPS = 4_000

# The arena to learn (the target dials are game-specific; difficulty sets opponent
# strength + zone geometry would normally come from a spec — here we dial it
# directly, same as prove_koth_learns.py dials KothArena fields).
ARENA = TargetKnockbackArena(
    platform_width=12.0, zone_half=1.6, zone_center_frac=0.5, difficulty=DIFFICULTY
)

# Difficulty grid for the "distinct outcomes" slide (cheap: references only, no PPO).
# Sampled finely through the transition (~0.45-0.65) so the smooth slide of the
# solvability proxy p — random's win-rate — is visible, not a 0/1 cliff.
DIFFICULTY_GRID = [0.0, 0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.8, 1.0]


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _make_env(arena: TargetKnockbackArena, seed: int):
    import gymnasium as gym

    class _Wrapped(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._rng = np.random.default_rng(seed)
            self._inner = TargetKnockbackEnv(
                arena,
                opponent_factory=lambda a: parametric_target_knockback(a, ego=1, seed=seed),
                seed=seed,
            )
            self.action_space = self._inner.action_space
            self.observation_space = self._inner.observation_space
            self.render_mode = None

        def reset(self, *, seed=None, options=None):
            ep_seed = int(self._rng.integers(1_000_000))
            self._inner = TargetKnockbackEnv(
                arena,
                opponent_factory=lambda a: parametric_target_knockback(a, ego=1, seed=ep_seed),
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


def _train_policy(arena: TargetKnockbackArena, seed: int):
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
        n_steps=1024,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.005,
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return _make_policy_from_model(model)


def _untrained_policy(arena: TargetKnockbackArena, seed: int):
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


def _difficulty_slide(eval_seeds: int) -> list[tuple[float, float, float]]:
    """For each difficulty: (random win-rate p, scripted win-rate). No PPO — cheap.

    Demonstrates "distinct learning outcomes": a smooth slide of the solvability
    proxy ``p`` (random win-rate) across the grid, not a 0/1 cliff.
    """
    rows: list[tuple[float, float, float]] = []
    for d in DIFFICULTY_GRID:
        arena = TargetKnockbackArena(
            platform_width=12.0, zone_half=1.6, zone_center_frac=0.5, difficulty=d
        )
        adapter = TargetKnockbackGameAdapter(eval_seeds=eval_seeds)
        held = adapter.arenas_from_configs([arena], curriculum_id=f"grid-{d}")
        p = _label_score(adapter.evaluate(adapter.random_policy(), held))
        s = _label_score(adapter.evaluate(adapter.scripted_expert(), held))
        rows.append((d, p, s))
    return rows


def main() -> int:
    print("Target-Knockback learnability: does a PPO Player learn to knock the")
    print("opponent into the target zone?")
    print(f"  ARENA: {ARENA}")
    print(
        f"  seeds={N_SEEDS}  ppo_episodes={PPO_EPISODES}  eval_seeds={EVAL_SEEDS}  difficulty={DIFFICULTY}"
    )
    print()

    adapter = TargetKnockbackGameAdapter(eval_seeds=EVAL_SEEDS)
    held = adapter.arenas_from_configs([ARENA], curriculum_id="heldout-tk")

    # Winnability check: scripted must clearly beat random.
    strong = _label_score(adapter.evaluate(adapter.scripted_expert(), held))
    weak = _label_score(adapter.evaluate(adapter.random_policy(), held))
    print(f"  winnability check: scripted={strong:.3f}  random={weak:.3f}")
    if strong <= weak:
        print(
            f"FAIL: arena not winnable in principle (scripted {strong:.3f} <= random {weak:.3f}).",
            file=sys.stderr,
        )
        return 2
    print()

    # PPO learning over N seeds.
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
        print(f"  seed {seed:>3}: before={before:.3f}  after={after:.3f}  Δ={after - before:+.3f}")

    before_mean = _mean(before_rates)
    after_mean = _mean(after_rates)
    delta = after_mean - before_mean

    # Difficulty slide (distinct outcomes).
    print()
    print("  difficulty slide (random win-rate p = solvability proxy; scripted = strong):")
    grid = _difficulty_slide(eval_seeds=max(20, EVAL_SEEDS // 2))
    for d, p, s in grid:
        band = p * (1.0 - p)
        flag = "  <- learnable band" if band > 0.2 else ""
        print(f"    d={d:.2f}:  random(p)={p:.3f}  scripted={s:.3f}  gap={s - p:+.3f}{flag}")

    print()
    print("=" * 64)
    print("  TARGET-KNOCKBACK MILESTONE — PPO learnability on game #2")
    print("=" * 64)
    print(f"  scripted={strong:.3f}  random={weak:.3f}")
    print(
        f"  held-out win-rate:  before={before_mean:.3f}  after={after_mean:.3f}  Δ={delta:+.3f}"
    )
    print("=" * 64)

    winnable = strong > weak
    learned = delta >= MIN_LEARN_DELTA and after_mean > before_mean
    # "Distinct outcomes": at least one grid difficulty lands in the learnable band
    # AND the random win-rate spans a wide range (a slide, not a cliff).
    ps = [p for _, p, _ in grid]
    any_in_band = any((p * (1.0 - p)) > 0.2 for p in ps)
    wide_slide = (max(ps) - min(ps)) >= 0.5
    distinct = any_in_band and wide_slide

    print("  Acceptance:")
    print(f"    [{'PASS' if winnable else 'FAIL'}] winnable: scripted({strong:.3f}) > random({weak:.3f})")
    print(
        f"    [{'PASS' if learned else 'FAIL'}] learned: after({after_mean:.3f}) > before({before_mean:.3f}) "
        f"by {delta:+.3f} (need >= +{MIN_LEARN_DELTA:.2f})"
    )
    print(
        f"    [{'PASS' if distinct else 'FAIL'}] distinct outcomes: random-p spans "
        f"[{min(ps):.2f},{max(ps):.2f}] with a learnable-band difficulty"
    )

    if winnable and learned and distinct:
        print("\n  RESULT: PASS — a PPO Player learns Target Knockback (Teacher game #2).")
        return 0
    print("\n  RESULT: FAIL — see numbers above (reported honestly).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
