#!/usr/bin/env python3
"""prove_ppo_learns.py — the Stage-1 milestone / acceptance test.

ONE prerequisite question, answered empirically:

  (1) Can a Player neural net actually LEARN in this simulator?
  (2) Do DIFFERENT training environments produce MEASURABLY DIFFERENT learning?

How it answers them, end to end, through the FROZEN contracts:

  * Build two arena configs that differ only in arena parameters
    (platform_width / gravity / knockback / spawn_gap): config A and config B.
  * For each config, train ``N_SEEDS`` small-MLP PPO Players against the FIXED
    scripted opponent (``PPOPlayerTrainer``, short budget).
  * Score win-rate vs the scripted opponent on a HELD-OUT arena of the same
    config (unseen seeds), BEFORE training (random-init net) and AFTER training,
    using the SAME scoring path as the reference policies
    (``FighterGameAdapter.evaluate``).
  * Assert AFTER > BEFORE for config A (it learned), and that config A's and
    config B's held-out after-win-rates differ measurably (different envs ->
    different learning).

Exit 0 iff BOTH hold; non-zero otherwise. Numbers are printed honestly — if PPO
does not learn, the table shows it and the script exits non-zero rather than
fudging.

Run:  python prove_ppo_learns.py            (defaults below)
Env overrides (for a quick smoke run):  N_SEEDS, PPO_EPISODES, EVAL_SEEDS
"""

from __future__ import annotations

import os
import sys
import warnings

# SB3/torch emit a few deprecation/UserWarnings that clutter the milestone
# output; the result table is what matters here.
warnings.filterwarnings("ignore")

from contracts import PlayerConfig, TrainingBudget
from games.fighter import FighterArena
from harness.fighter_adapter import FighterGameAdapter
from harness.ppo_trainer import PPOPlayerTrainer

# --- knobs (env-overridable so a smoke run is fast) ------------------------
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
PPO_EPISODES = int(os.environ.get("PPO_EPISODES", "1000"))  # -> ~60k timesteps
EVAL_SEEDS = int(os.environ.get("EVAL_SEEDS", "50"))

# A measurable A-vs-B difference must clear this margin to count.
MIN_AB_DIFF = float(os.environ.get("MIN_AB_DIFF", "0.20"))
# AFTER must beat BEFORE by at least this to count as "it learned".
MIN_LEARN_DELTA = float(os.environ.get("MIN_LEARN_DELTA", "0.10"))

# --- the two environments (differ ONLY in arena params) --------------------
# A: squarely in the learnable band — short PPO climbs to a high win-rate.
CONFIG_A = FighterArena(platform_width=10.0, gravity=0.6, knockback=2.5, spawn_gap=4.0)
# B: a harder regime — wider platform, weaker knockback, larger spawn gap. The
# game is still winnable in principle (scripted_expert wins), but the same short
# PPO budget reaches a measurably different win-rate.
CONFIG_B = FighterArena(platform_width=16.0, gravity=0.6, knockback=1.4, spawn_gap=7.0)


def _held_out(arena: FighterArena) -> FighterArena:
    """A held-out arena of the same config family.

    The adapter scores over a fixed *seed set* (env seeds 0..EVAL_SEEDS-1), and
    the trainer trains on independently-drawn episode seeds, so the eval seeds
    are unseen during training even at identical arena params. Returned as a
    distinct object to make the "held-out" intent explicit.
    """
    return FighterArena(
        platform_width=arena.platform_width,
        gravity=arena.gravity,
        knockback=arena.knockback,
        spawn_gap=arena.spawn_gap,
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _evaluate_config(
    name: str,
    arena: FighterArena,
    adapter: FighterGameAdapter,
) -> dict:
    """Train N_SEEDS PPO Players on ``arena``; return before/after held-out rates.

    BEFORE = random-init (untrained) Player's held-out win-rate, averaged over
    seeds. AFTER = trained Player's held-out win-rate, averaged over seeds. Both
    go through ``adapter.evaluate`` on a held-out arena — the identical scoring
    path used for the scripted/random references.
    """
    held = adapter.arenas_from_configs([_held_out(arena)], curriculum_id=f"heldout-{name}")
    train_arenas = adapter.arenas_from_configs([arena], curriculum_id=f"train-{name}")
    trainer = PPOPlayerTrainer(eval_seeds=EVAL_SEEDS)

    before_rates: list[float] = []
    after_rates: list[float] = []

    for i in range(N_SEEDS):
        seed = 100 * (1 if name == "A" else 2) + i  # disjoint seed bands per config
        config = PlayerConfig(
            architecture="mlp",
            num_seeds=1,
            budget=TrainingBudget(episodes=PPO_EPISODES),
            seed=seed,
            modal_parallel=False,
        )

        # BEFORE: an untrained PPO policy on the SAME net, scored on held-out.
        before_policy = _untrained_policy(arena, seed)
        before = adapter.evaluate(before_policy, held)
        before_rates.append(_label_score(before))

        # AFTER: train, then score the trained policy on held-out.
        job = trainer.submit(config, train_arenas)
        job.result()  # contract: collect the MatchResult (in-distribution score)
        after = adapter.evaluate(job.policy, held)
        after_rates.append(_label_score(after))

        print(
            f"  [{name}] seed {seed:>3}: before={before_rates[-1]:.3f} "
            f"after={after_rates[-1]:.3f}"
        )

    return {
        "name": name,
        "arena": arena,
        "before_mean": _mean(before_rates),
        "after_mean": _mean(after_rates),
        "before_rates": before_rates,
        "after_rates": after_rates,
    }


def _label_score(bundle: dict) -> float:
    """Pull the single mean_score out of a one-key ScoreBundle."""
    (entry,) = bundle.values()
    return float(entry["mean_score"])


def _untrained_policy(arena: FighterArena, seed: int):
    """A random-init PPO net (no training) wrapped as an obs->action policy.

    This is the honest BEFORE baseline: the exact same architecture the AFTER
    policy uses, just before any learning. In this game an untrained/random
    agent never beats the scripted opponent (confirmed: random win-rate is 0),
    so any positive AFTER rate is unambiguous learning.
    """
    from stable_baselines3 import PPO

    from harness.ppo_trainer import _MultiArenaFighterEnv, _make_policy_from_model

    env = _MultiArenaFighterEnv([arena], seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [64, 64]},
        device="cpu",
    )
    return _make_policy_from_model(model)


def _print_table(res_a: dict, res_b: dict) -> None:
    print()
    print("=" * 64)
    print("  STAGE-1 MILESTONE RESULTS — PPO learnability + arena difference")
    print("=" * 64)
    print(f"  seeds/config = {N_SEEDS}   ppo_episodes = {PPO_EPISODES}   eval_seeds = {EVAL_SEEDS}")
    print("-" * 64)
    header = f"  {'config':<8}{'platform_w':>11}{'knockback':>11}{'before':>9}{'after':>9}{'Δ':>8}"
    print(header)
    print("-" * 64)
    for r in (res_a, res_b):
        a = r["arena"]
        delta = r["after_mean"] - r["before_mean"]
        print(
            f"  {r['name']:<8}{a.platform_width:>11.1f}{a.knockback:>11.1f}"
            f"{r['before_mean']:>9.3f}{r['after_mean']:>9.3f}{delta:>+8.3f}"
        )
    print("-" * 64)
    ab_diff = abs(res_a["after_mean"] - res_b["after_mean"])
    print(f"  held-out AFTER win-rate:  A={res_a['after_mean']:.3f}  B={res_b['after_mean']:.3f}"
          f"  |A-B|={ab_diff:.3f}")
    print("=" * 64)


def main() -> int:
    print("Stage-1 milestone: does a PPO Player learn, and do arenas differ?")
    print(f"  CONFIG A: {CONFIG_A}")
    print(f"  CONFIG B: {CONFIG_B}")
    print()

    adapter = FighterGameAdapter(eval_seeds=EVAL_SEEDS)

    # Sanity: both arenas must be winnable in principle (scripted beats random),
    # else a low PPO score would mean "unwinnable", not "hard to learn".
    for name, arena in (("A", CONFIG_A), ("B", CONFIG_B)):
        arenas = adapter.arenas_from_configs([arena])
        strong = _label_score(adapter.evaluate(adapter.scripted_expert(), arenas))
        weak = _label_score(adapter.evaluate(adapter.random_policy(), arenas))
        print(f"  [{name}] winnability check: scripted={strong:.3f}  random={weak:.3f}")
        if strong <= weak:
            print(
                f"FAIL: config {name} is not winnable in principle "
                f"(scripted {strong:.3f} <= random {weak:.3f}).",
                file=sys.stderr,
            )
            return 2
    print()

    print("Training config A ...")
    res_a = _evaluate_config("A", CONFIG_A, adapter)
    print("Training config B ...")
    res_b = _evaluate_config("B", CONFIG_B, adapter)

    _print_table(res_a, res_b)

    # --- acceptance criteria ----------------------------------------------
    learn_delta_a = res_a["after_mean"] - res_a["before_mean"]
    learned_a = learn_delta_a >= MIN_LEARN_DELTA and res_a["after_mean"] > res_a["before_mean"]
    ab_diff = abs(res_a["after_mean"] - res_b["after_mean"])
    differ = ab_diff >= MIN_AB_DIFF

    print()
    print("  Acceptance:")
    print(
        f"    [{'PASS' if learned_a else 'FAIL'}] config A learned: "
        f"after({res_a['after_mean']:.3f}) > before({res_a['before_mean']:.3f}) "
        f"by {learn_delta_a:+.3f} (need >= +{MIN_LEARN_DELTA:.2f})"
    )
    print(
        f"    [{'PASS' if differ else 'FAIL'}] A vs B differ measurably: "
        f"|A-B| = {ab_diff:.3f} (need >= {MIN_AB_DIFF:.2f})"
    )

    if learned_a and differ:
        print("\n  RESULT: PASS — a PPO Player learns, and different arenas learn differently.")
        return 0

    print("\n  RESULT: FAIL — see numbers above (reported honestly, not fudged).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
