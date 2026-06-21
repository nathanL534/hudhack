"""
Falsification harness for the anti-camping shaping fix (v2).

Validates four claims:
  1. CROSSOVER GONE: approach_punch >= approach_only at EVERY difficulty tested.
  2. CAMPING STILL PENALIZED: idle/punch_spam below approach_punch.
  3. NO NEW REGRESSION from the cap.
  4. KNOCKBACK INFLATION FIXED: me_after = post-own-move, pre-knockback.

Strategies driven against the REAL parametric_fighter opponent via the
REAL FighterEnv step + _anti_camping_shaping path.
"""

from __future__ import annotations

import sys
import statistics
from dataclasses import dataclass

sys.path.insert(0, ".")
from games.fighter import (
    Action,
    FighterArena,
    FighterEnv,
    FighterSim,
    parametric_fighter,
)
import numpy as np

# ---------------------------------------------------------------------------
# Strategy factories — each returns a callable(obs) -> int
# ---------------------------------------------------------------------------

def policy_idle():
    def act(obs): return int(Action.IDLE)
    return act

def policy_punch_spam():
    def act(obs): return int(Action.PUNCH)
    return act

def policy_approach_only(arena: FighterArena):
    """Always move toward the opponent — never punch."""
    w = arena.platform_width
    def act(obs):
        me_x = obs[0] * w
        opp_x = obs[3] * w
        return int(Action.RIGHT) if opp_x > me_x else int(Action.LEFT)
    return act

def policy_approach_punch(arena: FighterArena):
    """Move toward the opponent; punch when in range."""
    w = arena.platform_width
    reach = arena.punch_range + arena.fighter_half_width
    def act(obs):
        me_x = obs[0] * w
        opp_x = obs[3] * w
        rel = obs[10] * w
        if abs(rel) <= reach:
            return int(Action.PUNCH)
        return int(Action.RIGHT) if opp_x > me_x else int(Action.LEFT)
    return act


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(arena: FighterArena, agent_policy, seed: int) -> tuple[float, int | None, int]:
    """Returns (total_reward, winner, steps) using anti_camping_reward=True."""
    env = FighterEnv(
        arena,
        lambda a: parametric_fighter(a, seed=seed + 10000),
        seed=seed,
        anti_camping_reward=True,
    )
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent_policy(obs)
        obs, rew, term, trunc, info = env.step(action)
        total_reward += rew
        done = term or trunc
    return total_reward, info["winner"], info["steps"]


# ---------------------------------------------------------------------------
# Monte Carlo table — Task 1 + 2
# ---------------------------------------------------------------------------

DIFFICULTIES = [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.85, 1.0]
N_EPISODES = 150


@dataclass
class Result:
    mean: float
    std: float
    wins: float   # win-rate (winner==0)
    n: int


def mc_run(difficulty: float, policy_factory, n: int) -> Result:
    arena = FighterArena(difficulty=difficulty)
    rewards, winners = [], []
    for s in range(n):
        if callable(policy_factory) and hasattr(policy_factory, "__call__"):
            try:
                policy = policy_factory(arena)
            except TypeError:
                policy = policy_factory()
        else:
            policy = policy_factory
        r, w, _ = run_episode(arena, policy, seed=s)
        rewards.append(r)
        winners.append(1 if w == 0 else 0)
    return Result(
        mean=statistics.mean(rewards),
        std=statistics.stdev(rewards),
        wins=statistics.mean(winners),
        n=n,
    )


print("=" * 80)
print("TASK 1+2 — Monte Carlo reward table (N={} episodes each)".format(N_EPISODES))
print("=" * 80)
print(f"{'d':>5} | {'IDLE':>10} | {'PUNCH_SPAM':>10} | {'APPR_ONLY':>10} | {'APPR_PUNCH':>10} | appr_punch>=appr_only? | IDLE<APPR_PUNCH?")
print("-" * 100)

crossover_violations = []
camping_not_penalized = []
approach_punch_winrates = {}

for d in DIFFICULTIES:
    r_idle      = mc_run(d, lambda a: policy_idle(), N_EPISODES)
    r_punch     = mc_run(d, lambda a: policy_punch_spam(), N_EPISODES)
    r_appr      = mc_run(d, policy_approach_only, N_EPISODES)
    r_appr_p    = mc_run(d, policy_approach_punch, N_EPISODES)

    ok_cross = r_appr_p.mean >= r_appr.mean
    ok_camp  = r_idle.mean < r_appr_p.mean

    if not ok_cross:
        crossover_violations.append(d)
    if not ok_camp:
        camping_not_penalized.append(d)

    approach_punch_winrates[d] = r_appr_p.wins

    flag_c = "OK" if ok_cross else "FAIL*** approach_only WINS"
    flag_k = "OK" if ok_camp  else "FAIL*** idle>=appr_punch"

    print(
        f"{d:>5.2f} | {r_idle.mean:>10.4f} | {r_punch.mean:>10.4f} | "
        f"{r_appr.mean:>10.4f} | {r_appr_p.mean:>10.4f} | "
        f"{flag_c:>22} | {flag_k}"
    )

print()

# ---------------------------------------------------------------------------
# Task 1 verdict
# ---------------------------------------------------------------------------
print("TASK 1 — CROSSOVER VERDICT:")
if not crossover_violations:
    print("  PASS: approach_punch >= approach_only at EVERY difficulty. Crossover is GONE.")
else:
    print(f"  FAIL: approach_only beats approach_punch at difficulties: {crossover_violations}")

print()
print("TASK 2 — CAMPING STILL PENALIZED:")
if not camping_not_penalized:
    print("  PASS: idle < approach_punch at every difficulty.")
else:
    print(f"  FAIL: idle >= approach_punch at difficulties: {camping_not_penalized}")

# ---------------------------------------------------------------------------
# Task 3a — Monotone win-rate for approach_punch
# ---------------------------------------------------------------------------
print()
print("TASK 3a — WIN-RATE MONOTONE-DECREASING for APPROACH_PUNCH:")
prev_d, prev_wr = None, None
mono_violations = []
for d, wr in sorted(approach_punch_winrates.items()):
    if prev_wr is not None and wr > prev_wr + 0.05:  # allow small noise
        mono_violations.append((prev_d, d, prev_wr, wr))
    prev_d, prev_wr = d, wr

print("  Win-rates:", {d: round(v, 3) for d, v in sorted(approach_punch_winrates.items())})
if not mono_violations:
    print("  PASS: win-rate is monotone-non-increasing across difficulties (within noise).")
else:
    print("  FAIL: win-rate increased from d={:.2f} (wr={:.3f}) to d={:.2f} (wr={:.3f})".format(
        *mono_violations[0]))

# ---------------------------------------------------------------------------
# Task 3b — Low-difficulty (d<=0.3) not broken
# ---------------------------------------------------------------------------
print()
print("TASK 3b — LOW DIFFICULTY (d<=0.3) APPROACH_PUNCH win-rates:")
for d in [0.0, 0.3]:
    wr = approach_punch_winrates[d]
    ok = wr >= 0.5
    print(f"  d={d}: win-rate={wr:.3f}  {'OK (>=0.5)' if ok else 'REGRESSION CANDIDATE (<0.5)'}")

# ---------------------------------------------------------------------------
# Task 3c — _init_sep > 0 at reset (no degenerate case)
# ---------------------------------------------------------------------------
print()
print("TASK 3c — _init_sep > 0 across 500 resets:")
min_sep, max_sep = float("inf"), 0.0
zero_seps = 0
for s in range(500):
    arena = FighterArena()
    sim = FighterSim(arena=arena, seed=s)
    sep = sim._init_sep
    if sep <= 1e-9:
        zero_seps += 1
    min_sep = min(min_sep, sep)
    max_sep = max(max_sep, sep)
print(f"  min_sep={min_sep:.4f}, max_sep={max_sep:.4f}, zero_seps={zero_seps}")
if zero_seps == 0:
    print("  PASS: _init_sep is always strictly positive.")
else:
    print("  FAIL: found zero/near-zero init_sep in {} resets.".format(zero_seps))

# ---------------------------------------------------------------------------
# Task 3d — vanilla _shaping_reward path unchanged
# ---------------------------------------------------------------------------
print()
print("TASK 3d — vanilla _shaping_reward path (anti_camping_reward=False) smoke test:")
arena = FighterArena(difficulty=0.5)
env_plain = FighterEnv(arena, lambda a: parametric_fighter(a, seed=999), seed=7, anti_camping_reward=False)
obs, _ = env_plain.reset()
rewards_plain = []
done = False
while not done:
    obs, rew, term, trunc, info = env_plain.step(int(Action.RIGHT))
    rewards_plain.append(rew)
    done = term or trunc

# All step rewards must be in range of shaping_reward output (tiny, +-0.05 max per step)
terminal_reward = rewards_plain[-1]
step_rewards = rewards_plain[:-1]
shaping_only = [r for r in step_rewards]
max_shaping = max(abs(r) for r in shaping_only) if shaping_only else 0
print(f"  Max |step reward|={max_shaping:.5f}, terminal_reward={terminal_reward:.4f}")
print(f"  shaping path used: {'_shaping_reward' if not env_plain._anti_camping_reward else '_anti_camping_shaping'}")
ok_3d = not env_plain._anti_camping_reward and max_shaping < 0.1
print(f"  {'PASS' if ok_3d else 'FAIL'}: vanilla path used and shaping stays small.")

# ---------------------------------------------------------------------------
# Task 4 — Knockback inflation: a fighter knocked THROUGH opponent gets no credit
# ---------------------------------------------------------------------------
print()
print("TASK 4 — KNOCKBACK INFLATION: credit ONLY for self-driven movement")
print("  Scenario: agent PUNCHes from in-range => defender flies back past attacker.")
print("  Pre-fix: attacker used to get approach credit for the opponent's departure.")
print("  Post-fix: x0_moved = position after OWN move only (before punch resolution).")

# Set up a controlled scenario: two fighters close together, f0 punches f1 backward
# We'll inspect the raw sim directly

arena = FighterArena(difficulty=0.0, knockback=2.5, platform_width=20.0, spawn_gap=1.5)
sim = FighterSim(arena=arena, seed=0)

# Manually place fighters: f0 at x=5, f1 at x=6.2 (within punch range 1.4+0.4=1.8)
sim.f0.x = 5.0
sim.f1.x = 6.2
sim.f0.facing = 1  # f0 faces right (toward f1)
sim.f1.facing = -1
sim._init_sep = abs(sim.f0.x - sim.f1.x)  # 1.2

approach_before = sim.approach[0]
x0_before = sim.f0.x
x1_before = sim.f1.x

# f0 punches, f1 idles
sim.step(int(Action.PUNCH), int(Action.IDLE))

approach_after = sim.approach[0]
approach_delta = approach_after - approach_before
x0_after = sim.f0.x
x1_after = sim.f1.x

print(f"  f0 before: x={x0_before}, f1 before: x={x1_before}")
print(f"  f0 after: x={x0_after:.4f}, f1 after: x={x1_after:.4f}")
print(f"  Knockback: f1 moved {x1_after - x1_before:.4f} (away from f0)")
print(f"  Approach delta for f0: {approach_delta:.6f}")
print(f"  Expected: ZERO (f0 did not MOVE, only punched; PUNCH is not a move action)")
ok_4 = abs(approach_delta) < 1e-9
print(f"  {'PASS' if ok_4 else 'FAIL (spurious knockback credit leaked through)'}")

# Also test: f0 moves RIGHT (toward f1), then the follow-through of punch doesn't inflate
arena2 = FighterArena(difficulty=0.0, knockback=4.0, platform_width=20.0, spawn_gap=3.0)
sim2 = FighterSim(arena=arena2, seed=0)
sim2.f0.x = 5.0
sim2.f1.x = 6.5
sim2.f0.facing = 1
sim2.f1.facing = -1
sim2._init_sep = abs(sim2.f0.x - sim2.f1.x)  # 1.5 — just outside punch range

# Move toward each other with separate step approach check
approach_before2 = sim2.approach[0]
sim2.step(int(Action.RIGHT), int(Action.IDLE))
approach_after2 = sim2.approach[0]
delta2 = approach_after2 - approach_before2

print(f"\n  f0 moves RIGHT toward f1 (gap=1.5 before move):")
print(f"  Approach credit: {delta2:.4f} (expected ~0.35 = move_speed)")
ok_4b = abs(delta2 - 0.35) < 0.05
print(f"  {'PASS' if ok_4b else 'WARN: unexpected credit delta'}")

# ---------------------------------------------------------------------------
# Extra adversarial: approach_cap stress — can a chase-heavy strategy EVER farm?
# ---------------------------------------------------------------------------
print()
print("ADVERSARIAL STRESS TEST — Can any strategy farm > _init_sep approach credit?")

arena_adv = FighterArena(difficulty=1.0, platform_width=20.0)  # strongest opponent = max fleeing
max_approach_seen = 0.0
max_ratio_seen = 0.0

for s in range(200):
    sim_a = FighterSim(arena=arena_adv, seed=s)
    init_sep = sim_a._init_sep
    # Approach-only policy (always move toward opponent)
    while not sim_a.done:
        me_x = sim_a.f0.x
        opp_x = sim_a.f1.x
        a0 = int(Action.RIGHT) if opp_x > me_x else int(Action.LEFT)
        opp_obs = sim_a.observe(ego=1)
        a1 = int(parametric_fighter(arena_adv, seed=s)(opp_obs))
        sim_a.step(a0, a1)
        if sim_a.approach[0] > max_approach_seen:
            max_approach_seen = sim_a.approach[0]
        if init_sep > 1e-9 and sim_a.approach[0] / init_sep > max_ratio_seen:
            max_ratio_seen = sim_a.approach[0] / init_sep

print(f"  Across 200 episodes vs d=1.0 opponent: max approach={max_approach_seen:.4f}, "
      f"max approach/init_sep ratio={max_ratio_seen:.4f}")
if max_ratio_seen <= 1.0 + 1e-6:
    print("  PASS: approach NEVER exceeds _init_sep (cap is enforced).")
else:
    print(f"  FAIL: approach exceeded _init_sep by factor {max_ratio_seen:.4f}.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
all_pass = (
    not crossover_violations and
    not camping_not_penalized and
    not mono_violations and
    zero_seps == 0 and
    ok_3d and
    ok_4 and
    ok_4b and
    max_ratio_seen <= 1.0 + 1e-6
)
print("Overall:", "ALL CHECKS PASS" if all_pass else "ONE OR MORE CHECKS FAILED")
if crossover_violations:
    print("  BLOCKER: crossover at d=", crossover_violations)
if camping_not_penalized:
    print("  FAIL: camping not penalized at d=", camping_not_penalized)
print("Done.")
