#!/usr/bin/env python3
"""inspect_policy.py — behavioural autopsy of the Stage-1 trained PPO fighter.

The Stage-1 milestone (``prove_ppo_learns.py``) proves a PPO Player *learns*
(win-rate climbs 0 -> high vs the scripted opponent on config A). It does NOT
say WHAT the net learned. This script answers that, honestly:

  Is the trained net a real multi-move strategy, or a degenerate one-move spam
  (e.g. "walk toward the opponent + punch", never jumps, never idles)?

It ADDS no behaviour to the game/trainer — it only USES the existing public
classes/functions:
  * trains a small PPO net on config A vs the FULL-STRENGTH scripted opponent
    (config A has difficulty=1.0, so the parametric opponent's epsilon is 0 ==
    exactly scripted_fighter), matching prove_ppo_learns.py's config A and the
    same short budget (PPO_EPISODES=1000 -> ~60k timesteps);
  * uses the SAME helpers prove_ppo_learns.py imports
    (_MultiArenaFighterEnv, _make_policy_from_model) so training is byte-for-byte
    the Stage-1 path, while ALSO keeping the SB3 model around so we can read
    action probabilities (for entropy + state probes);
  * scores it through the public play_match / scripted_fighter / random_policy.

Then it measures and prints:
  (a) action histogram over all eval frames (% idle/left/right/jump/punch);
  (b) avg match length + win-rate (sanity it's still the winning net);
  (c) entropy of the action distribution + hand-built state probes
      (does the chosen action change with the state, or is it near-constant?);
  (d) sample trajectories (first ~20 actions of a few matches);
  (e) the same action histogram for scripted_fighter and random_policy.

Finally a one-line VERDICT: REAL multi-move / PARTIAL / ONE-MOVE SPAM. The
numbers drive the verdict; a degenerate policy is called degenerate.

Run:  python inspect_policy.py
Env overrides (smoke): PPO_EPISODES, N_EVAL_MATCHES, INSPECT_SEED
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import Counter

# SB3/torch emit deprecation/UserWarnings that clutter the report; the numbers
# are what matter here (same suppression prove_ppo_learns.py uses).
warnings.filterwarnings("ignore")

import numpy as np

from games.fighter import (
    STAGE1_ACTIONS,
    Action,
    FighterArena,
    FighterSim,
    play_match,
    random_policy,
    scripted_fighter,
)

# --- config A, matched to prove_ppo_learns.py CONFIG_A ----------------------
CONFIG_A = FighterArena(platform_width=10.0, gravity=0.6, knockback=2.5, spawn_gap=4.0)

# Stage-1 short budget (same default as prove_ppo_learns.py's PPO_EPISODES).
PPO_EPISODES = int(os.environ.get("PPO_EPISODES", "1000"))  # -> ~60k timesteps
N_EVAL_MATCHES = int(os.environ.get("N_EVAL_MATCHES", "80"))  # ~50-100 matches
INSPECT_SEED = int(os.environ.get("INSPECT_SEED", "101"))  # matches config A seed band

ACTION_NAMES = ["idle", "left", "right", "jump", "punch"]
assert [int(a) for a in STAGE1_ACTIONS] == [0, 1, 2, 3, 4], "action layout drifted"


# ---------------------------------------------------------------------------
# Train the Stage-1 net (public path) and keep the SB3 model for probe access
# ---------------------------------------------------------------------------


def train_stage1_net(seed: int):
    """Train the Stage-1 PPO net on config A vs the full-strength scripted opp.

    Reuses the EXACT helpers prove_ppo_learns.py imports and the EXACT PPO
    hyperparameters PPOPlayerTrainer.submit uses, so the resulting net is the
    Stage-1 winner. Returns (policy, model): the obs->action callable (the thing
    the milestone scores) and the underlying SB3 model (so we can read action
    probabilities for entropy + state probes).
    """
    from stable_baselines3 import PPO

    from harness.ppo_trainer import (
        _MIN_TIMESTEPS,
        _STEPS_PER_EPISODE,
        _MultiArenaFighterEnv,
        _make_policy_from_model,
    )

    total_timesteps = max(_MIN_TIMESTEPS, PPO_EPISODES * _STEPS_PER_EPISODE)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    env = _MultiArenaFighterEnv([CONFIG_A], seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        # Identical to PPOPlayerTrainer.submit's hyperparameters.
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        learning_rate=3e-4,
        device="cpu",
    )
    print(
        f"Training Stage-1 PPO net on config A "
        f"(platform_width={CONFIG_A.platform_width}, knockback={CONFIG_A.knockback}, "
        f"spawn_gap={CONFIG_A.spawn_gap}, difficulty={CONFIG_A.difficulty}) "
        f"vs full-strength scripted opponent, {total_timesteps} timesteps ..."
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    policy = _make_policy_from_model(model)
    return policy, model


# ---------------------------------------------------------------------------
# Eval: roll out matches, collect every frame's (obs, action), winners, lengths
# ---------------------------------------------------------------------------


def rollout_matches(policy, arena: FighterArena, n_matches: int):
    """Play ``n_matches`` against the full-strength scripted opponent.

    Drives the sim directly (same loop as play_match) so we can log EVERY frame's
    chosen action for the histogram and the FULL action sequence per match. The
    opponent is scripted_fighter (config A difficulty=1.0 => parametric epsilon 0
    == scripted), the full-strength reference.

    Returns: (frame_actions, winners, lengths, full_sequences). The first ~20
    actions of each full_sequence are the eyeball-able trajectory; the full
    sequences also let us check open-loop-ness (are all matches identical?).
    """
    frame_actions: list[int] = []
    winners: list = []
    lengths: list[int] = []
    full_sequences: list[tuple[int, ...]] = []

    for s in range(n_matches):
        sim = FighterSim(arena=arena, seed=s)
        opponent = scripted_fighter(arena, ego=1)
        seq: list[int] = []
        while not sim.done:
            a0 = int(policy(sim.observe(ego=0)))
            a1 = int(opponent(sim.observe(ego=1)))
            frame_actions.append(a0)
            seq.append(a0)
            sim.step(a0, a1)
        winners.append(sim.winner)
        lengths.append(sim.steps)
        full_sequences.append(tuple(seq))

    return frame_actions, winners, lengths, full_sequences


def baseline_frame_actions(policy_factory, arena: FighterArena, n_matches: int):
    """Action histogram for a baseline policy (scripted / random) over the same
    n_matches vs the full-strength scripted opponent — apples-to-apples with the
    trained net's histogram. ``policy_factory(seed)`` returns a fresh fighter-0
    policy per match (random_policy needs a fresh seed; scripted ignores it)."""
    frame_actions: list[int] = []
    for s in range(n_matches):
        sim = FighterSim(arena=arena, seed=s)
        agent = policy_factory(s)
        opponent = scripted_fighter(arena, ego=1)
        while not sim.done:
            a0 = int(agent(sim.observe(ego=0)))
            a1 = int(opponent(sim.observe(ego=1)))
            frame_actions.append(a0)
            sim.step(a0, a1)
    return frame_actions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def histogram(frame_actions: list[int]) -> dict[str, float]:
    """Percent usage of each of the 5 moves over all frames."""
    n = len(frame_actions)
    counts = Counter(frame_actions)
    return {
        ACTION_NAMES[i]: (100.0 * counts.get(i, 0) / n if n else 0.0)
        for i in range(len(ACTION_NAMES))
    }


def empirical_entropy(frame_actions: list[int]) -> float:
    """Shannon entropy (bits) of the EMPIRICAL action distribution over frames.

    Max is log2(5) ~= 2.322 bits (perfectly uniform over 5 moves). ~0 means the
    net plays essentially one action regardless of frame — the degenerate signal.
    """
    n = len(frame_actions)
    if n == 0:
        return 0.0
    counts = Counter(frame_actions)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def mean_policy_entropy(model, frame_obs: np.ndarray) -> float:
    """Mean entropy (bits) of the net's per-state action *distribution*.

    The empirical-action entropy can be near 0 either because the net is
    genuinely one-move, OR because the states it visits happen to call for one
    move. This reads the net's OWN softmax over actions at each visited state and
    averages its entropy. Near 0 => the net is internally near-deterministic
    toward a single action (the strong degenerate signal); higher => it spreads
    probability across moves depending on the state.
    """
    if len(frame_obs) == 0:
        return 0.0
    import torch

    obs_t, _ = model.policy.obs_to_tensor(frame_obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)
        # Categorical entropy is in nats; convert to bits for the log2(5) scale.
        ent_nats = dist.entropy().cpu().numpy()
    return float(np.mean(ent_nats) / math.log(2))


def probe_states(model, arena: FighterArena):
    """Hand-built states: does the chosen action change with the game state?

    Each probe is a full 11-dim observation (same layout as FighterSim.observe)
    placing the ego fighter and opponent in a specific situation. If the net
    returns the SAME action for every probe it is state-blind (fully degenerate);
    if the action varies it is at least reacting to state.
    """
    w = arena.platform_width

    def obs(me_x, opp_x, me_y=0.0, opp_y=0.0, me_vy=0.0, opp_vy=0.0):
        me_on = 1.0 if (0.0 <= me_x <= w and me_y <= 1e-9) else 0.0
        opp_on = 1.0 if (0.0 <= opp_x <= w and opp_y <= 1e-9) else 0.0
        facing = 1.0 if opp_x >= me_x else -1.0
        opp_facing = 1.0 if me_x >= opp_x else -1.0
        return np.array(
            [me_x / w, me_y, me_vy, opp_x / w, opp_y, opp_vy,
             me_on, opp_on, facing, opp_facing, (opp_x - me_x) / w],
            dtype=np.float32,
        )

    centre = w / 2.0
    probes = {
        "opp far to my RIGHT (center)":   obs(centre, centre + 3.0),
        "opp far to my LEFT (center)":    obs(centre, centre - 3.0),
        "opp in punch-range RIGHT":       obs(centre, centre + 1.0),
        "opp in punch-range LEFT":        obs(centre, centre - 1.0),
        "I'm near LEFT edge, opp right":  obs(0.6, 3.0),
        "I'm near RIGHT edge, opp left":  obs(w - 0.6, w - 3.0),
        "opp ABOVE me (mid-jump), close": obs(centre, centre + 0.8, opp_y=2.0, opp_vy=0.5),
        "I'm airborne over center":       obs(centre, centre + 2.0, me_y=2.0, me_vy=0.5),
    }

    rows = []
    distinct = set()
    for label, o in probes.items():
        a, _ = model.predict(o, deterministic=True)
        a = int(a)
        rows.append((label, ACTION_NAMES[a]))
        distinct.add(a)
    return rows, distinct


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def decide_verdict(
    hist: dict[str, float],
    distinct_probe_actions: int,
    n_distinct_sequences: int,
    n_matches: int,
    policy_entropy_bits: float,
) -> tuple[str, str]:
    """Classify the trained net's behaviour HONESTLY from histogram + probes +
    open-loop check + per-state entropy. A rich-looking histogram is NOT enough:
    a net that plays the same fixed action cadence every match, ignores the
    opponent, and collapses its probes to one action is degenerate even if that
    cadence touches 3 motor primitives.

    Signals:
      * n_used      = moves used >= 5% of frames (histogram richness).
      * open_loop   = every match produced the byte-identical action sequence
                      (n_distinct_sequences == 1 across many seeds). This means
                      the policy is NOT reacting to state at all — it replays one
                      memorised motor loop. The single strongest degeneracy flag.
      * probe_blind = the net returns essentially one action across hand-built
                      states (distinct_probe_actions <= 1, or <= 2 with one
                      dominating).
      * det_entropy = per-state policy entropy ~ 0 (net is internally argmax-
                      deterministic toward a single action per state).

    Verdict ladder (conservative; degeneracy wins ties):
      * open_loop OR probe collapse to 1 action  -> ONE-MOVE SPAM (degenerate):
        an open-loop fixed cycle is functionally one-move spam dressed up in 2-3
        motor primitives — it is NOT a strategy, it ignores the game state.
      * 0/1 move used >= 5%                       -> ONE-MOVE SPAM (degenerate).
      * 2-3 moves used, but reacts to state       -> PARTIAL (2-3 moves).
      * 4+ moves used and reacts to state         -> REAL multi-move strategy.
    """
    used = {k: v for k, v in hist.items() if v >= 5.0}
    n_used = len(used)
    jump_pct, idle_pct = hist["jump"], hist["idle"]
    open_loop = n_matches > 1 and n_distinct_sequences == 1
    probe_blind = distinct_probe_actions <= 1

    if open_loop or probe_blind or n_used <= 1:
        verdict = "ONE-MOVE SPAM (degenerate)"
    elif n_used <= 3:
        verdict = "PARTIAL (2-3 moves)"
    else:
        verdict = "REAL multi-move strategy"

    note_bits = [f"{n_used} move(s) used >=5% of frames"]
    if open_loop:
        note_bits.append(
            f"OPEN-LOOP: all {n_matches} matches gave the IDENTICAL action "
            f"sequence (net ignores opponent/spawn — a fixed memorised cycle)"
        )
    note_bits.append(f"per-state policy entropy {policy_entropy_bits:.3f} bits "
                     f"({'near-deterministic' if policy_entropy_bits < 0.2 else 'spreads probability'})")
    if jump_pct < 1.0:
        note_bits.append(f"jump effectively unused ({jump_pct:.1f}%)")
    if idle_pct < 1.0:
        note_bits.append(f"idle effectively unused ({idle_pct:.1f}%)")
    note_bits.append(f"{distinct_probe_actions} distinct action(s) across 8 state probes")
    return verdict, "; ".join(note_bits)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_hist(hist: dict[str, float]) -> str:
    return "  ".join(f"{name}={hist[name]:5.1f}%" for name in ACTION_NAMES)


def main() -> int:
    print("=" * 70)
    print("  BEHAVIOURAL AUTOPSY — what does the Stage-1 PPO fighter actually do?")
    print("=" * 70)
    print(f"  config A vs full-strength scripted opponent | eval matches = {N_EVAL_MATCHES}")
    print(f"  ppo_episodes = {PPO_EPISODES}   seed = {INSPECT_SEED}")
    print("-" * 70)

    policy, model = train_stage1_net(INSPECT_SEED)

    # (a)+(b)+(d): roll out the trained net.
    frame_actions, winners, lengths, full_sequences = rollout_matches(
        policy, CONFIG_A, N_EVAL_MATCHES
    )
    n_distinct_sequences = len(set(full_sequences))
    trained_hist = histogram(frame_actions)
    wins = sum(1 for w in winners if w == 0)
    losses = sum(1 for w in winners if w == 1)
    draws = sum(1 for w in winners if w is None)
    win_rate = wins / len(winners) if winners else 0.0
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0

    # (e): baselines.
    scripted_actions = baseline_frame_actions(
        lambda _s: scripted_fighter(CONFIG_A, ego=0), CONFIG_A, N_EVAL_MATCHES
    )
    random_actions = baseline_frame_actions(
        lambda s: random_policy(seed=s), CONFIG_A, N_EVAL_MATCHES
    )
    scripted_hist = histogram(scripted_actions)
    random_hist = histogram(random_actions)

    # (c): entropy + state probes.
    emp_entropy = empirical_entropy(frame_actions)
    frame_obs = _collect_eval_obs(policy, CONFIG_A, N_EVAL_MATCHES)
    policy_entropy = mean_policy_entropy(model, frame_obs)
    probe_rows, distinct = probe_states(model, CONFIG_A)
    max_entropy = math.log2(len(ACTION_NAMES))

    # ---- print report ----
    print()
    print("(a) ACTION HISTOGRAM — % of all frames (trained net)")
    print(f"    TRAINED   {_fmt_hist(trained_hist)}   [{len(frame_actions)} frames]")
    print()
    print("(e) BASELINES — same histogram, for comparison")
    print(f"    SCRIPTED  {_fmt_hist(scripted_hist)}   [{len(scripted_actions)} frames]")
    print(f"    RANDOM    {_fmt_hist(random_hist)}   [{len(random_actions)} frames]")
    print()
    print("(b) MATCH STATS (trained net vs full-strength scripted opponent)")
    print(f"    win-rate = {win_rate:.3f}  ({wins} W / {losses} L / {draws} D of {len(winners)})")
    print(f"    avg match length = {avg_len:.1f} steps (max_steps={CONFIG_A.max_steps})")
    print()
    print("(c) STATE-DEPENDENCE")
    print(f"    empirical action entropy = {emp_entropy:.3f} bits  (max {max_entropy:.3f} = uniform/5)")
    print(f"    mean per-state policy entropy = {policy_entropy:.3f} bits  "
          f"(near 0 => net is internally near-deterministic toward one action)")
    print(f"    distinct FULL action sequences across {N_EVAL_MATCHES} matches = {n_distinct_sequences}"
          f"  ({'OPEN-LOOP: same loop every match, ignores state' if n_distinct_sequences == 1 else 'varies with state'})")
    print(f"    state probes ({len(probe_rows)} hand-built states):")
    for label, act in probe_rows:
        print(f"      - {label:<34} -> {act}")
    print(f"    distinct actions across probes: {len(distinct)} "
          f"({'STATE-BLIND' if len(distinct) <= 1 else 'limited reaction'})")
    print()
    print("(d) SAMPLE TRAJECTORIES — first <=20 actions of 3 matches")
    for i in range(min(3, len(full_sequences))):
        seq = " ".join(ACTION_NAMES[a] for a in full_sequences[i][:20])
        print(f"    match {i} (winner={_winner_str(winners[i])}, len={lengths[i]}): {seq}")
    print()

    verdict, note = decide_verdict(
        trained_hist, len(distinct), n_distinct_sequences, N_EVAL_MATCHES, policy_entropy
    )
    print("=" * 70)
    print(f"  VERDICT: {verdict}")
    print(f"  basis: {note}")
    print("=" * 70)

    # ---- persist results ----
    results = {
        "config_a": {
            "platform_width": CONFIG_A.platform_width,
            "gravity": CONFIG_A.gravity,
            "knockback": CONFIG_A.knockback,
            "spawn_gap": CONFIG_A.spawn_gap,
            "difficulty": CONFIG_A.difficulty,
        },
        "ppo_episodes": PPO_EPISODES,
        "n_eval_matches": N_EVAL_MATCHES,
        "seed": INSPECT_SEED,
        "trained_action_histogram_pct": trained_hist,
        "scripted_action_histogram_pct": scripted_hist,
        "random_action_histogram_pct": random_hist,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_match_length_steps": avg_len,
        "empirical_action_entropy_bits": emp_entropy,
        "max_action_entropy_bits": max_entropy,
        "mean_per_state_policy_entropy_bits": policy_entropy,
        "distinct_full_match_action_sequences": n_distinct_sequences,
        "open_loop_all_matches_identical": n_distinct_sequences == 1,
        "example_full_sequence": [ACTION_NAMES[a] for a in full_sequences[0]] if full_sequences else [],
        "state_probes": [{"state": label, "action": act} for label, act in probe_rows],
        "distinct_probe_actions": len(distinct),
        "verdict": verdict,
        "verdict_basis": note,
        "n_frames_trained": len(frame_actions),
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspect_policy_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


def _collect_eval_obs(policy, arena: FighterArena, n_matches: int) -> np.ndarray:
    """All ego-0 observations the trained net actually visits across eval matches,
    for the per-state policy-entropy estimate (same trajectory distribution as the
    histogram, just keeping the obs instead of the action)."""
    obs_list = []
    for s in range(n_matches):
        sim = FighterSim(arena=arena, seed=s)
        opponent = scripted_fighter(arena, ego=1)
        while not sim.done:
            o = sim.observe(ego=0)
            obs_list.append(o)
            a0 = int(policy(o))
            a1 = int(opponent(sim.observe(ego=1)))
            sim.step(a0, a1)
    return np.array(obs_list, dtype=np.float32)


def _winner_str(w) -> str:
    if w == 0:
        return "net"
    if w == 1:
        return "scripted"
    return "draw"


if __name__ == "__main__":
    raise SystemExit(main())
