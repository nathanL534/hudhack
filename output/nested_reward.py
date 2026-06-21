"""output/nested_reward.py — THE nested-RL Teacher reward (held-out transfer).

The project's core loop, finally assembled from the REAL pieces:

    Teacher arena  ->  train N fresh Players on it (Modal-parallel)  ->
    measure each Player's before/after win-rate on a FIXED HELD-OUT reference set
    ->  reward = avg held-out improvement across seeds (clamped to [0, 1]).

WHY held-out (the whole point). Raw before/after improvement is MAXIMISED on
TRIVIAL (easy) arenas: an easy arena lets a Player climb fast on its OWN arena,
so a Teacher graded on in-distribution improvement games the reward by emitting
trivial arenas (we measured exactly this: d=0.2 gives +0.27 "improvement" on its
own easy arena). The fix is to grade transfer to a STANDARD benchmark the Teacher
cannot move: train the Player wherever the Teacher likes, but EVALUATE before/after
on a FIXED held-out reference set (a couple of reference difficulties, default
geometry). A trivial training arena transfers little; a genuinely good training
arena transfers real skill -> higher reward. The fixed yardstick removes the
easy-arena degeneracy.

The reward worker is ``modal_player.train_player_transfer`` (deployed) /
``modal_player.local_transfer_worker`` (local fallback). This module fans seeds
out through it, averages the held-out improvement, and clamps to [0, 1].

Run from repo root (.venv has modal + sb3):

    # the reward of ONE arena (3 Modal seeds):
    .venv/bin/python output/nested_reward.py reward --difficulty 0.6

    # VALIDATE the reward distinguishes good vs trivial arenas (~6 candidates):
    .venv/bin/python output/nested_reward.py validate

    # local fallback (no Modal credits; same code path):
    .venv/bin/python output/nested_reward.py validate --backend local

The transfer worker must be deployed for backend=modal:
    .venv/bin/modal deploy modal_player.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Repo root on path (script lives in output/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import ArenaSpec, CurriculumSpec  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
REPLAYS_DIR = Path(__file__).resolve().parents[1] / "replays"

# The deployed reward-worker app. Defaults to the SHARED ``crucible-player`` (every
# existing caller / the demo). An isolated run can point at its OWN redeployed worker by
# setting ``CRUCIBLE_PLAYER_APP`` (e.g. ``crucible-player-behavior-diverse-B256-final``)
# so it never has to touch or redeploy the shared app — the byte-identical isolation rule.
MODAL_APP = (os.environ.get("CRUCIBLE_PLAYER_APP") or "").strip() or "crucible-player"
MODAL_FN = "train_player_transfer"

# Reward defaults. Short-but-real PPO budget per Player (a trend check, not a
# tuned competitor); enough held-out eval seeds for a stable win-rate.
DEFAULT_SEEDS = (1, 2, 3)            # ~3 fresh Players per arena
DEFAULT_EPISODES = 1000
DEFAULT_EVAL_SEEDS = 50


# ---------------------------------------------------------------------------
# Curriculum-spec -> reward payload
# ---------------------------------------------------------------------------


def _arena_payload(spec: ArenaSpec, *, episodes: int, eval_seeds: int, curriculum_id: str,
                   capture_replay_id: str | None = None,
                   held_out_arenas: list[dict] | None = None,
                   geometry_override: dict | None = None,
                   student_cfg: dict | None = None) -> dict:
    """Translate one ArenaSpec into the transfer-worker payload.

    The fighter knobs the worker reads are ``difficulty`` (the Teacher's primary
    dial) plus the optional geometry overrides. ArenaSpec.difficulty drives both
    the physical hardness and the parametric opponent's strength, so a single dial
    is a valid arena; map_size widens the platform (mirrors fighter_adapter's
    deterministic gridworld->geometry mapping) so the candidates differ in more
    than just difficulty.

    ``geometry_override`` (optional): when the caller has the full FIGHTER knob set
    (``platform_width`` / ``gravity`` / ``knockback`` / ``spawn_gap``) it is threaded
    straight into the worker payload instead of being derived from ``map_size`` — so
    the Teacher's emitted geometry trains the Player verbatim (the FIGHTER game path).

    ``held_out_arenas`` (optional): a list of full arena-spec dicts (the BROAD,
    structurally-diverse held-out population from ``output/broad_eval_set.py``). The
    worker (``modal_player._held_out_reference_arenas``) honors this key and scores
    transfer on it instead of the two narrow default-geometry turtle references.

    ``student_cfg`` (optional): the FOCUSED-population spec for the INNER Student the
    Ring-Out reward trains — wider MLP (``arch``), a focused opponent league
    (70% aggressive / 30% prior_student), the gated decisive ring-out reward, and a
    higher ``ent_coef``. Threaded VERBATIM into the worker as ``payload["student_cfg"]``
    so ``modal_player._real_ppo_transfer_result`` trains a FIGHTING (not camping)
    inner Student. When ``None``/absent the key is simply not set, so the worker runs
    the ORIGINAL byte-identical vanilla inner reward (demo-safe default).
    """
    payload: dict = {
        "difficulty": float(spec.difficulty),
        "platform_width": round(6.0 + 0.5 * spec.map_size, 3),
        "ppo_episodes": int(episodes),
        "eval_seeds": int(eval_seeds),
        "curriculum_id": curriculum_id,
        "architecture": "mlp",
    }
    if geometry_override:
        for k in ("platform_width", "gravity", "knockback", "spawn_gap"):
            if k in geometry_override and geometry_override[k] is not None:
                payload[k] = float(geometry_override[k])
    if held_out_arenas:
        payload["held_out_arenas"] = held_out_arenas
    if capture_replay_id:
        payload["capture_replay_id"] = capture_replay_id
    if student_cfg:
        payload["student_cfg"] = student_cfg
    return payload


def _run_seeds(payloads: list[dict], seeds: list[int], *, backend: str) -> list[dict]:
    """Fan ``seeds`` out through the transfer worker (Modal-parallel or local).

    ``payloads`` is per-seed (one payload per seed) so a single seed can carry a
    one-off flag (e.g. ``capture_replay_id``) without forcing every seed to repeat
    the work. Modal: one ``fn.map`` over the per-seed payloads (concurrent remote
    PPO). Local: the same body in-process (a ThreadPool would just contend on one
    CPU's torch, so the local fallback runs sequentially — correctness, not speed).
    """
    if backend == "modal":
        import modal

        fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
        return list(fn.map(payloads, seeds))
    if backend == "local":
        from modal_player import local_transfer_worker

        return [local_transfer_worker(p, s) for p, s in zip(payloads, seeds)]
    raise ValueError(f"unknown backend {backend!r} (use 'modal' or 'local')")


# ---------------------------------------------------------------------------
# THE nested-RL Teacher reward
# ---------------------------------------------------------------------------


def teacher_reward(
    curriculum_spec: CurriculumSpec,
    *,
    backend: str = "modal",
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    episodes: int = DEFAULT_EPISODES,
    eval_seeds: int = DEFAULT_EVAL_SEEDS,
    capture_replay: bool = False,
    held_out_arenas: list[dict] | None = None,
    geometry_override: dict | None = None,
    student_cfg: dict | None = None,
    _detail_sink: dict | None = None,
) -> float:
    """The nested-RL Teacher reward in [0, 1] for a curriculum spec.

    Trains N fresh Players (one per seed) on the curriculum's arena(s) via the
    Modal-parallel transfer worker, then returns the AVERAGE held-out improvement
    across seeds, clamped to [0, 1]. Held-out improvement is the Player's win-rate
    gain on the FIXED reference benchmark (after-training minus an untrained net),
    so a trivial arena (little transfer) scores LOW and a genuinely good training
    arena (real transfer) scores HIGH — the reward is NOT maxed by the easiest
    arena.

    For a multi-arena curriculum each arena is evaluated and the per-arena rewards
    are averaged (the Teacher's curriculum-level signal). ``_detail_sink``, if
    given, is populated with the raw per-seed rows for reporting.

    ``student_cfg`` (optional): the FOCUSED-population spec for the INNER Student.
    Passed straight through to the transfer worker (see ``_arena_payload``); when it
    is ``None``/absent the worker runs the ORIGINAL vanilla inner reward, so the old
    Teacher / demo path is byte-identical.
    """
    curriculum_spec.validate()
    per_arena_rewards: list[float] = []
    detail: list[dict] = []

    for ai, arena in enumerate(curriculum_spec.arenas):
        replay_id = None
        if capture_replay and ai == 0:
            replay_id = f"{curriculum_spec.curriculum_id}_a{ai}"
        seed_list = list(seeds)
        # Per-seed payloads: only the FIRST seed carries the replay-capture id, so
        # the (slow) trained-Player rollout happens once, not once per seed.
        payloads = [
            _arena_payload(
                arena,
                episodes=episodes,
                eval_seeds=eval_seeds,
                curriculum_id=f"{curriculum_spec.curriculum_id}-a{ai}",
                capture_replay_id=(replay_id if (replay_id and si == 0) else None),
                held_out_arenas=held_out_arenas,
                geometry_override=geometry_override,
                student_cfg=student_cfg,
            )
            for si in range(len(seed_list))
        ]
        rows = _run_seeds(payloads, seed_list, backend=backend)
        for r in rows:
            assert r.get("status") == "ppo_transfer", f"non-transfer result: {r!r}"

        improvements = [float(r["held_out_improvement"]) for r in rows]
        mean_improvement = sum(improvements) / len(improvements)
        # Map to [0, 1]: negative transfer (forgetting) floors at 0; the held-out
        # improvement is naturally bounded by 1.0 (a win-rate delta), so a clamp is
        # the whole mapping.
        arena_reward = max(0.0, min(1.0, mean_improvement))
        per_arena_rewards.append(arena_reward)

        detail.append(
            {
                "arena_index": ai,
                "difficulty": float(arena.difficulty),
                "map_size": int(arena.map_size),
                "mean_held_out_improvement": round(mean_improvement, 4),
                "arena_reward": round(arena_reward, 4),
                "mean_before": round(sum(r["before_winrate"] for r in rows) / len(rows), 4),
                "mean_after": round(sum(r["after_winrate"] for r in rows) / len(rows), 4),
                "mean_train_arena_winrate": round(
                    sum(r["train_arena_winrate"] for r in rows) / len(rows), 4
                ),
                "held_out_difficulties": rows[0].get("held_out_difficulties"),
                "per_seed_improvement": [round(x, 4) for x in improvements],
                "rows": rows,
            }
        )

    reward = sum(per_arena_rewards) / len(per_arena_rewards) if per_arena_rewards else 0.0
    if _detail_sink is not None:
        _detail_sink["arenas"] = detail
        _detail_sink["reward"] = round(reward, 4)
    return float(reward)


# ---------------------------------------------------------------------------
# Replay persistence (write whatever the worker captured)
# ---------------------------------------------------------------------------


def _persist_captured_replays(detail: list[dict]) -> list[str]:
    """Write any replays the transfer worker captured to replays/modal_trained_<id>.json."""
    written: list[str] = []
    for arena_detail in detail:
        for row in arena_detail.get("rows", []):
            cap = row.get("replay")
            if not cap:
                continue
            out = REPLAYS_DIR / f"modal_trained_{cap['replay_id']}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(cap["data"], indent=2))
            written.append(str(out))
    return written


# ---------------------------------------------------------------------------
# Single-arena reward (CLI: reward)
# ---------------------------------------------------------------------------


def _spec_for(difficulty: float, *, map_size: int = 10, curriculum_id: str | None = None) -> CurriculumSpec:
    cid = curriculum_id or f"arena-d{difficulty}-m{map_size}"
    return CurriculumSpec(
        game_id="ring-out-duel",
        curriculum_id=cid,
        arenas=[
            ArenaSpec(
                map_size=int(map_size),
                doors=0,
                keys=0,
                hazard_density=0.0,
                difficulty=float(difficulty),
            )
        ],
    )


def run_single(args) -> dict:
    spec = _spec_for(args.difficulty, map_size=args.map_size)
    sink: dict = {}
    t0 = time.time()
    reward = teacher_reward(
        spec,
        backend=args.backend,
        seeds=tuple(args.seeds),
        episodes=args.episodes,
        eval_seeds=args.eval_seeds,
        capture_replay=args.capture_replay,
        _detail_sink=sink,
    )
    wall = time.time() - t0
    replays = _persist_captured_replays(sink["arenas"]) if args.capture_replay else []
    print(f"arena d={args.difficulty} map_size={args.map_size}: "
          f"TEACHER REWARD = {reward:.4f}  ({wall:.1f}s, {len(args.seeds)} seeds)")
    a = sink["arenas"][0]
    print(f"  held-out before={a['mean_before']:.3f} after={a['mean_after']:.3f} "
          f"improvement={a['mean_held_out_improvement']:+.3f}")
    print(f"  (in-distribution train-arena win-rate={a['mean_train_arena_winrate']:.3f} "
          f"-- high here + low reward == a trivial arena)")
    for p in replays:
        print(f"  wrote replay {p}")
    return {"reward": reward, "wall_s": round(wall, 2), "detail": sink, "replays": replays}


# ---------------------------------------------------------------------------
# VALIDATION (CLI: validate) — does the reward distinguish good vs trivial?
# ---------------------------------------------------------------------------

# Six candidate arenas spanning the spectrum the Teacher would explore. Labels
# are PREDICTIONS we are testing — the reward must rank "good" arenas above the
# trivial/impossible ones for the held-out fix to be real.
#   * TRIVIAL : very easy arena. A Player trivially wins on it, but that teaches
#               nothing that transfers to the harder held-out set -> LOW reward.
#   * GOOD    : learnable-band arena. A Player learns transferable skill -> HIGH.
#   * IMPOSSIBLE: an undefeatable-turtle arena. The Player can't learn anything
#               useful (it mostly draws) -> LOW reward.
VALIDATION_CANDIDATES = [
    {"difficulty": 0.10, "map_size": 10, "label": "trivial",    "note": "very easy opponent (self-edging); the Player trivially wins its own arena but learns little that transfers to the turtle-active references"},
    {"difficulty": 0.30, "map_size": 10, "label": "good",       "note": "learnable band (calibrated peak): the Player learns transferable skill -> strongest held-out transfer"},
    {"difficulty": 0.50, "map_size": 10, "label": "good",       "note": "learnable band: solid transferable skill"},
    {"difficulty": 0.70, "map_size": 10, "label": "good",       "note": "upper learnable band: a second transfer mode (timing a partly-engaged turtle)"},
    {"difficulty": 0.85, "map_size": 10, "label": "impossible", "note": "turtle reliably forces draws; the Player can't learn a win here -> ~0 transfer"},
    {"difficulty": 0.95, "map_size": 10, "label": "impossible", "note": "undefeatable turtle: almost all draws, nothing to learn -> ~0 transfer"},
]


def run_validate(args) -> dict:
    print("=== VALIDATION: does the held-out reward distinguish good vs trivial arenas? ===")
    print(f"    backend={args.backend}  seeds={list(args.seeds)}  episodes={args.episodes}  "
          f"eval_seeds={args.eval_seeds}")
    print(f"    held-out reference difficulties (FIXED): see worker "
          f"(default {tuple(__import__('modal_player').HELD_OUT_REFERENCE_DIFFICULTIES)})")
    print()

    results = []
    t0 = time.time()
    for i, cand in enumerate(VALIDATION_CANDIDATES):
        spec = _spec_for(cand["difficulty"], map_size=cand["map_size"],
                         curriculum_id=f"val-{i:02d}-d{cand['difficulty']}")
        sink: dict = {}
        # Capture a replay from the FIRST 'good' arena's trained Player so the
        # viewer can play a real Modal-trained fighter.
        capture = (cand["label"] == "good" and not any(r.get("replays") for r in results))
        reward = teacher_reward(
            spec,
            backend=args.backend,
            seeds=tuple(args.seeds),
            episodes=args.episodes,
            eval_seeds=args.eval_seeds,
            capture_replay=capture,
            _detail_sink=sink,
        )
        a = sink["arenas"][0]
        replays = _persist_captured_replays(sink["arenas"]) if capture else []
        results.append(
            {
                "candidate_id": f"val-{i:02d}",
                "difficulty": cand["difficulty"],
                "map_size": cand["map_size"],
                "label": cand["label"],
                "note": cand["note"],
                "reward": round(reward, 4),
                "held_out_improvement": a["mean_held_out_improvement"],
                "held_out_before": a["mean_before"],
                "held_out_after": a["mean_after"],
                "train_arena_winrate": a["mean_train_arena_winrate"],
                "per_seed_improvement": a["per_seed_improvement"],
                "replays": replays,
            }
        )
        print(f"    [{i+1}/{len(VALIDATION_CANDIDATES)}] d={cand['difficulty']:<5} "
              f"({cand['label']:>10}) reward={reward:.4f}  "
              f"held_out {a['mean_before']:.2f}->{a['mean_after']:.2f} "
              f"({a['mean_held_out_improvement']:+.3f})  "
              f"train_arena_wr={a['mean_train_arena_winrate']:.2f}")
    wall = time.time() - t0

    # --- the reward-per-arena table ---
    print()
    print("    === REWARD-PER-ARENA TABLE ===")
    print(f"    {'diff':>5} {'label':>11} {'reward':>8} {'heldout_impr':>13} "
          f"{'train_wr':>9}")
    for r in results:
        print(f"    {r['difficulty']:>5} {r['label']:>11} {r['reward']:>8.4f} "
              f"{r['held_out_improvement']:>+13.3f} {r['train_arena_winrate']:>9.2f}")

    # --- the degeneracy check ---
    # The anti-degeneracy property the FIXED held-out test must buy us is: the
    # reward is NOT monotonically "easier = better" — the easiest arena must not
    # max the reward, and impossible arenas must collapse to ~0. (A reward measured
    # on the TRAINING arena instead is maxed by the easiest arena; that is the
    # degeneracy we are removing.)
    good = [r for r in results if r["label"] == "good"]
    impossible = [r for r in results if r["label"] == "impossible"]
    mean_good = sum(r["reward"] for r in good) / len(good) if good else 0.0
    mean_impossible = sum(r["reward"] for r in impossible) / len(impossible) if impossible else 0.0

    best = max(results, key=lambda r: r["reward"])
    easiest = min(results, key=lambda r: r["difficulty"])
    # FIXED iff (1) the highest-reward arena is NOT the easiest (easy doesn't max
    # the reward), (2) the best arena is a 'good' (learnable-band) arena, and (3)
    # good arenas clearly beat impossible ones (impossible collapses to ~0).
    easiest_is_not_best = best["candidate_id"] != easiest["candidate_id"]
    best_is_good = best["label"] == "good"
    good_beats_impossible = mean_good > mean_impossible
    degeneracy_fixed = bool(easiest_is_not_best and best_is_good and good_beats_impossible)

    print()
    print(f"    mean reward  good={mean_good:.4f}  impossible={mean_impossible:.4f}")
    print(f"    highest-reward arena: d={best['difficulty']} ({best['label']}), "
          f"reward={best['reward']:.4f}")
    print(f"    easiest arena:        d={easiest['difficulty']} ({easiest['label']}), "
          f"reward={easiest['reward']:.4f}")
    print(f"    highest-reward arena is a GOOD (learnable-band) arena: {best_is_good}")
    print(f"    GOOD beats IMPOSSIBLE: {good_beats_impossible}")
    print(f"    easiest arena is NOT the best (no 'easier = better' degeneracy): {easiest_is_not_best}")
    print()
    if degeneracy_fixed:
        print("    VERDICT: the FIXED held-out test removed the easy-arena degeneracy.")
        print("             The reward PEAKS on a genuinely learnable arena, is NOT maxed")
        print("             by the easiest arena, and collapses to ~0 on impossible arenas.")
    else:
        print("    VERDICT: degeneracy NOT clearly removed under this budget/seeds — inspect the table.")

    replays_written = [p for r in results for p in r["replays"]]
    summary = {
        "backend": args.backend,
        "seeds": list(args.seeds),
        "episodes": args.episodes,
        "eval_seeds": args.eval_seeds,
        "held_out_reference_difficulties": list(
            __import__("modal_player").HELD_OUT_REFERENCE_DIFFICULTIES
        ),
        "wall_clock_s": round(wall, 2),
        "candidates": results,
        "mean_reward_good": round(mean_good, 4),
        "mean_reward_impossible": round(mean_impossible, 4),
        "highest_reward_arena": {"difficulty": best["difficulty"], "label": best["label"],
                                 "reward": best["reward"]},
        "easiest_arena": {"difficulty": easiest["difficulty"], "label": easiest["label"],
                          "reward": easiest["reward"]},
        "best_is_good": best_is_good,
        "good_beats_impossible": good_beats_impossible,
        "easiest_is_not_best": easiest_is_not_best,
        "degeneracy_fixed": degeneracy_fixed,
        "replays_written": replays_written,
    }
    out = OUTPUT_DIR / "nested_reward_validation.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"    wrote {out}")
    if replays_written:
        for p in replays_written:
            print(f"    wrote replay {p}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser, *, default_seeds: tuple[int, ...] = DEFAULT_SEEDS) -> None:
    p.add_argument("--backend", choices=["modal", "local"], default="modal")
    p.add_argument("--seeds", type=int, nargs="+", default=list(default_seeds))
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    p.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=DEFAULT_EVAL_SEEDS)


# Validation defaults to MORE seeds than a single reward call: the held-out
# transfer of a SINGLE Player is bimodal (it either "gets" the turtle timing or
# does not), so a 3-seed mean is noisy enough to occasionally rank a borderline
# arena above a good one. Averaging 6 seeds stabilises the per-arena reward so the
# good > impossible ordering is robust run-to-run.
VALIDATE_SEEDS = (1, 2, 3, 4, 5, 6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The nested-RL Teacher reward (held-out transfer).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("reward", help="reward of a single arena")
    pr.add_argument("--difficulty", type=float, required=True)
    pr.add_argument("--map-size", dest="map_size", type=int, default=10)
    pr.add_argument("--capture-replay", action="store_true",
                    help="also capture+save a trained-Player replay")
    _add_common(pr)
    pr.set_defaults(func=run_single)

    pv = sub.add_parser("validate", help="run ~6 candidates; show reward-per-arena table")
    _add_common(pv, default_seeds=VALIDATE_SEEDS)
    pv.set_defaults(func=run_validate)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
