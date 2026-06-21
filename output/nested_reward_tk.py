"""output/nested_reward_tk.py — THE nested-RL Teacher reward for Target Knockback.

This is the Target-Knockback twin of ``output/nested_reward.py``. It assembles the
SAME nested-RL loop the fighter uses, but on the ISOLATED ``crucible-player-tk``
Modal worker (``modal_player_tk.train_tk_player``):

    Teacher arena -> train N fresh PPO Players on it (crucible-player-tk, parallel)
    -> measure each Player's held-out before/after win-rate
    -> reward = avg held-out improvement across seeds (clamped to [0, 1]).

WHY A SEPARATE MODULE (not a flag on nested_reward.teacher_reward):
  * The fighter worker returns ``{"status": "ppo_transfer", "held_out_improvement"}``;
    the TK worker (modal_player_tk) returns ``{"status": "ppo_tk", "improvement"}``.
    Reading the wrong key/status would silently mis-score. Keeping a dedicated TK
    reward makes the two reward paths COEXIST without touching each other — the
    explicit isolation requirement: a live Ring-Out RFT and a TK RFT can run at the
    same time and neither module imports the other's worker.
  * The fighter's broad held-out grid (output/broad_eval_set.py) is fighter-shaped.
    TK's held-out reference is the SAME arena family the worker already trains/scores
    against (the worker builds its own held-out arena from the payload), so TK does
    not need a separate broad-population file to be load-bearing — the worker's
    before/after IS the held-out transfer signal validated at scale
    (tk_modal_scale_validation_results.json: d=0.55 is the clean learnable band).

The reward THIS module returns is the SAME number the TK scale validation measured,
so the integrated round-trip's reward is directly comparable to the d=0.55 band.

Run from repo root (.venv has modal + sb3):

    # reward of ONE TK arena (3 Modal seeds on crucible-player-tk):
    .venv/bin/python output/nested_reward_tk.py reward --difficulty 0.55

    # local fallback (no Modal credits; SAME code path via local_tk_worker):
    .venv/bin/python output/nested_reward_tk.py reward --difficulty 0.55 --backend local
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Repo root on path (script lives in output/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPLAYS_DIR = Path(__file__).resolve().parents[1] / "replays"

# The ISOLATED Target-Knockback worker — NEVER the fighter's crucible-player.
MODAL_APP = "crucible-player-tk"
MODAL_FN = "train_tk_player"

# Reward defaults — match the scale-validation budget so the integrated reward is
# directly comparable to the d=0.55 learnable-band result.
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_PPO_EPISODES = 2000
DEFAULT_EVAL_SEEDS = 50


def tk_arena_payload(
    params: dict[str, float],
    *,
    episodes: int,
    eval_seeds: int,
    curriculum_id: str,
    capture_replay_id: str | None = None,
) -> dict:
    """Translate a validated TK Teacher param set into the worker payload.

    The TK worker (modal_player_tk._tk_arena_from_payload) reads ``difficulty`` plus
    the TK arena dials (``platform_width`` / ``zone_half`` / ``zone_center_frac``)
    directly. Geometry knobs the worker also accepts (``gravity`` / ``knockback`` /
    ``spawn_gap``) are passed verbatim too so the Teacher's emitted physics trains
    the Player exactly as emitted (the same contract the fighter's geometry_override
    uses). Missing knobs fall back to TargetKnockbackArena defaults inside the worker.
    """
    payload: dict = {
        "difficulty": float(params["difficulty"]),
        "ppo_episodes": int(episodes),
        "eval_seeds": int(eval_seeds),
        "curriculum_id": curriculum_id,
        "architecture": "mlp",
    }
    for k in ("platform_width", "gravity", "knockback", "spawn_gap", "zone_half", "zone_center_frac"):
        if k in params and params[k] is not None:
            payload[k] = float(params[k])
    if capture_replay_id:
        payload["capture_replay_id"] = capture_replay_id
    return payload


def _run_seeds(payloads: list[dict], seeds: list[int], *, backend: str) -> list[dict]:
    """Fan ``seeds`` out through the TK transfer worker (Modal-parallel or local).

    Modal: ONE ``fn.map`` over the per-seed payloads on ``crucible-player-tk`` (the
    isolated worker). Local: the SAME body in-process via ``local_tk_worker`` (no
    Modal credentials), sequentially (a ThreadPool would just contend on one CPU's
    torch).
    """
    if backend == "modal":
        import modal

        fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
        return list(fn.map(payloads, seeds))
    if backend == "local":
        from modal_player_tk import local_tk_worker

        return [local_tk_worker(p, s) for p, s in zip(payloads, seeds)]
    raise ValueError(f"unknown backend {backend!r} (use 'modal' or 'local')")


def tk_teacher_reward(
    params: dict[str, float],
    *,
    backend: str = "modal",
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    episodes: int = DEFAULT_PPO_EPISODES,
    eval_seeds: int = DEFAULT_EVAL_SEEDS,
    capture_replay: bool = False,
    curriculum_id: str = "tk-rft",
    _detail_sink: dict | None = None,
) -> float:
    """The nested-RL Target-Knockback Teacher reward in [0, 1] for a TK param set.

    Trains N fresh PPO Players (one per seed) on the TK arena via the isolated
    ``crucible-player-tk`` worker, then returns the AVERAGE held-out improvement
    across seeds, clamped to [0, 1]. Reads the TK worker's ``improvement`` field and
    asserts ``status == "ppo_tk"`` (the TK worker's schema), so it can NEVER pick up
    a fighter worker's row by mistake.
    """
    seed_list = list(seeds)
    replay_id = f"{curriculum_id}_a0" if capture_replay else None
    payloads = [
        tk_arena_payload(
            params,
            episodes=episodes,
            eval_seeds=eval_seeds,
            curriculum_id=curriculum_id,
            capture_replay_id=(replay_id if (replay_id and si == 0) else None),
        )
        for si in range(len(seed_list))
    ]
    rows = _run_seeds(payloads, seed_list, backend=backend)
    for r in rows:
        assert r.get("status") == "ppo_tk", f"non-TK result: {r!r}"

    improvements = [float(r["improvement"]) for r in rows]
    mean_improvement = sum(improvements) / len(improvements) if improvements else 0.0
    reward = max(0.0, min(1.0, mean_improvement))

    if _detail_sink is not None:
        _detail_sink["difficulty"] = float(params.get("difficulty", 0.0))
        _detail_sink["mean_improvement"] = round(mean_improvement, 4)
        _detail_sink["mean_before"] = round(sum(r["before_winrate"] for r in rows) / len(rows), 4)
        _detail_sink["mean_after"] = round(sum(r["after_winrate"] for r in rows) / len(rows), 4)
        _detail_sink["per_seed_improvement"] = [round(x, 4) for x in improvements]
        _detail_sink["reward"] = round(reward, 4)
        _detail_sink["rows"] = rows
    return float(reward)


def persist_captured_replays(sink: dict) -> list[str]:
    """Write any replay the TK worker captured to replays/<id>.json (driver-side)."""
    written: list[str] = []
    for row in sink.get("rows", []):
        cap = row.get("replay")
        if not cap:
            continue
        out = REPLAYS_DIR / f"{cap['replay_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cap["data"], indent=2))
        written.append(str(out))
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_single(args) -> dict:
    params = {
        "difficulty": args.difficulty,
        "platform_width": args.platform_width,
        "zone_half": args.zone_half,
    }
    sink: dict = {}
    t0 = time.time()
    reward = tk_teacher_reward(
        params,
        backend=args.backend,
        seeds=tuple(args.seeds),
        episodes=args.episodes,
        eval_seeds=args.eval_seeds,
        capture_replay=args.capture_replay,
        curriculum_id=f"tk-d{args.difficulty}",
        _detail_sink=sink,
    )
    wall = time.time() - t0
    replays = persist_captured_replays(sink) if args.capture_replay else []
    print(
        f"TK arena d={args.difficulty} pw={args.platform_width} zone_half={args.zone_half}: "
        f"TEACHER REWARD = {reward:.4f}  ({wall:.1f}s, {len(args.seeds)} seeds)"
    )
    print(
        f"  held-out before={sink['mean_before']:.3f} after={sink['mean_after']:.3f} "
        f"improvement={sink['mean_improvement']:+.3f}"
    )
    for p in replays:
        print(f"  wrote replay {p}")
    return {"reward": reward, "wall_s": round(wall, 2), "detail": sink, "replays": replays}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The nested-RL Target-Knockback Teacher reward (held-out transfer)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("reward", help="reward of a single TK arena")
    pr.add_argument("--difficulty", type=float, required=True)
    pr.add_argument("--platform-width", dest="platform_width", type=float, default=12.0)
    pr.add_argument("--zone-half", dest="zone_half", type=float, default=1.6)
    pr.add_argument("--capture-replay", action="store_true",
                    help="also capture+save a trained-Player TK replay")
    pr.add_argument("--backend", choices=["modal", "local"], default="modal")
    pr.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    pr.add_argument("--episodes", type=int, default=DEFAULT_PPO_EPISODES)
    pr.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=DEFAULT_EVAL_SEEDS)
    pr.set_defaults(func=run_single)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
