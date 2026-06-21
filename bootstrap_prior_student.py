"""bootstrap_prior_student.py — forge the VERIFIED-ACTIVE frozen prior Student.

The focused inner-Student population (``modal_player._real_ppo_transfer_result`` with a
``student_cfg``) wants its 30% ``prior_student`` league member to be a real, ACTIVE
earlier Student — one that MOVES and actually knocks the opponent off — not a passive
camper. The very first focused Teacher has no predecessor, so we bootstrap one here:

  1. Train a fresh Student (the SAME PPO knobs the focused population uses: [128,128]
     MLP, ent_coef=0.03, decisive ring-out reward) against a 100%-aggressive league —
     the scripted_fighter that closes and punches. A Student that learns to beat the
     aggressor learns to MOVE and ring-out, which is exactly the self-play sparring
     partner we want.
  2. VERIFY it is ACTIVE with ``play_match_trace`` over fresh seeds vs the aggressor:
       * moves on > MIN_MOVE_FRAC of steps (default 15%), and
       * wins by REAL ring-out on > MIN_RINGOUT_FRAC of matches (default 30%).
     If it fails either gate it is NOT frozen (a passive camper would teach the focused
     Student nothing) and the script exits non-zero.
  3. Freeze the verified Student to a ``PolicyArtifact`` JSON the worker loads via
     ``student_cfg["prior_student_path"]``.

If verification fails (or you want the documented fallback), the focused population still
works WITHOUT a prior_student: ``resolve_league`` drops the 30% member and renormalises to
100% aggressive — never a silent turtle/random substitution. So this script is a
best-effort UPGRADE from 100%-aggressive to a 70/30 focused league, never a hard gate.

Run from repo root (.venv has sb3 + gymnasium + torch):

    .venv/bin/python bootstrap_prior_student.py \
        --out replays/prior_student.json --episodes 2000 --verify-matches 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# The same focused-population knobs the inner Student trains under.
PRIOR_STUDENT_ARCH = [128, 128]
PRIOR_STUDENT_ENT_COEF = 0.03
# Aggressive-only league: the documented base the prior Student spars against.
AGGRESSIVE_ONLY_LEAGUE = [{"id": "aggressive", "weight": 1.0}]

# ACTIVITY GATES — a frozen prior Student must clear BOTH to be worth using.
MIN_MOVE_FRAC = 0.15      # moves (LEFT/RIGHT/JUMP) on > 15% of its steps
MIN_RINGOUT_FRAC = 0.30   # wins by REAL ring-out on > 30% of verify matches


def _train_prior_student(*, difficulty: float, episodes: int, eval_seeds: int, seed: int):
    """Train one Student vs 100%-aggressive with the focused-population knobs.

    Returns the trained SB3 model + the runnable obs->action policy. The arena
    carries ``decisive_timeout=True`` so the decisive reward's +0.1 tiebreak tier is
    live (mirrors the inner-Student arena).
    """
    import dataclasses

    from contracts import PlayerConfig, TrainingBudget
    from games.fighter import FighterArena
    from games.opponents_league import resolve_league
    from harness.fighter_adapter import FighterGameAdapter
    from harness.ppo_trainer import PPOPlayerTrainer

    arena = dataclasses.replace(FighterArena(difficulty=difficulty), decisive_timeout=True)
    league = resolve_league(AGGRESSIVE_ONLY_LEAGUE, has_prior_student=False)

    trainer = PPOPlayerTrainer(
        eval_seeds=eval_seeds,
        ent_coef=PRIOR_STUDENT_ENT_COEF,
        opponent_league=league,
        prior_student=None,           # the bootstrap Student has no predecessor.
        decisive_reward=True,
        net_arch=PRIOR_STUDENT_ARCH,
    )
    adapter = FighterGameAdapter(eval_seeds=eval_seeds)
    train_arenas = adapter.arenas_from_configs([arena], curriculum_id="bootstrap-prior")
    config = PlayerConfig(
        architecture="mlp",
        num_seeds=1,
        budget=TrainingBudget(episodes=episodes),
        seed=int(seed),
        modal_parallel=False,
    )
    job = trainer.submit(config, train_arenas)
    job.result()
    return job.model, job.policy, arena


def _verify_active(policy, arena, *, n_matches: int, seed0: int) -> dict:
    """Fight the candidate vs the aggressor over fresh seeds; measure activity.

    Uses ``play_match_trace`` (read-only behaviour probe) so the numbers are the
    real movement fraction and real-ring-out rate — the two gates that separate an
    ACTIVE sparring partner from a passive camper.
    """
    from games.fighter import play_match_trace, scripted_fighter

    aggressor = scripted_fighter(arena, ego=1)
    total_steps = move_steps = 0
    ringout_wins = timeout_wins = losses = draws = 0
    for i in range(n_matches):
        tr = play_match_trace(arena, policy, aggressor, seed=seed0 + i, ego=0)
        total_steps += tr["steps"]
        move_steps += tr["move_steps"]
        w, by = tr["winner"], tr["ended_by"]
        if w is None:
            draws += 1
        elif w == 0 and by == "ringout":
            ringout_wins += 1
        elif w == 0:
            timeout_wins += 1
        else:
            losses += 1
    move_frac = move_steps / total_steps if total_steps else 0.0
    ringout_frac = ringout_wins / n_matches if n_matches else 0.0
    return {
        "n_matches": n_matches,
        "move_frac": round(move_frac, 4),
        "ringout_win_frac": round(ringout_frac, 4),
        "ringout_wins": ringout_wins,
        "timeout_wins": timeout_wins,
        "losses": losses,
        "draws": draws,
        "moves_enough": move_frac > MIN_MOVE_FRAC,
        "ringouts_enough": ringout_frac > MIN_RINGOUT_FRAC,
    }


def bootstrap(args) -> dict:
    t0 = time.time()
    from output.stage6.policy import serialize_policy

    print(f"[bootstrap] training prior Student vs 100% aggressive "
          f"(arch={PRIOR_STUDENT_ARCH}, ent_coef={PRIOR_STUDENT_ENT_COEF}, "
          f"decisive_reward=True, episodes={args.episodes}) ...")
    model, policy, arena = _train_prior_student(
        difficulty=args.difficulty, episodes=args.episodes,
        eval_seeds=args.eval_seeds, seed=args.seed,
    )
    print(f"[bootstrap] verifying activity over {args.verify_matches} matches ...")
    audit = _verify_active(policy, arena, n_matches=args.verify_matches, seed0=10_000)
    active = audit["moves_enough"] and audit["ringouts_enough"]

    print(f"[bootstrap]   move_frac      = {audit['move_frac']:.3f} "
          f"(> {MIN_MOVE_FRAC} required: {audit['moves_enough']})")
    print(f"[bootstrap]   ringout_win_frac = {audit['ringout_win_frac']:.3f} "
          f"(> {MIN_RINGOUT_FRAC} required: {audit['ringouts_enough']})")
    print(f"[bootstrap]   wins(ringout/timeout)={audit['ringout_wins']}/{audit['timeout_wins']} "
          f"losses={audit['losses']} draws={audit['draws']}")

    out = {"verified_active": active, "activity_audit": audit,
           "wall_s": round(time.time() - t0, 1)}
    if not active:
        print("[bootstrap] VERDICT: candidate is NOT verified-active — NOT freezing it.")
        print("[bootstrap] FALLBACK: focused league runs at 100% aggressive "
              "(resolve_league drops the 30% prior_student member). No artifact written.")
        out["artifact_path"] = None
        return out

    artifact = serialize_policy(
        model, teacher="bootstrap", curriculum_id="bootstrap-prior",
        seed=int(args.seed), game="fighter", obs_dim=11,
        net_arch=PRIOR_STUDENT_ARCH,
        extra={"net_arch": PRIOR_STUDENT_ARCH, "role": "prior_student",
               "bootstrap": True, "activity_audit": audit},
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact.as_dict(), indent=2))
    print(f"[bootstrap] VERDICT: VERIFIED-ACTIVE. Froze prior Student -> {out_path}")
    print(f"[bootstrap] pass this path as student_cfg['prior_student_path'] for a 70/30 league.")
    out["artifact_path"] = str(out_path)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bootstrap a verified-active frozen prior Student.")
    p.add_argument("--out", default="replays/prior_student.json",
                   help="where to write the frozen PolicyArtifact JSON")
    p.add_argument("--difficulty", type=float, default=0.6,
                   help="training-arena difficulty (the aggressor strength dial)")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=25)
    p.add_argument("--verify-matches", dest="verify_matches", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    result = bootstrap(args)
    print(json.dumps({k: v for k, v in result.items() if k != "activity_audit"}, indent=2))
    return 0 if result.get("verified_active") else 1


if __name__ == "__main__":
    raise SystemExit(main())
