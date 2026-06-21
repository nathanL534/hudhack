"""output/eval_base_vs_trained.py — Stage 6: the "does trained beat base" decider (CLI).

THE FINAL COMPARISON. The PRIMARY (headline) metric is a DIRECT Student-vs-Student
head-to-head: train a fresh PPO Student on the BASE Teacher's curricula and another
on the TRAINED Teacher's, freeze both, and fight them on UNSEEN arenas (side-swapped,
many fresh seeds). The trained Teacher wins only if ITS Student beats the base
Student with a curriculum-level 95% CI lower bound above zero AND every anti-
circularity check passes. The legacy fixed-bot before->after held-out-transfer
number is now the SECONDARY diagnostic (evidence only; never sets the headline).

This is a THIN CLI over the reusable ``output/stage6`` architecture:

  * handles  — base/trained are HANDLES resolving to a Fireworks id, a LOCAL LoRA
               adapter path, or a ``modal:`` generation handle.
  * games    — a registry/router (fighter / target_knockback / koth). Ring-Out Duel
               (fighter) is the headline Teacher game; KOTH is the cross-game test
               (FRESH KOTH Students from KOTH-mapped curricula); TK reports "not
               installed" here.
  * config   — ONE fairness invariant builds EVERY payload for BOTH Students.
  * the headline verdict is "trained-Student beats base-Student head-to-head, CI
               lower bound > 0, anti-circularity passes" — NOT "RFT completed".

Commands
--------
    # fastest unit-level smoke: the test suite (NO Modal, NO PPO):
    .venv/bin/python -m pytest test_stage6.py test_stage6_primary.py -q

    # base-vs-base NULL test (MUST give ~zero advantage + no side bias; local):
    .venv/bin/python output/eval_base_vs_trained.py --null-test --backend local \
        --base offline --no-secondary

    # cheapest head-to-head smoke (real PPO, 2 replicates, tiny budget):
    .venv/bin/python output/eval_base_vs_trained.py --h2h-smoke --backend local \
        --base offline --trained offline --no-secondary

    # tiny REAL base-vs-trained head-to-head on Modal (deployed crucible-player):
    .venv/bin/python output/eval_base_vs_trained.py --h2h-smoke --backend modal \
        --base modal:base --trained modal:update2

    # the FULL Gate-4 decider (primary head-to-head + secondary + KOTH cross-game):
    .venv/bin/python output/eval_base_vs_trained.py --backend modal \
        --base modal:base --trained modal:update2 \
        --replicates 6 --curriculum-arenas 4 --match-seeds 16 \
        --h2h-grid full --koth-cross-game

    # report-only: regenerate charts + Markdown + dashboard from a saved JSON:
    .venv/bin/python output/eval_base_vs_trained.py --report-only \
        --result output/eval_base_vs_trained.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except Exception:  # pragma: no cover
    pass

# .env ships EMPTY MODAL_TOKEN_ID/SECRET placeholders that break Modal env-var
# auth; drop the blanks so the active ~/.modal.toml profile is used.
for _k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    if os.environ.get(_k, "").strip() == "":
        os.environ.pop(_k, None)

from output.stage6 import report as report_mod  # noqa: E402
from output.stage6.config import (  # noqa: E402
    EvalConfig,
    head_to_head_smoke_config,
    smoke_config,
)
from output.stage6.evaluator import (  # noqa: E402
    EvalRequest,
    OUTPUT_DIR,
    run_eval,
    run_full_eval,
    write_result,
)
from output.stage6.games import all_games, get_game  # noqa: E402

# The default BASE handle is the MODAL base-no-adapter Teacher: Fireworks gives a 404
# for serverless qwen3-4b inference on this account and the host venv has no
# transformers/peft, so ``modal:base`` is the only base Teacher that actually
# generates. (Pass ``--base accounts/fireworks/models/qwen3-4b`` to force Fireworks.)
DEFAULT_BASE = "modal:base"
RESULT_PATH = OUTPUT_DIR / "eval_base_vs_trained.json"
REPORT_DIR = OUTPUT_DIR / "stage6_report"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_config(args, *, smoke: bool) -> EvalConfig:
    if args.h2h_smoke:
        base = head_to_head_smoke_config()
    elif smoke:
        base = smoke_config()
    else:
        base = EvalConfig()
    return EvalConfig(
        temperature=args.temperature,
        ppo_episodes=args.episodes if args.episodes is not None else base.ppo_episodes,
        eval_seeds=args.eval_seeds if args.eval_seeds is not None else base.eval_seeds,
        architecture=base.architecture,
        arenas_per_model=args.arenas if args.arenas is not None else base.arenas_per_model,
        player_seeds=tuple(args.seeds) if args.seeds is not None else base.player_seeds,
        held_out_grid=args.grid if args.grid is not None else base.held_out_grid,
        n_replicates=args.replicates if args.replicates is not None else base.n_replicates,
        curriculum_arenas=args.curriculum_arenas if args.curriculum_arenas is not None else base.curriculum_arenas,
        match_seeds_per_arena=args.match_seeds if args.match_seeds is not None else base.match_seeds_per_arena,
        head_to_head_grid=args.h2h_grid if args.h2h_grid is not None else base.head_to_head_grid,
        opponent_league_enabled=bool(getattr(args, "opponent_league", False)) or base.opponent_league_enabled,
        net_arch=tuple(args.net_arch) if getattr(args, "net_arch", None) else base.net_arch,
    )


def _validate_cli(args) -> str | None:
    """Return an error string if the CLI args are invalid, else None."""
    if args.report_only:
        if not args.result:
            return "--report-only requires --result <path>"
        return None
    if args.list_games:
        return None
    base_vs_base = args.smoke or args.h2h_smoke or args.null_test
    if not base_vs_base and not args.trained:
        return ("--trained <handle> is required for the real decider "
                "(or use --smoke / --h2h-smoke / --null-test for a base-vs-base run)")
    if args.arenas is not None and args.arenas < 1:
        return "--arenas must be >= 1"
    if args.replicates is not None and args.replicates < 1:
        return "--replicates must be >= 1"
    if args.seeds is not None and len(args.seeds) < 1:
        return "--seeds needs at least one seed"
    if args.grid is not None and args.grid not in ("full", "diagonal"):
        return "--grid must be 'full' or 'diagonal'"
    try:
        game = get_game(args.game)
    except KeyError as exc:
        return str(exc)
    if not args.report_only and not game.installed:
        return (f"game {args.game!r} is not installed: {game.not_installed_reason}")
    if not args.report_only and game.role != "teacher":
        return (f"game {args.game!r} has role {game.role!r}; only role='teacher' games "
                f"can train a Teacher (KOTH is a held-out probe, not a Teacher game)")
    return None


def _cmd_list_games() -> int:
    print("Stage-6 game registry:")
    for name, e in all_games().items():
        status = "installed" if e.installed else f"NOT INSTALLED ({e.not_installed_reason})"
        print(f"  {name:<18} role={e.role:<8} {status}")
    return 0


def _cmd_report_only(args) -> int:
    result_path = Path(args.result)
    if not result_path.exists():
        print(f"ERROR: result file not found: {result_path}", file=sys.stderr)
        return 2
    result = json.loads(result_path.read_text())
    out_dir = Path(args.report_dir) if args.report_dir else REPORT_DIR
    artifacts = report_mod.generate_report(result, out_dir)
    print(f"Report regenerated from {result_path} -> {out_dir}")
    print(f"  markdown: {artifacts['report_md']}")
    for k, p in artifacts["charts"].items():
        print(f"  chart {k}: {p}")
    if artifacts["replay_manifest"]:
        print(f"  replay manifest: {artifacts['replay_manifest']}")
    if artifacts.get("dashboard"):
        print(f"  dashboard: {artifacts['dashboard']}")
    if not artifacts["charts_available"]:
        print("  (matplotlib unavailable — charts skipped, Markdown only)")
    return 0


def _cmd_run(args) -> int:
    smoke = bool(args.smoke or args.h2h_smoke or args.null_test)
    cfg = _build_config(args, smoke=smoke)

    base_handle = args.base or DEFAULT_BASE
    if args.null_test:
        # base-vs-base NULL test: SAME teacher both sides; must produce ~zero
        # advantage. If it systematically favors one side, the evaluator is broken.
        trained_handle = base_handle
    elif smoke and not args.trained:
        trained_handle = base_handle   # base vs base concurrency smoke
    else:
        trained_handle = args.trained

    # The KOTH cross-game is now run INSIDE ``run_full_eval`` so its trained-vs-base
    # delta feeds the train-game-overfit anti-gaming gate (H3) and gates the verdict.
    # ``--no-koth-cross-game`` opts out (cheap train-game-only run); the legacy
    # ``--koth-cross-game`` flag is a no-op kept for backward compatibility.
    req = EvalRequest(
        base_handle=base_handle,
        trained_handle=trained_handle,
        game=args.game,
        backend=args.backend,
        config=cfg,
        is_smoke=smoke,
        enable_hud_traces=bool(args.hud_traces),
        base_model=args.base_model,
        run_secondary=not args.no_secondary,
        run_cross_game=not args.no_koth_cross_game,
        verbose=True,
    )
    summary = run_full_eval(req)

    out = Path(args.out) if args.out else RESULT_PATH
    write_result(summary, out)
    print(f"\n    wrote {out}")

    if not args.no_report:
        out_dir = Path(args.report_dir) if args.report_dir else REPORT_DIR
        artifacts = report_mod.generate_report(summary, out_dir)
        print(f"    report -> {artifacts['report_md']}")
        if artifacts.get("dashboard"):
            print(f"    dashboard -> {artifacts['dashboard']}")

    # Exit non-zero if the real decider FAILS (so CI / scripts can gate on it).
    # The NULL test is special: it PASSES when advantage is ~zero (no win expected).
    if args.null_test:
        adv = summary["primary"]["mean_paired_advantage"]
        p0 = summary["primary"]["trained_win_rate_as_p0"]
        p1 = summary["primary"]["trained_win_rate_as_p1"]
        side_gap = abs(p0 - p1)
        ok = abs(adv) < 0.10 and side_gap < 0.25
        print(f"\n    NULL TEST: |advantage|={abs(adv):.4f} (<0.10?), side_gap={side_gap:.4f} (<0.25?) "
              f"-> {'PASS (evaluator unbiased)' if ok else 'FAIL (evaluator favors a side — FIX before real use)'}")
        return 0 if ok else 1
    if not smoke and not summary["overall_pass"]:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 6: base vs trained Teacher decider.")
    p.add_argument("--game", default="fighter", help="game name (default: fighter / Ring-Out Duel)")
    p.add_argument("--backend", choices=["modal", "local"], default="modal")
    p.add_argument("--base", default=None,
                   help="BASE Teacher handle (default modal:base = Qwen3-4B base on Modal; "
                        "also accepts a Fireworks id/deployment OR local:adapter path)")
    p.add_argument("--trained", default=None,
                   help="TRAINED Teacher handle (modal:<adapter-tag> e.g. modal:update2, "
                        "OR a Fireworks deployment id OR a local:adapter path)")
    p.add_argument("--base-model", dest="base_model", default=None,
                   help="HF base-model id the modal:/local: backends load and the trained "
                        "LoRA adapter attaches to (default Qwen/Qwen3-4B via the backend; "
                        "ignored by Fireworks handles, which keep their own model id)")
    p.add_argument("--arenas", type=int, default=None, help="arenas generated per model")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="fresh PPO Player seeds")
    p.add_argument("--episodes", type=int, default=None, help="PPO episodes per Player")
    p.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=None)
    p.add_argument("--grid", choices=["full", "diagonal"], default=None,
                   help="secondary held-out population grid (default: full)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--net-arch", dest="net_arch", type=int, nargs="+", default=None,
                   help="Student MLP hidden sizes, e.g. --net-arch 256 256 (default 64 64). "
                        "Bigger nets let Students actually learn decisive play. Applied "
                        "identically to base and trained.")
    # -- PRIMARY (Student-vs-Student head-to-head) knobs --
    p.add_argument("--replicates", type=int, default=None,
                   help="independent curriculum replicates (CI is over these)")
    p.add_argument("--curriculum-arenas", dest="curriculum_arenas", type=int, default=None,
                   help="arenas each Teacher generates per replicate (Student trains on the set)")
    p.add_argument("--match-seeds", dest="match_seeds", type=int, default=None,
                   help="match seeds per held-out arena per side (paired across the swap)")
    p.add_argument("--h2h-grid", dest="h2h_grid", choices=["full", "diagonal"], default=None,
                   help="primary head-to-head held-out grid (default: full)")
    p.add_argument("--opponent-league", dest="opponent_league", action="store_true",
                   help="enable the Stage-6 Student opponent league (aggressive/turtle/"
                        "random opponents) so Students fight decisively -> fewer draws. "
                        "Applied IDENTICALLY to base and trained Students (fairness).")
    p.add_argument("--no-secondary", dest="no_secondary", action="store_true",
                   help="skip the secondary fixed-bot transfer diagnostic (primary only)")
    p.add_argument("--koth-cross-game", dest="koth_cross_game", action="store_true",
                   help="(no-op; the KOTH cross-game now runs by default inside "
                        "run_full_eval and feeds the train-game-overfit gate)")
    p.add_argument("--no-koth-cross-game", dest="no_koth_cross_game", action="store_true",
                   help="skip the KOTH cross-game (cheap train-game-only run; the "
                        "train-game-overfit anti-gaming gate then SKIPs, non-gating)")
    p.add_argument("--null-test", dest="null_test", action="store_true",
                   help="base-vs-base NULL test: SAME teacher both sides; must give "
                        "~zero advantage and no side bias (else the evaluator is broken)")
    p.add_argument("--h2h-smoke", dest="h2h_smoke", action="store_true",
                   help="cheapest head-to-head smoke config (2 replicates, tiny PPO)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny concurrency smoke: base vs base, small matrix")
    p.add_argument("--hud-traces", dest="hud_traces", action="store_true",
                   help="capture HUD traces of generated curricula (best-effort; needs HUD)")
    p.add_argument("--out", default=None, help="result JSON path")
    p.add_argument("--report-dir", dest="report_dir", default=None, help="report output dir")
    p.add_argument("--no-report", dest="no_report", action="store_true",
                   help="skip chart/markdown generation after the run")
    p.add_argument("--report-only", dest="report_only", action="store_true",
                   help="regenerate report from --result WITHOUT rerunning PPO")
    p.add_argument("--result", default=None, help="saved result JSON (for --report-only)")
    p.add_argument("--list-games", dest="list_games", action="store_true",
                   help="list the game registry and exit")
    args = p.parse_args(argv)

    err = _validate_cli(args)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    if args.list_games:
        return _cmd_list_games()
    if args.report_only:
        return _cmd_report_only(args)
    return _cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
