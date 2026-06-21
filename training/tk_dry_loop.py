"""training/tk_dry_loop.py — ONE complete dry round-trip on Target Knockback (game #2).

The Target-Knockback twin of ``training/dry_loop.py``. It proves the WIRED TK path
round-trips end to end on the REAL pieces, mirroring how Ring-Out's dry loop proves the
fighter path:

    a TK Teacher arena spec (a curriculum JSON the Teacher would emit)
        -> strict validate + clamp to TK_BOUNDS  (the TK HUD grading core)
        -> the TK adapter / crucible-player-tk worker trains N fresh PPO Players on it
        -> each Player's held-out improvement is measured (output/nested_reward_tk.py)
        -> reward = mean held-out improvement across seeds (the nested TK reward)
        -> that SAME reward is recorded through a REAL TK HUD task/trace
           (LocalRuntime + design_tk_curriculum, CRUCIBLE_HUD_SCORER=nested)
        -> and returned through the Eval Protocol /init contract (execute_rollout)
        -> ASSERT Modal == HUD == Eval-Protocol (tolerance 1e-6); fail loudly
        -> capture ONE trained-Player TK replay, confirm it is viewer-discoverable.

CRITICAL ISOLATION: this driver NEVER touches the live deployed fighter EP bridge
(``crucible-ep-bridge-web.modal.run``). The Eval-Protocol leg runs ``execute_rollout``
IN-PROCESS with a fake completion client (the answer is the supplied TK arena JSON), so
no live bridge, no live Fireworks tracing, nothing deployed is mutated. The PPO reward
fans out on the ISOLATED ``crucible-player-tk`` worker (``--backend modal``) or runs
fully local (``--backend local``, no Modal credits) — the fighter's ``crucible-player``
is never called.

Run (from repo root so .env is picked up):

    # local backend — no Modal credits, real PPO on CPU (slow but self-contained):
    .venv/bin/python -m training.tk_dry_loop --backend local --seeds 1 --episodes 400

    # modal backend — fan out on the isolated crucible-player-tk worker:
    .venv/bin/python -m training.tk_dry_loop --backend modal --seeds 1 2 3

STOP CONDITIONS (the runner reports which fired and exits non-zero):
  * any PPO job failed (non-TK result)    -> exit 13
  * reward mismatch Modal/HUD/EP          -> exit 14
  * missing HUD trace                     -> exit 15
  * total dry run exceeded the timeout    -> exit 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except Exception:  # pragma: no cover
    pass

# .env ships EMPTY MODAL_TOKEN_ID/SECRET that break Modal env-var auth; drop them so the
# ~/.modal.toml njlee007 profile is used (same fix as the fighter dry loop).
for _k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    if os.environ.get(_k, "").strip() == "":
        os.environ.pop(_k, None)

QWEN_BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

OVERALL_TIMEOUT_S = 30 * 60


def _stop(code: int, reason: str) -> int:
    print("\n" + "=" * 78)
    print(f"STOP CONDITION FIRED (exit {code}): {reason}")
    print("=" * 78)
    return code


# A realistic TK Teacher answer in the validated learnable band (d=0.55). This stands in
# for the Qwen completion so the dry loop exercises the FULL wired path deterministically
# without burning a dedicated Qwen deploy (the fighter dry loop calls live Qwen; the TK
# round-trip's job is to prove the WIRING, so a fixed in-band arena is the right probe).
DEFAULT_TK_ANSWER = {
    "difficulty": 0.55,
    "platform_width": 12.0,
    "gravity": 0.6,
    "knockback": 2.5,
    "spawn_gap": 4.0,
    "zone_half": 1.6,
    "zone_center_frac": 0.5,
}


# ---------------------------------------------------------------------------
# HUD leg (load-bearing) — record the nested TK reward through a real trace
# ---------------------------------------------------------------------------


async def _run_hud(clamped: dict, bounds) -> tuple[float, str, str]:
    import mcp.types as mcp_types
    from hud.agents.base import Agent
    from hud.eval import LocalRuntime
    from hud.types import Step

    from training.tk_hud_teacher_env import design_tk_curriculum

    answer = json.dumps(clamped, sort_keys=True)

    class FixedCurriculumAgent(Agent):
        async def __call__(self, run) -> None:  # noqa: ANN001
            _ = run.prompt_text
            run.record(
                Step(
                    source="agent",
                    messages=[
                        mcp_types.PromptMessage(
                            role="assistant",
                            content=mcp_types.TextContent(type="text", text=answer),
                        )
                    ],
                )
            )
            run.trace.content = answer

    env_source = os.path.join(str(_REPO_ROOT), "training", "tk_hud_teacher_env.py")
    runtime = LocalRuntime(env_source, env="crucible-teacher-tk")
    bounds_list = {k: [lo, hi] for k, (lo, hi) in bounds.items()}
    task = design_tk_curriculum("Target Knockback; pick a learnable TK curriculum", bounds_list)
    job = await task.run(FixedCurriculumAgent(), runtime=runtime)
    run = job.runs[0]
    return float(run.reward), str(run.trace_id) if run.trace_id else "", str(run.trace.status)


# ---------------------------------------------------------------------------
# Eval Protocol leg — return the same reward (IN-PROCESS, never the live bridge)
# ---------------------------------------------------------------------------


async def _run_eval_protocol(clamped: dict, bounds, *, rollout_id: str) -> float:
    from types import SimpleNamespace

    from eval_protocol import InitRequest

    from training import tk_dry_loop_sidecar as side
    from training.ep_remote_server import execute_rollout

    answer = json.dumps(clamped, sort_keys=True)

    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    req = InitRequest.model_validate(
        {
            "completion_params": {"model": QWEN_BASE_MODEL, "temperature": 0.0},
            "messages": [{"role": "user", "content": "Design a TK curriculum"}],
            "model_base_url": FIREWORKS_BASE_URL,
            "api_key": os.getenv("FIREWORKS_API_KEY") or "k",
            "metadata": {
                "invocation_id": "tk-dryrun-inv",
                "experiment_id": "tk-dryrun-exp",
                "rollout_id": rollout_id,
                "run_id": "tk-dryrun-run",
                "row_id": "tk-dryrun-row",
            },
        }
    )
    result = await execute_rollout(
        req,
        bounds=bounds,
        scorer=side.served_tk_nested_scorer,
        client_factory=lambda b, k: FakeClient(),
        reporter=lambda rid, extras: None,  # no live Fireworks tracing in the dry run
    )
    return float(result.reward)


# ---------------------------------------------------------------------------
# Replay capture -> viewer discovery
# ---------------------------------------------------------------------------


def _save_replay(sink: dict, rollout_id: str) -> tuple[str | None, bool]:
    from output.nested_reward_tk import persist_captured_replays

    replays_dir = _REPO_ROOT / "replays"
    written = persist_captured_replays(sink)
    if not written:
        return None, False
    src = Path(written[0])
    out = replays_dir / f"tk_dryrun_{rollout_id}.json"
    out.write_text(src.read_text())

    names = sorted(p.name for p in replays_dir.glob("*.json") if p.name != "manifest.json")
    (replays_dir / "manifest.json").write_text(json.dumps(names, indent=2))
    manifest = json.loads((replays_dir / "manifest.json").read_text())
    return str(out), out.name in manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from training import tk_dry_loop_sidecar as side
    from training.hud_teacher_env import score_teacher_answer
    from training.tk_hud_teacher_env import TK_BOUNDS, tk_gap_scorer

    parser = argparse.ArgumentParser(description="TK dry round-trip (Teacher->TK->HUD->EP).")
    parser.add_argument("--backend", choices=["modal", "local"], default="modal")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=50)
    parser.add_argument("--answer", type=str, default=None,
                        help="JSON TK arena spec to grade (default: in-band d=0.55).")
    args = parser.parse_args(argv)

    bounds = TK_BOUNDS
    t_start = time.time()
    rollout_id = f"tk-dryrun-{int(t_start)}"

    raw_answer = args.answer or json.dumps(DEFAULT_TK_ANSWER, sort_keys=True)

    print("=" * 78)
    print("CRUCIBLE — DRY ROUND-TRIP (TARGET KNOCKBACK, game #2)")
    print("=" * 78)
    print(f"  rollout id     : {rollout_id}")
    print(f"  TK schema      : {list(bounds)}")
    print(f"  worker         : crucible-player-tk / train_tk_player  (ISOLATED)")
    print(f"  backend        : {args.backend}  seeds={args.seeds} episodes={args.episodes}")
    print(f"  reward         : nested PPO held-out transfer (TK)")
    print("=" * 78)

    # --- 1. Validate/clamp the Teacher arena spec ---------------------------
    print("\n[1/6] Validating + clamping the TK Teacher arena spec ...")
    # Use the SHARED grading core just to parse+clamp (gap scorer is cheap; the value
    # is discarded — we only want the clamped params here).
    _, clamped = score_teacher_answer(raw_answer, bounds=bounds, scorer=tk_gap_scorer)
    print(f"    raw answer      : {raw_answer}")
    print(f"    clamped params  : {json.dumps(clamped, sort_keys=True)}")

    # --- 2. Sidecar config + reset ------------------------------------------
    side.reset()
    side.write_config(
        backend=args.backend, seeds=list(args.seeds),
        episodes=args.episodes, eval_seeds=args.eval_seeds,
    )

    # --- 3. Nested TK reward (ONE computation) ------------------------------
    print(f"\n[2/6] Training {len(args.seeds)} fresh PPO Players ({args.backend}) on crucible-player-tk ...")
    from output.nested_reward_tk import tk_teacher_reward

    sink: dict = {}
    modal_t0 = time.time()
    try:
        modal_reward = tk_teacher_reward(
            clamped,
            backend=args.backend,
            seeds=tuple(args.seeds),
            episodes=args.episodes,
            eval_seeds=args.eval_seeds,
            capture_replay=True,
            curriculum_id="tk-dry-loop",
            _detail_sink=sink,
        )
    except AssertionError as exc:
        return _stop(13, f"a PPO job failed (non-TK result): {exc}")
    except Exception as exc:
        return _stop(13, f"TK reward computation failed: {type(exc).__name__}: {exc}")
    modal_s = time.time() - modal_t0

    improvements = sink["per_seed_improvement"]
    mean_imp = statistics.mean(improvements) if improvements else 0.0
    seed_std = statistics.pstdev(improvements) if len(improvements) > 1 else 0.0
    print(f"    per-seed improvement : {improvements}")
    print(f"    mean improvement     : {mean_imp:+.4f}   seed std = {seed_std:.4f}")
    print(f"    held-out before={sink['mean_before']:.3f} after={sink['mean_after']:.3f}")
    print(f"    NESTED TK reward (clamped [0,1]) = {modal_reward:.6f}")

    # Publish so HUD + EP read the EXACT same number (one computation).
    side.publish(clamped, modal_reward)

    # --- 4. HUD (load-bearing) ----------------------------------------------
    print("\n[3/6] Recording the nested TK reward through a REAL HUD task/trace ...")
    os.environ["CRUCIBLE_HUD_SCORER"] = "nested"
    os.environ.setdefault("CRUCIBLE_TK_SIDECAR_DIR", str(side.SIDECAR_DIR))
    try:
        hud_reward, trace_id, hud_status = asyncio.run(_run_hud(clamped, bounds))
    except Exception as exc:
        return _stop(15, f"HUD task failed: {type(exc).__name__}: {exc}")
    if not trace_id:
        return _stop(15, "HUD produced no trace id")
    print(f"    HUD-recorded reward = {hud_reward:.6f}")
    print(f"    HUD trace id        = {trace_id}")
    print(f"    HUD trace status    = {hud_status}")

    # --- 5. Eval Protocol (in-process; NEVER the live bridge) ----------------
    print("\n[4/6] Returning the same reward through the Eval Protocol /init contract ...")
    try:
        ep_reward = asyncio.run(_run_eval_protocol(clamped, bounds, rollout_id=rollout_id))
    except Exception as exc:
        return _stop(14, f"Eval Protocol rollout failed: {type(exc).__name__}: {exc}")
    print(f"    Eval-Protocol reward = {ep_reward:.6f}")

    # --- 6. Assert + replay --------------------------------------------------
    print("\n[5/6] Asserting Modal == HUD == Eval-Protocol (tol 1e-6) ...")
    tol = 1e-6
    ok_hud = abs(modal_reward - hud_reward) < tol
    ok_ep = abs(modal_reward - ep_reward) < tol
    print(f"    Modal         = {modal_reward:.6f}")
    print(f"    HUD           = {hud_reward:.6f}   (== Modal: {ok_hud})")
    print(f"    EvalProtocol  = {ep_reward:.6f}   (== Modal: {ok_ep})")
    rewards_match = ok_hud and ok_ep

    print("\n[6/6] Capturing trained-Player TK replay and confirming viewer discovery ...")
    replay_path, discoverable = _save_replay(sink, rollout_id)
    if replay_path:
        print(f"    wrote {replay_path}")
        print(f"    viewer-discoverable (in replays/manifest.json): {discoverable}")
    else:
        print("    WARNING: no replay was captured by the TK worker.")

    total_s = time.time() - t_start

    print("\n" + "=" * 78)
    print("TK DRY-ROUND-TRIP SUMMARY")
    print("=" * 78)
    print(f"  rollout id        : {rollout_id}")
    print(f"  clamped params    : {json.dumps(clamped, sort_keys=True)}")
    print(f"  mean improvement  : {mean_imp:+.4f}  (seed std {seed_std:.4f})")
    print(f"  Modal reward      : {modal_reward:.6f}")
    print(f"  HUD reward        : {hud_reward:.6f}   trace={trace_id}")
    print(f"  EvalProtocol rwd  : {ep_reward:.6f}")
    print(f"  rewards match     : {rewards_match}  (Modal==HUD: {ok_hud}, Modal==EP: {ok_ep})")
    print(f"  replay path       : {replay_path}  (viewer-discoverable: {discoverable})")
    print(f"  latency           : {total_s:.1f}s  (ppo {modal_s:.1f}s)")
    print("=" * 78)

    if total_s > OVERALL_TIMEOUT_S:
        return _stop(16, f"total dry run took {total_s:.0f}s > {OVERALL_TIMEOUT_S}s timeout")
    if not rewards_match:
        return _stop(14, "reward mismatch Modal/HUD/Eval-Protocol — see the numbers above")

    print("\nRESULT: PASS — TK Teacher spec -> crucible-player-tk PPO -> HUD -> Eval-Protocol all agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
