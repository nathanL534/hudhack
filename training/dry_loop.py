"""training/dry_loop.py — ONE complete pre-RFT dry loop on the FIGHTER game.

This is the dress rehearsal for Teacher RFT, end to end, on the REAL pieces:

    Qwen3-4B (dedicated Fireworks deploy)
        -> generates a bounded FIGHTER curriculum JSON (5 knobs)
        -> strict validate + clamp to FIGHTER_BOUNDS
        -> Modal trains N fresh PPO Players in parallel on that arena
        -> each Player's held-out TRANSFER improvement is measured on the BROAD,
           structurally-diverse held-out population (output/broad_eval_set.py)
        -> reward = mean held-out improvement across seeds (output/nested_reward.py)
        -> that SAME reward is recorded through a REAL HUD task/trace
           (LocalRuntime + design_curriculum, CRUCIBLE_HUD_SCORER=nested)
        -> and returned through the Eval Protocol /init contract (execute_rollout)
        -> ASSERT Modal == HUD == Eval-Protocol (tolerance 1e-6); fail loudly
        -> capture ONE replay from seed 1, confirm it is viewer-discoverable.

The reward is the VALIDATED nested PPO held-out improvement — NOT the placeholder
triangle scorer in modal_ep_bridge.py, and NOT the cheap strong-vs-weak gap proxy.
HUD stays load-bearing: the number HUD records IS this nested reward (the served
grader reads it through the dry-loop sidecar; see training/dry_loop_sidecar.py).

Run (from repo root so .env is picked up; Qwen deploy + Modal worker must be up)::

    .venv/bin/python -m training.dry_loop

STOP CONDITIONS (the runner reports which fired and exits non-zero):
  * Qwen never READY                      -> exit 10
  * invalid Qwen JSON after retries       -> exit 11
  * >1 parameter needed clamping          -> exit 12
  * any Modal job failed                  -> exit 13
  * reward mismatch Modal/HUD/EP          -> exit 14
  * missing HUD trace                     -> exit 15
  * total dry-run exceeded the timeout    -> exit 16
"""

from __future__ import annotations

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
except Exception:  # pragma: no cover - dotenv should be present in .venv
    pass

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

QWEN_DEPLOYMENT_ID = "qwen3-4b-dedicated"
QWEN_BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Nested-reward budget: 3 fresh PPO Players (seeds), a short-but-real PPO budget,
# enough held-out eval seeds for a stable win-rate, the BROAD (full) held-out grid.
SEEDS = (1, 2, 3)
PPO_EPISODES = 1000
EVAL_SEEDS = 50
HELD_OUT_GRID = "full"  # 5 physics variants x 3 difficulties = 15 held-out arenas
BACKEND = "modal"

CLAMP_BUDGET = 1            # STOP if MORE than this many keys need clamping.
OVERALL_TIMEOUT_S = 15 * 60  # STOP if the whole dry run exceeds this.

# Rough cost model (order-of-magnitude, for the report only).
QWEN_DOLLARS_PER_HR = 7.0           # dedicated qwen3-4b deployment (idle+serving)
MODAL_CPU_DOLLARS_PER_HR = 0.20     # ~one CPU container-hour, conservative


def _stop(code: int, reason: str) -> int:
    print("\n" + "=" * 78)
    print(f"STOP CONDITION FIRED (exit {code}): {reason}")
    print("=" * 78)
    return code


# ---------------------------------------------------------------------------
# 1. Qwen readiness
# ---------------------------------------------------------------------------


def _resolve_fireworks():
    from fireworks import Fireworks

    key = (os.getenv("FIREWORKS_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("FIREWORKS_API_KEY missing from environment/.env")
    client = Fireworks(api_key=key)
    page = client.accounts.list()
    items = getattr(page, "accounts", None) or list(page)
    if not items:
        raise RuntimeError("could not resolve a Fireworks account from the API key")
    acct = (getattr(items[0], "name", None) or str(items[0])).split("/")[-1]
    return Fireworks(api_key=key, account_id=acct), acct, key


def _wait_qwen_ready(fw, *, timeout_s: int = 60) -> bool:
    """Poll the dedicated deployment until READY (it is already provisioning)."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        state = None
        for d in fw.deployments.list():
            if (getattr(d, "name", "") or "").split("/")[-1] == QWEN_DEPLOYMENT_ID:
                state = getattr(d, "state", None)
                break
        if state != last:
            print(f"    qwen {QWEN_DEPLOYMENT_ID} state={state}")
            last = state
        if state == "READY":
            return True
        if state in ("FAILED", "DELETED", "DELETING"):
            return False
        time.sleep(20)
    return False


# ---------------------------------------------------------------------------
# 2-3. Qwen -> bounded fighter JSON -> validate/clamp
# ---------------------------------------------------------------------------


def _fighter_game(bounds: dict[str, tuple[float, float]]):
    from engine.games import Game

    defaults = {
        "difficulty": 0.5,
        "platform_width": 12.0,
        "gravity": 0.6,
        "knockback": 2.5,
        "spawn_gap": 4.0,
    }
    return Game(
        name="fighter",
        param_schema=bounds,
        to_gridworld_params=lambda p: dict(p),
        description="Platform-fighter arena (Crucible Teacher target).",
        defaults={k: defaults[k] for k in bounds},
    )


def _generate_and_validate(handle: str, api_key: str, bounds) -> tuple[dict, dict, str, int]:
    """Drive the production Teacher client at the deployed Qwen endpoint.

    Returns (raw_parsed_json, clamped_params, raw_completion_str, n_clamped).
    Raises on invalid JSON after retries (caught by the caller as a stop condition).
    """
    from training.fireworks_teacher import FireworksTeacher, _strict_validate_params
    from training.teacher import _clamp_to_schema

    game = _fighter_game(bounds)
    captured: dict = {}
    teacher = FireworksTeacher(model=handle, api_key=api_key, base_url=FIREWORKS_BASE_URL)
    real_transport = teacher._http_transport

    def _capturing(payload: dict) -> dict:
        captured.setdefault("request_payload", payload)
        resp = real_transport(payload)
        try:
            captured["raw_completion"] = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            captured["raw_completion"] = repr(resp)
        return resp

    teacher._transport = _capturing
    clamped = teacher.generate(game)  # strict validate + clamp; raises after retries

    raw = captured.get("raw_completion")
    parsed = json.loads(raw) if isinstance(raw, str) else raw

    # Count how many in-bounds keys the model emitted OUT of range (needed clamping).
    n_clamped = 0
    if isinstance(parsed, dict):
        for key, (low, high) in bounds.items():
            if key in parsed:
                try:
                    v = float(parsed[key])
                except (TypeError, ValueError):
                    n_clamped += 1
                    continue
                if v < low or v > high:
                    n_clamped += 1
    return parsed, clamped, (raw if isinstance(raw, str) else repr(raw)), n_clamped


# ---------------------------------------------------------------------------
# 5. Nested reward (Modal fan-out, ONE computation) + per-seed detail
# ---------------------------------------------------------------------------


def _compute_modal_reward(clamped: dict, sink: dict) -> float:
    from output.broad_eval_set import build_broad_eval_arenas, payload_arenas
    from output.nested_reward import teacher_reward
    from training.hud_teacher_env import fighter_geometry_override, params_to_curriculum

    spec = params_to_curriculum(clamped, curriculum_id="dryrun")
    held_out = payload_arenas(build_broad_eval_arenas(grid=HELD_OUT_GRID))
    return teacher_reward(
        spec,
        backend=BACKEND,
        seeds=SEEDS,
        episodes=PPO_EPISODES,
        eval_seeds=EVAL_SEEDS,
        capture_replay=True,           # capture a trained-Player replay (seed 1)
        held_out_arenas=held_out,
        geometry_override=fighter_geometry_override(clamped),
        _detail_sink=sink,
    )


# ---------------------------------------------------------------------------
# 7. HUD task (load-bearing) — record the nested reward through a real trace
# ---------------------------------------------------------------------------


async def _run_hud(clamped: dict, bounds) -> tuple[float, str, str, str | None]:
    """Drive the served crucible-teacher env; return (reward, trace_id, status, job_url)."""
    import mcp.types as mcp_types
    from hud.agents.base import Agent
    from hud.eval import LocalRuntime
    from hud.settings import settings
    from hud.types import Step

    from training.hud_teacher_env import design_curriculum

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

    env_source = os.path.join(str(_REPO_ROOT), "training", "hud_teacher_env.py")
    runtime = LocalRuntime(env_source, env="crucible-teacher")
    bounds_list = {k: [lo, hi] for k, (lo, hi) in bounds.items()}
    task = design_curriculum("Stage-1 2D fighter; pick a learnable FIGHTER curriculum", bounds_list)
    job = await task.run(FixedCurriculumAgent(), runtime=runtime)
    run = job.runs[0]
    job_url = None
    if settings.api_key and run.trace_id:
        job_url = f"{settings.hud_web_url}/jobs/{job.id}"
    return float(run.reward), str(run.trace_id) if run.trace_id else "", str(run.trace.status), job_url


# ---------------------------------------------------------------------------
# 8. Eval Protocol /init contract — return the same reward
# ---------------------------------------------------------------------------


async def _run_eval_protocol(clamped: dict, bounds, *, rollout_id: str) -> float:
    from types import SimpleNamespace

    from eval_protocol import InitRequest

    from training import dry_loop_sidecar as side
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
            "messages": [{"role": "user", "content": "Design a FIGHTER curriculum"}],
            "model_base_url": FIREWORKS_BASE_URL,
            "api_key": os.getenv("FIREWORKS_API_KEY") or "k",
            "metadata": {
                "invocation_id": "dryrun-inv",
                "experiment_id": "dryrun-exp",
                "rollout_id": rollout_id,
                "run_id": "dryrun-run",
                "row_id": "dryrun-row",
            },
        }
    )
    # The EP scorer reads the same published nested reward as HUD/Modal.
    result = await execute_rollout(
        req,
        bounds=bounds,
        scorer=side.served_nested_scorer,
        client_factory=lambda b, k: FakeClient(),
        reporter=lambda rid, extras: None,  # no live Fireworks tracing in the dry run
        # NOTE: execute_rollout accepts client_factory/reporter for exactly this.
    )
    return float(result.reward)


# ---------------------------------------------------------------------------
# 10. Replay capture -> viewer discovery
# ---------------------------------------------------------------------------


def _save_seed1_replay(sink: dict, rollout_id: str) -> tuple[str | None, bool]:
    """Write seed-1's captured replay to replays/dryrun_<rollout-id>_seed1.json and
    confirm the viewer can discover it (regenerate the manifest fallback)."""
    replays_dir = _REPO_ROOT / "replays"
    cap = None
    for arena in sink.get("arenas", []):
        for row in arena.get("rows", []):
            if row.get("replay"):
                cap = row["replay"]
                break
        if cap:
            break
    if not cap:
        return None, False

    out = replays_dir / f"dryrun_{rollout_id}_seed1.json"
    out.write_text(json.dumps(cap["data"], indent=2))

    # Regenerate the manifest fallback so the viewer lists it even where directory
    # indexing is disabled (the viewer prefers the live /replays/ listing, then this).
    names = sorted(p.name for p in replays_dir.glob("*.json") if p.name != "manifest.json")
    (replays_dir / "manifest.json").write_text(json.dumps(names, indent=2))

    # Viewer-discoverable iff it shows up in the same discovery the viewer uses:
    # the manifest array (the robust, server-agnostic path).
    manifest = json.loads((replays_dir / "manifest.json").read_text())
    discoverable = out.name in manifest
    return str(out), discoverable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    from training import dry_loop_sidecar as side
    from training.hud_teacher_env import FIGHTER_BOUNDS

    bounds = FIGHTER_BOUNDS
    t_start = time.time()
    rollout_id = f"dryrun-{int(t_start)}"
    cost = {"qwen_s": 0.0, "modal_s": 0.0}

    print("=" * 78)
    print("CRUCIBLE — pre-RFT DRY LOOP (FIGHTER game)")
    print("=" * 78)
    print(f"  rollout id     : {rollout_id}")
    print(f"  FIGHTER schema : {list(bounds)}")
    print(f"  reward         : nested PPO held-out transfer (broad '{HELD_OUT_GRID}' population)")
    print(f"  budget         : seeds={list(SEEDS)} episodes={PPO_EPISODES} eval_seeds={EVAL_SEEDS} backend={BACKEND}")
    print(f"  overall timeout: {OVERALL_TIMEOUT_S}s")
    print("=" * 78)

    # --- 1. Qwen readiness --------------------------------------------------
    print("\n[1/8] Polling Qwen deployment readiness ...")
    try:
        fw, acct, api_key = _resolve_fireworks()
        print(f"    account: {acct}")
        qwen_t0 = time.time()
        if not _wait_qwen_ready(fw, timeout_s=12 * 60):
            return _stop(10, f"Qwen deployment {QWEN_DEPLOYMENT_ID} never reached READY")
        handle = f"{QWEN_BASE_MODEL}#accounts/{acct}/deployments/{QWEN_DEPLOYMENT_ID}"
        print(f"    READY. inference handle: {handle}")
    except Exception as exc:
        return _stop(10, f"Qwen readiness/resolve failed: {type(exc).__name__}: {exc}")

    # --- 2-3. Qwen -> JSON -> validate/clamp --------------------------------
    print("\n[2/8] Calling Qwen for a bounded FIGHTER curriculum, then validating ...")
    try:
        raw_parsed, clamped, raw_str, n_clamped = _generate_and_validate(handle, api_key, bounds)
    except Exception as exc:
        cost["qwen_s"] = time.time() - qwen_t0
        return _stop(11, f"invalid Qwen JSON after retries: {type(exc).__name__}: {exc}")
    cost["qwen_s"] = time.time() - qwen_t0

    print("    RAW Qwen completion (verbatim):")
    print(f"      {raw_str!r}")
    print("    parsed JSON object:")
    print(json.dumps(raw_parsed, indent=6, sort_keys=True))
    print("    validated + clamped params (what the reward consumes):")
    print(json.dumps(clamped, indent=6, sort_keys=True))
    print(f"    keys that needed clamping: {n_clamped}")
    for key, (low, high) in bounds.items():
        v = clamped.get(key)
        print(f"      {key:<15} = {v:<10} in [{low}, {high}]")
    if n_clamped > CLAMP_BUDGET:
        return _stop(12, f"{n_clamped} parameters needed clamping (budget is {CLAMP_BUDGET})")

    # --- 4. Sidecar config + reset ------------------------------------------
    side.reset()
    side.write_config(
        backend=BACKEND, seeds=list(SEEDS), episodes=PPO_EPISODES,
        eval_seeds=EVAL_SEEDS, held_out_grid=HELD_OUT_GRID,
    )

    # --- 5. Nested reward via Modal (ONE computation) -----------------------
    print(f"\n[3/8] Training {len(SEEDS)} fresh PPO Players on Modal (parallel fan-out) ...")
    print(f"    held-out population: broad '{HELD_OUT_GRID}' grid "
          f"({len(__import__('output.broad_eval_set', fromlist=['build_broad_eval_arenas']).build_broad_eval_arenas(grid=HELD_OUT_GRID))} arenas)")
    sink: dict = {}
    modal_t0 = time.time()
    try:
        modal_reward = _compute_modal_reward(clamped, sink)
    except AssertionError as exc:
        cost["modal_s"] = time.time() - modal_t0
        return _stop(13, f"a Modal job failed (non-transfer result): {exc}")
    except Exception as exc:
        cost["modal_s"] = time.time() - modal_t0
        return _stop(13, f"Modal reward computation failed: {type(exc).__name__}: {exc}")
    cost["modal_s"] = time.time() - modal_t0

    arena = sink["arenas"][0]
    rows = arena["rows"]
    per_seed = []
    for s, row in zip(SEEDS, rows):
        per_seed.append((s, row["before_winrate"], row["after_winrate"], row["held_out_improvement"]))
    improvements = [imp for _, _, _, imp in per_seed]
    mean_imp = sum(improvements) / len(improvements)
    seed_std = statistics.pstdev(improvements) if len(improvements) > 1 else 0.0

    print("    per-seed held-out transfer:")
    print(f"      {'seed':>4} {'before':>8} {'after':>8} {'improvement':>12}")
    for s, b, a, imp in per_seed:
        print(f"      {s:>4} {b:>8.4f} {a:>8.4f} {imp:>+12.4f}")
    print(f"    mean improvement = {mean_imp:+.4f}   seed std = {seed_std:.4f}")
    print(f"    MODAL nested reward (clamped [0,1]) = {modal_reward:.6f}")

    # Publish so HUD + EP read the EXACT same number (one computation).
    side.publish(clamped, modal_reward)

    # --- 6. HUD (load-bearing) ----------------------------------------------
    print("\n[4/8] Recording the nested reward through a REAL HUD task/trace ...")
    os.environ["CRUCIBLE_HUD_SCORER"] = "nested"  # child subprocess reads this
    os.environ.setdefault("CRUCIBLE_SIDECAR_DIR", str(side.SIDECAR_DIR))
    try:
        hud_reward, trace_id, hud_status, job_url = asyncio.run(_run_hud(clamped, bounds))
    except Exception as exc:
        return _stop(15, f"HUD task failed: {type(exc).__name__}: {exc}")
    if not trace_id:
        return _stop(15, "HUD produced no trace id")
    print(f"    HUD-recorded reward = {hud_reward:.6f}")
    print(f"    HUD trace id        = {trace_id}")
    print(f"    HUD trace status    = {hud_status}")
    print(f"    HUD job url         = {job_url or '(no HUD_API_KEY — local trace only)'}")

    # --- 7. Eval Protocol ----------------------------------------------------
    print("\n[5/8] Returning the same reward through the Eval Protocol /init contract ...")
    try:
        ep_reward = asyncio.run(_run_eval_protocol(clamped, bounds, rollout_id=rollout_id))
    except Exception as exc:
        return _stop(14, f"Eval Protocol rollout failed: {type(exc).__name__}: {exc}")
    print(f"    Eval-Protocol reward = {ep_reward:.6f}")

    # --- 8. Assert + replay --------------------------------------------------
    print("\n[6/8] Asserting Modal == HUD == Eval-Protocol (tol 1e-6) ...")
    tol = 1e-6
    ok_hud = abs(modal_reward - hud_reward) < tol
    ok_ep = abs(modal_reward - ep_reward) < tol
    print(f"    Modal         = {modal_reward:.6f}")
    print(f"    HUD           = {hud_reward:.6f}   (== Modal: {ok_hud})")
    print(f"    EvalProtocol  = {ep_reward:.6f}   (== Modal: {ok_ep})")
    rewards_match = ok_hud and ok_ep

    print("\n[7/8] Capturing seed-1 replay and confirming viewer discovery ...")
    replay_path, discoverable = _save_seed1_replay(sink, rollout_id)
    if replay_path:
        print(f"    wrote {replay_path}")
        print(f"    viewer-discoverable (in replays/manifest.json): {discoverable}")
    else:
        print("    WARNING: no replay was captured by the transfer worker.")

    total_s = time.time() - t_start

    # --- Cost estimate -------------------------------------------------------
    qwen_cost = QWEN_DOLLARS_PER_HR * (cost["qwen_s"] / 3600.0)
    modal_cost = MODAL_CPU_DOLLARS_PER_HR * (cost["modal_s"] / 3600.0) * len(SEEDS)

    print("\n" + "=" * 78)
    print("[8/8] DRY-LOOP SUMMARY")
    print("=" * 78)
    print(f"  rollout id        : {rollout_id}")
    print(f"  Qwen raw JSON     : {raw_str}")
    print(f"  clamped params    : {json.dumps(clamped, sort_keys=True)}  (clamped keys: {n_clamped})")
    print(f"  mean reward       : {mean_imp:+.4f}  (seed std {seed_std:.4f})")
    print(f"  Modal reward      : {modal_reward:.6f}")
    print(f"  HUD reward        : {hud_reward:.6f}   trace={trace_id}")
    print(f"  HUD job url       : {job_url or '(local trace only)'}")
    print(f"  EvalProtocol rwd  : {ep_reward:.6f}")
    print(f"  rewards match     : {rewards_match}  (Modal==HUD: {ok_hud}, Modal==EP: {ok_ep})")
    print(f"  replay path       : {replay_path}  (viewer-discoverable: {discoverable})")
    print(f"  latency           : {total_s:.1f}s  (qwen {cost['qwen_s']:.1f}s, modal {cost['modal_s']:.1f}s)")
    print(f"  rough cost est    : ${qwen_cost + modal_cost:.4f}  "
          f"(qwen ${qwen_cost:.4f} + modal ${modal_cost:.4f})")
    print("=" * 78)

    if total_s > OVERALL_TIMEOUT_S:
        return _stop(16, f"total dry run took {total_s:.0f}s > {OVERALL_TIMEOUT_S}s timeout")
    if not rewards_match:
        return _stop(14, "reward mismatch Modal/HUD/Eval-Protocol — see the numbers above")

    print("\nRESULT: PASS — Qwen -> validate -> Modal PPO -> HUD -> Eval-Protocol all agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
