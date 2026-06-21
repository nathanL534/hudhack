"""output/stage6/evaluator.py — the Stage-6 base-vs-trained orchestrator.

This is the decider. It glues the pieces together while keeping the FAIRNESS
INVARIANT structural: one ``EvalConfig`` builds EVERY payload for BOTH models, and
the only thing that differs between base and trained is the resolved Teacher.

Pipeline (mirrors the validated reward path; preserves the single-fan-out design):

  1. Resolve base + trained HANDLES -> Teachers (Fireworks id OR local LoRA adapter).
  2. Each Teacher generates ``arenas_per_model`` arenas with IDENTICAL prompt +
     sampling (clamp/validation counts tracked for the anti-gaming check).
  3. Flatten EVERY (model x arena x seed) job and dispatch them in ONE concurrent
     fan-out (Modal ``fn.map`` remotely, or a sequential local fallback).
  4. Aggregate per-model mean/std/CI held-out transfer + parameter diversity.
  5. Run the anti-gaming checks; the verdict is "trained beats base on held-out
     transfer AND all anti-gaming checks pass".
  6. Write ONE structured result JSON (handles, arenas, validation counts, per-seed
     before/after, CI, diversity, latency/cost, HUD trace ids, replay paths).

The orchestrator never assumes a specific model family — Qwen-specific defaults
were removed; everything flows from the handle + game registry + config.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from output.stage6 import aggregate, anti_gaming
from output.stage6.config import EvalConfig
from output.stage6.games import GameEntry, GameNotInstalled, get_game
from output.stage6.handles import ResolvedTeacher, resolve_handle

OUTPUT_DIR = Path(__file__).resolve().parents[1]

MODAL_APP = "crucible-player"
MODAL_FN = "train_player_transfer"

# Rough per-job cost model (PPO container-seconds). Tunable; recorded as an
# ESTIMATE only. Modal A-class CPU ~ $0.000038/s; a transfer job ~ wall/ n_parallel.
COST_PER_JOB_USD_DEFAULT = 0.01


# ---------------------------------------------------------------------------
# Generation with clamp/validation tracking
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    arenas: list[dict]
    total: int
    invalid: int     # generations that failed strict validation (retried/raised)
    clamped: int     # generations where clamping moved at least one value


def _clamp_count(raw_params: dict, clamped: dict, schema: dict) -> bool:
    """True iff clamping moved any value (the raw value was out of range)."""
    for k, (low, high) in schema.items():
        if k in raw_params:
            try:
                v = float(raw_params[k])
            except (TypeError, ValueError):
                return True
            if v < low or v > high:
                return True
    return False


def generate_arenas(teacher, game: GameEntry, *, n: int, label: str,
                    verbose: bool = True, decode_temperature: float = 1.3,
                    min_unique: int = 4, max_attempts: int = 15) -> GenerationResult:
    """Generate ``n`` validated+clamped arenas, tracking invalid/clamped counts.

    Uses the Teacher's ``generate`` (identical prompt/sampling for base + trained).
    To measure clamping we wrap the Teacher's transport so we can inspect the RAW
    completion before clamping; if no transport is exposed we still count invalid
    generations (those that make ``generate`` raise) and treat clamped=0.

    DECODE-TIME ANTI-COLLAPSE (applied IDENTICALLY to base + trained, so neither
    side is advantaged): generation samples at a higher ``decode_temperature``
    (default 1.3) and REJECTS exact-duplicate arenas — dedup on the rounded knob
    tuple — resampling until at least ``min(min_unique, n)`` UNIQUE arenas exist,
    capping at ``max_attempts`` total generation attempts and then taking whatever
    unique set was found. This fixes the curriculum collapse where the Teacher
    emits one config N times (a point-mass curriculum), without retraining.
    """
    teacher_game = game.teacher_game()
    schema = teacher_game.param_schema

    # Bump decode temperature on the live Teacher (settable attr on both the Modal
    # and Fireworks Teachers). Restored in the finally block so the override is
    # scoped to this generation call only.
    orig_temperature = getattr(teacher, "temperature", None)
    if orig_temperature is not None:
        teacher.temperature = decode_temperature  # type: ignore[attr-defined]

    arenas: list[dict] = []
    seen: set = set()
    invalid = 0
    clamped = 0

    def _key(params: dict) -> tuple:
        return tuple(round(float(params.get(k, 0.0)), 6) for k in schema)

    target_unique = min(min_unique, n)

    raw_capture: dict = {}
    orig_transport = getattr(teacher, "_transport", None)

    if callable(orig_transport):
        def _wrap(payload):
            resp = orig_transport(payload)
            try:
                content = resp["choices"][0]["message"]["content"]
                raw_capture["last"] = json.loads(content) if isinstance(content, str) else content
            except Exception:
                raw_capture["last"] = None
            return resp
        teacher._transport = _wrap  # type: ignore[attr-defined]

    try:
        attempts = 0
        # Keep sampling until we have n UNIQUE arenas OR we hit at least
        # target_unique uniques after the first n draws, capped at max_attempts.
        while attempts < max_attempts and len(arenas) < n:
            raw_capture["last"] = None
            attempts += 1
            try:
                params = teacher.generate(teacher_game)
            except Exception as exc:  # invalid JSON exhausted retries
                invalid += 1
                if verbose:
                    print(f"    [{label}] attempt {attempts}: INVALID ({exc})")
                continue
            params = {k: float(params[k]) for k in schema if k in params}
            key = _key(params)
            if key in seen:
                if verbose:
                    print(f"    [{label}] attempt {attempts}: DUPLICATE (skipped)")
                # Stop spending attempts on dups once we already have enough unique.
                if len(arenas) >= target_unique and attempts >= n:
                    break
                continue
            seen.add(key)
            raw = raw_capture.get("last")
            if isinstance(raw, dict) and _clamp_count(raw, params, schema):
                clamped += 1
            arenas.append(params)
            if verbose:
                print(f"    [{label}] arena {len(arenas) - 1}: {json.dumps(params, sort_keys=True)}")
    finally:
        if callable(orig_transport):
            teacher._transport = orig_transport  # type: ignore[attr-defined]
        if orig_temperature is not None:
            teacher.temperature = orig_temperature  # type: ignore[attr-defined]

    return GenerationResult(arenas=arenas, total=len(arenas), invalid=invalid, clamped=clamped)


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def run_all_jobs(payloads: list[dict], seeds: list[int], *, backend: str) -> list[dict]:
    """Dispatch EVERY job at once and return rows IN ORDER.

    Modal: ONE ``fn.map`` over all payloads -> fully concurrent remote PPO. Local:
    sequential in-process (no credits; correctness fallback)."""
    if backend == "modal":
        import modal

        fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
        return list(fn.map(payloads, seeds))
    if backend == "local":
        from modal_player import local_transfer_worker

        return [local_transfer_worker(p, s) for p, s in zip(payloads, seeds)]
    raise ValueError(f"unknown backend {backend!r} (use 'modal' or 'local')")


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRequest:
    base_handle: str
    trained_handle: str
    game: str = "fighter"
    backend: str = "modal"
    config: EvalConfig = EvalConfig()
    is_smoke: bool = False
    cost_per_job_usd: float = COST_PER_JOB_USD_DEFAULT
    enable_hud_traces: bool = False
    verbose: bool = True
    # The HF/Fireworks base-model id the resolvers load under the handle. The
    # resolvers already accept ``base_model`` — this just threads it from the CLI so
    # a ``modal:``/``local:`` adapter loads ``Qwen/Qwen3-4B`` (the adapter's target),
    # not the Fireworks id. ``None`` -> each backend's own default.
    base_model: str | None = None
    # Run the SECONDARY fixed-bot transfer diagnostic too. The PRIMARY metric
    # (Student-vs-Student) is always run by ``run_full_eval``; the secondary is
    # extra evidence and can be skipped for a cheap primary-only run.
    run_secondary: bool = True
    # Run the KOTH cross-game head-to-head and FEED its trained-vs-base delta into
    # the primary anti-gaming gate (H3): a run that wins the train game but LOSES
    # KOTH is flagged train-game-overfit. On by default so the cross-game signal
    # actually gates the verdict; can be skipped for a cheap train-game-only run.
    run_cross_game: bool = True


def run_eval(req: EvalRequest, *, resolve=resolve_handle, run_jobs=run_all_jobs) -> dict:
    """Run the full base-vs-trained decider and return the structured result dict.

    ``resolve`` and ``run_jobs`` are injectable so unit tests can drive the whole
    orchestration with stub Teachers and a stub worker (no Modal, no Fireworks).
    """
    cfg = req.config
    game = get_game(req.game)
    if not game.installed:
        raise GameNotInstalled(
            f"game {req.game!r} is not installed: {game.not_installed_reason}"
        )
    if game.role != "teacher":
        raise ValueError(
            f"game {req.game!r} has role {game.role!r}; only role='teacher' games can "
            f"train a Teacher (use it as a probe instead)"
        )

    # --- 0. resolve handles (Modal / Fireworks id / local adapter), recorded verbatim ---
    # Thread the base-model id (when set) so a modal:/local: adapter loads its real
    # target (Qwen/Qwen3-4B). The injectable stub resolvers swallow it via **kw.
    resolve_kw = {"base_model": req.base_model} if req.base_model else {}
    base = resolve("base", req.base_handle, **resolve_kw)
    trained = resolve(
        "trained" if not req.is_smoke else "base(smoke)", req.trained_handle, **resolve_kw
    )

    # --- the held-out yardstick: IDENTICAL for both models (fairness invariant) ---
    eval_arenas = game.build_held_out(grid=cfg.held_out_grid)
    game.disjoint_guard(eval_arenas, [a["difficulty"] for a in eval_arenas])
    held_out = game.payload_arenas(eval_arenas)

    if req.verbose:
        print("=== Stage 6: BASE vs TRAINED Teacher — held-out transfer decider ===")
        print(f"    game={game.name}  backend={req.backend}  smoke={req.is_smoke}")
        print(f"    base    handle: {base.handle} -> {base.resolved_id} [{base.kind}]")
        print(f"    trained handle: {trained.handle} -> {trained.resolved_id} [{trained.kind}]")
        print(f"    config fingerprint: {cfg.fingerprint()}")
        print(f"    arenas/model={cfg.arenas_per_model}  seeds/arena={len(cfg.player_seeds)}  "
              f"ppo_episodes={cfg.ppo_episodes}  eval_seeds={cfg.eval_seeds}")
        print(f"    held-out population: {len(eval_arenas)} arenas (grid={cfg.held_out_grid})\n")

    # --- 1. each model generates its arenas (IDENTICAL prompt/sampling) ---
    if req.verbose:
        print("[1/3] Generating curricula from each Teacher (identical prompt/sampling) ...")
    models: list[tuple[str, ResolvedTeacher]] = [("base", base), (trained.label, trained)]
    gen_by_model: dict[str, GenerationResult] = {}
    for label, resolved in models:
        teacher = resolved.build()
        gen_by_model[label] = generate_arenas(
            teacher, game, n=cfg.arenas_per_model, label=label, verbose=req.verbose,
        )

    # --- optional: capture HUD traces of the generated curricula (best-effort) ---
    hud_traces: list[dict] = []
    if req.enable_hud_traces:
        from output.stage6 import hud_traces as hud_mod

        teacher_game = game.teacher_game()
        hud_traces = hud_mod.capture_traces(
            {label: gen_by_model[label].arenas for label, _ in models},
            teacher_game.param_schema, enabled=True,
        )

    # --- 2. flatten EVERY (model x arena x seed) job; ONE config builds them all ---
    flat: list[tuple[str, int, int, dict]] = []
    for label, _ in models:
        for ai, params in enumerate(gen_by_model[label].arenas):
            for s in cfg.player_seeds:
                cid = f"{label}-a{ai}-s{s}"
                flat.append((label, ai, s, cfg.train_payload(
                    params, held_out, param_keys=game.param_keys, curriculum_id=cid,
                )))

    n_jobs = len(flat)
    if not n_jobs:
        raise RuntimeError("no jobs to run (every generation was invalid?)")
    if req.verbose:
        print(f"\n[2/3] Dispatching ALL {n_jobs} (model x arena x seed) PPO jobs "
              f"in ONE concurrent fan-out ...")
    t0 = time.time()
    rows = run_jobs([p for _, _, _, p in flat], [s for _, _, s, _ in flat], backend=req.backend)
    wall = time.time() - t0
    for (label, ai, s, _), r in zip(flat, rows):
        assert r.get("status") == "ppo_transfer", f"non-transfer result: {r!r}"
    if req.verbose:
        print(f"    parallel wall-clock for {n_jobs} jobs: {wall:.1f}s")

    # --- 3. aggregate ---
    if req.verbose:
        print("\n[3/3] Aggregating held-out transfer per model ...")
    by_model: dict[str, list[float]] = {label: [] for label, _ in models}
    detail_by_model: dict[str, list[dict]] = {label: [] for label, _ in models}
    replays: list[str] = []
    for (label, ai, s, _), r in zip(flat, rows):
        impr = float(r["held_out_improvement"])
        by_model[label].append(impr)
        detail_by_model[label].append({
            "arena_index": ai, "seed": s,
            "held_out_improvement": round(impr, 4),
            "before": round(float(r["before_winrate"]), 4),
            "after": round(float(r["after_winrate"]), 4),
            "train_arena_winrate": round(float(r["train_arena_winrate"]), 4),
        })

    per_model = []
    for label, resolved in models:
        imps = by_model[label]
        ci = aggregate.mean_confidence_interval(imps)
        div = aggregate.parameter_diversity(gen_by_model[label].arenas, game.param_keys)
        per_model.append({
            "model": label,
            "resolved_id": resolved.resolved_id,
            "kind": resolved.kind,
            "mean_transfer": round(aggregate.mean(imps), 4),
            "std_transfer": round(aggregate.sample_std(imps), 4),
            "ci": ci.as_dict(),
            "n_jobs": len(imps),
            "arenas": [dict(a) for a in gen_by_model[label].arenas],
            "diversity": div,
            "generation": {
                "total": gen_by_model[label].total,
                "invalid": gen_by_model[label].invalid,
                "clamped": gen_by_model[label].clamped,
            },
            "per_job": detail_by_model[label],
        })

    base_row = next(m for m in per_model if m["model"] == "base")
    trained_row = next(m for m in per_model if m["model"] != "base")
    base_mean = base_row["mean_transfer"]
    trained_mean = trained_row["mean_transfer"]
    trained_beats_base = trained_mean > base_mean

    # JSON validation / clamping across BOTH models (the clamp-fraction check).
    total_gen = sum(m["generation"]["total"] for m in per_model)
    total_clamped = sum(m["generation"]["clamped"] + m["generation"]["invalid"] for m in per_model)
    clamp_fraction = (total_clamped / total_gen) if total_gen else 0.0

    # --- anti-gaming ---
    gaming = anti_gaming.run_all(
        base_mean=base_mean,
        trained_mean=trained_mean,
        trained_diversity=trained_row["diversity"],
        clamp_fraction=clamp_fraction,
        seed_std=trained_row["std_transfer"],
        base_resolved_id=base.resolved_id,
        trained_resolved_id=trained.resolved_id,
        is_smoke=req.is_smoke,
        train_signal_rose=None,
        probe_deltas=None,  # populated by the optional KOTH probe (out of band)
    )

    overall_pass = bool(trained_beats_base) and gaming.passed

    estimated_cost = round(n_jobs * req.cost_per_job_usd, 4)

    # Collect any replay paths the rows produced (worker returns dicts; the local
    # driver is responsible for writing them — recorded here when present).
    for (label, ai, s, _), r in zip(flat, rows):
        rep = r.get("replay")
        if isinstance(rep, dict) and rep.get("path"):
            replays.append(str(rep["path"]))

    summary = {
        "experiment": "stage6_base_vs_trained",
        "game": game.name,
        "game_role": game.role,
        "backend": req.backend,
        "is_smoke": req.is_smoke,
        "base_handle": base.handle,
        "base_resolved_id": base.resolved_id,
        "base_kind": base.kind,
        "trained_handle": trained.handle,
        "trained_resolved_id": trained.resolved_id,
        "trained_kind": trained.kind,
        "config": cfg.as_dict(),
        "n_held_out_arenas": len(eval_arenas),
        "held_out_arenas": held_out,
        "n_jobs": n_jobs,
        "wall_clock_s": round(wall, 2),
        "estimated_cost_usd": estimated_cost,
        "per_model": per_model,
        "per_game": [{
            "game": game.name, "role": game.role,
            "base_mean": base_mean, "trained_mean": trained_mean,
            "delta": round(trained_mean - base_mean, 4),
        }],
        "json_validation": {
            "total": total_gen, "clamped": total_clamped,
            "clamp_fraction": round(clamp_fraction, 4),
        },
        "base_mean_transfer": base_mean,
        "trained_mean_transfer": trained_mean,
        "trained_beats_base": trained_beats_base,
        "delta": round(trained_mean - base_mean, 4),
        "anti_gaming": gaming.as_dict(),
        "overall_pass": overall_pass,
        "hud_traces": hud_traces,   # populated when --hud-traces is enabled
        "replays": replays,
    }

    if req.verbose:
        _print_verdict(summary)
    return summary


def _print_verdict(summary: dict) -> None:
    print("\n    === BASE vs TRAINED VERDICT ===")
    print(f"    {'model':>14} {'mean':>10} {'std':>8} {'95% CI':>22} {'n':>4}")
    for m in summary["per_model"]:
        ci = m["ci"]
        print(f"    {m['model']:>14} {m['mean_transfer']:>+10.4f} {m['std_transfer']:>8.4f} "
              f"  [{ci['low']:+.4f},{ci['high']:+.4f}] {m['n_jobs']:>4}")
    print(f"\n    base    mean transfer = {summary['base_mean_transfer']:+.4f}")
    print(f"    trained mean transfer = {summary['trained_mean_transfer']:+.4f}")
    print(f"    delta = {summary['delta']:+.4f}  TRAINED BEATS BASE: {summary['trained_beats_base']}")
    print(f"    anti-gaming passed: {summary['anti_gaming']['passed']}  "
          f"(failed: {summary['anti_gaming']['failed']})")
    print(f"    OVERALL PASS (transfer + anti-gaming): {summary['overall_pass']}")


def write_result(summary: dict, path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(summary, indent=2))
    return path


# ---------------------------------------------------------------------------
# The FULL decider: PRIMARY (Student-vs-Student) headline + SECONDARY diagnostic
# ---------------------------------------------------------------------------


def run_full_eval(
    req: EvalRequest, *, resolve=resolve_handle, run_jobs=run_all_jobs,
    run_train=None, run_h2h=None, run_cross_game=None,
) -> dict:
    """The Stage-6 decider whose HEADLINE is the Student-vs-Student head-to-head.

    Runs:
      * the PRIMARY metric — two fresh Students (one per Teacher's curricula) fought
        head-to-head on unseen arenas, side-swapped, with a curriculum-level 95% CI.
        This ALONE decides the headline verdict.
      * the SECONDARY metric (when ``req.run_secondary``) — the legacy fixed-bot
        before->after held-out transfer diagnostic, kept as extra evidence under a
        clearly-labeled "Secondary transfer diagnostic" section. It does NOT affect
        the headline.
      * the CROSS-GAME probe (when ``req.run_cross_game``) — a KOTH Student-vs-Student
        head-to-head whose trained-vs-base delta is fed into the PRIMARY anti-gaming
        gate (H3). A run that WINS the train game but LOSES KOTH is flagged
        train-game-overfit, and that gate DOES contribute to the headline verdict.

    The injected ``run_train`` / ``run_h2h`` thread through to BOTH the train-game
    head-to-head and the KOTH cross-game so tests can stub the fan-outs;
    ``run_cross_game`` lets a test inject the cross-game result directly (and the CLI
    avoid a double-run); ``resolve`` / ``run_jobs`` stub the secondary path exactly as
    ``run_eval`` already supports.
    """
    from output.stage6 import anti_gaming, head_to_head as h2h_mod
    from output.stage6 import koth_cross_game as koth_mod

    cfg = req.config
    game = get_game(req.game)
    if not game.installed:
        raise GameNotInstalled(f"game {req.game!r} is not installed: {game.not_installed_reason}")
    if game.role != "teacher":
        raise ValueError(
            f"game {req.game!r} has role {game.role!r}; only role='teacher' games can train a Teacher"
        )

    resolve_kw = {"base_model": req.base_model} if req.base_model else {}
    base = resolve("base", req.base_handle, **resolve_kw)
    trained = resolve(
        "trained" if not req.is_smoke else "base(smoke)", req.trained_handle, **resolve_kw
    )

    # --- PRIMARY: Student-vs-Student head-to-head (the headline) ---
    h2h_req = h2h_mod.HeadToHeadRequest(
        base=base, trained=trained, game=game, config=cfg,
        backend=req.backend, is_smoke=req.is_smoke, verbose=req.verbose,
    )
    h2h_kwargs = {}
    if run_train is not None:
        h2h_kwargs["run_train"] = run_train
    if run_h2h is not None:
        h2h_kwargs["run_h2h"] = run_h2h
    primary = h2h_mod.run_head_to_head(h2h_req, **h2h_kwargs)

    # --- SECONDARY: fixed-bot before->after transfer diagnostic (extra evidence) ---
    secondary = None
    if req.run_secondary:
        if req.verbose:
            print("\n--- Secondary transfer diagnostic (fixed-bot before->after) ---")
        secondary = run_eval(req, resolve=resolve, run_jobs=run_jobs)

    # --- CROSS-GAME (KOTH): the trained-vs-base KOTH delta feeds the anti-gaming gate.
    # H3 — improvement must not be confined to the TRAIN game. We run a fresh-KOTH-
    # Students head-to-head and extract its primary advantage (trained - base). That
    # delta becomes the ``koth`` probe in ``check_train_game_only``, so a run that wins
    # the train-game head-to-head but LOSES KOTH is flagged train-game-overfit. The
    # gate is folded into ``overall_pass`` below, so it DOES contribute to the verdict.
    cross_game = None
    if req.run_cross_game:
        if run_cross_game is not None:
            cross_game = run_cross_game(base, trained, config=cfg, backend=req.backend,
                                        is_smoke=req.is_smoke, verbose=req.verbose)
        else:
            if req.verbose:
                print("\n--- Cross-game generalization: KOTH (fresh KOTH Students) ---")
            koth_kwargs = {}
            if run_train is not None:
                koth_kwargs["run_train"] = run_train
            if run_h2h is not None:
                koth_kwargs["run_h2h"] = run_h2h
            cross_game = koth_mod.run_koth_cross_game(
                base, trained, config=cfg, backend=req.backend,
                is_smoke=req.is_smoke, verbose=req.verbose, **koth_kwargs,
            )

    # Build the cross-game anti-gaming gate. ``probe_deltas`` carries the KOTH
    # trained-vs-base delta when KOTH actually ran (and was supported); otherwise the
    # check SKIPs (non-gating), exactly as ``check_train_game_only`` already handles a
    # no-probe run. The train-game delta it is checked against is the PRIMARY
    # Student-vs-Student advantage (the headline metric), not the secondary transfer.
    probe_deltas = None
    if cross_game and cross_game.get("supported"):
        probe_deltas = {"koth": float(cross_game.get("mean_paired_advantage", 0.0))}
    train_game_advantage = float(primary["mean_paired_advantage"])
    cross_game_gate = anti_gaming.check_train_game_only(train_game_advantage, probe_deltas)
    cross_game_gate_passed = bool(
        cross_game_gate.passed or cross_game_gate.severity != "fail"
    )

    # The HEADLINE verdict uses the primary AND the cross-game (train-game-overfit)
    # gate. A run that wins the train-game head-to-head but loses KOTH now FAILS.
    overall_pass = bool(primary["primary_pass"]) and cross_game_gate_passed

    summary = {
        "experiment": "stage6_student_vs_student",
        "game": game.name,
        "backend": req.backend,
        "is_smoke": req.is_smoke,
        "base_handle": base.handle,
        "base_resolved_id": base.resolved_id,
        "base_kind": base.kind,
        "trained_handle": trained.handle,
        "trained_resolved_id": trained.resolved_id,
        "trained_kind": trained.kind,
        "config": cfg.as_dict(),
        # --- the headline ---
        "headline_metric": "primary_student_vs_student_head_to_head",
        "primary": primary,
        "overall_pass": overall_pass,
        # --- cross-game (KOTH) probe + the anti-gaming gate it feeds (H3) ---
        # ``cross_game_koth`` is the full KOTH head-to-head summary; ``cross_game_gate``
        # is the train-game-overfit check whose verdict is folded into overall_pass.
        "cross_game_koth": cross_game,
        "cross_game_gate": cross_game_gate.as_dict(),
        # --- secondary evidence (never sets the headline) ---
        "secondary_transfer_diagnostic": secondary,
        # Convenience top-level mirrors of the headline numbers (for the dashboard).
        "trained_student_win_rate": primary["trained_student_win_rate"],
        "base_student_win_rate": primary["base_student_win_rate"],
        "mean_paired_advantage": primary["mean_paired_advantage"],
        "ci_lower_bound": primary["ci_lower_bound"],
        "replays": primary.get("replays", []),
    }
    if req.verbose:
        _print_full_verdict(summary)
    return summary


def _print_full_verdict(summary: dict) -> None:
    p = summary["primary"]
    print("\n    ===================== STAGE 6 HEADLINE =====================")
    print(f"    PRIMARY (Student vs Student): trained beats base = "
          f"{p['mean_paired_advantage'] > 0} (advantage {p['mean_paired_advantage']:+.4f})")
    print(f"    curriculum-level 95% CI lower bound = {p['ci_lower_bound']:+.4f} "
          f"(must be > 0)")
    print(f"    anti-circularity = {'PASS' if p['anti_circularity']['passed'] else 'FAIL'}")
    cg_gate = summary.get("cross_game_gate")
    if cg_gate:
        cg = summary.get("cross_game_koth") or {}
        if cg.get("supported"):
            koth_delta = cg.get("mean_paired_advantage")
            print(f"    cross-game (KOTH) delta = {koth_delta:+.4f} "
                  f"-> train-game-overfit gate = "
                  f"{'PASS' if cg_gate['passed'] else 'FAIL'}")
        else:
            print(f"    cross-game (KOTH) gate = SKIP "
                  f"({(cg.get('reason') or 'not run')})")
    print(f"    OVERALL HEADLINE PASS = {summary['overall_pass']}")
    if summary.get("secondary_transfer_diagnostic"):
        s = summary["secondary_transfer_diagnostic"]
        print(f"    [secondary diagnostic] fixed-bot transfer delta = {s.get('delta'):+.4f} "
              f"(evidence only, not the headline)")
