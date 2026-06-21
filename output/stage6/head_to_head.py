"""output/stage6/head_to_head.py — the PRIMARY metric: Student vs Student.

THE HEADLINE. The original thesis is "the TRAINED Teacher generates better
curricula than the BASE Teacher". The most direct test of that is NOT a fixed-bot
before/after diagnostic (that is the SECONDARY metric) — it is to take the two
Students each Teacher's curricula PRODUCE and fight them HEAD-TO-HEAD on unseen
ground:

    Base Teacher curricula  -> train fresh Base Student
    Trained Teacher curricula -> train fresh Trained Student
    freeze BOTH
    fight them on UNSEEN arenas (side-swapped, many fresh seeds)
    report trained-Student win rate + a curriculum-level 95% CI

Primary success criterion (the headline verdict):
    "The Student trained on the TRAINED Teacher's curricula beats the Student
     trained on the BASE Teacher's curricula on held-out arenas, with a
     curriculum-level 95% CI lower bound above zero."

Pipeline (one replicate):
  1. BOTH Teachers generate ``curriculum_arenas`` arenas (identical prompt/sampling,
     validated+clamped identically — the fairness invariant).
  2. Train ONE fresh Student per Teacher on its curriculum SET. IDENTICAL
     architecture, PPO budget, seed policy, curriculum count. Freeze both
     (serialize policy weights). No model gets extra episodes or easier opponents.
  3. Fight them on the held-out arena population (disjoint from BOTH curricula).
     For each held-out arena: trained-as-P0 vs base, THEN base-as-P0 vs trained
     (the SIDE SWAP) with matched seeds. Record win/loss/draw per side.

Replicates are INDEPENDENT (fresh curricula + fresh Students per replicate), so
the curriculum-level CI is computed across replicate paired-advantages — NOT
across frames or duplicate matches. Success requires mean advantage > 0 AND CI
lower bound > 0 AND no side bias AND benefit across >1 arena family AND >1
replicate (all enforced by ``anti_gaming.run_head_to_head_checks``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from output.stage6 import aggregate, anti_gaming
from output.stage6.config import EvalConfig
from output.stage6.evaluator import generate_arenas
from output.stage6.games import GameEntry
from output.stage6.handles import ResolvedTeacher

# Modal app + function names for the PRIMARY workers (deployed in modal_player.py).
MODAL_APP = "crucible-player"
MODAL_TRAIN_FN = "train_student_policy"
MODAL_H2H_FN = "head_to_head_match"


# ---------------------------------------------------------------------------
# Held-out population (disjoint from BOTH Teachers' curricula)
# ---------------------------------------------------------------------------


def held_out_population(game: GameEntry, *, grid: str) -> list[dict]:
    """Build the UNSEEN arena population for the head-to-head.

    Reuses the game's fixed held-out builder (structurally diverse: varied
    difficulty AND physics) — the same disjoint-by-construction set the secondary
    metric uses, so disjointness is guaranteed the same way. ``name`` is kept here
    (so a per-arena-FAMILY breakdown can group by it); strip it for the worker.
    """
    return game.build_held_out(grid=grid)


_REPLAY_DIR = Path(__file__).resolve().parents[2] / "replays"


def _write_replay(game: str, outcome: str, side: str, data: dict) -> str:
    """Persist one captured head-to-head replay to ``replays/h2h_<game>_<outcome>_<side>.json``."""
    _REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    name = f"h2h_{game}_{outcome}_{side}.json"
    path = _REPLAY_DIR / name
    path.write_text(json.dumps(data))
    return str(path)


# Which games carry a fighter-style viewer replay the match WORKER captures. KOTH is
# NOT here: its obs space differs and the canonical worker recorder is fighter-only,
# so the match worker deliberately returns no KOTH frames. We capture KOTH replays in
# the orchestration instead (below), reusing the exact same restore + record + write
# path the worker would — never forcing the worker to fabricate frames it can't.
_WORKER_REPLAY_GAMES = frozenset({"fighter", "ring-out-duel", "ring_out_duel"})


def _capture_koth_replays(
    game_name: str, arena_spec: dict, trained_policy: dict, base_policy: dict,
    match_seeds: list[int],
) -> list[dict]:
    """Roll out representative KOTH Student-vs-Student replays on the DRIVER.

    The match worker (``modal_player._capture_h2h_replays``) only records the
    fighter's viewer replay — KOTH has a different obs space and the canonical
    recorder is fighter-only, so the worker returns no KOTH frames. We restore the
    two FROZEN Students here (the SAME ``PolicyArtifact`` -> ``restore_policy`` path
    the worker uses, into a throwaway KOTH env that only supplies the obs/action
    spaces) and record one representative match per outcome category with
    ``record_replay.record_match_koth`` — the same recorder home, same replay schema,
    same ``_write_replay`` sink. This keeps capture byte-identical to the canonical
    recorder while never touching the match worker.

    Returns capture rows shaped exactly like the worker's (``side`` / ``outcome`` /
    ``data`` + self-describing source-Teacher metadata) so the existing persist loop
    writes them unchanged. Best-effort: any failure yields ``[]`` (capture never
    gates the headline metric).
    """
    # Capture is best-effort and runs on the driver (torch/SB3 + KOTH must be
    # importable, and the policy dicts must carry real weights). Any failure —
    # missing deps, a stub policy without ``state_dict_b64``, a restore error —
    # yields ``[]`` so capture NEVER gates the headline metric or breaks a stubbed
    # orchestration test.
    try:
        import dataclasses

        from record_replay import record_match_koth
        from output.stage6.policy import DEFAULT_NET_ARCH, PolicyArtifact, restore_policy
        from games.koth import KothArena
        from harness.koth_trainer import _MultiArenaKothEnv

        field_names = {f.name for f in dataclasses.fields(KothArena)}
        arena = KothArena(**{k: float(v) for k, v in arena_spec.items() if k in field_names})
        seed0 = int(match_seeds[0]) if match_seeds else 0

        # A throwaway env only supplies obs/action spaces to restore the frozen nets
        # — exactly what the worker's ``_head_to_head_match`` does.
        restore_env = _MultiArenaKothEnv([arena], seed=seed0)
        trained = restore_policy(
            PolicyArtifact.from_dict(trained_policy), restore_env,
            seed=seed0,  # net_arch read from artifact
        )
        base = restore_policy(
            PolicyArtifact.from_dict(base_policy), restore_env,
            seed=seed0 + 1,  # net_arch read from artifact
        )

        _winner_of = {"p1": 0, "p2": 1, "draw": None}
        want = {"trained_win": None, "base_win": None, "draw": None, "side_swap": None}
        meta_common = {
            "game": "koth",
            "source_teacher_trained": trained_policy.get("teacher"),
            "source_teacher_base": base_policy.get("teacher"),
            "trained_curriculum_id": trained_policy.get("curriculum_id"),
            "base_curriculum_id": base_policy.get("curriculum_id"),
            "trained_policy_seed": trained_policy.get("seed"),
            "base_policy_seed": base_policy.get("seed"),
            "arena": arena_spec,
            "zone": {
                "center": round(float(arena.zone_center), 4),
                "half": round(float(arena.zone_half), 4),
                "center_frac": round(float(arena.zone_center_frac), 4),
            },
        }

        def _roll(p0, p1, seed, p0_label, p1_label):
            data = record_match_koth(arena, p0, p1, config="h2h-koth",
                                     p1_policy=p0_label, p2_policy=p1_label, seed=seed)
            return data, _winner_of[data["meta"]["winner"]]

        for ms in match_seeds:
            # trained as P0 vs base as P1.
            data_a, w_a = _roll(trained, base, ms, "trained_student", "base_student")
            common = {**meta_common, "match_seed": ms}
            if w_a == 0 and want["trained_win"] is None:
                want["trained_win"] = {**common, "side": "trained_as_P0", "outcome": "trained_win", "data": data_a}
            elif w_a == 1 and want["base_win"] is None:
                want["base_win"] = {**common, "side": "trained_as_P0", "outcome": "base_win", "data": data_a}
            elif w_a is None and want["draw"] is None:
                want["draw"] = {**common, "side": "trained_as_P0", "outcome": "draw", "data": data_a}
            # side swap: trained as P1.
            if want["side_swap"] is None:
                data_b, w_b = _roll(base, trained, ms, "base_student", "trained_student")
                want["side_swap"] = {**common, "side": "trained_as_P1",
                                     "outcome": ("trained_win" if w_b == 1 else
                                                 "base_win" if w_b == 0 else "draw"),
                                     "data": data_b}
            if all(v is not None for v in want.values()):
                break

        return [v for v in want.values() if v is not None]
    except Exception:  # pragma: no cover - best-effort capture, never gates
        return []


def _arena_family(spec: dict) -> str:
    """Group key for the per-arena-FAMILY breakdown (physics archetype)."""
    name = spec.get("name", "")
    if name:
        # broad_eval names look like "narrow_ledge_d0.4" -> family "narrow_ledge".
        return name.rsplit("_d", 1)[0]
    return f"pw{spec.get('platform_width', '?')}"


def _round_spec(spec: dict, keys) -> tuple:
    return tuple(round(float(spec.get(k, 0.0)), 6) for k in keys)


def count_held_out_overlap(
    held_out: list[dict], curricula: list[dict], param_keys,
) -> int:
    """How many held-out arenas exactly match a curriculum arena (param tuple).

    A non-zero count is a training-on-the-test circularity (H5). Compared on the
    full param tuple so a shared difficulty alone (different physics) is fine.
    """
    cur = {_round_spec(c, param_keys) for c in curricula}
    return sum(1 for h in held_out if _round_spec(h, param_keys) in cur)


# ---------------------------------------------------------------------------
# Fan-out dispatch (Modal map / local sequential)
# ---------------------------------------------------------------------------


def _run_train(payloads: list[dict], seeds: list[int], *, backend: str) -> list[dict]:
    if backend == "modal":
        import modal

        fn = modal.Function.from_name(MODAL_APP, MODAL_TRAIN_FN)
        return list(fn.map(payloads, seeds))
    if backend == "local":
        from modal_player import local_train_student_worker

        return [local_train_student_worker(p, s) for p, s in zip(payloads, seeds)]
    raise ValueError(f"unknown backend {backend!r}")


def _run_h2h(payloads: list[dict], seeds: list[int], *, backend: str) -> list[dict]:
    if backend == "modal":
        import modal

        fn = modal.Function.from_name(MODAL_APP, MODAL_H2H_FN)
        return list(fn.map(payloads, seeds))
    if backend == "local":
        from modal_player import local_head_to_head_worker

        return [local_head_to_head_worker(p, s) for p, s in zip(payloads, seeds)]
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# The primary evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadToHeadRequest:
    base: ResolvedTeacher
    trained: ResolvedTeacher
    game: GameEntry
    config: EvalConfig
    backend: str = "modal"
    is_smoke: bool = False
    verbose: bool = True


def run_head_to_head(
    req: HeadToHeadRequest, *, run_train=_run_train, run_h2h=_run_h2h,
) -> dict:
    """Run the full Student-vs-Student head-to-head and return the structured dict.

    ``run_train`` / ``run_h2h`` are injectable so unit tests drive the orchestration
    with stub workers (no Modal, no PPO). The result dict is the PRIMARY headline:
    curriculum-level paired advantage + CI + side-specific win rates + per-arena
    breakdown + the anti-circularity verdict.
    """
    cfg = req.config
    game = req.game
    param_keys = game.param_keys

    held_full = held_out_population(game, grid=cfg.head_to_head_grid)
    held_payload = game.payload_arenas(held_full)

    if req.verbose:
        print("=== Stage 6 PRIMARY: BASE-Student vs TRAINED-Student head-to-head ===")
        print(f"    game={game.name}  backend={req.backend}  smoke={req.is_smoke}")
        print(f"    replicates={cfg.n_replicates}  curriculum_arenas={cfg.curriculum_arenas}  "
              f"held-out={len(held_full)}  match_seeds/arena/side={cfg.match_seeds_per_arena}")

    t0 = time.time()
    replicate_rows: list[dict] = []
    all_overlap = 0
    base_checksums: list[str] = []
    trained_checksums: list[str] = []
    # Per-arena-family accumulation across replicates (for the multi-arena check).
    family_adv: dict[str, list[float]] = {}
    side_p0_wins = side_p1_wins = 0
    side_p0_total = side_p1_total = 0
    replays: list[dict] = []

    for r in range(cfg.n_replicates):
        rep = _run_one_replicate(
            req, r, held_full, held_payload, param_keys,
            run_train=run_train, run_h2h=run_h2h,
        )
        replicate_rows.append(rep)
        all_overlap += rep["held_out_overlap"]
        base_checksums.append(rep["base_student_checksum"])
        trained_checksums.append(rep["trained_student_checksum"])
        side_p0_wins += rep["_side_p0_wins"]; side_p0_total += rep["_side_p0_total"]
        side_p1_wins += rep["_side_p1_wins"]; side_p1_total += rep["_side_p1_total"]
        for fam, adv in rep["per_family_advantage"].items():
            family_adv.setdefault(fam, []).append(adv)
        replays.extend(rep.get("replays", []))
        if req.verbose:
            print(f"    [replicate {r}] paired advantage = {rep['paired_advantage']:+.4f} "
                  f"(trained {rep['trained_win_rate']:.3f} / base {rep['base_win_rate']:.3f} "
                  f"/ draw {rep['draw_rate']:.3f})")

    wall = time.time() - t0

    # --- curriculum-level aggregation: CI across INDEPENDENT replicates ---
    rep_advantages = [rep["paired_advantage"] for rep in replicate_rows]
    mean_adv = aggregate.mean(rep_advantages)
    t_ci = aggregate.mean_confidence_interval(rep_advantages)
    boot_ci = aggregate.bootstrap_ci(rep_advantages, seed=12345)

    trained_p0_winrate = (side_p0_wins / side_p0_total) if side_p0_total else 0.0
    trained_p1_winrate = (side_p1_wins / side_p1_total) if side_p1_total else 0.0

    # Per-arena-family advantage = mean across replicates for that family.
    per_family = {fam: round(aggregate.mean(vals), 6) for fam, vals in family_adv.items()}
    per_family_advantages = list(per_family.values())

    # Distinct-Students check uses the FIRST replicate's two checksums (each
    # replicate trains fresh Students; if base==trained even once, that's the lie).
    distinct_violation = any(b == t for b, t in zip(base_checksums, trained_checksums))
    base_cs = base_checksums[0] if base_checksums else ""
    trained_cs = base_cs if distinct_violation else (trained_checksums[0] if trained_checksums else "")

    gaming = anti_gaming.run_head_to_head_checks(
        advantage_mean=mean_adv,
        ci_low=t_ci.low,
        trained_p0_winrate=trained_p0_winrate,
        trained_p1_winrate=trained_p1_winrate,
        per_arena_advantages=per_family_advantages,
        base_checksum=base_cs,
        trained_checksum=trained_cs,
        held_out_overlap_count=all_overlap,
        n_replicates=cfg.n_replicates,
        is_smoke=req.is_smoke,
    )

    overall_trained_winrate = (
        sum(rep["_trained_wins"] for rep in replicate_rows)
        / max(1, sum(rep["_total_matches"] for rep in replicate_rows))
    )
    overall_base_winrate = (
        sum(rep["_base_wins"] for rep in replicate_rows)
        / max(1, sum(rep["_total_matches"] for rep in replicate_rows))
    )
    overall_draw_rate = (
        sum(rep["_draws"] for rep in replicate_rows)
        / max(1, sum(rep["_total_matches"] for rep in replicate_rows))
    )

    # The PRIMARY verdict: mean advantage > 0 AND all anti-circularity checks pass.
    primary_pass = bool(mean_adv > 0.0) and gaming.passed

    # Effect size: Cohen's d on the replicate paired-advantages (vs 0).
    std = aggregate.sample_std(rep_advantages)
    effect_size = round(mean_adv / std, 4) if std > 0 else None

    summary = {
        "metric": "primary_student_vs_student_head_to_head",
        "game": game.name,
        "n_replicates": cfg.n_replicates,
        "curriculum_arenas_per_teacher": cfg.curriculum_arenas,
        "n_held_out_arenas": len(held_full),
        "match_seeds_per_arena_per_side": cfg.match_seeds_per_arena,
        "held_out_arenas": held_payload,
        # --- the headline numbers ---
        "trained_student_win_rate": round(overall_trained_winrate, 6),
        "base_student_win_rate": round(overall_base_winrate, 6),
        "draw_rate": round(overall_draw_rate, 6),
        "mean_paired_advantage": round(mean_adv, 6),
        "ci_t": t_ci.as_dict(),
        "ci_bootstrap": boot_ci.as_dict(),
        "ci_lower_bound": round(t_ci.low, 6),
        "effect_size_cohens_d": effect_size,
        "trained_win_rate_as_p0": round(trained_p0_winrate, 6),
        "trained_win_rate_as_p1": round(trained_p1_winrate, 6),
        "per_arena_family_advantage": per_family,
        "held_out_overlap_total": all_overlap,
        "anti_circularity": gaming.as_dict(),
        "primary_pass": primary_pass,
        "wall_clock_s": round(wall, 2),
        "replicates": [
            {k: v for k, v in rep.items() if not k.startswith("_")}
            for rep in replicate_rows
        ],
        "replays": replays,
    }

    if req.verbose:
        _print_primary_verdict(summary)
    return summary


def _run_one_replicate(
    req: HeadToHeadRequest, r: int, held_full: list[dict], held_payload: list[dict],
    param_keys, *, run_train, run_h2h,
) -> dict:
    """One independent replicate: generate -> train 2 Students -> freeze -> fight."""
    cfg = req.config
    game = req.game

    # --- 1. both Teachers generate curricula (identical prompt/sampling) ---
    # Each replicate uses a DISTINCT generation seed (``curriculum_seed_base + r``),
    # applied IDENTICALLY to base and trained, so the two Teachers are rebuilt fresh
    # per replicate and sample DIFFERENT curricula across replicates. Without this,
    # every replicate would rebuild the Teacher at the same seed and emit identical
    # arenas, so the curriculum-level CI would capture only Student-training noise.
    gen_seed = cfg.curriculum_seed_base + r
    base_teacher = req.base.build(gen_seed=gen_seed)
    trained_teacher = req.trained.build(gen_seed=gen_seed)
    base_gen = generate_arenas(base_teacher, game, n=cfg.curriculum_arenas,
                               label=f"base-r{r}", verbose=False)
    trained_gen = generate_arenas(trained_teacher, game, n=cfg.curriculum_arenas,
                                  label=f"trained-r{r}", verbose=False)

    # --- disjointness: held-out must not overlap EITHER curriculum ---
    overlap = (count_held_out_overlap(held_full, base_gen.arenas, param_keys)
               + count_held_out_overlap(held_full, trained_gen.arenas, param_keys))

    # --- 2. train ONE fresh Student per Teacher on its curriculum SET ---
    # IDENTICAL budget/seed-policy for both (fairness invariant). The training seed
    # is the SAME for base and trained in a replicate, derived from the replicate
    # index, so neither side gets a luckier seed.
    train_seed = 1000 + r
    train_payloads = [
        _student_payload("base", f"base-r{r}", base_gen.arenas, cfg, game.name),
        _student_payload("trained", f"trained-r{r}", trained_gen.arenas, cfg, game.name),
    ]
    rows = run_train(train_payloads, [train_seed, train_seed], backend=req.backend)
    base_row = next(x for x in rows if x["teacher"] == "base")
    trained_row = next(x for x in rows if x["teacher"] == "trained")
    base_policy = base_row["policy"]
    trained_policy = trained_row["policy"]

    # --- 3. fight on every held-out arena, side-swapped, fresh seeds ---
    match_seeds = list(range(r * 1000, r * 1000 + cfg.match_seeds_per_arena))
    # Capture representative replays on the FIRST replicate's FIRST held-out arena
    # only (one capture per run; the rest are score-only to keep the fan-out cheap).
    h2h_payloads = []
    for ai, payload_arena in enumerate(held_payload):
        h2h_payloads.append({
            "game": game.name,
            "trained_policy": trained_policy,
            "base_policy": base_policy,
            "arena": payload_arena,
            "match_seeds": match_seeds,
            "capture_replays": bool(r == 0 and ai == 0),
        })
    h2h_rows = run_h2h(h2h_payloads, [r] * len(h2h_payloads), backend=req.backend)

    # --- tally win/loss/draw per side + per arena family ---
    trained_wins = base_wins = draws = 0
    p0_wins = p0_total = p1_wins = p1_total = 0
    per_family_counts: dict[str, dict] = {}
    for spec, hrow in zip(held_full, h2h_rows):
        fam = _arena_family(spec)
        fc = per_family_counts.setdefault(fam, {"t": 0, "b": 0, "d": 0})
        for d in hrow["per_seed"]:
            for side_key, is_p0 in (("trained_p0", True), ("trained_p1", False)):
                out = d[side_key]
                if out == "win":
                    trained_wins += 1; fc["t"] += 1
                    if is_p0: p0_wins += 1
                    else: p1_wins += 1
                elif out == "loss":
                    base_wins += 1; fc["b"] += 1
                else:
                    draws += 1; fc["d"] += 1
                if is_p0: p0_total += 1
                else: p1_total += 1

    # --- persist any captured representative replays (data -> disk, path -> JSON) ---
    replays_meta: list[dict] = []
    for hrow in h2h_rows:
        for rep in hrow.get("replays", []) or []:
            data = rep.pop("data", None)
            if data is None:
                continue
            path = _write_replay(game.name, rep["outcome"], rep["side"], data)
            replays_meta.append({k: v for k, v in rep.items() if k != "data"} | {"path": path})

    # KOTH cross-game capture: the match worker only records the FIGHTER viewer
    # replay (different obs space + fighter-only recorder), so for KOTH it returns
    # no frames and ``replays_meta`` is empty above. Capture the SAME representative
    # KOTH matches here on the driver — restore the frozen Students and record with
    # ``record_match_koth`` — when capture was requested for this replicate (the
    # FIRST replicate's FIRST held-out arena, mirroring the worker's capture point)
    # and the worker produced nothing. Reuses the exact ``_write_replay`` sink; the
    # fighter path is untouched (it already captured, so this branch is skipped).
    if r == 0 and not replays_meta and game.name not in _WORKER_REPLAY_GAMES and h2h_payloads:
        first = h2h_payloads[0]
        koth_caps = _capture_koth_replays(
            game.name, first["arena"], trained_policy, base_policy, match_seeds,
        )
        for rep in koth_caps:
            data = rep.pop("data", None)
            if data is None:
                continue
            path = _write_replay(game.name, rep["outcome"], rep["side"], data)
            replays_meta.append({k: v for k, v in rep.items() if k != "data"} | {"path": path})

    total = trained_wins + base_wins + draws
    adv = aggregate.paired_advantage(trained_wins, base_wins, draws)
    per_family_advantage = {
        fam: aggregate.paired_advantage(c["t"], c["b"], c["d"])["advantage"]
        for fam, c in per_family_counts.items()
    }

    return {
        "replicate": r,
        "base_curriculum": [dict(a) for a in base_gen.arenas],
        "trained_curriculum": [dict(a) for a in trained_gen.arenas],
        "base_student_checksum": base_policy["checksum"],
        "trained_student_checksum": trained_policy["checksum"],
        "base_student_train_winrate": round(float(base_row.get("train_arena_winrate", 0.0)), 4),
        "trained_student_train_winrate": round(float(trained_row.get("train_arena_winrate", 0.0)), 4),
        "held_out_overlap": overlap,
        "trained_win_rate": adv["trained_win_rate"],
        "base_win_rate": adv["base_win_rate"],
        "draw_rate": adv["draw_rate"],
        "paired_advantage": adv["advantage"],
        "n_matches": total,
        "per_family_advantage": per_family_advantage,
        "replays": replays_meta,
        "clamp": {
            "base": {"total": base_gen.total, "invalid": base_gen.invalid, "clamped": base_gen.clamped},
            "trained": {"total": trained_gen.total, "invalid": trained_gen.invalid, "clamped": trained_gen.clamped},
        },
        # Private accumulators (stripped before JSON).
        "_trained_wins": trained_wins, "_base_wins": base_wins, "_draws": draws,
        "_total_matches": total,
        "_side_p0_wins": p0_wins, "_side_p0_total": p0_total,
        "_side_p1_wins": p1_wins, "_side_p1_total": p1_total,
    }


def _student_payload(teacher: str, curriculum_id: str, arenas: list[dict],
                     cfg: EvalConfig, game: str) -> dict:
    """Build the train-Student payload (the SAME budget knobs for both Teachers)."""
    payload = {
        "game": game,
        "teacher": teacher,
        "curriculum_id": curriculum_id,
        "ppo_episodes": int(cfg.ppo_episodes),
        "eval_seeds": int(cfg.eval_seeds),
        "architecture": cfg.architecture,
        "net_arch": list(cfg.net_arch),
        "curriculum_arenas": [dict(a) for a in arenas],
    }
    # Opponent league (default OFF — a no-op when disabled). Applied here, in the
    # SINGLE Student-payload builder, so the base and trained Students receive the
    # IDENTICAL league config by construction (the fairness invariant).
    return cfg.apply_opponent_league(payload)


def _print_primary_verdict(summary: dict) -> None:
    print("\n    === PRIMARY VERDICT (Student vs Student) ===")
    print(f"    trained-Student win rate = {summary['trained_student_win_rate']:.4f}")
    print(f"    base-Student    win rate = {summary['base_student_win_rate']:.4f}")
    print(f"    draw rate                = {summary['draw_rate']:.4f}")
    print(f"    mean paired advantage    = {summary['mean_paired_advantage']:+.4f}")
    ci = summary["ci_t"]
    print(f"    curriculum-level 95% CI  = [{ci['low']:+.4f}, {ci['high']:+.4f}]  (n={ci['n']} replicates)")
    print(f"    trained win-rate P0={summary['trained_win_rate_as_p0']:.3f}  "
          f"P1={summary['trained_win_rate_as_p1']:.3f}")
    print(f"    anti-circularity passed  = {summary['anti_circularity']['passed']}  "
          f"(failed: {summary['anti_circularity']['failed']})")
    print(f"    PRIMARY PASS             = {summary['primary_pass']}")
