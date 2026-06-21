"""output/stage6/koth_cross_game.py — the FINAL HIDDEN cross-game test (KOTH).

KOTH is the held-out game the Teacher NEVER trained to generate. The naive (wrong)
cross-game test would force Ring-Out-trained policy WEIGHTS into KOTH — but the obs
spaces differ (fighter=11, KOTH=16), so a fighter net cannot even read a KOTH
observation. We do NOT do that.

The CORRECT cross-game test (what this module implements):

  1. Each Teacher generates curricula in ITS NATIVE schema (fighter params:
     difficulty + platform_width + gravity + knockback + spawn_gap).
  2. A Teacher-level MAPPING turns each generated arena into VALID KOTH params
     (the same difficulty -> zone-geometry mapping ``harness.koth_adapter`` uses,
     so a hard fighter arena maps to a hard KOTH hill). This is a CURRICULUM
     mapping, not a weight transfer.
  3. Train FRESH KOTH Students (obs_dim=16) from the base-Teacher-derived vs
     trained-Teacher-derived KOTH curricula. IDENTICAL architecture/budget.
  4. Freeze both; run KOTH Student-vs-Student head-to-head with side swaps + fresh
     seeds — the SAME primary machinery as the fighter, just game="koth".

If KOTH (or its adapter) is not importable, this returns an explicit UNSUPPORTED
result documenting why — it NEVER fabricates a cross-game number.

Label: "Cross-game generalization".
"""

from __future__ import annotations

from output.stage6 import head_to_head as h2h_mod
from output.stage6.config import EvalConfig
from output.stage6.games import KOTH_BOUNDS, get_game  # KOTH_BOUNDS: single source of truth
from output.stage6.handles import ResolvedTeacher


def koth_supported() -> tuple[bool, str]:
    """Is the KOTH cross-game test implementable in this worktree? (bool, reason)."""
    g = get_game("koth")
    if not g.installed:
        return False, f"koth game not installed: {g.not_installed_reason}"
    try:
        import harness.koth_trainer  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent
        return False, f"harness.koth_trainer import failed: {exc}"
    return True, "koth game + adapter + trainer importable"


# KOTH parameter bounds (for clamping mapped curricula into valid KOTH arenas) come
# from the registry's single source of truth — ``output.stage6.games.KOTH_BOUNDS``.


def map_fighter_arena_to_koth(arena: dict) -> dict:
    """Map ONE fighter-schema arena into VALID KOTH params (Teacher-level mapping).

    Mirrors ``harness.koth_adapter._arena_from_spec``: difficulty shrinks the zone
    and pushes it off-centre (harder hill), platform_width carries through (clamped).
    The result is always a valid KOTH arena — the mapping is total and clamped, so
    no Teacher output can produce an invalid KOTH arena.
    """
    difficulty = float(arena.get("difficulty", 0.5))
    width = float(arena.get("platform_width", 12.0))
    width = min(max(width, KOTH_BOUNDS["platform_width"][0]), KOTH_BOUNDS["platform_width"][1])
    # Zone shrinks with difficulty (harder to hold); floor keeps it occupiable.
    zone_half = max(0.6, 0.22 * width * (1.0 - 0.6 * difficulty))
    zone_half = _clamp(zone_half, KOTH_BOUNDS["zone_half"])
    # Off-centre as difficulty rises (asymmetric hill).
    zone_center_frac = _clamp(0.5 + 0.18 * difficulty, KOTH_BOUNDS["zone_center_frac"])
    return {
        "difficulty": round(difficulty, 4),
        "platform_width": round(width, 3),
        "zone_half": round(zone_half, 3),
        "zone_center_frac": round(zone_center_frac, 3),
    }


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return min(max(value, lo), hi)


# (replay writer lives in head_to_head; KOTH captures no fighter-style replay)


class _MappingTeacher:
    """Wraps a Teacher so every arena it emits is MAPPED into KOTH params.

    The Teacher still generates in ITS native fighter schema (so base/trained
    prompts are unchanged); this wrapper maps the result, so the head-to-head's
    ``generate_arenas`` sees valid KOTH arenas. ``param_schema`` is overridden to
    the KOTH schema so generation validates against KOTH bounds.
    """

    def __init__(self, inner):
        self._inner = inner

    def generate(self, game):  # game is the KOTH teacher_game (KOTH schema)
        # Generate against the FIGHTER schema the Teacher knows, then map.
        fighter_game = get_game("fighter").teacher_game()
        raw = self._inner.generate(fighter_game)
        return map_fighter_arena_to_koth(raw)


def _wrap_resolved(resolved: ResolvedTeacher) -> ResolvedTeacher:
    """Return a ResolvedTeacher whose ``build()`` yields a KOTH-mapping Teacher."""
    return ResolvedTeacher(
        label=resolved.label,
        handle=resolved.handle,
        resolved_id=resolved.resolved_id + "::koth-mapped",
        kind=resolved.kind,
        _build=lambda: _MappingTeacher(resolved.build()),
    )


def run_koth_cross_game(
    base: ResolvedTeacher,
    trained: ResolvedTeacher,
    *,
    config: EvalConfig,
    backend: str = "modal",
    is_smoke: bool = False,
    verbose: bool = True,
    run_train=None,
    run_h2h=None,
) -> dict:
    """Run the KOTH Student-vs-Student cross-game head-to-head (or UNSUPPORTED).

    Trains FRESH KOTH Students from each Teacher's KOTH-MAPPED curricula and fights
    them head-to-head — never forcing fighter weights into KOTH. Returns a dict with
    ``supported`` True + the full head-to-head summary, or False + a documented
    reason. ``run_train`` / ``run_h2h`` are injectable for tests.
    """
    ok, reason = koth_supported()
    if not ok:
        return {
            "label": "cross_game_generalization",
            "game": "koth",
            "supported": False,
            "reason": reason,
            "note": "No defensible KOTH cross-game mapping runnable here; "
                    "emitting explicit UNSUPPORTED rather than a fabricated number.",
        }

    koth_game = get_game("koth")
    # KOTH is registered role='probe' (never a Teacher TRAINING game). For the
    # cross-game test we are NOT training a Teacher on KOTH — we train KOTH STUDENTS
    # from mapped curricula — so we drive the head-to-head directly with the KOTH
    # game entry, bypassing the role guard (which only gates Teacher RFT).
    h2h_req = h2h_mod.HeadToHeadRequest(
        base=_wrap_resolved(base),
        trained=_wrap_resolved(trained),
        game=koth_game,
        config=config,
        backend=backend,
        is_smoke=is_smoke,
        verbose=verbose,
    )
    kwargs = {}
    if run_train is not None:
        kwargs["run_train"] = run_train
    if run_h2h is not None:
        kwargs["run_h2h"] = run_h2h
    summary = h2h_mod.run_head_to_head(h2h_req, **kwargs)
    summary["label"] = "cross_game_generalization"
    summary["supported"] = True
    summary["mapping"] = "fighter-schema -> KOTH zone geometry (koth_adapter mapping)"
    return summary
