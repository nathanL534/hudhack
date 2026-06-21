"""output/broad_eval_set.py — the DIVERSE held-out eval population.

WHY this module exists. The nested-RL Teacher reward grades a training arena by
how much a freshly-trained Player TRANSFERS to a FIXED held-out reference set
(``modal_player.HELD_OUT_REFERENCE_DIFFICULTIES`` = two default-geometry turtle
opponents, d=0.75/0.85). A prior multi-seed sweep found a trivial-looking arena
(training d=0.25) scored the PEAK reward. That raises THE open question: is
d=0.25 a genuinely good *curriculum* (its Players transfer to a BROAD range of
unseen fighter tasks), or did it just learn to beat those two specific turtle
references? The 2-opponent, same-archetype, same-physics held-out set is too
narrow to tell them apart.

This builds a STRUCTURALLY DIVERSE held-out population — a grid that varies BOTH
arena PHYSICS (platform_width / gravity / knockback / spawn_gap) AND opponent
strength (difficulty) — so a Player's measured transfer reflects broad skill,
not a narrow benchmark exploit. It is a NEW eval set used ONLY for scoring,
layered behind the worker's ``held_out_arenas`` payload key: it does NOT touch
the reward FORMULA, add any penalty, or change the default reward behavior or the
existing held-out reference set. Every arena here is FIXED and guaranteed not to
equal any training arena (see ``assert_disjoint_from_training``).

Physics bounds are taken from values the codebase already exercises
(``prove_ppo_learns.CONFIG_A/B`` and ``record_batch``'s stage variants), which
span the schema's plausible range:

    platform_width  6.0 .. 16.0   (narrow ledge .. wide arena)
    gravity         0.35 .. 0.8   (floaty .. heavy)
    knockback       1.4 .. 4.0    (light .. heavy hits)
    spawn_gap       2.5 .. 7.0    (cramped .. spread out)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path (script lives in output/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.fighter import FighterArena  # noqa: E402

# Default fighter geometry, for reference / collision tests.
_DEFAULTS = FighterArena()

# Difficulty axis: a BROAD opponent-strength sweep, deliberately spanning the
# learnable band (0.4/0.6) up to a turtle-active opponent (0.8). None of these
# equals either narrow reference difficulty (0.75/0.85), so transfer here is
# transfer to UNSEEN opponent strengths, not the benchmark itself.
DIFFICULTIES: tuple[float, ...] = (0.4, 0.6, 0.8)

# Physics axis: five distinct geometry archetypes spanning the plausible range.
# Each is a NAMED structural variant the Player has never trained on. Geometry is
# given explicitly (no default platform_width=10 row that could collide with a
# training arena's geometry — training arenas are width=11 / default physics).
PHYSICS_VARIANTS: tuple[dict, ...] = (
    {"name": "narrow_ledge", "platform_width": 7.0, "gravity": 0.6, "knockback": 2.5, "spawn_gap": 3.0},
    {"name": "wide_arena", "platform_width": 16.0, "gravity": 0.6, "knockback": 2.0, "spawn_gap": 6.0},
    {"name": "floaty_lowg", "platform_width": 12.0, "gravity": 0.35, "knockback": 2.5, "spawn_gap": 5.0},
    {"name": "heavy_knock", "platform_width": 12.0, "gravity": 0.6, "knockback": 4.0, "spawn_gap": 5.0},
    {"name": "high_grav", "platform_width": 9.0, "gravity": 0.8, "knockback": 1.6, "spawn_gap": 4.5},
)


def _arena_spec(variant: dict, difficulty: float) -> dict:
    """One held-out arena spec (worker-payload shape) from a variant + difficulty."""
    return {
        "name": f"{variant['name']}_d{difficulty}",
        "platform_width": float(variant["platform_width"]),
        "gravity": float(variant["gravity"]),
        "knockback": float(variant["knockback"]),
        "spawn_gap": float(variant["spawn_gap"]),
        "difficulty": float(difficulty),
    }


def build_broad_eval_arenas(
    *,
    difficulties: tuple[float, ...] = DIFFICULTIES,
    physics_variants: tuple[dict, ...] = PHYSICS_VARIANTS,
    grid: str = "diagonal",
) -> list[dict]:
    """Build the diverse FIXED held-out population (a list of arena-spec dicts).

    ``grid``:
      * ``"diagonal"`` (default) — one physics variant per difficulty, cycled, so
        EVERY arena differs in BOTH physics and difficulty. With 5 variants and 3
        difficulties this yields 5 arenas where each pairs a distinct geometry
        with a difficulty (geometry is the fast axis), covering all 5 geometries
        and all 3 difficulties. Cheap (5 eval points) yet structurally diverse.
      * ``"full"`` — the full Cartesian product (variants x difficulties), i.e.
        5 x 3 = 15 arenas. Used for the headline experiment so each training
        difficulty is scored against the SAME broad 15-arena population.

    Determinism: the order is fixed, so the same call always yields byte-identical
    arenas — a stable, non-gameable yardstick.
    """
    if grid == "full":
        arenas = [
            _arena_spec(v, d) for v in physics_variants for d in difficulties
        ]
    elif grid == "diagonal":
        arenas = [
            _arena_spec(v, difficulties[i % len(difficulties)])
            for i, v in enumerate(physics_variants)
        ]
    else:
        raise ValueError(f"unknown grid {grid!r} (use 'diagonal' or 'full')")
    return arenas


# --- collision guard: held-out arenas must never equal a training arena -------

# Training arenas in this experiment come from ``nested_reward._spec_for(d)`` ->
# ArenaSpec(map_size=10, difficulty=d) -> worker payload geometry
# platform_width = 6.0 + 0.5*map_size = 11.0, gravity/knockback/spawn_gap at the
# FighterArena defaults (0.6 / 2.5 / 4.0). So a held-out arena "equals a training
# arena" iff its full (geometry, difficulty) tuple matches a width-11 default-
# physics arena at one of the training difficulties.
_TRAINING_GEOMETRY = (11.0, _DEFAULTS.gravity, _DEFAULTS.knockback, _DEFAULTS.spawn_gap)


def _geometry_tuple(spec: dict) -> tuple[float, float, float, float, float]:
    return (
        float(spec.get("platform_width", _DEFAULTS.platform_width)),
        float(spec.get("gravity", _DEFAULTS.gravity)),
        float(spec.get("knockback", _DEFAULTS.knockback)),
        float(spec.get("spawn_gap", _DEFAULTS.spawn_gap)),
        float(spec["difficulty"]),
    )


def assert_disjoint_from_training(
    arenas: list[dict], training_difficulties: tuple[float, ...]
) -> None:
    """Raise if any held-out arena equals a training arena (full tuple match).

    A held-out arena is "the same as training" only if its WHOLE (geometry,
    difficulty) tuple matches the training geometry at a training difficulty. A
    shared difficulty alone is fine (different physics is still unseen), and a
    shared geometry alone is fine (different opponent strength is still unseen) —
    the eval set's whole job is to probe transfer to unseen *combinations*.
    """
    training_tuples = {(*_TRAINING_GEOMETRY, float(d)) for d in training_difficulties}
    for a in arenas:
        if _geometry_tuple(a) in training_tuples:
            raise ValueError(
                f"held-out arena {a.get('name', a)!r} equals a training arena "
                f"(geometry+difficulty collide): {_geometry_tuple(a)}"
            )


def payload_arenas(arenas: list[dict]) -> list[dict]:
    """Strip the human-readable ``name`` so the list is a clean worker payload."""
    return [{k: v for k, v in a.items() if k != "name"} for a in arenas]


if __name__ == "__main__":  # pragma: no cover - manual inspection
    import json

    full = build_broad_eval_arenas(grid="full")
    diag = build_broad_eval_arenas(grid="diagonal")
    print(f"diagonal: {len(diag)} arenas")
    for a in diag:
        print("  ", a)
    print(f"full: {len(full)} arenas")
    for a in full:
        print("  ", a["name"])
    assert_disjoint_from_training(full, (0.25, 0.55, 0.85))
    print(json.dumps({"diagonal": diag, "full": full}, indent=2))
