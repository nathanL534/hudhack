"""record_batch.py — generate a DIVERSE, uniquely-named set of fighter replays.

The single-replay recorder (``record_replay.py``) only ever writes 3 fixed
filenames, so re-running it overwrites instead of accumulating — the viewer
therefore only ever shows 3 traces. This script reuses the same
``record_match`` machinery but sweeps difficulty + arena geometry and writes a
UNIQUE filename per match, so the viewer's dynamic ``/replays/`` discovery has a
whole gallery to list.

All matchups here are scripted/parametric/random (instant — no PPO training),
so the full batch lands in a second or two. The trained replay still comes from
``record_replay.py --trained``.

    .venv/bin/python record_batch.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from games.fighter import (
    FighterArena,
    parametric_fighter,
    random_policy,
    scripted_fighter,
)
from record_replay import REPLAYS_DIR, record_match
from replay import validate_replay


def _write(name: str, data: dict) -> Path:
    problems = validate_replay(data)
    if problems:
        raise ValueError(f"replay {name!r} failed validation: {problems}")
    out = REPLAYS_DIR / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    w = data["meta"]["winner"]
    print(f"  wrote {out.relative_to(REPLAYS_DIR.parent)}  "
          f"({data['meta']['frames']} frames, winner={w})")
    return out


def main() -> int:
    print(f"Recording batch into {REPLAYS_DIR}/")

    # 1) Difficulty sweep: full-strength scripted p1 vs a parametric p2 whose
    #    competence scales with arena.difficulty. Shows the SAME hero against
    #    progressively tougher opponents — the most legible "what difficulty
    #    means" gallery for a judge.
    for d in (0.2, 0.4, 0.6, 0.8, 1.0):
        arena = FighterArena(difficulty=d)
        data = record_match(
            arena,
            scripted_fighter(arena, ego=0),
            parametric_fighter(arena, ego=1),
            config=f"d{d:.1f}",
            p1_policy="scripted",
            p2_policy=f"parametric@{d:.1f}",
            seed=11 + int(d * 10),
        )
        _write(f"sweep_scripted_vs_d{int(d * 100):03d}", data)

    # 2) Geometry variety: same matchup, different stages (narrow ledge, wide
    #    arena, low gravity floaty, heavy knockback). Visually distinct stages.
    geometries = {
        "narrow_ledge": FighterArena(platform_width=6.0, spawn_gap=2.5, difficulty=0.7),
        "wide_arena":   FighterArena(platform_width=16.0, spawn_gap=6.0, difficulty=0.7),
        "floaty_lowg":  FighterArena(gravity=0.35, jump_impulse=2.6, difficulty=0.7),
        "heavy_knock":  FighterArena(knockback=4.0, difficulty=0.7),
    }
    for label, arena in geometries.items():
        data = record_match(
            arena,
            scripted_fighter(arena, ego=0),
            parametric_fighter(arena, ego=1),
            config=label,
            p1_policy="scripted",
            p2_policy="parametric@0.7",
            seed=hash(label) % 1000,
        )
        _write(f"stage_{label}", data)

    # 3) Two chaotic random-vs-random matches (cheap entertainment + shows the
    #    untrained baseline flailing — the "before" half of the learning story).
    for seed in (5, 23):
        arena = FighterArena(difficulty=1.0)
        data = record_match(
            arena,
            random_policy(seed=seed),
            random_policy(seed=seed + 100),
            config="random",
            p1_policy="random",
            p2_policy="random",
            seed=seed,
        )
        _write(f"random_vs_random_s{seed:02d}", data)

    n = len(list(REPLAYS_DIR.glob("*.json")))
    print(f"done. {n} total replays now in {REPLAYS_DIR.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
