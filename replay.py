"""replay.py — Stage-1 fighter replay format + writer (ADD-ONLY, no sim edits).

A replay is a self-contained JSON document of one ``ring-out-duel`` match: the
platform geometry plus a per-frame trace of both fighters' positions, facing,
chosen action, and any notable events (punches, ring-outs). It is everything the
``viewer/viewer.html`` canvas needs to animate a match — and nothing about the
sim's internals, so it stays valid even as the fighter physics evolves.

This module ONLY reads the fighter through its public interface (``FighterSim``
positions are sampled, ``play_match``-style stepping). It never imports private
harness helpers and never mutates any existing file.

Schema (v1):

    {
      "meta":     {"game": "ring-out-duel", "config": "A",
                    "winner": "p1" | "p2" | "draw", "frames": N,
                    "p1_policy": "...", "p2_policy": "...", "seed": int,
                    "schema": 1},
      "platform": {"x_left": float, "x_right": float, "y": float},
      "frames":   [ {"p1": {"x": .., "y": .., "facing": 1, "action": "punch"},
                     "p2": {"x": .., "y": .., "facing": -1, "action": "idle"},
                     "events": ["p1_punch", "p2_ringout"]}, ... ]
    }

Frame ``i`` is the post-step state after tick ``i`` (frame 0 = initial spawn,
before any action). Coordinates are in raw sim units (same space as the
platform), so the viewer can map ``[x_left, x_right]`` straight onto the canvas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from games.fighter import Action, FighterArena, FighterSim

SCHEMA_VERSION = 1

# fighter 0 -> "p1", fighter 1 -> "p2" everywhere in the replay.
_ACTION_NAME = {
    int(Action.IDLE): "idle",
    int(Action.LEFT): "left",
    int(Action.RIGHT): "right",
    int(Action.JUMP): "jump",
    int(Action.PUNCH): "punch",
}


def action_name(a: int) -> str:
    """Public action int -> stable lowercase label used in the replay JSON."""
    return _ACTION_NAME.get(int(a), f"a{int(a)}")


@dataclass
class ReplayBuilder:
    """Accumulates per-frame fighter state into a replay dict.

    Construct from a ``FighterArena`` (gives the platform geometry), call
    ``capture`` once per frame with both fighters' bodies and the actions that
    PRODUCED that frame, then ``to_dict``/``write``. Knowing the winner is
    deferred to ``finalize`` because it is only decided at the last tick.
    """

    arena: FighterArena
    config: str = "?"
    p1_policy: str = "p1"
    p2_policy: str = "p2"
    seed: int = 0
    frames: list[dict] = field(default_factory=list)
    _prev_alive: tuple[bool, bool] = (True, True)

    def _body_dict(self, body, action: int) -> dict:
        return {
            "x": round(float(body.x), 4),
            "y": round(float(body.y), 4),
            "facing": int(body.facing),
            "action": action_name(action),
        }

    def capture(self, sim: FighterSim, a0: int, a1: int) -> None:
        """Record one frame from the CURRENT sim state.

        ``a0``/``a1`` are the actions fighters 0/1 took on the tick that led to
        this state (for frame 0 — the spawn — pass IDLE/IDLE). Events are derived
        from the actions (a punch this tick) and from alive-state transitions (a
        fighter that was alive last frame and is dead now => ring-out).
        """
        events: list[str] = []
        if int(a0) == int(Action.PUNCH):
            events.append("p1_punch")
        if int(a1) == int(Action.PUNCH):
            events.append("p2_punch")

        alive = (sim.f0.alive, sim.f1.alive)
        if self._prev_alive[0] and not alive[0]:
            events.append("p1_ringout")
        if self._prev_alive[1] and not alive[1]:
            events.append("p2_ringout")
        self._prev_alive = alive

        self.frames.append(
            {
                "p1": self._body_dict(sim.f0, a0),
                "p2": self._body_dict(sim.f1, a1),
                "events": events,
            }
        )

    def to_dict(self, winner: Optional[int]) -> dict:
        """Assemble the full replay dict. ``winner`` is 0, 1, or None (draw)."""
        winner_label = {0: "p1", 1: "p2", None: "draw"}[winner]
        return {
            "meta": {
                "game": "ring-out-duel",
                "config": self.config,
                "winner": winner_label,
                "frames": len(self.frames),
                "p1_policy": self.p1_policy,
                "p2_policy": self.p2_policy,
                "seed": self.seed,
                "schema": SCHEMA_VERSION,
            },
            "platform": {
                "x_left": 0.0,
                "x_right": round(float(self.arena.platform_width), 4),
                "y": 0.0,
            },
            "frames": self.frames,
        }

    def write(self, path: str | Path, winner: Optional[int]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(winner), indent=2))
        return out


def validate_replay(data: dict) -> list[str]:
    """Cheap structural check. Returns a list of problems ([] == well-formed).

    Not a schema library — just enough to assert a written replay is internally
    consistent (the recorder self-checks with this, and it documents the
    contract the viewer relies on).
    """
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["top-level is not an object"]

    meta = data.get("meta")
    if not isinstance(meta, dict):
        problems.append("missing/invalid 'meta'")
    else:
        if meta.get("winner") not in ("p1", "p2", "draw"):
            problems.append(f"meta.winner invalid: {meta.get('winner')!r}")
        if not isinstance(meta.get("frames"), int):
            problems.append("meta.frames not an int")

    plat = data.get("platform")
    if not isinstance(plat, dict) or not all(k in plat for k in ("x_left", "x_right", "y")):
        problems.append("missing/invalid 'platform'")

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        problems.append("'frames' missing or empty")
        return problems

    if isinstance(meta, dict) and meta.get("frames") != len(frames):
        problems.append(f"meta.frames ({meta.get('frames')}) != len(frames) ({len(frames)})")

    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            problems.append(f"frame {i} not an object")
            continue
        for who in ("p1", "p2"):
            b = fr.get(who)
            if not isinstance(b, dict) or not all(k in b for k in ("x", "y", "facing", "action")):
                problems.append(f"frame {i}.{who} malformed")
        if not isinstance(fr.get("events"), list):
            problems.append(f"frame {i}.events not a list")

    return problems
