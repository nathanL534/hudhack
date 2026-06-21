"""output/stage6/games.py — the Stage-6 game registry / router.

Stage 6 must run the base-vs-trained decider over MULTIPLE games, with a clean
adapter interface per game, so a new game plugs in WITHOUT touching the evaluator.
A game entry pins everything the pipeline needs:

  * ``teacher_game``      — the ``engine.games.Game`` the Teacher is prompted with
                            (its ``param_schema`` is the prompt + clamp surface).
  * ``param_keys``        — the float knobs that flow into the Modal worker payload.
  * ``build_held_out``    — builds the FIXED held-out eval population (the yardstick).
  * ``installed``         — whether the underlying game modules are importable. If
                            a game is declared but not installed, the evaluator
                            reports "not installed" instead of crashing.
  * ``role``              — ``teacher`` (eligible to TRAIN a Teacher on) or
                            ``probe`` (held-out transfer probe ONLY, never trained).

Games
-----
* ``fighter`` (Ring-Out Duel) — FULLY supported now; the headline Teacher game.
* ``target_knockback`` (TK / "Game-2") — adapter interface DEFINED but the modules
  are absent in this worktree, so it resolves as ``installed=False``. We do NOT
  copy or merge the Game-2 branch here; we only declare the seam it will plug into.
* ``koth`` (King of the Hill) — registered as a held-out transfer PROBE only
  (``role="probe"``), never as a Teacher training game.

The router (``get_game`` / ``installed_games`` / ``teacher_games`` /
``probe_games``) is the only thing the evaluator imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Game entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameEntry:
    name: str
    role: str                                   # "teacher" | "probe"
    installed: bool
    param_keys: tuple[str, ...]                 # float knobs sent to the worker
    description: str = ""
    not_installed_reason: str = ""
    # Lazily-built so importing the registry never pulls heavy game modules.
    _teacher_game: Optional[Callable[[], object]] = None
    _build_held_out: Optional[Callable[..., list[dict]]] = None
    _disjoint_guard: Optional[Callable[..., None]] = None
    _payload_arenas: Optional[Callable[[list[dict]], list[dict]]] = None

    # -- accessors that raise a clean error when the game is absent -----------

    def teacher_game(self) -> object:
        self._require_installed()
        assert self._teacher_game is not None
        return self._teacher_game()

    def build_held_out(self, **kw) -> list[dict]:
        self._require_installed()
        assert self._build_held_out is not None
        return self._build_held_out(**kw)

    def disjoint_guard(self, arenas: list[dict], training_difficulties) -> None:
        if self._disjoint_guard is not None:
            self._disjoint_guard(arenas, training_difficulties)

    def payload_arenas(self, arenas: list[dict]) -> list[dict]:
        if self._payload_arenas is not None:
            return self._payload_arenas(arenas)
        return [dict(a) for a in arenas]

    def _require_installed(self) -> None:
        if not self.installed:
            raise GameNotInstalled(
                f"game {self.name!r} is not installed: "
                f"{self.not_installed_reason or 'modules absent in this worktree'}"
            )


class GameNotInstalled(RuntimeError):
    """Raised when a declared game's modules are not present in this worktree."""


# ---------------------------------------------------------------------------
# fighter (Ring-Out Duel) — fully supported
# ---------------------------------------------------------------------------

# The Teacher's FIGHTER schema (identical to the legacy evaluator's bounds).
FIGHTER_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "gravity": (0.2, 1.2),
    "knockback": (0.5, 6.0),
    "spawn_gap": (1.0, 12.0),
}
FIGHTER_PARAM_KEYS = ("difficulty", "platform_width", "gravity", "knockback", "spawn_gap")


def _fighter_teacher_game() -> object:
    from engine.games import Game

    defaults = {
        "difficulty": 0.5, "platform_width": 12.0, "gravity": 0.6,
        "knockback": 2.5, "spawn_gap": 4.0,
    }
    return Game(
        name="fighter",
        param_schema=FIGHTER_BOUNDS,
        to_gridworld_params=lambda p: dict(p),
        description="Platform-fighter arena (Ring-Out Duel; Crucible Teacher target).",
        defaults=defaults,
    )


def _fighter_held_out(*, grid: str = "full") -> list[dict]:
    from output.broad_eval_set import build_broad_eval_arenas

    return build_broad_eval_arenas(grid=grid)


def _fighter_disjoint(arenas: list[dict], training_difficulties) -> None:
    from output.broad_eval_set import assert_disjoint_from_training

    assert_disjoint_from_training(arenas, tuple(training_difficulties))


def _fighter_payload_arenas(arenas: list[dict]) -> list[dict]:
    from output.broad_eval_set import payload_arenas

    return payload_arenas(arenas)


def _fighter_installed() -> bool:
    try:
        import games.fighter  # noqa: F401
        import output.broad_eval_set  # noqa: F401

        return True
    except Exception:  # pragma: no cover - fighter is always present here
        return False


# ---------------------------------------------------------------------------
# koth (King of the Hill) — held-out PROBE only, never a Teacher training game
# ---------------------------------------------------------------------------


def _koth_installed() -> bool:
    try:
        import games.koth  # noqa: F401
        import harness.koth_adapter  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


# KOTH's REAL param schema (what a KOTH Student trains on / a held-out arena clamps
# to). Used by ``generate_arenas`` to validate+clamp the Teacher-mapped curricula.
KOTH_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "zone_half": (0.6, 4.0),
    "zone_center_frac": (0.2, 0.8),
}


def _koth_teacher_game() -> object:
    """The KOTH 'teacher game' — only its ``param_schema`` is used (for clamping the
    KOTH-mapped curricula). The cross-game mapping Teacher generates in the FIGHTER
    schema and maps the result, so this schema validates the MAPPED KOTH arenas."""
    from engine.games import Game

    defaults = {"difficulty": 0.5, "platform_width": 12.0, "zone_half": 1.5,
                "zone_center_frac": 0.5}
    return Game(
        name="koth",
        param_schema=KOTH_BOUNDS,
        to_gridworld_params=lambda p: dict(p),
        description="King of the Hill cross-game (zone geometry).",
        defaults=defaults,
    )


def _koth_held_out(*, grid: str = "full") -> list[dict]:
    """The FIXED KOTH held-out arena population (real KOTH params, structurally diverse).

    KOTH is the cross-game test: FRESH KOTH Students (obs_dim=16) are trained from
    each Teacher's KOTH-mapped curricula and fought head-to-head HERE. These specs
    are in KOTH's REAL param schema (platform_width + zone_half + zone_center_frac +
    difficulty), varied across BOTH zone geometry and difficulty so a per-arena-
    family breakdown is meaningful. ``diagonal`` is the cheap 4-arena set; ``full``
    is the 8-arena population for the headline cross-game run. Disjoint by
    construction from the Teacher-mapped curricula (which centre on the default
    width-12 mapping at the Teacher's difficulties).
    """
    families = (
        {"name": "koth_tight", "platform_width": 9.0, "zone_half": 1.0, "zone_center_frac": 0.5},
        {"name": "koth_wide", "platform_width": 16.0, "zone_half": 2.2, "zone_center_frac": 0.5},
        {"name": "koth_offcentre", "platform_width": 12.0, "zone_half": 1.4, "zone_center_frac": 0.68},
        {"name": "koth_pinhole", "platform_width": 13.0, "zone_half": 0.8, "zone_center_frac": 0.5},
    )
    diffs = (0.35, 0.65)
    if grid == "diagonal":
        return [dict(families[i], difficulty=diffs[i % len(diffs)]) for i in range(len(families))]
    return [dict(f, difficulty=d) for f in families for d in diffs]


def _koth_payload_arenas(arenas: list[dict]) -> list[dict]:
    return [{k: v for k, v in a.items() if k != "name"} for a in arenas]


# ---------------------------------------------------------------------------
# target_knockback (TK / "Game-2") — adapter interface DECLARED, modules ABSENT
# ---------------------------------------------------------------------------


def _tk_installed() -> bool:
    """TK ships as ``games/target_knockback.py`` + ``harness/target_knockback_adapter.py``.

    Neither exists in this worktree (Game-2 lives on a separate branch we must NOT
    merge here), so this returns False and the evaluator reports "not installed".
    The moment those two modules land, this flips to True with no evaluator change.
    """
    try:
        import games.target_knockback  # noqa: F401
        import harness.target_knockback_adapter  # noqa: F401

        return True
    except Exception:
        return False


def _tk_teacher_game() -> object:  # pragma: no cover - TK not installed here
    """The TK Teacher game seam.

    When TK lands it should expose ``games.target_knockback.TK_BOUNDS`` (a 5-knob
    param schema) and a ``TargetKnockbackGameAdapter`` mirroring the fighter/KOTH
    adapters. This builder wraps that schema into an ``engine.games.Game`` exactly
    like the fighter, so the Teacher prompt path is unchanged.
    """
    from engine.games import Game
    from games.target_knockback import TK_BOUNDS, TK_DEFAULTS  # type: ignore

    return Game(
        name="target_knockback",
        param_schema=TK_BOUNDS,
        to_gridworld_params=lambda p: dict(p),
        description="Target Knockback (Game-2).",
        defaults=TK_DEFAULTS,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, GameEntry]:
    fighter_ok = _fighter_installed()
    koth_ok = _koth_installed()
    tk_ok = _tk_installed()
    return {
        "fighter": GameEntry(
            name="fighter",
            role="teacher",
            installed=fighter_ok,
            param_keys=FIGHTER_PARAM_KEYS,
            description="Ring-Out Duel — the headline Teacher training game.",
            not_installed_reason="" if fighter_ok else "games.fighter / output.broad_eval_set import failed",
            _teacher_game=_fighter_teacher_game,
            _build_held_out=_fighter_held_out,
            _disjoint_guard=_fighter_disjoint,
            _payload_arenas=_fighter_payload_arenas,
        ),
        "koth": GameEntry(
            name="koth",
            role="probe",
            installed=koth_ok,
            param_keys=("difficulty", "platform_width", "zone_half", "zone_center_frac"),
            description="King of the Hill — cross-game Student-vs-Student test (fresh KOTH students).",
            not_installed_reason="" if koth_ok else "games.koth / harness.koth_adapter import failed",
            _teacher_game=_koth_teacher_game,
            _build_held_out=_koth_held_out,
            _payload_arenas=_koth_payload_arenas,
        ),
        "target_knockback": GameEntry(
            name="target_knockback",
            role="teacher",
            installed=tk_ok,
            param_keys=("difficulty",),  # placeholder until the real schema lands
            description="Target Knockback (Game-2) — adapter seam declared; modules absent.",
            not_installed_reason="games.target_knockback + harness.target_knockback_adapter "
            "absent in this worktree (Game-2 branch not merged)",
            _teacher_game=_tk_teacher_game,
        ),
    }


_REGISTRY: dict[str, GameEntry] = _build_registry()

# Public aliases so callers can say "ring-out-duel" or "tk".
_ALIASES = {
    "ring-out-duel": "fighter",
    "ring_out_duel": "fighter",
    "tk": "target_knockback",
    "game2": "target_knockback",
    "king-of-the-hill": "koth",
}


def get_game(name: str) -> GameEntry:
    key = _ALIASES.get(name, name)
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown game {name!r} (known: {known})")
    return _REGISTRY[key]


def all_games() -> dict[str, GameEntry]:
    return dict(_REGISTRY)


def installed_games() -> list[str]:
    return [n for n, e in _REGISTRY.items() if e.installed]


def teacher_games() -> list[str]:
    """Games eligible to TRAIN a Teacher on (installed AND role=='teacher')."""
    return [n for n, e in _REGISTRY.items() if e.installed and e.role == "teacher"]


def probe_games() -> list[str]:
    """Held-out transfer PROBE games (role=='probe')."""
    return [n for n, e in _REGISTRY.items() if e.role == "probe"]
