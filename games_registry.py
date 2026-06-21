"""games_registry.py — the SHARED Teacher game schema/router (multi-game seam).

WHY THIS EXISTS
---------------
Crucible started as ONE game (Ring-Out, the Stage-1 2D platform fighter). Its
Teacher schema (the 5-knob FIGHTER param set), its HUD grader, its nested-RL
reward, and its RFT dataset row were all hard-wired to "the fighter" with no
discriminator. Adding a SECOND game (Target Knockback) the wrong way would mean
forking each of those — two schemas drifting apart.

This module is the RIGHT seam: a tiny registry keyed by a ``game`` discriminator.
Each game registers ONE :class:`GameSpec` describing everything the shared
Teacher/HUD/EP machinery needs to route to it:

  * ``game``            — the stable discriminator string (e.g. ``"ring_out"``).
  * ``prompt``          — the Teacher curriculum-design prompt for this game.
  * ``bounds``          — the param schema (key -> (low, high)) the boundary
                          validator clamps to. This IS the per-game param range.
  * ``modal_app`` /
    ``modal_fn``        — the ISOLATED Modal PPO worker that trains this game's
                          Players (Ring-Out -> ``crucible-player``; Target
                          Knockback -> ``crucible-player-tk``). The two reward
                          paths never touch because each game names its own app.
  * ``bridge_app`` /
    ``bridge_url_env``  — the deployed EP bridge app + the env var the evaluator
                          reads to find it. Each game gets its own bridge so a
                          live RFT on one game is never disturbed by the other.

Nothing here imports a game engine or torch — it is pure metadata + the schema
router. The Ring-Out wiring (training/hud_teacher_env.py, training/modal_ep_bridge.py,
training/rft_evaluator.py once it lands) can adopt this registry incrementally; the
Target-Knockback wiring is built ON it from day one (the second game proves the
seam). Registering a third game is one ``register(GameSpec(...))`` call — no fork.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A param schema: key -> (low, high). The boundary validator clamps every emitted
# key into its range; a missing key is rejected (never silently defaulted).
Bounds = dict[str, tuple[float, float]]


@dataclass(frozen=True)
class GameSpec:
    """Everything the shared Teacher/HUD/EP router needs to serve ONE game.

    Frozen + serializable-by-field so a spec can be logged or shipped. The actual
    scorer/curriculum-mapper are NOT stored here (they pull in heavy engine/torch
    deps); they live in the per-game wiring module, keyed by ``game``. This keeps
    the registry import cheap (a credential-free template smoke can read the bounds
    + prompt without importing numpy/gymnasium/torch).
    """

    game: str
    prompt: str
    bounds: Bounds
    # The isolated Modal PPO worker for this game's nested reward. The two games'
    # reward paths coexist precisely because each names its OWN app/function.
    modal_app: str
    modal_fn: str
    # The deployed EP bridge app for this game + the env var the RFT evaluator reads
    # to locate it. Each game's RFT job targets its own bridge.
    bridge_app: str
    bridge_url_env: str
    # Human-facing description used in HUD task framing / logs.
    description: str = ""
    # Geometry keys (everything except the opponent-strength ``difficulty`` dial)
    # that are threaded VERBATIM into the Modal worker payload. Per game because the
    # geometry differs (fighter physics vs TK target-zone).
    geometry_keys: tuple[str, ...] = field(default_factory=tuple)

    def bounds_list(self) -> dict[str, list[float]]:
        """Bounds as the JSON-friendly ``{key: [low, high]}`` shape HUD wants."""
        return {k: [lo, hi] for k, (lo, hi) in self.bounds.items()}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, GameSpec] = {}


def register(spec: GameSpec) -> GameSpec:
    """Register a game. Idempotent re-register of the SAME spec is allowed; a
    conflicting re-register (same id, different spec) is an error so two games can
    never silently clobber each other's routing."""
    existing = _REGISTRY.get(spec.game)
    if existing is not None and existing != spec:
        raise ValueError(
            f"game {spec.game!r} already registered with a different spec; "
            "two games must not share a discriminator"
        )
    _REGISTRY[spec.game] = spec
    return spec


def get(game: str) -> GameSpec:
    """Resolve a game spec by its discriminator, or raise with the known set."""
    try:
        return _REGISTRY[game]
    except KeyError:
        raise KeyError(
            f"unknown game {game!r}; registered games: {sorted(_REGISTRY)}"
        ) from None


def games() -> list[str]:
    """The registered game discriminators (sorted, stable)."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

# Ring-Out (game #1): the Stage-1 2D platform fighter. The schema/prompt/worker
# here MIRROR the existing fighter wiring (training/hud_teacher_env.FIGHTER_BOUNDS,
# the crucible-player worker, the deployed crucible-ep-bridge). Registered so the
# router knows about BOTH games; the fighter's own modules keep working untouched
# and can read their bounds from here when they choose to.
RING_OUT = register(
    GameSpec(
        game="ring_out",
        prompt=(
            "You design RL training curricula for a Stage-1 2D platform fighter "
            "(the Ring-Out duel: two fighters, knock the opponent off the platform). "
            "Return ONE JSON object with EXACTLY these keys, each a number in range: "
            "difficulty [0.0,1.0], platform_width [8.0,30.0], gravity [0.2,1.2], "
            "knockback [0.5,6.0], spawn_gap [1.0,12.0]. Choose a LEARNABLE arena whose "
            "trained Player transfers broadly. Output strict JSON only; no prose, no code."
        ),
        bounds={
            "difficulty": (0.0, 1.0),
            "platform_width": (8.0, 30.0),
            "gravity": (0.2, 1.2),
            "knockback": (0.5, 6.0),
            "spawn_gap": (1.0, 12.0),
        },
        modal_app="crucible-player",
        modal_fn="train_player_transfer",
        bridge_app="crucible-ep-bridge",
        bridge_url_env="CRUCIBLE_BRIDGE_URL",
        description="Stage-1 2D platform fighter; knock the opponent off the platform.",
        geometry_keys=("platform_width", "gravity", "knockback", "spawn_gap"),
    )
)

# Target Knockback (game #2): same physics engine, different objective — score for
# every tick the OPPONENT is in a marked target zone. The Teacher turns the SAME
# difficulty dial plus the fighter physics knobs AND the two TK-specific zone dials
# (zone_half / zone_center_frac). Its reward fans out on the ISOLATED
# crucible-player-tk worker, behind its OWN bridge — so it coexists with an in-flight
# Ring-Out RFT without touching it. d=0.55 is the validated clean learnable band
# (tk_modal_scale_validation_results.json).
TARGET_KNOCKBACK = register(
    GameSpec(
        game="target_knockback",
        prompt=(
            "You design RL training curricula for a 2D platform game called Target "
            "Knockback: two fighters on a platform with a marked TARGET ZONE; you score "
            "for every tick your OPPONENT is knocked into the zone (punch them in, they "
            "jump to escape). Return ONE JSON object with EXACTLY these keys, each a "
            "number in range: difficulty [0.0,1.0], platform_width [8.0,30.0], "
            "gravity [0.2,1.2], knockback [0.5,6.0], spawn_gap [1.0,12.0], "
            "zone_half [0.6,4.0], zone_center_frac [0.3,0.7]. Choose a LEARNABLE arena "
            "whose trained Player transfers broadly. Output strict JSON only; no prose, "
            "no code."
        ),
        bounds={
            "difficulty": (0.0, 1.0),
            "platform_width": (8.0, 30.0),
            "gravity": (0.2, 1.2),
            "knockback": (0.5, 6.0),
            "spawn_gap": (1.0, 12.0),
            "zone_half": (0.6, 4.0),
            "zone_center_frac": (0.3, 0.7),
        },
        modal_app="crucible-player-tk",
        modal_fn="train_tk_player",
        bridge_app="crucible-ep-bridge-tk",
        bridge_url_env="CRUCIBLE_TK_BRIDGE_URL",
        description=(
            "2D platform game; punch/knock the opponent INTO a target zone and keep "
            "them there. Same physics engine as Ring-Out, different objective."
        ),
        geometry_keys=(
            "platform_width",
            "gravity",
            "knockback",
            "spawn_gap",
            "zone_half",
            "zone_center_frac",
        ),
    )
)
