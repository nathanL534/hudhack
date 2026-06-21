"""games/opponents_league.py — the Stage-6 Student training "opponent league".

THE PROBLEM THIS FIXES. A Stage-6 fighter Student trained against a SINGLE
opponent learns to exploit that one bot's quirks rather than to fight. We
measured the symptom directly: a 48.5%-draw head-to-head, the signature of a
policy that has memorised how to neutralise one specific opponent (turtle-vs-
turtle stalemates) instead of learning a transferable fighting strategy.

THE FIX. Train each Student against MULTIPLE opponent STYLES drawn from a
weighted league, so no single bot can be over-fit. This module owns three pure,
deterministic pieces of that machinery:

  * ``resolve_league``     — turn a declared league into the ACTIVE league for
                             this run (dropping ``prior_student`` when there is
                             no frozen prior Student to play, renormalising the
                             remaining weights so they sum to ~1.0).
  * ``select_opponent_id`` — a DETERMINISTIC weighted pick from the active
                             league, driven by a passed NumPy ``Generator`` so
                             the same rng state always yields the same id (the
                             whole run replays for a fixed seed).
  * ``make_opponent``      — build an ``obs -> action`` policy for a chosen
                             style, matching ``games/fighter.py``'s opponent
                             contract (``Callable[[np.ndarray], int]``).

REUSE, NOT REINVENTION. Every style here wraps an EXISTING fighter policy — we
do not re-implement the sim or any behaviour:

  * ``aggressive``     -> ``scripted_fighter`` (the unmodified, fully aggressive
                          heuristic: advance toward the opponent, punch in range).
  * ``turtle``         -> ``parametric_fighter(difficulty=1.0)`` (the jump_turtle
                          archetype already living inside the parametric opponent:
                          jump-dodges incoming punches, backs away from edges,
                          refuses the bait — a defensive keep-distance style).
  * ``random``         -> ``random_policy`` (uniform over the Stage-1 actions).
  * ``prior_student``  -> the frozen prior-Student policy passed straight through.

DEFAULT BEHAVIOUR IS UNCHANGED. This module is opt-in: a trainer that keeps
wiring ``parametric_fighter`` directly behaves exactly as before. The league is
only active once a trainer chooses to select through it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from games.fighter import (
    FighterArena,
    Policy,
    parametric_fighter,
    random_policy,
    scripted_fighter,
)

# A Student's opponents always control fighter 1 in FighterEnv/FighterSim, so the
# scripted/parametric styles are built with ego=1 to match the env's wiring.
_OPPONENT_EGO = 1

# The id every league uses for the frozen prior-Student member. Centralised so
# resolve_league's drop test and the DEFAULT_LEAGUE literal never drift apart.
PRIOR_STUDENT_ID = "prior_student"

# The default league: four styles, equal weight. ``prior_student`` is dropped at
# resolve time when no frozen prior Student exists (e.g. the very first Student),
# leaving an equal-weight aggressive/turtle/random league.
DEFAULT_LEAGUE: list[dict] = [
    {"id": "aggressive", "weight": 1},
    {"id": "turtle", "weight": 1},
    {"id": "random", "weight": 1},
    {"id": PRIOR_STUDENT_ID, "weight": 1},
]

# The FOCUSED league (the 80%-draw-backfire fix). The equal-weight DEFAULT_LEAGUE
# above is a documented NEGATIVE CONTROL: turtle taught survival/passivity and
# random supplied weak noisy lessons, splitting the fixed PPO budget across
# incompatible styles -> ~80% draws. The focused league removes BOTH and trains the
# Student on a decisive distribution instead:
#   * 70% aggressive    — the scripted_fighter that actually closes and punches.
#   * 30% prior_student — active self-play vs a frozen earlier Student.
# When no frozen prior Student is supplied, ``resolve_league`` drops the
# prior_student entry and renormalises -> 100% aggressive (the documented fallback,
# never a silent turtle/random substitution).
FOCUSED_LEAGUE: list[dict] = [
    {"id": "aggressive", "weight": 0.70},
    {"id": PRIOR_STUDENT_ID, "weight": 0.30},
]


def resolve_league(league: list[dict], has_prior_student: bool) -> list[dict]:
    """Resolve a declared league into the ACTIVE league for this run.

    Drops the ``{"id": "prior_student"}`` member when ``has_prior_student`` is
    False (so a Student with no frozen predecessor never tries to play one), then
    renormalises the remaining weights so they sum to ~1.0. Returns a fresh list
    of ``{"id", "weight"}`` dicts; the input is not mutated.

    The renormalisation makes ``select_opponent_id`` a clean weighted draw over
    whatever members survived, independent of how the raw weights were declared
    (they need not sum to 1 going in).
    """
    members = [dict(m) for m in league]
    if not has_prior_student:
        members = [m for m in members if m.get("id") != PRIOR_STUDENT_ID]

    if not members:
        raise ValueError(
            "resolve_league produced an empty league: every member was dropped. "
            "A league must keep at least one non-prior_student style."
        )

    total = float(sum(float(m.get("weight", 0.0)) for m in members))
    if total <= 0.0:
        # Degenerate input (all weights 0 / missing): fall back to a uniform
        # split so the active league is always a valid probability vector.
        uniform = 1.0 / len(members)
        return [{"id": m["id"], "weight": uniform} for m in members]

    return [{"id": m["id"], "weight": float(m.get("weight", 0.0)) / total} for m in members]


def select_opponent_id(active_league: list[dict], rng: np.random.Generator) -> str:
    """Deterministically pick one opponent id from the active league.

    Weighted by each member's (already-normalised) ``weight``, using the passed
    NumPy ``Generator`` ``rng`` for the single random draw. The pick is a pure
    function of the rng state: two calls from rng's at the same state return the
    same id, and a fixed-seed rng produces an identical id SEQUENCE across runs.
    Weights are re-normalised defensively here too, so this is correct even if
    handed a raw (un-resolved) league.
    """
    if not active_league:
        raise ValueError("select_opponent_id got an empty active_league.")

    ids = [m["id"] for m in active_league]
    weights = np.array([float(m.get("weight", 0.0)) for m in active_league], dtype=np.float64)
    total = weights.sum()
    if total <= 0.0:
        weights = np.full(len(ids), 1.0 / len(ids), dtype=np.float64)
    else:
        weights = weights / total

    # rng.choice over indices keeps the draw on plain ints (avoids any dtype
    # surprises from choosing over a Python-str array) while still being a single
    # deterministic consumption of rng state.
    idx = int(rng.choice(len(ids), p=weights))
    return ids[idx]


def make_opponent(
    opponent_id: str,
    arena: FighterArena,
    *,
    ego: int = _OPPONENT_EGO,
    seed: int,
    prior_student: Optional[Policy] = None,
) -> Policy:
    """Build an ``obs -> action`` policy for ``opponent_id``.

    The returned callable matches ``games/fighter.py``'s opponent contract
    (``Callable[[np.ndarray], int]``) and reuses an existing fighter policy for
    every style — nothing about the sim or the policy contract is reinvented:

      * ``"aggressive"``     advance toward the opponent and punch when in range
                             (the unmodified ``scripted_fighter`` heuristic).
      * ``"turtle"``         defensive / keep-distance: the jump_turtle archetype
                             from ``parametric_fighter`` at full difficulty
                             (jump-dodges punches, backs off edges, refuses bait).
      * ``"random"``         uniform random over the valid Stage-1 actions,
                             seeded so a fixed seed replays.
      * ``"prior_student"``  the frozen prior-Student policy passed as
                             ``prior_student``. If that is None this id should
                             never have been selected (``resolve_league`` drops
                             it when ``has_prior_student`` is False); we raise
                             rather than silently degrade.

    ``seed`` makes the stochastic styles (``random``, and the parametric
    ``turtle``'s per-match dodge draw / dodge rolls) reproducible.
    """
    if opponent_id == "aggressive":
        # scripted_fighter IS the fully-aggressive heuristic: self-preserve at the
        # edge, punch in range, otherwise close distance toward the opponent.
        return scripted_fighter(arena, ego=ego)

    if opponent_id == "turtle":
        # The defensive turtle is the high-difficulty face of parametric_fighter:
        # at difficulty=1.0 epsilon is 0 (no self-edging) and the jump_turtle
        # engages fully — jump-dodging punches and keeping distance. Seeded so the
        # per-match dodge draw and dodge rolls replay for a fixed seed.
        return parametric_fighter(arena, ego=ego, difficulty=1.0, seed=seed)

    if opponent_id == "random":
        return random_policy(seed=seed)

    if opponent_id == PRIOR_STUDENT_ID:
        if prior_student is None:
            raise ValueError(
                "make_opponent('prior_student', ...) requires a non-None "
                "prior_student policy. resolve_league should have dropped this "
                "member when has_prior_student is False."
            )
        return prior_student

    raise ValueError(
        f"Unknown opponent_id {opponent_id!r}. "
        f"Known styles: aggressive, turtle, random, {PRIOR_STUDENT_ID}."
    )


# Re-exported so a trainer can validate a style id against the supported set
# without reaching into make_opponent's branch logic.
KNOWN_OPPONENT_IDS: tuple[str, ...] = ("aggressive", "turtle", "random", PRIOR_STUDENT_ID)
