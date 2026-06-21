"""games/target_knockback.py — Target Knockback on the SAME engine as the fighter.

This is Teacher-training game #2: a structurally-different objective built on the
fighter's exact physics. A Teacher that learns to generate learnable Ring-Out Duel
arenas (``games/fighter.py``) should also be trainable on, and generalise across,
this game. Target Knockback reuses the fighter's 2D physics, its EXACT five actions
(idle/left/right/jump/punch), and the same structured-observation style — but the
WIN CONDITION is different again from both prior games:

  * Ring-Out Duel (fighter): knock the OPPONENT off the platform; match ends the
    instant someone leaves ``[0, width]``.
  * King of the Hill (koth): YOU accumulate score for every tick YOU occupy a
    zone; most self-occupancy wins.
  * Target Knockback (here): there is a marked TARGET ZONE on the platform. You
    score for every tick the **OPPONENT** is inside it. The objective is to
    PUNCH/KNOCK the opponent INTO the zone and keep them there — you score off the
    *other* player's position, not your own. There is NO ring-out (players are
    clamped to the platform, never eliminated). The match runs the full step
    budget; the winner is whoever banked the most "knock-in" time on their rival.

Why this is a genuinely different game (not KotH relabelled): in KotH the reward
comes from the controlling player's OWN position, so the policy optimises "get me
to the zone and hold". Here the reward comes from the OPPONENT's position, so the
optimal policy is "manoeuvre to the side of the opponent away from the zone, then
punch the opponent ACROSS into it" — knockback is the *primary* scoring tool, not
a defensive shove. Jump becomes the opponent's escape (airborne dodges the
knockback and banks no knock-in time), so the same five actions carry yet another
distinct trade-off.

Engine reuse (NOT a fork): we import the fighter's ``Action`` / ``STAGE1_ACTIONS``
and reuse a single ``FighterSim`` instance as the physics core, exactly as
``games/koth.py`` does. Movement, jumping, facing, punch/knockback and gravity are
the fighter's own helpers — Target Knockback only changes (a) edge handling (clamp
instead of ring-out) and (b) the outcome rule (knock-in time instead of last-alive).

Coordinate system (inherited from the fighter): x increases to the right; the
platform is ``[0, platform_width]`` at ground height ``y = 0``; ``y > 0`` is
airborne. The target zone is the segment ``[zone_center - zone_half, zone_center +
zone_half]``. The opponent "is in the target" when its x is inside that segment AND
it is on the ground (an airborne opponent banks no knock-in time for you — being
mid-jump is the dodge, which is what makes JUMP a real escape rather than a free
way to sit in the target while safe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Reuse the fighter's action set + physics core verbatim. This is the literal
# "same engine" claim: Target Knockback does not redefine an action enum or a
# body; it drives the very objects the fighter uses.
from games.fighter import (
    STAGE1_ACTIONS,
    Action,
    FighterArena,
    FighterSim,
    Policy,
    _Body,
)

try:  # Gymnasium is a dependency, but keep the sim importable without it.
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_BASE = gym.Env
except Exception:  # pragma: no cover - exercised only if gymnasium is missing.
    gym = None
    spaces = None
    _GYM_BASE = object


# ---------------------------------------------------------------------------
# Arena (parametric — same param-schema style as FighterArena / KothArena)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetKnockbackArena:
    """Configurable parameters for one Target-Knockback match.

    Mirrors ``FighterArena`` / ``KothArena``'s style: a small set of geometry dials
    plus a single ``difficulty`` knob the Teacher turns. The zone dials
    (``zone_center_frac`` / ``zone_half``) are this game's analogue of the KotH zone
    — they are what makes an arena easy or hard to score in, and they are exactly
    what a Teacher generates when it emits a Target-Knockback curriculum.

    The physics dials (``platform_width`` / ``gravity`` / ``knockback`` /
    ``spawn_gap`` / movement constants) are kept identical to ``FighterArena`` so
    the SAME physics core runs all three games. Target Knockback simply adds the
    target zone on top and scores the OPPONENT's occupancy of it.
    """

    platform_width: float = 10.0   # length of the platform segment [0, width]
    gravity: float = 0.6           # downward accel applied each step while airborne
    knockback: float = 2.5         # horizontal impulse a landed punch imparts
    spawn_gap: float = 4.0         # initial horizontal distance between players
    max_steps: int = 200           # full step budget; most knock-in time wins

    # --- the Target-Knockback-specific dials (the target zone) -------------
    # Zone center as a FRACTION of platform_width in [0, 1]; 0.5 == platform
    # centre. Stored as a fraction (not absolute x) so the zone scales with the
    # platform when the Teacher widens it.
    zone_center_frac: float = 0.5
    # Half-width of the target in ABSOLUTE units. The full zone is
    # [center - zone_half, center + zone_half]. Smaller == harder to knock into.
    zone_half: float = 1.5

    # Difficulty dial in [0, 1] — the single knob the Teacher (via
    # ArenaSpec.difficulty) turns to make the opponent weaker (low) or stronger
    # (high). Same direction/meaning as FighterArena.difficulty / KothArena.
    # difficulty: it feeds the parametric opponent's strength only, NOT physics,
    # so a fixed difficulty replays identically.
    difficulty: float = 1.0

    # Fixed constants (identical to FighterArena's — the shared physics core
    # reads these). Kept here so the arena is self-contained and can be passed
    # straight into a FighterSim via _as_fighter_arena().
    move_speed: float = 0.35
    jump_impulse: float = 2.2
    punch_range: float = 1.4
    fighter_half_width: float = 0.4

    # -- derived zone geometry ----------------------------------------------

    @property
    def zone_center(self) -> float:
        return float(self.zone_center_frac) * self.platform_width

    @property
    def zone_lo(self) -> float:
        return self.zone_center - self.zone_half

    @property
    def zone_hi(self) -> float:
        return self.zone_center + self.zone_half

    def in_target(self, body: _Body) -> bool:
        """True if ``body`` is in the target zone (inside the segment AND grounded).

        An airborne body banks no knock-in time — being mid-jump is the opponent's
        DODGE. This is what keeps JUMP meaningful in Target Knockback (it lets the
        defender escape the target), so the same five actions carry real trade-offs
        in all three games.
        """
        return bool(self.zone_lo <= body.x <= self.zone_hi and body.on_ground)

    def _as_fighter_arena(self) -> FighterArena:
        """The physics-only view of this arena, for driving a ``FighterSim``.

        Target Knockback's physics constants ARE the fighter's, so we hand the
        shared sim a ``FighterArena`` carrying exactly the physics dials. The zone
        and the knock-in outcome rule live in ``TargetKnockbackSim`` on top of this
        — the sim core never knows it is running a different game.
        """
        return FighterArena(
            platform_width=self.platform_width,
            gravity=self.gravity,
            knockback=self.knockback,
            spawn_gap=self.spawn_gap,
            max_steps=self.max_steps,
            difficulty=self.difficulty,
            move_speed=self.move_speed,
            jump_impulse=self.jump_impulse,
            punch_range=self.punch_range,
            fighter_half_width=self.fighter_half_width,
        )


# ---------------------------------------------------------------------------
# Sim — reuses FighterSim physics, swaps edges + outcome for knock-in time
# ---------------------------------------------------------------------------


@dataclass
class TargetKnockbackSim:
    """Target-Knockback physics + match logic on top of the fighter's engine.

    Composition, not a fork: a private ``FighterSim`` is the physics core. Each tick
    delegates movement / jump / facing / punch-knockback / gravity to that core (so
    the motor model is byte-identical to the fighter), then applies its OWN two
    differences:

      1. EDGE handling: no ring-out. After the core integrates, any player that
         walked/got-knocked past an edge is CLAMPED back onto the platform and
         marked alive. (The fighter would have killed it; here we keep it in play —
         the point is to knock the rival into the TARGET, not off the stage.)
      2. OUTCOME: each tick, player 0 scores one unit if the OPPONENT (f1) is in the
         target; player 1 scores one unit if ITS opponent (f0) is in the target.
         The match runs the FULL ``max_steps``; the winner is whoever knocked their
         rival into the target for the most ticks. Ties -> draw (None).

    Note the score INVERSION vs KotH: ``score0`` counts f1-in-target (player 0 wants
    its rival in the zone), ``score1`` counts f0-in-target. You score off the OTHER
    player's position. Deterministic under a seed (spawn jitter is the fighter's
    seeded logic, reused verbatim).
    """

    arena: TargetKnockbackArena
    seed: int = 0
    _core: FighterSim = field(init=False)
    # score0 = ticks the OPPONENT (f1) spent in the target (player 0's points).
    # score1 = ticks the OPPONENT (f0) spent in the target (player 1's points).
    score0: int = field(default=0, init=False)
    score1: int = field(default=0, init=False)
    steps: int = field(default=0, init=False)
    winner: Optional[int] = field(default=None, init=False)  # 0, 1, or None
    done: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._core = FighterSim(arena=self.arena._as_fighter_arena(), seed=self.seed)
        self.reset(self.seed)

    # -- convenient handles onto the shared bodies --------------------------

    @property
    def f0(self) -> _Body:
        return self._core.f0

    @property
    def f1(self) -> _Body:
        return self._core.f1

    # -- lifecycle ----------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        # Reuse the fighter's seeded spawn so the same seed gives the same start
        # across all three games (and the spawn is randomised => the policy must be
        # reactive, not an open-loop memoriser — exactly the fighter's reasoning).
        self._core.reset(self.seed)
        self.score0 = 0
        self.score1 = 0
        self.steps = 0
        self.winner = None
        self.done = False

    # -- one tick -----------------------------------------------------------

    def step(self, a0: int, a1: int) -> None:
        """Advance one tick: fighter physics, then clamp edges + knock-in scoring.

        We drive the SHARED physics helpers directly (movement, facing, jump,
        punch-knockback, gravity) in the fighter's exact order, then apply the two
        game-specific rules (clamp instead of ring-out; accumulate knock-in time off
        the OPPONENT's position).
        """
        if self.done:
            return

        core = self._core
        f0, f1 = core.f0, core.f1

        # --- identical fighter motor model (reused helpers) ----------------
        core._apply_move(f0, a0)
        core._apply_move(f1, a1)
        core._update_facing(f0, f1, a0)
        core._update_facing(f1, f0, a1)
        core._apply_jump(f0, a0)
        core._apply_jump(f1, a1)
        if a0 == Action.PUNCH:
            core._resolve_punch(attacker=f0, defender=f1)
        if a1 == Action.PUNCH:
            core._resolve_punch(attacker=f1, defender=f0)
        core._integrate_gravity(f0)
        core._integrate_gravity(f1)

        # --- difference #1: clamp to platform (NO ring-out) ----------------
        # The fighter would kill a body that left [0, width]; here we keep it in
        # play by clamping x back onto the platform. The game is about knocking the
        # rival into the TARGET, not off the stage, so nobody is eliminated.
        self._clamp_to_platform(f0)
        self._clamp_to_platform(f1)

        # --- difference #2: accumulate knock-in time (OPPONENT in target) ---
        # The score INVERSION: player 0 scores when f1 (its OPPONENT) is in target;
        # player 1 scores when f0 is. You score off the OTHER player's position.
        if self.arena.in_target(f1):
            self.score0 += 1
        if self.arena.in_target(f0):
            self.score1 += 1

        self.steps += 1
        self._resolve_outcome()

    def _clamp_to_platform(self, body: _Body) -> None:
        w = self.arena.platform_width
        if body.x < 0.0:
            body.x = 0.0
        elif body.x > w:
            body.x = w
        body.alive = True  # Target Knockback never rings out; keep the body in play.

    def _resolve_outcome(self) -> None:
        # Runs to the full step budget; the winner is the most knock-in time.
        if self.steps >= self.arena.max_steps:
            self.done = True
            if self.score0 > self.score1:
                self.winner = 0
            elif self.score1 > self.score0:
                self.winner = 1
            else:
                self.winner = None  # tie -> draw

    # -- observation (same style + shape as FighterSim.observe) -------------

    def observe(self, ego: int) -> np.ndarray:
        """Structured observation from ``ego``'s point of view (ego in {0, 1}).

        The FIRST 11 entries are byte-for-byte the fighter's observation layout
        (reused via ``FighterSim.observe``) so a policy reads the same physical
        state in all three games. Target Knockback then APPENDS the target-zone
        information THIS objective needs. Crucially — because you score off the
        OPPONENT's position — the appended flags describe where the OPPONENT sits
        relative to the target, not where ego does:

          [ ...fighter's 11 dims...,
            zone_center_norm, zone_half_norm,
            opp_in_target, ego_in_target,
            signed_dist_opp_to_zone ]

        ``opp_in_target`` (index 13) comes FIRST among the flags because it is the
        quantity ego is trying to drive up (its own points). ``signed_dist_opp_to_
        zone`` (index 15) is the signed normalised distance from the OPPONENT to the
        nearest target edge (0 if the opponent is already inside) — the single most
        useful scalar for "which way do I need to push them". Positions are
        normalised by platform_width, same as the fighter, so one policy is
        scale-stable across arena widths.
        """
        base = self._core.observe(ego)  # the fighter's 11-dim physical obs
        me, opp = (self.f0, self.f1) if ego == 0 else (self.f1, self.f0)
        w = self.arena.platform_width
        zc = self.arena.zone_center
        # Signed distance from the OPPONENT to the nearest point of the target
        # (0 if the opponent is inside), normalised. This is the quantity ego acts
        # on: it tells ego which side of the zone the opponent is on and how far.
        if opp.x < self.arena.zone_lo:
            signed = (opp.x - self.arena.zone_lo) / w
        elif opp.x > self.arena.zone_hi:
            signed = (opp.x - self.arena.zone_hi) / w
        else:
            signed = 0.0
        extra = np.array(
            [
                zc / w,
                self.arena.zone_half / w,
                1.0 if self.arena.in_target(opp) else 0.0,  # opp_in_target (ego scores)
                1.0 if self.arena.in_target(me) else 0.0,   # ego_in_target (rival scores)
                signed,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, extra]).astype(np.float32)


OBS_DIM = 16  # 11 fighter dims + 5 target-zone dims


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def random_policy(seed: Optional[int] = None) -> Policy:
    """Uniform-random action over the SAME five-action set the fighter uses."""
    rng = np.random.default_rng(seed)

    def act(_obs: np.ndarray) -> int:
        return int(STAGE1_ACTIONS[rng.integers(len(STAGE1_ACTIONS))])

    return act


def scripted_target_knockback(arena: TargetKnockbackArena, ego: int = 0) -> Policy:
    """A simple but effective Target-Knockback heuristic over the structured obs.

    Strategy (knock the opponent INTO the target and pin them there):

      * If I'M the one drifting into the target, step out of it (a free point
        against me is worth avoiding).
      * If the opponent is ALREADY in the target and within punch range in front,
        PUNCH — knockback that keeps shoving them holds them in / re-knocks them and
        banks more knock-in time. This is the core scoring move once they're in.
      * If the opponent is OUTSIDE the target, position so a punch pushes them
        TOWARD it: stand on the side of the opponent AWAY from the zone, then close
        to range and PUNCH so the knockback drives them across into the zone.

    Reads only the observation vector (the same input the Player gets), so it is a
    fair fixed opponent. It must clearly beat ``random_policy``.
    """
    w = arena.platform_width
    reach = arena.punch_range + arena.fighter_half_width
    move = arena.move_speed
    zone_lo, zone_hi = arena.zone_lo, arena.zone_hi
    # A small inside-the-reach standoff: aim to sit ~`stand` from the opponent so a
    # punch reliably connects, but never closer (overshooting risks crossing into
    # the zone). One move-step of slack keeps it stable against the opponent moving.
    stand = max(0.0, reach - move)

    def act(obs: np.ndarray) -> int:
        # Indices: [0..10] fighter dims; [11]=zone_center, [12]=zone_half,
        # [13]=opp_in_target, [14]=ego_in_target, [15]=signed_dist_opp_to_zone.
        me_x = obs[0] * w
        opp_x = obs[3] * w
        rel = obs[10] * w            # opp.x - me.x (fighter's relative-x)
        zc = arena.zone_center
        ego_in_target = obs[14] > 0.5

        # The ONE geometric idea: to knock the opponent INTO the target, stand on the
        # side of the opponent AWAY from the zone and punch toward them. Zone to the
        # opponent's LEFT (zc < opp_x) => stand to the opponent's RIGHT, punch left;
        # mirror otherwise. The standoff x sits one `stand` past the opponent on my
        # away-from-zone side, so it is itself never inside the zone.
        want_me_right = zc < opp_x
        target_x = opp_x + stand if want_me_right else opp_x - stand
        on_correct_side = (want_me_right and me_x >= opp_x) or (not want_me_right and me_x <= opp_x)

        # 0. Don't get scored on: if I'm in the target, leave it toward the side that
        #    is my ATTACKING side (away from the zone relative to the opponent) when
        #    possible, so I exit AND end up correctly placed. Falling back to the
        #    nearer edge minimises self-points if the attacking exit is blocked.
        if ego_in_target:
            exit_right_ok = (zone_hi - me_x) <= (me_x - zone_lo) + move
            if want_me_right and (me_x + move <= w):
                return int(Action.RIGHT)        # attack-side exit is to the right
            if (not want_me_right) and (me_x - move >= 0.0):
                return int(Action.LEFT)         # attack-side exit is to the left
            return int(Action.RIGHT) if exit_right_ok else int(Action.LEFT)

        # 1. Correctly placed AND in punch range -> PUNCH: knock the opponent toward
        #    (and deeper into) the zone. Fires whether or not they're already in —
        #    the re-knock banks more knock-in time and denies their escape.
        if on_correct_side and abs(rel) <= reach:
            return int(Action.PUNCH)

        # 2. NAVIGATE to the standoff, one step toward target_x, but NEVER step INTO
        #    the zone (that scores a free point on myself). If the direct line to the
        #    standoff would cross the zone, stop at my own edge of it and POISE there
        #    (IDLE): from the zone boundary I'm one step from punching the opponent
        #    back through the zone the moment they drift into range, without ever
        #    standing in it. A draw from a true stalemate beats scoring on myself.
        if target_x > me_x + 1e-6:
            nxt = me_x + move
            if not (zone_lo <= nxt <= zone_hi):
                return int(Action.RIGHT)
            return int(Action.RIGHT) if me_x + move < zone_lo else int(Action.IDLE)
        if target_x < me_x - 1e-6:
            nxt = me_x - move
            if not (zone_lo <= nxt <= zone_hi):
                return int(Action.LEFT)
            return int(Action.LEFT) if me_x - move > zone_hi else int(Action.IDLE)
        return int(Action.IDLE)

    return act


# ---------------------------------------------------------------------------
# Parametric opponent (difficulty -> strength), mirroring the fighter's dial
# ---------------------------------------------------------------------------

# Low-end lever: epsilon "self-target". With probability ``eps`` the opponent
# drops its skilled behaviour and WALKS INTO the target zone itself (toward the
# zone centre), handing the Player free knock-in time. ``eps`` is HIGH at low
# difficulty and decays to 0 by ``WEAK_ZERO`` — so the bottom of the dial slides
# the Player's win-rate up toward 1.0 (same shape as the fighter's self-edging and
# KotH's abandon-hill levers, just "walk into the target" here).
EPS_MAX = 0.85
WEAK_ZERO = 0.6

# High-end lever: the jump_turtle archetype (same name/role as the fighter's). Once
# past ``STRONG_START`` the competent branch becomes a jump_turtle that (a) JUMPS to
# dodge an incoming punch — airborne it banks no knock-in time AND dodges the
# knockback (``_resolve_punch`` requires both at the same height) — and (b) flees
# the target zone, walking AWAY from it. A reliable jump_turtle therefore can be
# neither knocked into the target nor caught standing in it: the match times out
# with little/no knock-in time for the Player, a draw or loss. As dodge reliability
# rises, the strong/trained Player's win-rate slides from 1.0 down toward 0.0.
STRONG_START = 0.45
# Difficulty at which the jump_turtle's base dodge reliability SATURATES at 1.0.
# Set just below 1.0 so the high-end slide finishes a touch before the top of the
# dial (the [STRONG_START, STRONG_FULL] band is where the turtle's dodge ramps in;
# above STRONG_FULL it is a clean, undefeatable draw machine). A GENTLE ramp here is
# deliberate: it keeps an intermediate band (~d=0.55-0.65) where the turtle is
# beatable by SKILLED play (scripted ~0.65-0.95) but still punishes a passive or
# flailing Player (random/untrained ~0.1-0.2), which is exactly the headroom the
# PPO learnability proof needs — a clean low 'before' with a high 'after' ceiling.
STRONG_FULL = 0.9
# Per-MATCH dodge spread, identical role to the fighter's: converts a single-match
# dodge THRESHOLD (a 1.0/0.0 cliff) into a SMOOTH MEAN win-rate across seeds. Some
# matches the turtle is impenetrable (draw), some it leaks (Player scores). Tapers
# to 0 as the dodge saturates so the very top of the dial locks to a clean
# undefeatable turtle (0.0 floor).
DODGE_SPREAD = 0.7


def epsilon_for_difficulty(difficulty: float) -> float:
    """difficulty in [0,1] -> opponent epsilon, monotonically DECREASING.

    Mirror of the fighter's ``epsilon_for_difficulty``: HIGH at low difficulty
    (opponent walks into the target itself, easy to beat), linearly to 0 at
    ``WEAK_ZERO``.
    """
    d = float(min(1.0, max(0.0, difficulty)))
    if d >= WEAK_ZERO:
        return 0.0
    return EPS_MAX * (1.0 - d / WEAK_ZERO)


def dodge_reliability_for_difficulty(difficulty: float) -> float:
    """difficulty -> the jump_turtle's base dodge reliability in [0, 1].

    0 below ``STRONG_START`` (no turtle, opponent plays the plain scripted
    heuristic), ramping LINEARLY to 1.0 at difficulty 1.0 (a perfectly impenetrable
    turtle that can't be knocked into the target). High-end mirror of
    ``epsilon_for_difficulty`` — the same shape as the fighter's lever of the same
    name.
    """
    d = float(min(1.0, max(0.0, difficulty)))
    if d <= STRONG_START:
        return 0.0
    return min(1.0, (d - STRONG_START) / (STRONG_FULL - STRONG_START))


def parametric_target_knockback(
    arena: TargetKnockbackArena,
    ego: int = 0,
    *,
    difficulty: Optional[float] = None,
    seed: Optional[int] = None,
) -> Policy:
    """Difficulty-scaled Target-Knockback opponent: one dial -> smooth strength.

    LOW-END (epsilon self-target): with probability ``eps`` the opponent drops its
    skill and walks INTO the target zone itself, handing the Player free knock-in
    time. ``eps`` is high at low difficulty and fades to 0 by ``WEAK_ZERO``, so the
    bottom of the dial slides the Player's win-rate up toward 1.0.

    HIGH-END (jump_turtle): once past ``STRONG_START`` the competent branch becomes
    a jump_turtle whose dodge reliability rises with difficulty. It JUMPS to dodge an
    incoming punch (airborne it banks no knock-in time and is immune to knockback)
    and flees the target zone, so a reliable turtle can be neither knocked in nor
    caught in the target — the match times out with little Player knock-in time, a
    draw or loss. The HIGH end of the dial therefore slides the Player's win-rate
    down toward 0.0.

    The two levers OVERLAP in the ``[STRONG_START, WEAK_ZERO]`` band (epsilon fading
    out while the turtle fades in), which is what makes the Player's win-rate a
    smooth, monotone function of ``difficulty`` (1.0 -> 0.0) with 2-3 values in the
    learnable band, instead of a 1.0/0.0 cliff.

    ``difficulty`` defaults to ``arena.difficulty`` (a single field on the arena).
    Every stochastic choice (eps roll, per-match dodge draw, dodge rolls) is driven
    by a SEEDED rng, so a fixed seed replays identically — determinism preserved,
    exactly like the fighter's and KotH's opponents.
    """
    d = float(min(1.0, max(0.0, arena.difficulty if difficulty is None else difficulty)))
    eps = epsilon_for_difficulty(d)
    base_dodge = dodge_reliability_for_difficulty(d)
    w = arena.platform_width
    zc = arena.zone_center
    zone_lo, zone_hi = arena.zone_lo, arena.zone_hi
    reach = arena.punch_range + arena.fighter_half_width
    rng = np.random.default_rng(seed)

    # Per-MATCH effective dodge reliability, drawn ONCE here (constant within a
    # match, varying across seeds). The spread tapers to 0 as the base dodge
    # saturates -> the top of the dial is a clean, variance-free draw.
    eff_spread = DODGE_SPREAD * (1.0 - base_dodge)
    dodge = float(np.clip(base_dodge + (rng.random() - 0.5) * eff_spread, 0.0, 1.0))

    base_scripted = scripted_target_knockback(arena, ego=ego)

    def _walk_into_target(obs: np.ndarray) -> int:
        # Walk toward the target-zone centre — self-inflicted knock-in time that
        # hands the Player points regardless of what the Player does. Keeps the
        # low-difficulty curve monotone (more eps -> more Player points -> higher
        # win-rate).
        me_x = obs[0] * w
        return int(Action.RIGHT) if me_x < zc else int(Action.LEFT)

    # SAFE-CAMP distance from the nearest zone edge: far enough that a single
    # punch's knockback can't shove me from here into the target. Anchored to the
    # knockback impulse so it scales with the arena. This is what makes the turtle
    # robust to a brainless PUNCH-SPAMMER (the common untrained-net attractor): if I
    # camp this far out and dodge the occasional close punch, a spammer simply can't
    # reach me into the zone — so a passive/flailing Player reliably FAILS to score,
    # giving the PPO 'before' baseline a clean low floor instead of a lucky ~0.5.
    safe_gap = arena.knockback + arena.fighter_half_width + 0.5
    # The two candidate camp spots: one safe_gap OUTSIDE each zone edge. The turtle
    # camps on the side of the zone it is ALREADY on, so it never has to cross the
    # zone (and self-score) to reach safety. Clamped to the platform.
    camp_left_x = max(0.0, zone_lo - safe_gap)
    camp_right_x = min(w, zone_hi + safe_gap)

    def _toward(x_from: float, x_to: float) -> int:
        if x_to > x_from + 1e-6:
            return int(Action.RIGHT)
        if x_to < x_from - 1e-6:
            return int(Action.LEFT)
        return int(Action.IDLE)

    def _competent(obs: np.ndarray) -> int:
        me_x = obs[0] * w
        rel = obs[10] * w           # opp.x - me.x
        on_ground = obs[6] > 0.5
        ego_in_target = obs[14] > 0.5
        within_punch = abs(rel) <= reach + 0.6

        # 1. DODGE: if an incoming punch could connect (opponent within reach) and
        #    we're grounded, JUMP with probability ``dodge`` to go airborne and dodge
        #    BOTH the knockback and the in-target scoring (same-height check in
        #    _resolve_punch; airborne banks no knock-in time). The turtle's core: the
        #    more reliably it dodges, the less the Player can knock it into the
        #    target, the more matches draw.
        if on_ground and within_punch and rng.random() < dodge:
            return int(Action.JUMP)

        # 2. GET OUT of the target NOW if I'm in it — exit via the NEAREST boundary
        #    (never "away from centre", which can march a body just inside the far
        #    edge back ACROSS the zone and self-score).
        if ego_in_target:
            return int(Action.LEFT) if (me_x - zone_lo) <= (zone_hi - me_x) else int(Action.RIGHT)

        # 3. SAFE-CAMP: with probability ``dodge`` commit to defence. If a punch can
        #    reach me right now, JUMP to dodge it unconditionally (airborne == no
        #    knockback, no knock-in); otherwise walk to (and hold at) the safe camp on
        #    MY CURRENT SIDE of the zone, far enough that a punch can't knock me in,
        #    NEVER crossing the zone to get there. This is the turtle being a turtle:
        #    deny the Player any knock-in, force a draw. (The unconditional in-range
        #    jump is what makes a committed turtle robust even to a brainless
        #    punch-spammer; camping on my own side stops self-scoring across the zone.)
        if rng.random() < dodge:
            if on_ground and within_punch:
                return int(Action.JUMP)
            camp_x = camp_right_x if me_x >= zc else camp_left_x
            return _toward(me_x, camp_x)

        # 4. Otherwise play the plain scripted heuristic — itself trying to knock the
        #    PLAYER into the target (a fair, aggressive baseline that punishes a
        #    passive Player at low/mid difficulty where this branch is reached often).
        return int(base_scripted(obs))

    def act(obs: np.ndarray) -> int:
        if eps > 0.0 and rng.random() < eps:
            return _walk_into_target(obs)
        return _competent(obs)

    return act


# ---------------------------------------------------------------------------
# Single-agent Gymnasium env (opponent baked into step) — mirrors FighterEnv
# ---------------------------------------------------------------------------


class TargetKnockbackEnv(_GYM_BASE):
    """Single-agent Gymnasium-style env over ``TargetKnockbackSim`` (mirrors
    ``FighterEnv`` / ``KothEnv``).

    The learning agent always controls player 0; the opponent (player 1) is a FIXED
    policy passed at construction and stepped INSIDE ``step`` — a standard
    single-agent MDP, no self-play. Same construction shape as ``FighterEnv`` so the
    same PPO wiring trains this game by swapping the env class.

    Reward: sparse terminal win/loss (+1 / -1, 0 on draw) plus a small dense shaping
    term that nudges the agent toward KNOCKING THE OPPONENT INTO THE TARGET (the
    objective) — the analogue of the fighter's close-distance shaping and KotH's
    occupancy shaping. Kept << terminal so it can't dominate; just enough gradient
    for a short PPO budget.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        arena: TargetKnockbackArena,
        opponent_factory: Callable[[TargetKnockbackArena], Policy],
        *,
        seed: int = 0,
    ):
        super().__init__()
        self.arena = arena
        self._opponent_factory = opponent_factory
        self._base_seed = seed
        self._episode = 0

        self.sim = TargetKnockbackSim(arena=arena, seed=seed)
        self._opponent = opponent_factory(arena)

        if spaces is not None:
            self.action_space = spaces.Discrete(len(STAGE1_ACTIONS))
            high = np.full(OBS_DIM, np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # -- Gym API ------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._base_seed = seed
            self._episode = 0
        ep_seed = self._base_seed + self._episode
        self._episode += 1

        self.sim.reset(ep_seed)
        self._opponent = self._opponent_factory(self.arena)
        obs = self.sim.observe(ego=0)
        return obs, {}

    def step(self, action: int):
        opp_obs = self.sim.observe(ego=1)
        opp_action = int(self._opponent(opp_obs))

        prev_opp_gap = self._opp_gap_to_zone()       # opponent's distance to target
        self.sim.step(int(action), opp_action)
        now_opp_in = self.arena.in_target(self.sim.f1)
        now_opp_gap = self._opp_gap_to_zone()

        obs = self.sim.observe(ego=0)
        terminated = self.sim.done
        truncated = False

        reward = self._shaping_reward(now_opp_in, prev_opp_gap, now_opp_gap)
        if terminated:
            if self.sim.winner == 0:
                reward += 1.0
            elif self.sim.winner == 1:
                reward += -1.0
            # draw -> +0

        info = {
            "winner": self.sim.winner,
            "steps": self.sim.steps,
            "score0": self.sim.score0,
            "score1": self.sim.score1,
        }
        return obs, reward, terminated, truncated, info

    def _opp_gap_to_zone(self) -> float:
        """Absolute distance from the OPPONENT (f1) to the nearest target edge (0 if
        the opponent is inside the zone). The quantity the Player wants to DRIVE
        DOWN — pushing the opponent toward, and into, the target."""
        x = self.sim.f1.x
        if x < self.arena.zone_lo:
            return self.arena.zone_lo - x
        if x > self.arena.zone_hi:
            return x - self.arena.zone_hi
        return 0.0

    def _shaping_reward(self, now_opp_in: bool, prev_opp_gap: float, now_opp_gap: float) -> float:
        """Small dense shaping (kept << terminal reward).

        Two terms, both pointing at the SAME objective (get the opponent into the
        target), so the gradient is informative on a short PPO budget:

          * Per-tick OCCUPANCY bonus while the OPPONENT is in the target (this is
            what banks points).
          * A PROGRESS term that rewards REDUCING the opponent's distance to the
            target this tick — i.e. a punch/knockback (or herding) that shoves the
            opponent closer to a knock-in earns reward the moment it happens, not
            only once they cross the boundary. Normalised by platform width and kept
            small so it can't dominate the terminal win/loss. This is the analogue of
            the fighter's close-distance shaping, but applied to the OPPONENT's
            position because that is what scores here.
        """
        w = self.arena.platform_width
        shaping = 0.0
        if now_opp_in:
            shaping += 0.01  # opponent-in-target is the objective; reward it
        # Reward shrinking the opponent's gap to the target (a good knockback);
        # symmetric penalty if the opponent gets farther (e.g. a wasted punch).
        shaping += 0.02 * ((prev_opp_gap - now_opp_gap) / w)
        return shaping


# ---------------------------------------------------------------------------
# Headless match runner (no Gym needed) — used by the adapter's evaluate()
# ---------------------------------------------------------------------------


def play_match(
    arena: TargetKnockbackArena,
    policy_a: Policy,
    policy_b: Policy,
    seed: int = 0,
) -> Optional[int]:
    """Run one full Target-Knockback match between two policies. Returns winner
    (0, 1) or None.

    ``policy_a`` controls player 0, ``policy_b`` controls player 1. Pure sim, no Gym
    — what the adapter uses to score a policy against the fixed opponent over a set
    of arenas/seeds. The match always runs the full step budget (no early
    termination); the winner is the most knock-in time on the rival.
    """
    sim = TargetKnockbackSim(arena=arena, seed=seed)
    while not sim.done:
        a0 = int(policy_a(sim.observe(ego=0)))
        a1 = int(policy_b(sim.observe(ego=1)))
        sim.step(a0, a1)
    return sim.winner
