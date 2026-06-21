"""games/koth.py — King of the Hill on the SAME engine as the fighter.

This is the structurally-DIFFERENT held-out game that proves the Teacher's
transfer claim. A Teacher trained to generate Ring-Out Duel arenas
(``games/fighter.py``) should generalise to a game it never trained on. King of
the Hill reuses the fighter's 2D physics, its EXACT five actions
(idle/left/right/jump/punch), and the same structured-observation style — but the
WIN CONDITION is different:

  * Ring-Out Duel (fighter): knock the opponent off the platform. Match ends the
    instant someone leaves ``[0, width]``.
  * King of the Hill (here): there is a ZONE (a sub-segment of the platform). A
    player accumulates score for every tick it OCCUPIES the zone. There is NO
    ring-out — players bounce off the edges and stay on the platform. The match
    runs the full step budget and the WINNER is the player with the most
    zone-time. Punch still works (knockback shoves a rival OUT of the zone), so
    the same motor skills transfer, but the objective is occupy-over-time, not
    eliminate.

Engine reuse (NOT a fork): we import the fighter's ``Action`` / ``STAGE1_ACTIONS``
and reuse a single ``FighterSim`` instance as the physics core. Movement, jumping,
facing, punch/knockback and gravity are exactly the fighter's helpers — KotH only
changes (a) edge handling (clamp instead of ring-out) and (b) the outcome rule
(zone-time instead of last-alive). Same physics, different game.

Coordinate system (inherited from the fighter): x increases to the right; the
platform is ``[0, platform_width]`` at ground height ``y = 0``; ``y > 0`` is
airborne. The zone is the segment ``[zone_center - zone_half, zone_center +
zone_half]``. A player "occupies" the zone when its x is inside that segment AND
it is on the ground (you can't bank zone-time mid-jump — being airborne is
neutral, which is what makes JUMP a real dodge/repositioning trade-off rather
than a free way to camp).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Reuse the fighter's action set + physics core verbatim. This is the literal
# "same engine" claim: KotH does not redefine an action enum or a body; it drives
# the very objects the fighter uses.
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
# Arena (parametric — same param-schema style as FighterArena)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KothArena:
    """Configurable parameters for one King-of-the-Hill match.

    Mirrors ``FighterArena``'s style: a small set of geometry dials plus a single
    ``difficulty`` knob the Teacher turns. The zone dials (``zone_center_frac`` /
    ``zone_half``) are KotH's analogue of the fighter's spawn/knockback dials —
    they are what makes a KotH arena easy or hard, and they are exactly what a
    Teacher generates when it emits a KotH curriculum.

    The physics dials (``platform_width`` / ``gravity`` / ``knockback`` /
    ``spawn_gap`` / movement constants) are kept identical to ``FighterArena`` so
    the SAME physics core runs both games. KotH simply adds the zone on top.
    """

    platform_width: float = 10.0   # length of the platform segment [0, width]
    gravity: float = 0.6           # downward accel applied each step while airborne
    knockback: float = 2.5         # horizontal impulse a landed punch imparts
    spawn_gap: float = 4.0         # initial horizontal distance between players
    max_steps: int = 200           # full step budget; most zone-time at the end wins

    # --- the KotH-specific dials (the zone) --------------------------------
    # Zone center as a FRACTION of platform_width in [0, 1]; 0.5 == platform
    # centre. Stored as a fraction (not absolute x) so the zone scales with the
    # platform when the Teacher widens it.
    zone_center_frac: float = 0.5
    # Half-width of the zone in ABSOLUTE units. The full zone is
    # [center - zone_half, center + zone_half]. Smaller == harder to hold.
    zone_half: float = 1.5

    # Difficulty dial in [0, 1] — the single knob the Teacher (via
    # ArenaSpec.difficulty) turns to make the opponent weaker (low) or stronger
    # (high). Same direction/meaning as FighterArena.difficulty: it feeds the
    # parametric opponent's strength only, NOT physics, so a fixed difficulty
    # replays identically.
    difficulty: float = 1.0

    # Fixed constants (identical to FighterArena's — the shared physics core
    # reads these). Kept here so KothArena is self-contained and can be passed
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

    def in_zone(self, body: _Body) -> bool:
        """True if ``body`` is occupying the zone (inside the segment AND grounded).

        Airborne players bank no zone-time — JUMP is a dodge/reposition, never a
        free way to camp. This is what keeps the JUMP action meaningful in KotH
        (it dodges a punch but costs you zone-time), so the same five actions
        carry real trade-offs in BOTH games.
        """
        return bool(self.zone_lo <= body.x <= self.zone_hi and body.on_ground)

    def _as_fighter_arena(self) -> FighterArena:
        """The physics-only view of this arena, for driving a ``FighterSim``.

        KotH's physics constants ARE the fighter's, so we hand the shared sim a
        ``FighterArena`` carrying exactly the physics dials. The zone and the KotH
        outcome rule live in ``KothSim`` on top of this — the sim core never knows
        it is running a different game.
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
# Sim — reuses FighterSim physics, swaps edges + outcome for zone-time
# ---------------------------------------------------------------------------


@dataclass
class KothSim:
    """King-of-the-Hill physics + match logic on top of the fighter's engine.

    Composition, not a fork: a private ``FighterSim`` is the physics core. Each
    KotH tick delegates movement / jump / facing / punch-knockback / gravity to
    that core (so the motor model is byte-identical to the fighter), then KotH
    applies its OWN two differences:

      1. EDGE handling: no ring-out. After the core integrates, any player that
         walked/got-knocked past an edge is CLAMPED back onto the platform and
         marked alive. (The fighter would have killed it; KotH keeps it in play.)
      2. OUTCOME: each tick, every player inside the zone (and grounded) gains one
         unit of zone-time. The match runs the FULL ``max_steps``; the winner is
         whoever banked the most zone-time. Ties -> draw (None).

    Deterministic under a seed (the spawn jitter is the fighter's seeded logic,
    reused verbatim). Punch still imparts knockback, so a player can shove a rival
    OUT of the zone — the same motor skill, a different purpose.
    """

    arena: KothArena
    seed: int = 0
    _core: FighterSim = field(init=False)
    zone_time0: int = field(default=0, init=False)
    zone_time1: int = field(default=0, init=False)
    steps: int = field(default=0, init=False)
    winner: Optional[int] = field(default=None, init=False)  # 0, 1, or None (draw/ongoing)
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
        # in both games (and the spawn is randomised => the policy must be
        # reactive, not an open-loop memoriser — exactly the fighter's reasoning).
        self._core.reset(self.seed)
        self.zone_time0 = 0
        self.zone_time1 = 0
        self.steps = 0
        self.winner = None
        self.done = False

    # -- one tick -----------------------------------------------------------

    def step(self, a0: int, a1: int) -> None:
        """Advance one tick: fighter physics, then KotH edges + zone scoring.

        We drive the SHARED physics helpers directly (movement, facing, jump,
        punch-knockback, gravity) in the fighter's exact order, then apply the two
        KotH-specific rules (clamp instead of ring-out; accumulate zone-time).
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

        # --- KotH difference #1: clamp to platform (NO ring-out) -----------
        # The fighter would kill a body that left [0, width]; KotH instead keeps
        # it in play by clamping x back onto the platform. Players stay alive the
        # whole match — the game is about occupancy, not survival.
        self._clamp_to_platform(f0)
        self._clamp_to_platform(f1)

        # --- KotH difference #2: accumulate zone-time ----------------------
        if self.arena.in_zone(f0):
            self.zone_time0 += 1
        if self.arena.in_zone(f1):
            self.zone_time1 += 1

        self.steps += 1
        self._resolve_outcome()

    def _clamp_to_platform(self, body: _Body) -> None:
        w = self.arena.platform_width
        if body.x < 0.0:
            body.x = 0.0
        elif body.x > w:
            body.x = w
        body.alive = True  # KotH never rings out; keep the body in play.

    def _resolve_outcome(self) -> None:
        # KotH runs to the full step budget; the winner is the most zone-time.
        if self.steps >= self.arena.max_steps:
            self.done = True
            if self.zone_time0 > self.zone_time1:
                self.winner = 0
            elif self.zone_time1 > self.zone_time0:
                self.winner = 1
            else:
                self.winner = None  # tie -> draw

    # -- observation (same style + shape as FighterSim.observe) -------------

    def observe(self, ego: int) -> np.ndarray:
        """Structured observation from ``ego``'s point of view (ego in {0, 1}).

        The FIRST 11 entries are byte-for-byte the fighter's observation layout
        (reused via ``FighterSim.observe``) so a policy reads the same physical
        state in both games. KotH then APPENDS the zone information the new
        objective needs (the zone is the only thing that differs):

          [ ...fighter's 11 dims...,
            zone_center_norm, zone_half_norm,
            ego_in_zone, opp_in_zone,
            signed_dist_to_zone(ego) ]

        Appending (rather than rewriting) keeps the shared physical sub-vector
        aligned across games — the same first-11 semantics — while giving the
        Player exactly what KotH adds. Positions are normalised by platform_width,
        same as the fighter, so one policy is scale-stable across arena widths.
        """
        base = self._core.observe(ego)  # the fighter's 11-dim physical obs
        me, opp = (self.f0, self.f1) if ego == 0 else (self.f1, self.f0)
        w = self.arena.platform_width
        zc = self.arena.zone_center
        # Signed distance from ego to the NEAREST point of the zone (0 if inside),
        # normalised — the single most useful scalar for "go to the zone".
        if me.x < self.arena.zone_lo:
            signed = (me.x - self.arena.zone_lo) / w
        elif me.x > self.arena.zone_hi:
            signed = (me.x - self.arena.zone_hi) / w
        else:
            signed = 0.0
        extra = np.array(
            [
                zc / w,
                self.arena.zone_half / w,
                1.0 if self.arena.in_zone(me) else 0.0,
                1.0 if self.arena.in_zone(opp) else 0.0,
                signed,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, extra]).astype(np.float32)


OBS_DIM = 16  # 11 fighter dims + 5 KotH zone dims


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def random_policy(seed: Optional[int] = None) -> Policy:
    """Uniform-random action over the SAME five-action set the fighter uses."""
    rng = np.random.default_rng(seed)

    def act(_obs: np.ndarray) -> int:
        return int(STAGE1_ACTIONS[rng.integers(len(STAGE1_ACTIONS))])

    return act


def scripted_koth(arena: KothArena, ego: int = 0) -> Policy:
    """A simple but effective KotH heuristic over the structured obs.

    Strategy (the KotH analogue of the fighter's scripted heuristic — go to and
    HOLD the objective, contest the rival):

      * If I'm OUTSIDE the zone, walk toward it (the signed-distance scalar tells
        me which way).
      * If I'm INSIDE the zone and the opponent is also in/near it within punch
        range and in front, PUNCH — knockback shoves the rival OUT of the zone,
        so I keep banking zone-time while they have to walk back.
      * If I'm INSIDE the zone and unthreatened, hold position (IDLE) so I don't
        drift out and lose occupancy.

    Reads only the observation vector (the same input the Player gets), so it is a
    fair fixed opponent. It must clearly beat ``random_policy``.
    """
    w = arena.platform_width
    reach = arena.punch_range + arena.fighter_half_width

    def act(obs: np.ndarray) -> int:
        # Indices: [0..10] fighter dims; [11]=zone_center, [12]=zone_half,
        # [13]=ego_in_zone, [14]=opp_in_zone, [15]=signed_dist_to_zone.
        rel = obs[10] * w           # opp.x - me.x (fighter's relative-x)
        in_zone = obs[13] > 0.5
        opp_in_zone = obs[14] > 0.5
        signed = obs[15] * w        # signed distance to the zone (0 if inside)

        if not in_zone:
            # Outside: head toward the zone. signed<0 => zone is to my right.
            return int(Action.RIGHT) if signed < 0 else int(Action.LEFT)

        # Inside the zone: contest the rival if it's a punchable threat.
        if opp_in_zone and abs(rel) <= reach:
            return int(Action.PUNCH)

        # Inside and safe: hold the hill.
        return int(Action.IDLE)

    return act


# ---------------------------------------------------------------------------
# Parametric opponent (difficulty -> strength), mirroring the fighter's dial
# ---------------------------------------------------------------------------

# Low-end lever: epsilon "wander". With probability ``eps`` the opponent drops
# its skilled behaviour and ABANDONS the zone (walks toward the nearer platform
# edge, away from the hill), handing the Player uncontested zone-time. ``eps`` is
# HIGH at low difficulty and decays to 0 by ``WEAK_ZERO`` — so the bottom of the
# dial slides the Player's win-rate up toward 1.0 (same shape as the fighter's
# self-edging lever, just "leave the hill" instead of "ring yourself out").
EPS_MAX = 0.85
WEAK_ZERO = 0.6

# High-end lever: contest reliability. Above ``STRONG_START`` the competent
# branch starts actively PUNCHING the Player out of the zone whenever it can,
# with a reliability that ramps to 1.0 at difficulty 1.0. A perfectly-contesting
# opponent splits the hill evenly (or edges ahead), so the strong/trained Player's
# win-rate slides from 1.0 down toward a coin-flip/loss at the top of the dial.
STRONG_START = 0.45


def epsilon_for_difficulty(difficulty: float) -> float:
    """difficulty in [0,1] -> opponent epsilon, monotonically DECREASING.

    Mirror of the fighter's ``epsilon_for_difficulty``: HIGH at low difficulty
    (opponent abandons the hill, easy to beat), linearly to 0 at ``WEAK_ZERO``.
    """
    d = float(min(1.0, max(0.0, difficulty)))
    if d >= WEAK_ZERO:
        return 0.0
    return EPS_MAX * (1.0 - d / WEAK_ZERO)


def contest_reliability_for_difficulty(difficulty: float) -> float:
    """difficulty -> how reliably the opponent contests the zone, in [0, 1].

    0 below ``STRONG_START`` (opponent plays the plain scripted heuristic),
    ramping LINEARLY to 1.0 at difficulty 1.0 (a relentless contester). High-end
    mirror of the fighter's ``dodge_reliability_for_difficulty``.
    """
    d = float(min(1.0, max(0.0, difficulty)))
    if d <= STRONG_START:
        return 0.0
    return min(1.0, (d - STRONG_START) / (1.0 - STRONG_START))


def parametric_koth(
    arena: KothArena,
    ego: int = 0,
    *,
    difficulty: Optional[float] = None,
    seed: Optional[int] = None,
) -> Policy:
    """Difficulty-scaled KotH opponent: one dial -> smooth strength.

    LOW-END (epsilon wander): with probability ``eps`` the opponent abandons the
    hill (steps toward its nearer edge), handing the Player uncontested zone-time.
    ``eps`` is high at low difficulty and fades to 0 by ``WEAK_ZERO``.

    HIGH-END (contest): once past ``STRONG_START`` the competent branch becomes an
    active contester that punches the Player out of the zone whenever in range,
    with reliability rising to 1.0. A relentless contester denies the Player
    occupancy, sliding the Player's win-rate down toward a draw/loss.

    ``difficulty`` defaults to ``arena.difficulty`` (a single field on the arena).
    Every stochastic choice is driven by a SEEDED rng, so a fixed seed replays
    identically — determinism preserved, exactly like the fighter's opponent.
    """
    d = float(min(1.0, max(0.0, arena.difficulty if difficulty is None else difficulty)))
    eps = epsilon_for_difficulty(d)
    contest = contest_reliability_for_difficulty(d)
    w = arena.platform_width
    centre = w / 2.0
    reach = arena.punch_range + arena.fighter_half_width
    rng = np.random.default_rng(seed)

    base_scripted = scripted_koth(arena, ego=ego)

    def _abandon_hill(obs: np.ndarray) -> int:
        # Walk toward the nearer platform edge — away from the hill — so the
        # Player gets uncontested zone-time. Keeps the low-difficulty curve
        # monotone (more eps -> more Player zone-time -> higher win-rate).
        me_x = obs[0] * w
        return int(Action.LEFT) if me_x <= centre else int(Action.RIGHT)

    def _competent(obs: np.ndarray) -> int:
        rel = obs[10] * w
        in_zone = obs[13] > 0.5
        opp_in_zone = obs[14] > 0.5
        signed = obs[15] * w

        # CONTEST: if reliable enough this match, punch the Player out of the zone
        # the moment it's a punchable threat (in/near zone, in range, in front).
        if contest > 0.0 and opp_in_zone and abs(rel) <= reach and rng.random() < contest:
            return int(Action.PUNCH)
        # Otherwise play the plain hold-the-hill heuristic.
        if not in_zone:
            return int(Action.RIGHT) if signed < 0 else int(Action.LEFT)
        if opp_in_zone and abs(rel) <= reach:
            return int(Action.PUNCH)
        return int(Action.IDLE)

    def act(obs: np.ndarray) -> int:
        if eps > 0.0 and rng.random() < eps:
            return _abandon_hill(obs)
        return _competent(obs)

    return act


# ---------------------------------------------------------------------------
# Single-agent Gymnasium env (opponent baked into step) — mirrors FighterEnv
# ---------------------------------------------------------------------------


class KothEnv(_GYM_BASE):
    """Single-agent Gymnasium-style env over ``KothSim`` (mirrors ``FighterEnv``).

    The learning agent always controls player 0; the opponent (player 1) is a
    FIXED policy passed at construction and stepped INSIDE ``step`` — a standard
    single-agent MDP, no self-play. Same construction shape as ``FighterEnv`` so
    the same PPO wiring trains either game by swapping the env class.

    Reward: sparse terminal win/loss (+1 / -1, 0 on draw) plus a small dense
    shaping term that nudges the agent toward OCCUPYING the zone (the KotH
    objective) — the analogue of the fighter's close-distance shaping. Kept
    << terminal so it can't dominate; just enough gradient for a short PPO budget.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        arena: KothArena,
        opponent_factory: Callable[[KothArena], Policy],
        *,
        seed: int = 0,
    ):
        super().__init__()
        self.arena = arena
        self._opponent_factory = opponent_factory
        self._base_seed = seed
        self._episode = 0

        self.sim = KothSim(arena=arena, seed=seed)
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

        prev_in_zone = self.arena.in_zone(self.sim.f0)
        self.sim.step(int(action), opp_action)
        now_in_zone = self.arena.in_zone(self.sim.f0)

        obs = self.sim.observe(ego=0)
        terminated = self.sim.done
        truncated = False

        reward = self._shaping_reward(obs, prev_in_zone, now_in_zone)
        if terminated:
            if self.sim.winner == 0:
                reward += 1.0
            elif self.sim.winner == 1:
                reward += -1.0
            # draw -> +0

        info = {
            "winner": self.sim.winner,
            "steps": self.sim.steps,
            "zone_time0": self.sim.zone_time0,
            "zone_time1": self.sim.zone_time1,
        }
        return obs, reward, terminated, truncated, info

    def _shaping_reward(self, obs: np.ndarray, prev_in_zone: bool, now_in_zone: bool) -> float:
        """Small dense shaping (kept << terminal reward).

        * Per-tick bonus for OCCUPYING the zone (the core KotH objective).
        * Small bonus for being closer to the zone when outside it (approach),
          read from the normalised signed-distance scalar.
        """
        shaping = 0.0
        if now_in_zone:
            shaping += 0.01  # occupancy is the objective; reward it each tick
        else:
            # Closer-to-zone is better; |signed| shrinks toward the zone.
            shaping += 0.002 * (1.0 - min(1.0, abs(float(obs[15]))))
        return shaping


# ---------------------------------------------------------------------------
# Headless match runner (no Gym needed) — used by the adapter's evaluate()
# ---------------------------------------------------------------------------


def play_match(
    arena: KothArena,
    policy_a: Policy,
    policy_b: Policy,
    seed: int = 0,
) -> Optional[int]:
    """Run one full KotH match between two policies. Returns winner (0, 1) or None.

    ``policy_a`` controls player 0, ``policy_b`` controls player 1. Pure sim, no
    Gym — what the adapter uses to score a policy against the fixed opponent over
    a set of arenas/seeds. The match always runs the full step budget (KotH has no
    early termination); the winner is the most zone-time.
    """
    sim = KothSim(arena=arena, seed=seed)
    while not sim.done:
        a0 = int(policy_a(sim.observe(ego=0)))
        a1 = int(policy_b(sim.observe(ego=1)))
        sim.step(a0, a1)
    return sim.winner
