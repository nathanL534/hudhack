"""games/fighter.py — Stage-1 headless 2D fighter (deterministic NumPy sim).

A minimal platform fighter for the Crucible milestone. Two fighters stand on a
horizontal platform segment; falling off either edge is a **ring-out** (a loss).
The whole point of this module is to answer one prerequisite question: can a
neural-net Player *learn* in this simulator, and do different arenas produce
different learning? So the sim is deliberately small, fully deterministic under a
seed, and dependency-free apart from NumPy.

What lives here:
  * ``Action``        — the Stage-1 action enum (idle/left/right/jump/punch),
                        designed so kick/dodge/block/duck can be ADDED later
                        without changing any signature.
  * ``FighterArena``  — the configurable arena params (platform_width, gravity,
                        knockback, spawn_gap) for one match.
  * ``FighterSim``    — the raw two-fighter physics + match logic. Both fighters
                        are driven explicitly; nothing about RL lives here.
  * ``FighterEnv``    — a single-agent Gymnasium-style env. The PPO Player is the
                        agent; the OPPONENT is baked into ``step`` (a fixed policy
                        passed at construction). This keeps it standard
                        single-agent RL — no self-play, no multi-agent.
  * ``random_policy`` / ``scripted_fighter`` — the two reference policies. The
                        scripted heuristic must clearly beat random.

Coordinate system: x increases to the right. The platform is the segment
``[0, platform_width]`` at height ``y = 0``. A fighter is "on the platform" while
``0 <= x <= platform_width`` and ``y <= 0`` (i.e. grounded). ``y > 0`` is in the
air (mid-jump); ``y < 0`` never happens — we clamp to the ground. A fighter rings
out the instant its x leaves ``[0, platform_width]`` while not safely airborne
above the platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional

import numpy as np

try:  # Gymnasium is a Stage-1 dependency, but keep the sim importable without it.
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_BASE = gym.Env
except Exception:  # pragma: no cover - exercised only if gymnasium is missing.
    gym = None
    spaces = None
    _GYM_BASE = object


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class Action(IntEnum):
    """Stage-1 action set.

    Only these five are *active* in Stage 1. The enum is laid out so later
    stages can append KICK / DODGE / BLOCK / DUCK with new integer values
    without renumbering the existing ones — every signature here uses
    ``Action`` / ``int``, never a hardcoded count, so adding members is
    additive. ``STAGE1_ACTIONS`` is the canonical Stage-1 subset the env and
    policies expose; the Gym action space is sized from it, so a future stage
    that wants more actions widens this tuple rather than touching signatures.
    """

    IDLE = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    PUNCH = 4
    # --- reserved for later stages (NOT active in Stage 1) ---
    # KICK = 5
    # DODGE = 6
    # BLOCK = 7
    # DUCK = 8


# The actions the Stage-1 env actually exposes. Sizing the action space from
# this tuple (not from len(Action)) is what makes the enum forward-compatible:
# adding a reserved member above does not silently grow the Stage-1 space.
STAGE1_ACTIONS: tuple[Action, ...] = (
    Action.IDLE,
    Action.LEFT,
    Action.RIGHT,
    Action.JUMP,
    Action.PUNCH,
)


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FighterArena:
    """Configurable parameters for one fighter match (the Stage-1 subset).

    These four dials are exactly the arena knobs the milestone varies between
    config A and config B to show that different environments produce different
    learning. Everything else (move speed, jump impulse, punch range, step
    budget) is held fixed so the *arena* is the independent variable.
    """

    platform_width: float = 10.0  # length of the platform segment [0, width]
    gravity: float = 0.6          # downward accel applied each step while airborne
    knockback: float = 2.5        # horizontal impulse a landed punch imparts
    spawn_gap: float = 4.0        # initial horizontal distance between fighters
    max_steps: int = 200          # step budget; no ring-out by then => draw

    # Difficulty dial in [0, 1] for the PARAMETRIC opponent. This is the single
    # knob the Teacher (via ArenaSpec.difficulty) turns to make the opponent
    # weaker (low) or stronger (high). It feeds opponent epsilon-mixing only — it
    # does NOT touch physics, so a fixed difficulty replays identically. At
    # difficulty=1.0 the parametric opponent is the full scripted heuristic
    # (epsilon 0); at 0.0 it is mostly random (a weak opponent the Player beats).
    difficulty: float = 1.0

    # Fixed Stage-1 constants (not varied per arena, but grouped here so the
    # sim has a single source of truth and a later stage could promote any of
    # them to an arena dial without changing call sites).
    move_speed: float = 0.35      # horizontal speed per left/right step
    jump_impulse: float = 2.2     # initial upward velocity of a jump
    punch_range: float = 1.4      # max horizontal distance a punch can connect
    fighter_half_width: float = 0.4  # half the body width (for contact tests)


# ---------------------------------------------------------------------------
# Sim state
# ---------------------------------------------------------------------------


@dataclass
class _Body:
    """One fighter's mutable physical state."""

    x: float
    y: float = 0.0          # height above the platform; 0 == grounded
    vy: float = 0.0         # vertical velocity (jump/gravity)
    facing: int = 1         # +1 faces right, -1 faces left
    alive: bool = True      # set False on ring-out

    @property
    def on_ground(self) -> bool:
        return self.y <= 1e-9


@dataclass
class FighterSim:
    """Deterministic two-fighter physics + match logic.

    The sim is agnostic about *who* controls each fighter — ``step`` takes the
    action for fighter 0 and fighter 1 explicitly. The single-agent env layers
    a baked-in opponent on top of this; the sim itself never references a
    policy. Seedable, though Stage-1 physics is fully deterministic so the seed
    only affects spawn jitter (kept tiny and reproducible).
    """

    arena: FighterArena
    seed: int = 0
    f0: _Body = field(init=False)
    f1: _Body = field(init=False)
    steps: int = field(default=0, init=False)
    winner: Optional[int] = field(default=None, init=False)  # 0, 1, or None (draw/ongoing)
    done: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    # -- lifecycle ----------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)

        # Spawn symmetrically around the platform centre, separated by spawn_gap.
        centre = self.arena.platform_width / 2.0
        half = self.arena.spawn_gap / 2.0
        # A tiny deterministic jitter (<= 0.05) so identical seeds reproduce and
        # different seeds differ, without making the match non-deterministic.
        jitter = (rng.random(2) - 0.5) * 0.1
        x0 = float(np.clip(centre - half + jitter[0], 0.1, self.arena.platform_width - 0.1))
        x1 = float(np.clip(centre + half + jitter[1], 0.1, self.arena.platform_width - 0.1))

        self.f0 = _Body(x=x0, facing=1)
        self.f1 = _Body(x=x1, facing=-1)
        self.steps = 0
        self.winner = None
        self.done = False

    # -- one tick -----------------------------------------------------------

    def step(self, a0: int, a1: int) -> None:
        """Advance the sim one tick given both fighters' actions.

        Order within a tick: face/update from movement intent, apply jumps,
        resolve punches (knockback), integrate gravity, then test ring-outs.
        Resolving punches before gravity means a punch that pushes an opponent
        off the edge registers this same tick.
        """
        if self.done:
            return

        self._apply_move(self.f0, a0)
        self._apply_move(self.f1, a1)

        # Keep facing pointed at the opponent unless actively moving away — this
        # makes "facing direction" a meaningful part of the observation and
        # gives punch a sensible default direction.
        self._update_facing(self.f0, self.f1, a0)
        self._update_facing(self.f1, self.f0, a1)

        self._apply_jump(self.f0, a0)
        self._apply_jump(self.f1, a1)

        # Punches resolve against the CURRENT positions (pre-gravity this tick).
        if a0 == Action.PUNCH:
            self._resolve_punch(attacker=self.f0, defender=self.f1)
        if a1 == Action.PUNCH:
            self._resolve_punch(attacker=self.f1, defender=self.f0)

        self._integrate_gravity(self.f0)
        self._integrate_gravity(self.f1)

        self._check_ringout(self.f0)
        self._check_ringout(self.f1)

        self.steps += 1
        self._resolve_outcome()

    # -- physics helpers ----------------------------------------------------

    def _apply_move(self, body: _Body, action: int) -> None:
        if not body.alive:
            return
        if action == Action.LEFT:
            body.x -= self.arena.move_speed
        elif action == Action.RIGHT:
            body.x += self.arena.move_speed
        # No platform clamp here on purpose: walking off the edge SHOULD ring
        # you out. The clamp would defeat the entire game.

    def _update_facing(self, body: _Body, opponent: _Body, action: int) -> None:
        if not body.alive:
            return
        if action == Action.LEFT:
            body.facing = -1
        elif action == Action.RIGHT:
            body.facing = 1
        else:
            # Default: face the opponent. Stable, and makes punch direction
            # well-defined when the fighter isn't actively walking.
            body.facing = 1 if opponent.x >= body.x else -1

    def _apply_jump(self, body: _Body, action: int) -> None:
        if not body.alive:
            return
        if action == Action.JUMP and body.on_ground:
            body.vy = self.arena.jump_impulse

    def _resolve_punch(self, attacker: _Body, defender: _Body) -> None:
        if not attacker.alive or not defender.alive:
            return
        dx = defender.x - attacker.x
        # The punch only connects if the defender is within range AND on the
        # side the attacker faces (you can't punch behind you).
        in_range = abs(dx) <= (self.arena.punch_range + self.arena.fighter_half_width)
        in_front = (dx >= 0 and attacker.facing > 0) or (dx <= 0 and attacker.facing < 0)
        # Both must be roughly at the same height to connect (no air-to-ground).
        same_height = abs(attacker.y - defender.y) <= 1.0
        if in_range and in_front and same_height:
            # Knockback pushes the defender AWAY in the attacker's facing dir.
            defender.x += self.arena.knockback * attacker.facing

    def _integrate_gravity(self, body: _Body) -> None:
        if not body.alive:
            return
        if not body.on_ground or body.vy > 0:
            body.y += body.vy
            body.vy -= self.arena.gravity
            if body.y <= 0.0:
                body.y = 0.0
                body.vy = 0.0

    def _check_ringout(self, body: _Body) -> None:
        if not body.alive:
            return
        # Ring-out only counts when grounded (or descending onto) outside the
        # platform. While airborne ABOVE the platform you're safe; the danger is
        # landing — or being knocked — past an edge.
        off_edge = body.x < 0.0 or body.x > self.arena.platform_width
        if off_edge and body.on_ground:
            body.alive = False

    def _resolve_outcome(self) -> None:
        f0_out = not self.f0.alive
        f1_out = not self.f1.alive
        if f0_out and f1_out:
            # Simultaneous double ring-out -> draw.
            self.winner = None
            self.done = True
        elif f1_out:
            self.winner = 0
            self.done = True
        elif f0_out:
            self.winner = 1
            self.done = True
        elif self.steps >= self.arena.max_steps:
            # Out of budget with both alive -> draw (no winner).
            self.winner = None
            self.done = True

    # -- observation --------------------------------------------------------

    def observe(self, ego: int) -> np.ndarray:
        """Structured observation from ``ego``'s point of view (ego in {0, 1}).

        Layout (float32, length 11):
          [ego.x, ego.y, ego.vy, opp.x, opp.y, opp.vy,
           ego_on_platform, opp_on_platform, ego.facing, opp.facing,
           relative_x(opp - ego)]
        Positions are normalised by platform_width so the same policy is
        scale-stable across arenas of different widths.
        """
        me, opp = (self.f0, self.f1) if ego == 0 else (self.f1, self.f0)
        w = self.arena.platform_width
        obs = np.array(
            [
                me.x / w,
                me.y,
                me.vy,
                opp.x / w,
                opp.y,
                opp.vy,
                1.0 if (0.0 <= me.x <= w and me.on_ground) else 0.0,
                1.0 if (0.0 <= opp.x <= w and opp.on_ground) else 0.0,
                float(me.facing),
                float(opp.facing),
                (opp.x - me.x) / w,
            ],
            dtype=np.float32,
        )
        return obs


OBS_DIM = 11  # length of FighterSim.observe()


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

# A Policy here is just ``Callable[[np.ndarray], int]``: structured obs -> action.
Policy = Callable[[np.ndarray], int]


def random_policy(seed: Optional[int] = None) -> Policy:
    """Uniform-random action over the Stage-1 action set."""
    rng = np.random.default_rng(seed)

    def act(_obs: np.ndarray) -> int:
        return int(STAGE1_ACTIONS[rng.integers(len(STAGE1_ACTIONS))])

    return act


def scripted_fighter(arena: FighterArena, ego: int = 0) -> Policy:
    """A simple but effective heuristic, expressed over the structured obs.

    Strategy:
      * If standing near YOUR OWN edge, step back toward centre (and jump if
        very close) so you don't get knocked off.
      * Else if the opponent is within punch range and in front, PUNCH.
      * Else move toward the opponent to close distance.

    This reads only the observation vector (the same input the Player gets), so
    it is a fair fixed opponent. It must clearly beat ``random_policy``.
    """
    w = arena.platform_width
    edge_margin = max(1.0, 0.12 * w)  # "near my edge" zone
    reach = arena.punch_range + arena.fighter_half_width

    def act(obs: np.ndarray) -> int:
        me_x = obs[0] * w
        opp_x = obs[3] * w
        rel = obs[10] * w  # opp.x - me.x

        # 1. Self-preservation: away from my own nearest edge.
        if me_x <= edge_margin:
            return int(Action.RIGHT)  # near left edge -> move right
        if me_x >= w - edge_margin:
            return int(Action.LEFT)   # near right edge -> move left

        # 2. In range and facing the opponent -> punch (try to knock them off).
        if abs(rel) <= reach:
            return int(Action.PUNCH)

        # 3. Otherwise close the distance toward the opponent.
        return int(Action.RIGHT) if opp_x > me_x else int(Action.LEFT)

    return act


# ---------------------------------------------------------------------------
# Parametric opponent (difficulty -> strength)
# ---------------------------------------------------------------------------

# --- Primary lever: epsilon-mixing -----------------------------------------
# ``eps`` is the per-step probability the opponent DROPS its skilled behaviour
# and plays a *self-defeating* move — it steps toward its OWN nearest edge,
# drifting toward a ring-out. ``eps`` is HIGH at low difficulty (the opponent
# keeps walking off the platform, so the Player wins easily) and 0 at high
# difficulty. A self-edging fail-branch (not uniform-random, not charge-the-
# Player) is what keeps the curve monotone and draw-free: it resolves the match
# as a Player win on its own, so the low-difficulty win-rate tracks ``eps``
# directly instead of falling into the time-out draw trap an idle/erratic
# opponent creates.
EPS_MAX = 0.85  # self-edge prob at difficulty = 0.0  (weakest opponent)
EPS_MIN = 0.0   # self-edge prob at difficulty = 1.0  (no self-sabotage)

# --- Secondary lever: defensive competence ---------------------------------
# Epsilon alone is not enough because the *base* scripted heuristic is fully
# exploitable: a trained Player learns to lure it to the edge and punch it off,
# so even at eps=0 a converged Player wins ~1.0 and the HIGH-difficulty end has
# no gradient (the cliff just moves). The second lever closes that: as difficulty
# rises the opponent's competent branch gets a WIDER edge-margin and STOPS
# over-pursuing (it holds toward centre instead of chasing into the Player's
# trap). A high-difficulty opponent therefore can't be forced off and out-trades
# the Player at the edge, dropping the Player's win-rate at the top of the dial.
# Margins are in platform-width fractions; pursuit-hold turns on past this
# difficulty.
DEF_MARGIN_FRAC_LOW = 0.12   # edge-margin fraction at difficulty 0 (= base scripted)
DEF_MARGIN_FRAC_HIGH = 0.25  # edge-margin fraction at difficulty 1 (fortress)


def epsilon_for_difficulty(difficulty: float) -> float:
    """Map difficulty in [0, 1] -> opponent epsilon, monotonically DECREASING.

    HIGH epsilon at LOW difficulty (the opponent self-defeats often, easy to
    beat) and LOW epsilon (-> 0) at HIGH difficulty. Linear in difficulty so the
    Player's achievable win-rate is a smooth function of the dial rather than a
    cliff. Clamped so out-of-range dials are well-defined.
    """
    d = float(min(1.0, max(0.0, difficulty)))
    return EPS_MAX + (EPS_MIN - EPS_MAX) * d


def parametric_fighter(
    arena: FighterArena,
    ego: int = 0,
    *,
    difficulty: Optional[float] = None,
    seed: Optional[int] = None,
) -> Policy:
    """Difficulty-scaled opponent: two levers turn one dial into smooth strength.

    PRIMARY (epsilon-mixing): each step, with probability ``eps`` the opponent
    drops its skill and steps toward its OWN nearest edge (a self-inflicted
    ring-out that hands the Player a win). ``eps`` is HIGH at low difficulty and
    0 at high, so the LOW end of the dial slides the Player's win-rate up.

    SECONDARY (defensive competence): on the (1-eps) competent branch, the
    opponent's edge-margin widens and its pursuit shuts off as difficulty rises
    — at high difficulty it guards a fat edge zone and refuses to chase past
    centre, so a trained Player can neither lure it off nor out-position it, and
    the HIGH end of the dial slides the Player's win-rate down. Without this the
    base scripted heuristic is fully exploitable and the high end pins at 1.0.

    Together the two levers make the Player's win-rate a smooth, monotone
    function of ``difficulty`` (high -> low) with a real learnable band in the
    middle, instead of the 1.0/0.0 cliff a fixed-strength opponent produces.

    ``difficulty`` defaults to ``arena.difficulty`` so wiring is a single field
    on the arena. The fail-branch is driven by a SEEDED RNG, so a fixed seed
    replays identically — determinism is preserved.

    At ``difficulty=1.0`` epsilon is 0; the competent branch is a hardened
    (fortress) scripted defender. ``scripted_fighter`` (the unmodified, fully
    aggressive heuristic) remains the strong reference for the gap proxy.
    """
    d = float(min(1.0, max(0.0, arena.difficulty if difficulty is None else difficulty)))
    eps = epsilon_for_difficulty(d)
    w = arena.platform_width
    reach = arena.punch_range + arena.fighter_half_width
    # Defensive competence scales with difficulty.
    margin_frac = DEF_MARGIN_FRAC_LOW + (DEF_MARGIN_FRAC_HIGH - DEF_MARGIN_FRAC_LOW) * d
    edge_margin = max(1.0, margin_frac * w)
    # Pursuit-discipline ramps in smoothly over the upper dial. ``hold_prob`` is
    # the per-step probability the opponent REFUSES the bait (holds toward centre
    # instead of chasing the Player to an edge). 0 below d=0.4 (base scripted
    # "always chase"), ramping to 1 at d=1.0. A *probability* (not a hard switch
    # at d=0.5) is what removes the mid-dial cliff: the opponent's positional
    # discipline — and the Player's win-rate against it — slides instead of snaps.
    hold_prob = max(0.0, (d - 0.4) / 0.6)
    rng = np.random.default_rng(seed)

    def _step_toward_own_edge(obs: np.ndarray) -> int:
        # Walk toward whichever platform edge is nearer to ME — a self-inflicted
        # ring-out that resolves the match as a Player win regardless of what the
        # Player does. Keeps the low-difficulty curve monotone and draw-free.
        me_x = obs[0] * w
        return int(Action.LEFT) if me_x <= (w / 2.0) else int(Action.RIGHT)

    def _competent(obs: np.ndarray) -> int:
        me_x = obs[0] * w
        opp_x = obs[3] * w
        rel = obs[10] * w  # opp.x - me.x
        # 1. Self-preservation: away from my own nearest edge (margin scales up
        #    with difficulty -> fortress at the top of the dial).
        if me_x <= edge_margin:
            return int(Action.RIGHT)
        if me_x >= w - edge_margin:
            return int(Action.LEFT)
        # 2. In range and facing -> punch.
        if abs(rel) <= reach:
            return int(Action.PUNCH)
        # 3. Pursuit. With probability ``hold_prob`` (rising with difficulty)
        #    hold toward centre instead of chasing the Player to the edge
        #    (refusing the bait); otherwise chase like the base scripted heuristic.
        if hold_prob > 0.0 and rng.random() < hold_prob:
            if me_x < (w / 2.0) - 1.0:
                return int(Action.RIGHT)
            if me_x > (w / 2.0) + 1.0:
                return int(Action.LEFT)
            return int(Action.IDLE)
        return int(Action.RIGHT) if opp_x > me_x else int(Action.LEFT)

    def act(obs: np.ndarray) -> int:
        if eps > 0.0 and rng.random() < eps:
            return _step_toward_own_edge(obs)
        return _competent(obs)

    return act


# ---------------------------------------------------------------------------
# Single-agent Gymnasium env (opponent baked into step)
# ---------------------------------------------------------------------------


class FighterEnv(_GYM_BASE):
    """Single-agent Gymnasium-style env over ``FighterSim``.

    The learning agent always controls fighter 0. The opponent (fighter 1) is a
    FIXED policy passed at construction and stepped INSIDE ``step`` — so from the
    agent's perspective this is a standard single-agent MDP. No self-play, no
    multi-agent API. The opponent factory receives the arena so a scripted
    opponent can read arena geometry.

    Reward: sparse terminal win/loss (+1 / -1, 0 on draw) plus a small dense
    shaping term that nudges the agent toward the opponent and toward landing
    punches. The shaping is tiny relative to the terminal reward so it can't
    dominate, but it gives a short PPO budget enough gradient to climb — this
    milestone is a *trend check*, not a tuned agent.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        arena: FighterArena,
        opponent_factory: Callable[[FighterArena], Policy],
        *,
        seed: int = 0,
    ):
        super().__init__()
        self.arena = arena
        self._opponent_factory = opponent_factory
        self._base_seed = seed
        self._episode = 0

        self.sim = FighterSim(arena=arena, seed=seed)
        self._opponent = opponent_factory(arena)

        if spaces is not None:
            self.action_space = spaces.Discrete(len(STAGE1_ACTIONS))
            # Generous bounds — obs is normalised but velocities/positions can
            # briefly exceed [0,1]; infinite bounds keep SB3 happy.
            high = np.full(OBS_DIM, np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # -- Gym API ------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Derive a fresh-but-reproducible per-episode seed so episodes vary yet
        # the whole run replays identically for a given base seed.
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
        # Opponent acts from ITS OWN observation (ego=1), baked in here.
        opp_obs = self.sim.observe(ego=1)
        opp_action = int(self._opponent(opp_obs))

        prev_dist = abs(self.sim.f1.x - self.sim.f0.x)
        self.sim.step(int(action), opp_action)
        new_dist = abs(self.sim.f1.x - self.sim.f0.x)

        obs = self.sim.observe(ego=0)
        terminated = self.sim.done
        truncated = False

        reward = self._shaping_reward(int(action), prev_dist, new_dist)
        if terminated:
            if self.sim.winner == 0:
                reward += 1.0
            elif self.sim.winner == 1:
                reward += -1.0
            # draw -> +0

        info = {"winner": self.sim.winner, "steps": self.sim.steps}
        return obs, reward, terminated, truncated, info

    def _shaping_reward(self, action: int, prev_dist: float, new_dist: float) -> float:
        """Small dense shaping (kept << terminal reward).

        * Reward closing distance toward the opponent (encourages engagement).
        * Reward a punch that connects (the defender got knocked = dist jumps),
          approximated by a punch issued while in range.
        * Tiny step penalty so stalling for a draw isn't free.
        """
        shaping = 0.0
        shaping += 0.01 * (prev_dist - new_dist)  # + when we got closer
        reach = self.arena.punch_range + self.arena.fighter_half_width
        if action == Action.PUNCH and prev_dist <= reach:
            shaping += 0.02
        shaping -= 0.001  # mild urgency
        return shaping


# ---------------------------------------------------------------------------
# Headless match runner (no Gym needed) — used by the adapter's evaluate()
# ---------------------------------------------------------------------------


def play_match(
    arena: FighterArena,
    policy_a: Policy,
    policy_b: Policy,
    seed: int = 0,
) -> Optional[int]:
    """Run one full match between two policies. Returns the winner (0, 1) or None.

    ``policy_a`` controls fighter 0, ``policy_b`` controls fighter 1. Pure sim,
    no Gym — this is what the adapter uses to score a policy against the fixed
    opponent over a set of arenas/seeds.
    """
    sim = FighterSim(arena=arena, seed=seed)
    while not sim.done:
        a0 = int(policy_a(sim.observe(ego=0)))
        a1 = int(policy_b(sim.observe(ego=1)))
        sim.step(a0, a1)
    return sim.winner
