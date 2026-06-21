"""record_replay.py — roll out fighter matches and write replay JSON (ADD-ONLY).

Drives the Stage-1 fighter through its PUBLIC interface (``FighterSim`` +
``observe`` + ``step``, the same loop ``play_match`` uses) and records every
frame into a ``replay.ReplayBuilder``. Writes the result to ``replays/<name>.json``.

It does NOT modify the fighter, the trainer, contracts, or any interface — it
only reads positions/facing/actions per frame.

Run it to (re)generate the example replays:

    .venv/bin/python record_replay.py            # scripted-vs-random + scripted-vs-scripted
    .venv/bin/python record_replay.py --trained  # also train a short PPO net and record it

Each generated replay is validated with ``replay.validate_replay`` before it is
considered done, so a written file is guaranteed parseable by the viewer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

from games.fighter import (
    Action,
    FighterArena,
    FighterSim,
    Policy,
    random_policy,
    scripted_fighter,
)
from replay import ReplayBuilder, action_name, validate_replay

# Matched to prove_ppo_learns.py / inspect_policy.py CONFIG_A so a recorded
# trained net is the same arena the Stage-1 milestone trains on.
CONFIG_A = FighterArena(platform_width=10.0, gravity=0.6, knockback=2.5, spawn_gap=4.0)

REPLAYS_DIR = Path(__file__).resolve().parent / "replays"


def record_match(
    arena: FighterArena,
    policy_a: Policy,
    policy_b: Policy,
    *,
    config: str,
    p1_policy: str,
    p2_policy: str,
    seed: int = 0,
) -> dict:
    """Roll out one match and return its replay dict.

    ``policy_a`` drives fighter 0 (p1), ``policy_b`` drives fighter 1 (p2). This
    mirrors ``games.fighter.play_match`` exactly, but captures a frame after each
    tick (plus a frame 0 for the spawn) instead of only returning the winner.
    """
    sim = FighterSim(arena=arena, seed=seed)
    builder = ReplayBuilder(
        arena=arena,
        config=config,
        p1_policy=p1_policy,
        p2_policy=p2_policy,
        seed=seed,
    )

    # Frame 0: the spawn, before anyone has acted.
    builder.capture(sim, int(Action.IDLE), int(Action.IDLE))

    while not sim.done:
        a0 = int(policy_a(sim.observe(ego=0)))
        a1 = int(policy_b(sim.observe(ego=1)))
        sim.step(a0, a1)
        builder.capture(sim, a0, a1)

    return builder.to_dict(sim.winner)


def record_match_koth(
    arena,
    policy_a,
    policy_b,
    *,
    config: str,
    p1_policy: str,
    p2_policy: str,
    seed: int = 0,
) -> dict:
    """Roll out one King-of-the-Hill match and return a viewer-compatible replay.

    The same recorder contract as ``record_match`` (one frame per tick, frame 0 =
    spawn), but driven by ``games.koth.KothSim`` instead of ``FighterSim``. KOTH has
    no ring-out — the winner is whoever banks the most zone-time over the full step
    budget — so the per-frame body state is captured directly (x/y/facing/action,
    identical schema to the fighter) and the match-deciding TARGET ZONE is carried in
    ``meta.zone`` AND ``meta.game = "koth"`` so the viewer can draw the hill the two
    Students are fighting to control. The frame schema is byte-identical to the
    fighter replay, so the existing viewer animates the bodies unchanged and only
    needs to ADD the zone overlay.

    ``policy_a`` drives player 0 (p1); ``policy_b`` drives player 1 (p2). Reuses the
    shared ``Action`` action-name table and the same x/y/facing fields the fighter
    recorder writes, so a KOTH replay is a strict superset of the fighter schema.
    """
    from games.koth import KothSim

    sim = KothSim(arena=arena, seed=seed)
    frames: list[dict] = []

    def _body_dict(body, action: int) -> dict:
        return {
            "x": round(float(body.x), 4),
            "y": round(float(body.y), 4),
            "facing": int(body.facing),
            "action": action_name(action),
        }

    def _capture(a0: int, a1: int) -> None:
        events: list[str] = []
        if int(a0) == int(Action.PUNCH):
            events.append("p1_punch")
        if int(a1) == int(Action.PUNCH):
            events.append("p2_punch")
        frames.append({
            "p1": _body_dict(sim.f0, a0),
            "p2": _body_dict(sim.f1, a1),
            "events": events,
            # Per-frame zone occupancy + banked zone-time so the viewer can show
            # who is holding the hill at any scrub position (purely additive).
            "p1_in_zone": bool(arena.in_zone(sim.f0)),
            "p2_in_zone": bool(arena.in_zone(sim.f1)),
            "zone_time": [int(sim.zone_time0), int(sim.zone_time1)],
        })

    # Frame 0: the spawn, before anyone has acted.
    _capture(int(Action.IDLE), int(Action.IDLE))
    while not sim.done:
        a0 = int(policy_a(sim.observe(ego=0)))
        a1 = int(policy_b(sim.observe(ego=1)))
        sim.step(a0, a1)
        _capture(a0, a1)

    winner_label = {0: "p1", 1: "p2", None: "draw"}[sim.winner]
    return {
        "meta": {
            "game": "koth",
            "config": config,
            "winner": winner_label,
            "frames": len(frames),
            "p1_policy": p1_policy,
            "p2_policy": p2_policy,
            "seed": seed,
            "schema": 1,
            # The TARGET ZONE geometry (the hill) in the SAME sim-x space as the
            # platform, so the viewer maps it straight onto the canvas. This is the
            # one thing KOTH adds over the fighter replay.
            "zone": {
                "center": round(float(arena.zone_center), 4),
                "half": round(float(arena.zone_half), 4),
                "lo": round(float(arena.zone_lo), 4),
                "hi": round(float(arena.zone_hi), 4),
                "center_frac": round(float(arena.zone_center_frac), 4),
            },
            "zone_time_final": [int(sim.zone_time0), int(sim.zone_time1)],
        },
        "platform": {
            "x_left": 0.0,
            "x_right": round(float(arena.platform_width), 4),
            "y": 0.0,
        },
        "frames": frames,
    }


def _write(name: str, data: dict) -> Path:
    problems = validate_replay(data)
    if problems:
        raise ValueError(f"replay {name!r} failed validation: {problems}")
    out = REPLAYS_DIR / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(data, indent=2))
    w = data["meta"]["winner"]
    print(f"  wrote {out.relative_to(REPLAYS_DIR.parent)}  "
          f"({data['meta']['frames']} frames, winner={w})")
    return out


def gen_scripted_vs_random(seed: int = 7) -> Path:
    """Full-strength scripted (p1) vs uniform-random (p2). Instant, no training."""
    arena = CONFIG_A
    data = record_match(
        arena,
        scripted_fighter(arena, ego=0),
        random_policy(seed=seed),
        config="A",
        p1_policy="scripted",
        p2_policy="random",
        seed=seed,
    )
    return _write("scripted_vs_random", data)


def gen_scripted_vs_scripted(seed: int = 3) -> Path:
    """Scripted (p1) vs scripted (p2) — a symmetric, strategic match."""
    arena = CONFIG_A
    data = record_match(
        arena,
        scripted_fighter(arena, ego=0),
        scripted_fighter(arena, ego=1),
        config="A",
        p1_policy="scripted",
        p2_policy="scripted",
        seed=seed,
    )
    return _write("scripted_vs_scripted", data)


def gen_trained_vs_scripted(seed: int = 101) -> Optional[Path]:
    """Train a short PPO net on config A vs the full-strength scripted opponent,
    then record one match of the trained net (p1) vs scripted (p2).

    This is the replay that shows whether the learned policy plays a real
    strategy or just spams one move. Training uses the PUBLIC ``FighterEnv`` with
    the same PPO hyperparameters the Stage-1 milestone uses — it does NOT import
    any private harness helper, so it cannot collide with agents editing those.

    Returns the written path, or ``None`` if SB3 isn't available / training
    fails (the scripted replays still stand on their own).
    """
    arena = CONFIG_A
    try:
        import os

        from stable_baselines3 import PPO

        from games.fighter import FighterEnv, parametric_fighter

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        timesteps = int(os.environ.get("PPO_TIMESTEPS", "60000"))

        # Opponent = full-strength scripted. CONFIG_A has difficulty=1.0, so the
        # parametric opponent's epsilon is 0 == exactly scripted_fighter. We use
        # the parametric factory (signature Callable[[arena], Policy]) so the env
        # constructor's opponent_factory contract is satisfied.
        def opponent_factory(a: FighterArena) -> Policy:
            return parametric_fighter(a, ego=1)

        env = FighterEnv(arena, opponent_factory, seed=seed)
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            policy_kwargs={"net_arch": [64, 64]},
            n_steps=512,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            learning_rate=3e-4,
            device="cpu",
        )
        print(f"  training PPO net on config A vs full-strength scripted "
              f"({timesteps} timesteps) ...")
        model.learn(total_timesteps=timesteps, progress_bar=False)

        def trained_policy(obs) -> int:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        # Record vs the SAME full-strength scripted opponent (deterministic).
        data = record_match(
            arena,
            trained_policy,
            scripted_fighter(arena, ego=1),
            config="A",
            p1_policy="trained_ppo",
            p2_policy="scripted",
            seed=seed,
        )
        return _write("trained_vs_scripted", data)
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"  [skip] trained_vs_scripted: {type(exc).__name__}: {exc}")
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record Stage-1 fighter replays.")
    parser.add_argument(
        "--trained",
        action="store_true",
        help="also train a short PPO net and record trained-vs-scripted",
    )
    args = parser.parse_args(argv)

    print(f"Recording replays into {REPLAYS_DIR}/")
    gen_scripted_vs_random()
    gen_scripted_vs_scripted()
    if args.trained:
        gen_trained_vs_scripted()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
