"""training/dry_loop_sidecar.py — cross-process glue for the pre-RFT dry loop.

THE PROBLEM IT SOLVES
---------------------
The dry loop demands ONE nested-RL PPO reward that is reported IDENTICALLY by
three transports:

  * Modal  — the raw ``nested_reward.teacher_reward`` fan-out value;
  * HUD    — what the served ``crucible-teacher`` env records on its trace;
  * Eval Protocol — what ``ep_remote_server.execute_rollout`` grades and returns.

HUD's ``LocalRuntime`` re-imports ``hud_teacher_env`` in a CHILD subprocess, so an
in-process scorer override never reaches the served grader, and re-running the
(expensive, non-deterministic) PPO fan-out in the child would produce a DIFFERENT
number than the one Modal already computed — the assertion would fail for a real
reason (PPO variance), not a wiring bug.

THE FIX — a tiny filesystem sidecar:

  1. The dry loop computes the nested reward ONCE (Modal fan-out) and ``publish``-es
     it keyed by the canonical params.
  2. The served HUD grader and the EP scorer both call ``served_nested_scorer``,
     which reads the published value for those params. Same number, by construction.
  3. If a scorer is asked for params that were never published (defensive: the
     dry loop always publishes first), it computes the nested reward itself using
     the shared config — still the REAL reward, never a placeholder.

The cache + config live under ``output/dry_loop/`` (gitignored output dir). The
cache key is the canonical FIGHTER param set, so cross-process lookups are exact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Sidecar state dir (under the gitignored output/). Overridable via env so a test
# or a parallel run can isolate its own cache.
SIDECAR_DIR = Path(os.getenv("CRUCIBLE_SIDECAR_DIR", str(_REPO_ROOT / "output" / "dry_loop")))
CACHE_PATH = SIDECAR_DIR / "reward_cache.json"
CONFIG_PATH = SIDECAR_DIR / "reward_config.json"

# The FIGHTER param keys that define a reward (everything the worker payload uses).
_REWARD_KEYS = ("difficulty", "platform_width", "gravity", "knockback", "spawn_gap")


def _canonical_key(params: dict) -> str:
    """A stable, exact cache key for a FIGHTER param set.

    Rounds to 6 dp (matches the assertion tolerance) so float formatting noise
    across processes never splits a key, and sorts keys for determinism.
    """
    rounded = {k: round(float(params[k]), 6) for k in _REWARD_KEYS if k in params}
    return json.dumps(rounded, sort_keys=True)


# ---------------------------------------------------------------------------
# Config (the shared reward parameters the child must reproduce)
# ---------------------------------------------------------------------------

# Conservative defaults; the dry loop overwrites these via ``write_config`` so the
# served child uses the exact same budget/seeds/held-out set as the Modal call.
_DEFAULT_CONFIG = {
    "backend": "modal",
    "seeds": [1, 2, 3],
    "episodes": 1000,
    "eval_seeds": 50,
    "held_out_grid": "full",
}


def write_config(*, backend: str, seeds: list[int], episodes: int, eval_seeds: int,
                 held_out_grid: str) -> None:
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({
        "backend": backend,
        "seeds": list(seeds),
        "episodes": int(episodes),
        "eval_seeds": int(eval_seeds),
        "held_out_grid": held_out_grid,
    }, indent=2))


def read_config() -> dict:
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        return {**_DEFAULT_CONFIG, **cfg}
    return dict(_DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Cache (params -> already-computed nested reward)
# ---------------------------------------------------------------------------


def reset() -> None:
    """Clear the sidecar cache (the dry loop calls this at the start of a run)."""
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({}))


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def publish(params: dict, reward: float) -> None:
    """Record the canonical nested reward for ``params`` (computed once, by Modal)."""
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    cache[_canonical_key(params)] = float(reward)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def lookup(params: dict) -> float | None:
    return _load_cache().get(_canonical_key(params))


# ---------------------------------------------------------------------------
# The scorer the served HUD env + the EP bridge call
# ---------------------------------------------------------------------------


def compute_nested_reward(params: dict, *, _detail_sink: dict | None = None) -> float:
    """Compute the REAL nested-RL PPO held-out-transfer reward for a FIGHTER param set.

    Builds a single-arena ``CurriculumSpec`` (difficulty dial) and threads the full
    FIGHTER geometry + the BROAD held-out population into the Modal transfer worker,
    exactly as the dry loop's Modal call does. This is the load-bearing computation;
    a served grader only reaches it on a cache miss.
    """
    import sys

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from output.broad_eval_set import build_broad_eval_arenas, payload_arenas
    from output.nested_reward import teacher_reward
    from training.hud_teacher_env import fighter_geometry_override, params_to_curriculum

    cfg = read_config()
    spec = params_to_curriculum(params, curriculum_id="dry-loop")
    held_out = payload_arenas(build_broad_eval_arenas(grid=cfg["held_out_grid"]))
    return teacher_reward(
        spec,
        backend=cfg["backend"],
        seeds=tuple(cfg["seeds"]),
        episodes=int(cfg["episodes"]),
        eval_seeds=int(cfg["eval_seeds"]),
        held_out_arenas=held_out,
        geometry_override=fighter_geometry_override(params),
        _detail_sink=_detail_sink,
    )


def served_nested_scorer(params: dict) -> float:
    """Scorer for the served HUD env / EP bridge: cache hit, else compute.

    The dry loop publishes the reward before driving HUD/EP, so this is a cache
    READ in the normal flow — guaranteeing HUD == Modal == EP bit-for-bit. The
    compute fallback keeps it correct (real nested reward) if ever called cold.
    """
    cached = lookup(params)
    if cached is not None:
        return float(cached)
    return float(compute_nested_reward(params))
