"""training/tk_dry_loop_sidecar.py — cross-process glue for the TK dry loop.

The Target-Knockback twin of ``training/dry_loop_sidecar.py``. The TK dry loop demands
ONE nested-RL PPO reward reported IDENTICALLY by three transports (Modal / HUD / Eval
Protocol). HUD's ``LocalRuntime`` re-imports ``tk_hud_teacher_env`` in a CHILD
subprocess, so an in-process scorer override never reaches the served grader, and
re-running the (expensive, non-deterministic) PPO fan-out in the child would produce a
DIFFERENT number than Modal already computed.

THE FIX — a tiny filesystem sidecar (its OWN cache dir, separate from the fighter's):

  1. The TK dry loop computes the nested TK reward ONCE (crucible-player-tk fan-out)
     and ``publish``-es it keyed by the canonical TK params.
  2. The served HUD grader and the EP scorer both call ``served_tk_nested_scorer``,
     which reads the published value for those params. Same number, by construction.
  3. On a (defensive) cache miss the scorer computes the real TK nested reward itself.

The cache lives under ``output/tk_dry_loop/`` so it never collides with the fighter
sidecar's ``output/dry_loop/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# TK sidecar state dir — DISTINCT from the fighter's output/dry_loop/.
SIDECAR_DIR = Path(
    os.getenv("CRUCIBLE_TK_SIDECAR_DIR", str(_REPO_ROOT / "output" / "tk_dry_loop"))
)
CACHE_PATH = SIDECAR_DIR / "reward_cache.json"
CONFIG_PATH = SIDECAR_DIR / "reward_config.json"

# The TK param keys that define a reward (everything the worker payload uses).
_REWARD_KEYS = (
    "difficulty",
    "platform_width",
    "gravity",
    "knockback",
    "spawn_gap",
    "zone_half",
    "zone_center_frac",
)

_DEFAULT_CONFIG = {
    "backend": "modal",
    "seeds": [1, 2, 3],
    "episodes": 2000,
    "eval_seeds": 50,
}


def _canonical_key(params: dict) -> str:
    rounded = {k: round(float(params[k]), 6) for k in _REWARD_KEYS if k in params}
    return json.dumps(rounded, sort_keys=True)


def write_config(*, backend: str, seeds: list[int], episodes: int, eval_seeds: int) -> None:
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "backend": backend,
                "seeds": list(seeds),
                "episodes": int(episodes),
                "eval_seeds": int(eval_seeds),
            },
            indent=2,
        )
    )


def read_config() -> dict:
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        return {**_DEFAULT_CONFIG, **cfg}
    return dict(_DEFAULT_CONFIG)


def reset() -> None:
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
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    cache[_canonical_key(params)] = float(reward)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def lookup(params: dict) -> float | None:
    return _load_cache().get(_canonical_key(params))


def compute_tk_nested_reward(params: dict, *, _detail_sink: dict | None = None) -> float:
    """Compute the REAL nested-RL TK PPO held-out-transfer reward for a TK param set.

    Threads the full TK geometry into the crucible-player-tk transfer worker, exactly
    as the TK dry loop's Modal call does. Load-bearing; a served grader only reaches
    it on a cache miss.
    """
    import sys

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from output.nested_reward_tk import tk_teacher_reward

    cfg = read_config()
    return tk_teacher_reward(
        params,
        backend=cfg["backend"],
        seeds=tuple(cfg["seeds"]),
        episodes=int(cfg["episodes"]),
        eval_seeds=int(cfg["eval_seeds"]),
        curriculum_id="tk-dry-loop",
        _detail_sink=_detail_sink,
    )


def served_tk_nested_scorer(params: dict) -> float:
    """Scorer for the served TK HUD env / EP bridge: cache hit, else compute.

    The TK dry loop publishes the reward before driving HUD/EP, so this is a cache
    READ in the normal flow — guaranteeing HUD == Modal == EP bit-for-bit.
    """
    cached = lookup(params)
    if cached is not None:
        return float(cached)
    return float(compute_tk_nested_reward(params))
