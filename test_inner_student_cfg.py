"""test_inner_student_cfg.py — the TEACHER inner Ring-Out reward ``student_cfg`` path.

Pure-logic gates (NO PPO, NO Modal) for the focused inner-Student population wired
into the Teacher's NESTED Ring-Out reward (``modal_player._real_ppo_transfer_result``
via ``_resolve_inner_student``, fed by ``output.nested_reward._arena_payload``).

The two load-bearing guarantees:

  A. DEFAULT (student_cfg None/absent) => the ORIGINAL vanilla inner reward, byte-
     identical: net_arch [64,64], NO league, NO decisive reward, NO ent_coef override,
     the train arena UNCHANGED (no decisive_timeout), and the payload carries no
     ``student_cfg`` key. This keeps the old-Teacher / demo path unchanged.

  B. FOCUSED (student_cfg provided) => arch [128,128], the 70/30 focused league
     (resolved to 100% aggressive when no prior_student artifact exists), decisive
     reward ON with decisive_timeout flipped on the train arena, and ent_coef=0.03.

These import only NumPy + the gym-guarded ``games.fighter`` (no SB3), exactly like
``test_opponent_league.py``.
"""

from __future__ import annotations

import math

from games.fighter import FighterArena
from modal_player import _resolve_inner_student
from output.nested_reward import _arena_payload
from contracts import ArenaSpec


# A representative focused student_cfg (the teammate's trainer passes this exact shape).
def _focused_cfg(prior_student_path=None) -> dict:
    return {
        "arch": [128, 128],
        "opponent_league_enabled": True,
        "opponent_league": [
            {"id": "aggressive", "weight": 0.7},
            {"id": "prior_student", "weight": 0.3},
        ],
        "decisive_reward": True,
        "ent_coef": 0.03,
        "prior_student_path": prior_student_path,
    }


# ---------------------------------------------------------------------------
# A. DEFAULT path is byte-identical vanilla
# ---------------------------------------------------------------------------


def test_default_resolve_is_vanilla():
    arena = FighterArena(difficulty=0.6)
    net_arch, kwargs, out_arena = _resolve_inner_student(None, arena, seed=1)
    assert net_arch == [64, 64]
    assert kwargs == {}, "default path must pass NO extra trainer kwargs (vanilla PPOPlayerTrainer)"
    assert out_arena is arena, "default path must NOT replace the train arena"
    assert out_arena.decisive_timeout is False


def test_empty_cfg_is_vanilla():
    # An explicitly-empty dict is treated as 'no focused population' (falsy).
    arena = FighterArena(difficulty=0.4)
    net_arch, kwargs, out_arena = _resolve_inner_student({}, arena, seed=1)
    assert net_arch == [64, 64]
    assert kwargs == {}
    assert out_arena is arena


def test_arena_payload_omits_student_cfg_by_default():
    spec = ArenaSpec(map_size=10, doors=0, keys=0, hazard_density=0.0, difficulty=0.6)
    payload = _arena_payload(spec, episodes=1000, eval_seeds=50, curriculum_id="c")
    assert "student_cfg" not in payload, "default reward payload must carry NO student_cfg key"


# ---------------------------------------------------------------------------
# B. FOCUSED path resolves arch / league / decisive / ent_coef
# ---------------------------------------------------------------------------


def test_focused_resolve_sets_arch_ent_decisive():
    arena = FighterArena(difficulty=0.6)
    net_arch, kwargs, out_arena = _resolve_inner_student(_focused_cfg(), arena, seed=1)
    assert net_arch == [128, 128]
    assert kwargs["net_arch"] == [128, 128]
    assert math.isclose(kwargs["ent_coef"], 0.03, abs_tol=1e-12)
    assert kwargs["decisive_reward"] is True
    # decisive reward REQUIRES decisive_timeout on the arena (else timeout is always a draw).
    assert out_arena.decisive_timeout is True
    assert out_arena is not arena, "focused+decisive must produce a NEW arena (decisive_timeout)"
    # The original arena is not mutated.
    assert arena.decisive_timeout is False


def test_focused_no_prior_student_falls_back_to_100pct_aggressive():
    arena = FighterArena(difficulty=0.6)
    _, kwargs, _ = _resolve_inner_student(_focused_cfg(prior_student_path=None), arena, seed=1)
    league = kwargs["opponent_league"]
    ids = {m["id"]: m["weight"] for m in league}
    assert "prior_student" not in ids, "prior_student must be dropped when no artifact is supplied"
    assert math.isclose(ids["aggressive"], 1.0, abs_tol=1e-9), "fallback must be 100% aggressive"
    assert kwargs["prior_student"] is None


def test_focused_missing_prior_student_path_falls_back():
    # A path that does not exist on disk is treated exactly like None (drop + renormalise).
    arena = FighterArena(difficulty=0.6)
    cfg = _focused_cfg(prior_student_path="/nonexistent/prior_student.json")
    _, kwargs, _ = _resolve_inner_student(cfg, arena, seed=1)
    ids = {m["id"]: m["weight"] for m in kwargs["opponent_league"]}
    assert "prior_student" not in ids
    assert math.isclose(ids["aggressive"], 1.0, abs_tol=1e-9)
    assert kwargs["prior_student"] is None


def test_focused_league_has_no_turtle_or_random():
    arena = FighterArena(difficulty=0.6)
    _, kwargs, _ = _resolve_inner_student(_focused_cfg(), arena, seed=1)
    ids = {m["id"] for m in kwargs["opponent_league"]}
    assert "turtle" not in ids, "focused league must NOT contain turtle"
    assert "random" not in ids, "focused league must NOT contain random"


def test_arena_payload_threads_student_cfg_when_provided():
    spec = ArenaSpec(map_size=10, doors=0, keys=0, hazard_density=0.0, difficulty=0.6)
    cfg = _focused_cfg()
    payload = _arena_payload(spec, episodes=1000, eval_seeds=50, curriculum_id="c", student_cfg=cfg)
    assert payload["student_cfg"] == cfg, "student_cfg must thread VERBATIM into the worker payload"


def test_decisive_off_does_not_flip_decisive_timeout():
    # A focused cfg WITHOUT decisive_reward must leave the arena's decisive_timeout alone.
    arena = FighterArena(difficulty=0.6)
    cfg = _focused_cfg()
    cfg["decisive_reward"] = False
    _, kwargs, out_arena = _resolve_inner_student(cfg, arena, seed=1)
    assert kwargs["decisive_reward"] is False
    assert out_arena.decisive_timeout is False


def test_arch_defaults_to_128_when_absent():
    # opponent_league_enabled with no explicit arch still defaults to the focused [128,128].
    arena = FighterArena(difficulty=0.6)
    cfg = {"opponent_league_enabled": True, "decisive_reward": True, "ent_coef": 0.03}
    net_arch, kwargs, _ = _resolve_inner_student(cfg, arena, seed=1)
    assert net_arch == [128, 128]
    assert kwargs["net_arch"] == [128, 128]
