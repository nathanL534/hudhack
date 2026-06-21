# Lateral Analysis: Stage-6 Diff Bloat Check
Date: 2026-06-21

## Scope
Audit the diff at /tmp/bloat_code_diff.txt for code added in the diff that duplicates
patterns, helpers, or classes that already existed in the repo before the diff landed.

---

## Finding 1: `harness/koth_trainer.py` duplicates `_MultiArenaFighterEnv`, `_make_policy_from_model`, and `_set_global_seeds` verbatim from `harness/ppo_trainer.py`

**What was added:**
- `harness/koth_trainer.py` — new file, 185 lines, lines 40-46 (`_set_global_seeds`),
  lines 63-113 (`_MultiArenaKothEnv`), lines 105-112 (`_make_policy_from_model`).

**What already existed:**
- `harness/ppo_trainer.py` lines 53-55: identical `_set_global_seeds` body
  (`random.seed` + `np.random.seed`).
- `harness/ppo_trainer.py` lines 77-121: `_MultiArenaFighterEnv` — the new
  `_MultiArenaKothEnv` (lines 63-113) is character-for-character the same gym.Env
  skeleton (same `__init__`, same `reset`, same `step`, same `close`) with only the
  env class name (`KothEnv` vs `FighterEnv`) and opponent factory substituted.
- `harness/ppo_trainer.py` lines 124-131: `_make_policy_from_model` — the new copy
  in `koth_trainer.py` lines 105-112 is byte-identical.

**Simpler alternative:**
`_make_policy_from_model` and `_set_global_seeds` could live in `harness/interfaces.py`
(the shared ABC module both trainers already import) and be imported by both trainers —
zero new lines. `_MultiArenaKothEnv` could be collapsed into a single generic
`_MultiArenaEnv(env_cls, arenas, seed)` factory that both trainers instantiate, or
`_MultiArenaFighterEnv` could accept an `env_cls` + `opponent_factory` argument.
The diff's own docstring concedes the class is a "structural twin" with "identical SB3
wiring" — that admission signals a shared base was the right call.

**Duplication size:** ~50 lines of identical logic split across two files.

---

## Finding 2: `output/stage6/koth_cross_game.py` re-declares `KOTH_BOUNDS` and `_clamp`, both of which already appear in `output/stage6/games.py`

**What was added:**
- `output/stage6/koth_cross_game.py` lines 48-53: `KOTH_BOUNDS` dict (same four
  keys and identical numeric bounds).
- `output/stage6/koth_cross_game.py` lines 80-82: `_clamp(value, bounds)` — a
  two-line `min(max(...))` wrapper.

**What already existed:**
- `output/stage6/games.py` lines 163-168: `KOTH_BOUNDS` — identical dict, already
  module-level, already used by `_koth_teacher_game` in the same file.
- `output/stage6/evaluator.py` lines 59-61: `_clamp_count` (different purpose) but
  the `min/max` clamp pattern itself is a Python built-in one-liner
  (`min(max(v, lo), hi)`) that needs no wrapper at all.

**Simpler alternative:**
`koth_cross_game.py` already imports from `output.stage6.games` (`get_game`); it
could import `KOTH_BOUNDS` from there with one extra name in the same import line.
The `_clamp` helper adds no abstraction over the built-in `min(max(v, lo), hi)` and
can be inlined at its two call sites (or just imported from `games.py` if a name is
wanted). The duplication means a future bounds change must be made in two files.

**Duplication size:** ~8 lines + silent drift risk on bounds values.

---

## Finding 3: `koth_cross_game.map_fighter_arena_to_koth` reimplements the mapping already in `harness/koth_adapter._arena_from_spec`

**What was added:**
- `output/stage6/koth_cross_game.py` lines 114-135: `map_fighter_arena_to_koth(arena)`
  — maps `difficulty` + `platform_width` to KOTH zone geometry using the formula
  `zone_half = max(0.6, 0.22 * width * (1.0 - 0.6 * difficulty))` and
  `zone_center_frac = 0.5 + 0.18 * difficulty`.

**What already existed:**
- `harness/koth_adapter.py` lines 69-96: `_arena_from_spec(spec: ArenaSpec)` — uses
  the IDENTICAL formulas (`zone_half = max(0.6, 0.22 * width * (1.0 - 0.6 * spec.difficulty))`
  and `zone_center_frac = 0.5 + 0.18 * spec.difficulty`). Only the input type
  differs: `_arena_from_spec` takes an `ArenaSpec` (with `.map_size` and
  `.difficulty` fields), while the new function takes a plain `dict`. The docstring
  in `koth_cross_game.py` even says "Mirrors `harness.koth_adapter._arena_from_spec`".

**Simpler alternative:**
Expose `_arena_from_spec` (or a renamed public `koth_adapter.arena_from_params`) that
accepts either an `ArenaSpec` or a `(difficulty, platform_width)` pair. The
`koth_cross_game` mapping then reduces to a one-line call. The risk of the current
situation: the two formulas diverge silently — a tuning change to one does not
propagate to the other, producing different curricula from the same Teacher output
depending on the code path taken.

**Duplication size:** ~20 lines of formula code + a proven drift risk (the comment
"mirrors X" is a documentation debt that compilers cannot enforce).

---

## Finding 4: `_capture_h2h_replays` in `modal_player.py` inlines the frame-capture loop already abstracted in `record_replay.record_match`

**What was added:**
- `modal_player.py` lines 751-795: `_capture_h2h_replays` — contains a nested `_roll`
  function (lines 751-764) that instantiates `FighterSim`, creates a `ReplayBuilder`,
  calls `builder.capture(sim, IDLE, IDLE)`, steps the sim in a `while not sim.done`
  loop calling `sim.observe`, `sim.step`, `builder.capture`, and finally
  `builder.to_dict(sim.winner)` and `validate_replay(data)`.

**What already existed:**
- `record_replay.py` lines 43-83: `record_match(arena, policy_a, policy_b, *, config,
  p1_policy, p2_policy, seed)` — identical frame-capture loop (frame 0 IDLE capture,
  `while not sim.done` step/capture, `builder.to_dict`, `validate_replay`). It was
  already exported and designed to be the single place where this pattern lives.

**Simpler alternative:**
`_capture_h2h_replays` could call `record_match(arena, trained, base, ...)` directly.
`record_replay` is already on the Modal image's Python path (it's in the root, alongside
`replay.py` which the worker already imports). The return type (`dict`) is the same.
The only adaptation needed is passing the two policy labels — which `record_match`
already accepts as `p1_policy` / `p2_policy` string args. This would reduce `_roll`
to zero new lines and keep the loop in one place.

**Note on `validate_replay` return value:** the diff's `_roll` inverts the truthiness
check — it returns `(None, winner)` when `validate_replay` returns truthy problems and
`(data, winner)` when it returns falsy. `record_replay._write` raises on problems.
This inversion is a latent correctness bug enabled by the duplication; a shared
function would have caught it.

**Duplication size:** ~20 lines + one correctness inversion.

---

## Finding 5: `harness/koth_trainer._KothTrainingJob` duplicates `harness/ppo_trainer._PPOTrainingJob` with zero functional difference after the diff's own change

**What was added:**
- `harness/koth_trainer.py` lines 113-125: `_KothTrainingJob(TrainingJob)` with three
  methods: `__init__`, `is_done`, `result`.

**What already existed (and was modified by the diff):**
- `harness/ppo_trainer.py` lines 134-155: `_PPOTrainingJob(TrainingJob)` — the diff
  itself updated this class (adding `model=None` to `__init__`) so it now has the
  identical constructor signature, identical `is_done` body (`return True`), and
  identical `result` body as `_KothTrainingJob`. The diff comment on line 475 of the
  diff even says `_KothTrainingJob` is "A synchronously-finished TrainingJob carrying
  the policy AND raw model" — the exact same description applied to `_PPOTrainingJob`.

**Simpler alternative:**
Since the diff updated `_PPOTrainingJob` to carry `.model`, the new `_KothTrainingJob`
is now byte-for-byte identical in contract and behavior. `KothPlayerTrainer.submit`
could instantiate `_PPOTrainingJob` directly (importing it from `harness.ppo_trainer`)
or the class could be lifted to `harness/interfaces.py` as `SyncTrainingJob` shared
by both trainers. This is 20 lines of dead-clone that must be kept in sync manually.

**Duplication size:** ~20 lines of identical class body.

---

## Summary Table

| Finding | What was added | Existing alternative | Location of existing code | LOC duplicated |
|---------|---------------|---------------------|--------------------------|----------------|
| 1 | `_set_global_seeds`, `_MultiArenaKothEnv`, `_make_policy_from_model` in koth_trainer | Same functions in ppo_trainer | `harness/ppo_trainer.py:53-131` | ~50 |
| 2 | `KOTH_BOUNDS` + `_clamp` in koth_cross_game | `KOTH_BOUNDS` in games.py; `min/max` builtin | `output/stage6/games.py:163` | ~8 |
| 3 | `map_fighter_arena_to_koth` formula | `_arena_from_spec` in koth_adapter | `harness/koth_adapter.py:69-96` | ~20 |
| 4 | `_roll` frame-capture loop in `_capture_h2h_replays` | `record_match` in record_replay | `record_replay.py:43-83` | ~20 |
| 5 | `_KothTrainingJob` class body | `_PPOTrainingJob` (updated by same diff) | `harness/ppo_trainer.py:134-155` | ~20 |

**Total: ~118 lines of duplicated logic introduced in this diff.**

## Confidence Level
[x] High — all findings verified by direct grep and line-by-line comparison of both
    the diff and the pre-existing source. No finding relies on inference.
