# Defensive Overkill Audit — Stage-6 RL Evaluation Harness

Audited diff at `/tmp/bloat_code_diff.txt`. All findings are genuinely unnecessary; no real defects are reported.

---

**Finding 1 — Redundant empty-arenas guard after an assertion already fires**

`harness/koth_trainer.py` line 168: `mean_score = sum(arena_scores) / len(arena_scores) if arena_scores else 0.0`.
`arena_scores` is built as `[self._winrate(a, policy) for a in koth_arenas]` on line 167. `koth_arenas` can only be non-empty at this point because `_MultiArenaKothEnv.__init__` raises `AssertionError` on line 75 (`assert arenas, "need at least one arena to train on"`) if `koth_arenas` were empty — and that constructor is called three lines earlier on line 150. An empty list cannot reach line 168. The identical pattern appears in the unchanged `ppo_trainer.py` line 209, so this was copied defensively rather than derived from a real risk. The `if arena_scores else 0.0` branch is dead code.

**Finding 2 — Redundant `isinstance` branch in `restore_policy` at a call site that always passes the concrete type**

`output/stage6/policy.py` line 166: `art = artifact if isinstance(artifact, PolicyArtifact) else PolicyArtifact.from_dict(artifact)`. Both call sites in this diff — `modal_player.py` lines 607 and 611 — already call `PolicyArtifact.from_dict(payload[...])` before passing the result to `restore_policy`, so `artifact` is always a `PolicyArtifact` by the time the isinstance check runs. The union type `PolicyArtifact | dict` in the signature and the runtime branch exist purely defensively against callers that do not exist. If the intent is to support dict-passing in the future, the type annotation should reflect that, but the runtime branch is dead weight today.

**Finding 3 — Triple-repeated `max(1, ...)` denominator guard on a value that cannot be zero**

`output/stage6/head_to_head.py` lines 244-253: `overall_trained_winrate`, `overall_base_winrate`, and `overall_draw_rate` each divide by `max(1, sum(rep["_total_matches"] for rep in replicate_rows))`. `replicate_rows` always has exactly `cfg.n_replicates` entries (the loop on line 190 appends one unconditionally per replicate). Each replicate's `_total_matches` is `trained_wins + base_wins + draws`, and those counts accumulate over `len(held_full) * cfg.match_seeds_per_arena * 2` match outcomes — a product that is positive for any legal config (both `n_held_out_arenas` and `match_seeds_per_arena` must be >= 1 for the run to reach this point, enforced by the CLI at `--replicates >= 1` and `--match-seeds >= 1`). The `max(1, ...)` guard for a zero-total case that cannot occur is applied three times in a row on the same denominator, adding noise without guarding anything real.

**Finding 4 — Validation of already-validated external data inside an internal function**

`modal_player.py` line 515-516: `_train_student_policy` raises `ValueError("no curriculum arenas to train the Student on")` if `arenas` is empty after building from `specs`. However, `specs` is derived from `payload.get("curriculum_arenas") or [payload]` on line 513 — the `or [payload]` fallback means `specs` is never empty, and `_arenas_from_specs` returns one arena per spec unless a spec has zero keys in `field_names`, which cannot happen because every concrete arena dataclass (`FighterArena`, `KothArena`) has at least `difficulty`. Every call site in this diff that invokes `_train_student_policy` or `local_train_student_worker` routes through `head_to_head._run_one_replicate`, which builds `curriculum_arenas` from teacher-generated arena dicts that are validated and clamped by `generate_arenas` before being placed in the payload. The validation at line 515 is checking a condition the two layers above it already exclude.

**Finding 5 — Error path added for a failure mode excluded by the call contract**

`modal_player.py` lines 643-650: `_model_of(job)` uses `getattr(job, "model", None)` and raises `AttributeError` with a detailed message if `.model` is `None`. In this diff, `_model_of` is called only once, on line 621, immediately after `job = trainer.submit(config, train_arenas)` on line 617. `trainer` is either `KothPlayerTrainer` or `PPOPlayerTrainer`, both of which construct `_KothTrainingJob` / `_PPOTrainingJob` with `model=model` attached unconditionally (koth_trainer.py line 177, ppo_trainer.py line 215 in the diff). The `None` branch inside `_model_of` cannot be reached from any in-process caller; it would only matter if a third trainer forgot to set `.model`, a case that does not exist in this codebase. The wrapper function and its error text are defensive against an imaginary third trainer rather than a real failure mode in the current call graph.
