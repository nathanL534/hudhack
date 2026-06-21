# Contrarian Analysis: Stage-6 Final Verifier Diff Bloat Check
Date: 2026-06-21
Researcher: contrarian

## The View I'm Challenging

The diff is a correct, complete implementation of the Student-vs-Student primary metric.
Every new file, new class, and new config knob was necessary to deliver the specified
change (replace primary metric, keep secondary, add KOTH cross-game, purge fabricated
numbers, update report/dashboard).

---

## Challenge 1: `harness/koth_trainer.py` is a structural twin with no unique logic

**Lines:** 272–461 (entire new file, 185 lines)

The file opens by calling itself "a structural twin of `harness/ppo_trainer.py`." It
copies `_set_global_seeds`, `_make_policy_from_model`, `_KothTrainingJob`, and the full
PPO hyperparameter block (`net_arch=[64,64]`, `n_steps=512`, `batch_size=128`,
`n_epochs=4`, `gamma=0.99`, `lr=3e-4`) verbatim. The only real differences are the env
class (`KothEnv` vs `_MultiArenaFighterEnv`) and the arena type (`KothArena` vs
`FighterArena`). `PPOPlayerTrainer` could have been generalized with an `env_factory`
callable injected at construction — `KothPlayerTrainer` would then be two lines (one
import, one instantiation). The docstring's own framing ("same engine, different game")
is the argument for parameterization, not duplication. As written, if `n_steps=512`
ever needs adjustment, it must change in two places.

**Could it be smaller?** Yes. A parameterized `PPOPlayerTrainer(env_factory=...,
arena_cls=...)` would have eliminated the entire file.

---

## Challenge 2: `EvalConfig` gets a fourth new knob that never needs to differ from the existing one

**Lines:** 1331–1384 (config.py additions, specifically `head_to_head_grid`)

Three new `EvalConfig` fields are genuine: `n_replicates`, `curriculum_arenas`,
`match_seeds_per_arena`. The fourth — `head_to_head_grid` — is structurally identical to
the existing `held_out_grid`. Both control the same "full vs diagonal" choice for the
same disjoint-arena-set concept; the difference is which metric reads the arenas. In
practice, `smoke_config()` and `head_to_head_smoke_config()` differ almost entirely in
`head_to_head_grid="diagonal"` vs not setting it. There is no scenario where you would
run a `full` secondary grid alongside a `diagonal` primary grid (or vice versa) — the
two are always aligned. The fourth field adds a config knob that never needs to vary
independently and generates a separate function (`head_to_head_smoke_config`) whose only
material difference from `smoke_config` is that one new field.

**Could it be smaller?** Yes. Re-use `held_out_grid` for both metrics, or at worst add
a single shared `grid` override. The new function `head_to_head_smoke_config` vanishes.

---

## Challenge 3: `restore_policy` demands a full env when the artifact already encodes the obs/action shape

**Lines:** 680–683 in `modal_player.py` (`_head_to_head_match`); 2382–2417 in
`output/stage6/policy.py` (`restore_policy`)

```python
restore_env = env_cls([arena], seed=int(seed))
trained = restore_policy(artifact, restore_env, ...)
base   = restore_policy(artifact, restore_env, ...)
```

`restore_policy` uses `env` solely so SB3 can infer observation and action spaces when
building the fresh `PPO("MlpPolicy", env, ...)`. But `PolicyArtifact` already carries
`obs_dim`, and the action count is always `Discrete(5)` for every game in this repo
(both fighter and KOTH expose 5 actions). The env construction is not free: it
instantiates a `_MultiArenaKothEnv` or `_MultiArenaFighterEnv` (each of which builds
an inner `KothEnv`/`FighterEnv`) for the sole purpose of being interrogated for its
spaces and then discarded. `restore_policy` could accept `obs_dim` and `n_actions`
directly (or read them from the artifact) and construct a `gymnasium.spaces.Box` /
`Discrete` inline, removing the env import chain from the head-to-head hot path entirely.

**Could it be smaller?** Yes. `restore_policy(artifact, obs_dim=art.obs_dim,
n_actions=5)` — no throwaway env, no `env_cls` import in the worker.

---

## Challenge 4: `_capture_h2h_replays` re-runs matches that `_head_to_head_match` already played

**Lines:** 733–795 in `modal_player.py`

`_head_to_head_match` (lines 653–730) plays every `(arena, seed, side)` pair and
records outcomes in `per_seed`. When `capture_replays=True`, it then calls
`_capture_h2h_replays`, which iterates over the same `match_seeds` and re-runs the
same matches a second time using `FighterSim` directly to capture frames. For a run with
`match_seeds_per_arena=12`, this doubles simulation time on the one captured arena.
Worse, the capture re-run is not guaranteed to replay identically: `_head_to_head_match`
uses the `play_match` primitive (which drives its own `FighterSim`), while
`_capture_h2h_replays` drives `FighterSim` manually with `ReplayBuilder` attached. If
`play_match` does any additional seed manipulation, the captured replay is not the same
match as the scored one. The correct design: attach an optional `ReplayBuilder` inside
`play_match` on the first pass, so outcomes and frames are captured in a single
simulation run.

**Could it be smaller?** Yes. One pass with an optional builder attached eliminates the
second loop entirely and avoids the divergence risk.

---

## Challenge 5: `bootstrap_ci` is computed but gates no decision

**Lines:** 1107–1135 in `output/stage6/aggregate.py`; lines 1823–1825 in
`output/stage6/head_to_head.py`

```python
t_ci    = aggregate.mean_confidence_interval(rep_advantages)
boot_ci = aggregate.bootstrap_ci(rep_advantages, seed=12345)
```

`anti_gaming.check_ci_lower_above_zero` uses `t_ci.low`. The headline field
`ci_lower_bound` uses `t_ci.low`. The dashboard JSON reads `ci_t.low`/`ci_t.high`.
`boot_ci` is written into the result dict as `ci_bootstrap` and nowhere else — no
threshold check, no anti-circularity gate, no dashboard key. The 53-line
`bootstrap_ci` function and its test (`test_bootstrap_ci_brackets_mean_and_single_sample`)
exist to produce a result field that no decision logic reads. If bootstrap is the
preferred estimator it should replace the t-CI as the gate; if t-CI is preferred,
`boot_ci` is dead weight. Carrying two CIs where only one has teeth creates an
appearance of double validation without the substance — reviewers may assume both
bounds must clear zero, but only one does.

**Could it be smaller?** Yes. Pick one CI estimator, gate on it, drop the other.
The 53-line function, its test, and the `ci_bootstrap` result key all disappear.

---

## Confidence in These Challenges

[x] Strong — each challenge identifies a concrete, eliminable redundancy with a clear
    minimal alternative. None of them is "this feature shouldn't exist" — all five
    accept the intent and show the implementation needed fewer lines.

## What Would Vindicate the Mainstream

- Challenge 1 vindicated if `KothPlayerTrainer` genuinely differs from `PPOPlayerTrainer`
  in ways not visible in this diff (e.g., different reward shaping, different opponent
  selection logic upstream).
- Challenge 2 vindicated if there is a documented test scenario requiring `held_out_grid`
  and `head_to_head_grid` to differ.
- Challenge 3 vindicated if SB3's `PPO("MlpPolicy", env, ...)` construction depends on
  env-specific state beyond spaces (e.g., custom wrappers affecting policy internals).
- Challenge 4 vindicated if `play_match` and `FighterSim` produce different frame-level
  state on the same seed, making a second pass necessary for capture.
- Challenge 5 vindicated if `boot_ci` is documented as a manual cross-check for human
  reviewers and is explicitly not intended to gate the automated verdict — but that
  intent is nowhere stated in the diff.

## Sources

No external sources required — findings are based entirely on internal diff analysis.
