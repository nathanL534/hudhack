# Stage 6 — Base vs Trained Teacher (held-out transfer decider)

**Claim under test:** the TRAINED Teacher generates environments that cause fresh PPO Players to improve MORE on held-out tasks than the BASE Teacher's environments.

## Verdict: FAIL

- trained beats base on held-out transfer: **False** (delta +0.0000)
- anti-gaming checks passed: **False**
- overall (both required): **False**

> A trained Teacher whose reward rose but that does NOT beat base on held-out transfer is a FAIL, not a success.

## Run

- game: `fighter`  backend: `local`
- base handle: `offline` -> `offline:base`
- trained handle: `offline` -> `offline:trained`
- config fingerprint: `017adf9e93b8` (arenas/model=2, seeds=[1], ppo_episodes=40, eval_seeds=5)
- jobs: 4  wall-clock: 13.32s  est. cost: $0.0400

## Per-model held-out transfer

| model | mean | std | 95% CI | n |
|---|---|---|---|---|
| base | +0.1200 | 0.0000 | [+0.1200, +0.1200] | 2 |
| trained | +0.1200 | 0.0000 | [+0.1200, +0.1200] | 2 |

## Parameter diversity (trained Teacher)

- unique configs: 1/2 (fraction 0.5)
- mean per-knob std: 0.0
- collapsed to one config: True

## JSON validation / clamping

- generations: 4  clamped/invalid: 0  fraction: 0.0%

## Anti-gaming checks

| check | passed | detail |
|---|---|---|
| reward_up_but_no_transfer | FAIL | held-out transfer trained(+0.1200) <= base(+0.1200) |
| arena_collapse | FAIL | trained Teacher emitted 1/2 unique arena configs (COLLAPSED to one config) |
| excessive_clamping | PASS | invalid/clamped JSON fraction 0.0% (limit 10%) |
| seed_noise_dominates | PASS | seed std 0.0000 <= |mean improvement| 0.0000 |
| train_game_only | PASS | no held-out probe games were run (check skipped) |
| same_model | PASS | base and trained resolved to distinct models: base=offline:base trained=offline:trained |

## Per-game

| game | role | base | trained | delta |
|---|---|---|---|---|
| fighter | teacher | +0.1200 | +0.1200 | +0.0000 |

## Charts

- base_vs_trained: `chart_base_vs_trained.png`
- per_game: `chart_per_game.png`
- seed_variance: `chart_seed_variance.png`
