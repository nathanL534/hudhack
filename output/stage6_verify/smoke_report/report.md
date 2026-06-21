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
- trained handle: `offline` -> `offline:base(smoke)`
- config fingerprint: `9feb2c3b4e5a` (arenas/model=2, seeds=[1, 2], ppo_episodes=300, eval_seeds=10)
- jobs: 8  wall-clock: 61.88s  est. cost: $0.0800

## Per-model held-out transfer

| model | mean | std | 95% CI | n |
|---|---|---|---|---|
| base | +0.1000 | 0.0462 | [+0.0265, +0.1735] | 4 |
| base(smoke) | +0.1000 | 0.0462 | [+0.0265, +0.1735] | 4 |

## Parameter diversity (trained Teacher)

- unique configs: 1/2 (fraction 0.5)
- mean per-knob std: 0.0
- collapsed to one config: True

## JSON validation / clamping

- generations: 4  clamped/invalid: 0  fraction: 0.0%

## Anti-gaming checks

| check | passed | detail |
|---|---|---|
| reward_up_but_no_transfer | FAIL | held-out transfer trained(+0.1000) <= base(+0.1000) |
| arena_collapse | FAIL | trained Teacher emitted 1/2 unique arena configs (COLLAPSED to one config) |
| excessive_clamping | PASS | invalid/clamped JSON fraction 0.0% (limit 10%) |
| seed_noise_dominates | FAIL | seed std 0.0462 > |mean improvement| 0.0000 |
| train_game_only | PASS | no held-out probe games were run (check skipped) |
| same_model | PASS | smoke mode: base==trained by design (offline:base) |

## Per-game

| game | role | base | trained | delta |
|---|---|---|---|---|
| fighter | teacher | +0.1000 | +0.1000 | +0.0000 |

## Charts

- base_vs_trained: `chart_base_vs_trained.png`
- per_game: `chart_per_game.png`
- seed_variance: `chart_seed_variance.png`
