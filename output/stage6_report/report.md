# Stage 6 — Base vs Trained Teacher (held-out transfer decider)

**Claim under test:** the TRAINED Teacher generates environments that cause fresh PPO Players to improve MORE on held-out tasks than the BASE Teacher's environments.

## Verdict: FAIL

- trained beats base on held-out transfer: **False** (delta +0.0000)
- anti-gaming checks passed: **False**
- overall (both required): **False**

> A trained Teacher whose reward rose but that does NOT beat base on held-out transfer is a FAIL, not a success.

## Run

- game: `fighter`  backend: `modal`
- base handle: `modal:base` -> `modal:Qwen/Qwen3-4B`
- trained handle: `modal:update2` -> `modal:Qwen/Qwen3-4B+update2`
- config fingerprint: `bfacbd5bc8a5` (arenas/model=3, seeds=[1, 2, 3], ppo_episodes=300, eval_seeds=12)
- jobs: 18  wall-clock: 65.47s  est. cost: $0.1800

## Per-model held-out transfer

| model | mean | std | 95% CI | n |
|---|---|---|---|---|
| base | +0.0889 | 0.0882 | [+0.0211, +0.1567] | 9 |
| trained | +0.0889 | 0.0882 | [+0.0211, +0.1567] | 9 |

## Parameter diversity (trained Teacher)

- unique configs: 1/3 (fraction 0.333333)
- mean per-knob std: 0.0
- collapsed to one config: True

## JSON validation / clamping

- generations: 6  clamped/invalid: 0  fraction: 0.0%

## Anti-gaming checks

| check | passed | detail |
|---|---|---|
| reward_up_but_no_transfer | FAIL | held-out transfer trained(+0.0889) <= base(+0.0889) |
| arena_collapse | FAIL | trained Teacher emitted 1/3 unique arena configs (COLLAPSED to one config) |
| excessive_clamping | PASS | invalid/clamped JSON fraction 0.0% (limit 10%) |
| seed_noise_dominates | FAIL | seed std 0.0882 > |mean improvement| 0.0000 |
| train_game_only | PASS | no held-out probe games were run (check skipped) |
| same_model | PASS | base and trained resolved to distinct models: base=modal:Qwen/Qwen3-4B trained=modal:Qwen/Qwen3-4B+update2 |

## Per-game

| game | role | base | trained | delta |
|---|---|---|---|---|
| fighter | teacher | +0.0889 | +0.0889 | +0.0000 |

## Charts

- base_vs_trained: `chart_base_vs_trained.png`
- per_game: `chart_per_game.png`
- seed_variance: `chart_seed_variance.png`
