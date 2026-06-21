# Stage 6 — Base vs Trained Teacher (held-out transfer decider)

**Claim under test:** the TRAINED Teacher generates environments that cause fresh PPO Players to improve MORE on held-out tasks than the BASE Teacher's environments.

## Verdict: FAIL

- trained beats base on held-out transfer: **None** (delta +0.0000)
- anti-gaming checks passed: **True**
- overall (both required): **False**

> A trained Teacher whose reward rose but that does NOT beat base on held-out transfer is a FAIL, not a success.

## Run

- game: `None`  backend: `None`
- base handle: `None` -> `None`
- trained handle: `None` -> `None`
- config fingerprint: `None` (arenas/model=None, seeds=None, ppo_episodes=None, eval_seeds=None)
- jobs: None  wall-clock: Nones  est. cost: $0.0000

## Per-model held-out transfer

| model | mean | std | 95% CI | n |
|---|---|---|---|---|

## JSON validation / clamping

- generations: 0  clamped/invalid: 0  fraction: 0.0%

## Anti-gaming checks

| check | passed | detail |
|---|---|---|

## Charts

- base_vs_trained: `chart_base_vs_trained.png`
- seed_variance: `chart_seed_variance.png`
