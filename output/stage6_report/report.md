# Stage 6 — Base Teacher's Student VS Trained Teacher's Student

**Primary claim under test:** the Student trained on the TRAINED Teacher's curricula beats the Student trained on the BASE Teacher's curricula on HELD-OUT arenas, with a curriculum-level 95% CI lower bound above zero.

## Headline verdict: FAIL

- trained-Student win rate: **0.4417**  vs base-Student **0.4417**  (draw 0.1167)
- mean paired advantage: **+0.0000**
- curriculum-level 95% CI: **[+0.0000, +0.0000]** over 2 independent replicates (lower bound must be > 0)
- effect size (Cohen's d): **None**
- anti-circularity checks passed: **False**
- OVERALL HEADLINE (advantage>0 AND CI lower>0 AND anti-circularity): **False**

> The fixed-bot before→after number is SECONDARY evidence only — it does NOT set this headline.

## Side-bias & per-arena-family

- trained-Student win rate as P0: **0.6000**  as P1: **0.2833** (a real edge is side-symmetric)

| arena family | trained paired advantage |
|---|---|
| narrow_ledge | +0.0000 |
| wide_arena | +0.0000 |
| floaty_lowg | +0.0000 |
| heavy_knock | +0.0000 |
| high_grav | +0.0000 |

## Anti-circularity checks (primary)

| check | result | detail |
|---|---|---|
| ci_lower_above_zero | FAIL | paired-advantage mean +0.0000, 95% CI lower bound +0.0000 <= 0 |
| no_side_bias | FAIL | trained win-rate P0=0.6000 P1=0.2833 (gap 0.3167, limit 0.25) — side-position dependency detected |
| multi_arena_benefit | FAIL | trained Student ahead on 0/5 arena families (need >1) |
| distinct_students | PASS | smoke mode: identical-checksum Students tolerated (d821d0f78f891572) |
| disjoint_held_out | PASS | 0 held-out arena(s) overlap a Teacher's curriculum (must be 0 — no training-on-the-test) |
| enough_replicates | PASS | 2 independent curriculum replicate(s) (smoke: single replicate is a smoke, not a verdict) |

## Curriculum replicates

| replicate | trained win | base win | draw | paired adv | overlap |
|---|---|---|---|---|---|
| 0 | 0.417 | 0.417 | 0.167 | +0.0000 | 0 |
| 1 | 0.467 | 0.467 | 0.067 | +0.0000 | 0 |

## Run

- game: `fighter`  backend: `modal`
- base handle: `modal:base` -> `modal:Qwen/Qwen3-4B`
- trained handle: `modal:base` -> `modal:Qwen/Qwen3-4B`
- config fingerprint: `8d0110320c89` (replicates=2, curriculum_arenas=3, match_seeds/arena/side=12, ppo_episodes=1500)

## Secondary transfer diagnostic (fixed-bot before→after)

> Evidence only — a fresh PPO Player trains on each Teacher's arena and is scored on a fixed held-out reference set. Does NOT set the headline.

- trained beats base on fixed-bot transfer: **False** (delta +0.0000)
- secondary anti-gaming passed: **False**

| model | mean transfer | std | 95% CI | n |
|---|---|---|---|---|
| base | +0.0983 | 0.0540 | [+0.0124, +0.1842] | 4 |
| base(smoke) | +0.0983 | 0.0540 | [+0.0124, +0.1842] | 4 |

## Charts

- head_to_head_winrate: `chart_head_to_head_winrate.png`
- per_arena_advantage: `chart_per_arena_advantage.png`
- side_bias: `chart_side_bias.png`
- secondary_transfer: `chart_base_vs_trained.png`
- secondary_seed_variance: `chart_seed_variance.png`

## Replays

- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_trained_win_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_base_win_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_base_win_trained_as_P1.json` 
