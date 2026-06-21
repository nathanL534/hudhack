# Stage 6 — Base Teacher's Student VS Trained Teacher's Student

**Primary claim under test:** the Student trained on the TRAINED Teacher's curricula beats the Student trained on the BASE Teacher's curricula on HELD-OUT arenas, with a curriculum-level 95% CI lower bound above zero.

## Headline verdict: FAIL

- trained-Student win rate: **0.4740**  vs base-Student **0.3480**  (draw 0.1780)
- mean paired advantage: **+0.1260**
- curriculum-level 95% CI: **[-0.1374, +0.3894]** over 5 independent replicates (lower bound must be > 0)
- effect size (Cohen's d): **0.5938**
- anti-circularity checks passed: **False**
- OVERALL HEADLINE (advantage>0 AND CI lower>0 AND anti-circularity): **False**

> The fixed-bot before→after number is SECONDARY evidence only — it does NOT set this headline.

## Side-bias & per-arena-family

- trained-Student win rate as P0: **0.4600**  as P1: **0.4880** (a real edge is side-symmetric)

| arena family | trained paired advantage |
|---|---|
| narrow_ledge | -0.3200 |
| wide_arena | +0.2900 |
| floaty_lowg | +0.4100 |
| heavy_knock | +0.4100 |
| high_grav | -0.1600 |

## Anti-circularity checks (primary)

| check | result | detail |
|---|---|---|
| ci_lower_above_zero | FAIL | paired-advantage mean +0.1260, 95% CI lower bound -0.1374 <= 0 |
| no_side_bias | PASS | trained win-rate P0=0.4600 P1=0.4880 (gap 0.0280, limit 0.25) |
| multi_arena_benefit | PASS | trained Student ahead on 3/5 arena families (need >1) |
| distinct_students | PASS | base and trained Students have distinct policy weights: base=92fa9e43a09e2db0 trained=a06ac88715e2d8c0 |
| disjoint_held_out | PASS | 0 held-out arena(s) overlap a Teacher's curriculum (must be 0 — no training-on-the-test) |
| enough_replicates | PASS | 5 independent curriculum replicate(s) (need >1 for a curriculum-level CI) |

## Curriculum replicates

| replicate | trained win | base win | draw | paired adv | overlap |
|---|---|---|---|---|---|
| 0 | 0.420 | 0.410 | 0.170 | +0.0100 | 0 |
| 1 | 0.430 | 0.310 | 0.260 | +0.1200 | 0 |
| 2 | 0.540 | 0.290 | 0.170 | +0.2500 | 0 |
| 3 | 0.590 | 0.190 | 0.220 | +0.4000 | 0 |
| 4 | 0.390 | 0.540 | 0.070 | -0.1500 | 0 |

## Run

- game: `fighter`  backend: `modal`
- base handle: `modal:base` -> `modal:Qwen/Qwen3-4B`
- trained handle: `modal:leagueB128/update3` -> `modal:Qwen/Qwen3-4B+leagueB128/update3`
- config fingerprint: `7ba2fd4e9495` (replicates=5, curriculum_arenas=4, match_seeds/arena/side=10, ppo_episodes=1500)

## Secondary transfer diagnostic (fixed-bot before→after)

> Evidence only — a fresh PPO Player trains on each Teacher's arena and is scored on a fixed held-out reference set. Does NOT set the headline.

- trained beats base on fixed-bot transfer: **False** (delta -0.0127)
- secondary anti-gaming passed: **False**

| model | mean transfer | std | 95% CI | n |
|---|---|---|---|---|
| base | +0.2170 | 0.1661 | [+0.1484, +0.2855] | 25 |
| trained | +0.2043 | 0.1000 | [+0.1630, +0.2455] | 25 |

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
