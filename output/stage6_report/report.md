# Stage 6 — Base Teacher's Student VS Trained Teacher's Student

**Primary claim under test:** the Student trained on the TRAINED Teacher's curricula beats the Student trained on the BASE Teacher's curricula on HELD-OUT arenas, with a curriculum-level 95% CI lower bound above zero.

## Headline verdict: FAIL

- trained-Student win rate: **0.4480**  vs base-Student **0.4180**  (draw 0.1340)
- mean paired advantage: **+0.0300**
- curriculum-level 95% CI: **[-0.3215, +0.3815]** over 5 independent replicates (lower bound must be > 0)
- effect size (Cohen's d): **0.106**
- anti-circularity checks passed: **False**
- OVERALL HEADLINE (advantage>0 AND CI lower>0 AND anti-circularity): **False**

> The fixed-bot before→after number is SECONDARY evidence only — it does NOT set this headline.

## Side-bias & per-arena-family

- trained-Student win rate as P0: **0.6280**  as P1: **0.2680** (a real edge is side-symmetric)

| arena family | trained paired advantage |
|---|---|
| narrow_ledge | +0.0600 |
| wide_arena | -0.1900 |
| floaty_lowg | +0.1500 |
| heavy_knock | +0.0800 |
| high_grav | +0.0500 |

## Anti-circularity checks (primary)

| check | result | detail |
|---|---|---|
| ci_lower_above_zero | FAIL | paired-advantage mean +0.0300, 95% CI lower bound -0.3215 <= 0 |
| no_side_bias | FAIL | trained win-rate P0=0.6280 P1=0.2680 (gap 0.3600, limit 0.25) — side-position dependency detected |
| multi_arena_benefit | PASS | trained Student ahead on 4/5 arena families (need >1) |
| distinct_students | PASS | base and trained Students have distinct policy weights: base=52bc844159f44ef1 trained=57b33ba292dade6f |
| disjoint_held_out | PASS | 0 held-out arena(s) overlap a Teacher's curriculum (must be 0 — no training-on-the-test) |
| enough_replicates | PASS | 5 independent curriculum replicate(s) (need >1 for a curriculum-level CI) |

## Curriculum replicates

| replicate | trained win | base win | draw | paired adv | overlap |
|---|---|---|---|---|---|
| 0 | 0.260 | 0.630 | 0.110 | -0.3700 | 0 |
| 1 | 0.650 | 0.280 | 0.070 | +0.3700 | 0 |
| 2 | 0.380 | 0.430 | 0.190 | -0.0500 | 0 |
| 3 | 0.410 | 0.430 | 0.160 | -0.0200 | 0 |
| 4 | 0.540 | 0.320 | 0.140 | +0.2200 | 0 |

## Run

- game: `fighter`  backend: `modal`
- base handle: `modal:base` -> `modal:Qwen/Qwen3-4B`
- trained handle: `modal:leagueB256/update3` -> `modal:Qwen/Qwen3-4B+leagueB256/update3`
- config fingerprint: `c389af9adda4` (replicates=5, curriculum_arenas=4, match_seeds/arena/side=10, ppo_episodes=1500)

## Secondary transfer diagnostic (fixed-bot before→after)

> Evidence only — a fresh PPO Player trains on each Teacher's arena and is scored on a fixed held-out reference set. Does NOT set the headline.

- trained beats base on fixed-bot transfer: **True** (delta +0.0570)
- secondary anti-gaming passed: **False**

| model | mean transfer | std | 95% CI | n |
|---|---|---|---|---|
| base | +0.1831 | 0.1496 | [+0.1131, +0.2531] | 20 |
| trained | +0.2401 | 0.1334 | [+0.1662, +0.3139] | 15 |

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
