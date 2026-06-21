# Stage 6 — Base Teacher's Student VS Trained Teacher's Student

**Primary claim under test:** the Student trained on the TRAINED Teacher's curricula beats the Student trained on the BASE Teacher's curricula on HELD-OUT arenas, with a curriculum-level 95% CI lower bound above zero.

## Headline verdict: FAIL

- trained-Student win rate: **0.3900**  vs base-Student **0.2967**  (draw 0.3133)
- mean paired advantage: **+0.0933**
- curriculum-level 95% CI: **[-0.8410, +1.0276]** over 3 independent replicates (lower bound must be > 0)
- effect size (Cohen's d): **0.2482**
- anti-circularity checks passed: **False**
- OVERALL HEADLINE (advantage>0 AND CI lower>0 AND anti-circularity): **False**

> The fixed-bot before→after number is SECONDARY evidence only — it does NOT set this headline.

## Side-bias & per-arena-family

- trained-Student win rate as P0: **0.3267**  as P1: **0.4533** (a real edge is side-symmetric)

| arena family | trained paired advantage |
|---|---|
| pw12.0 | +0.2167 |
| pw18.0 | +0.0333 |
| pw10.0 | +0.2500 |
| pw20.0 | -0.1000 |
| pw14.0 | +0.0667 |

## Anti-circularity checks (primary)

| check | result | detail |
|---|---|---|
| ci_lower_above_zero | FAIL | paired-advantage mean +0.0933, 95% CI lower bound -0.8410 <= 0 |
| no_side_bias | PASS | trained win-rate P0=0.3267 P1=0.4533 (gap 0.1267, limit 0.25) |
| multi_arena_benefit | PASS | trained Student ahead on 4/5 arena families (need >1) |
| distinct_students | PASS | base and trained Students have distinct policy weights: base=a5777a401da41ecd trained=b50c6115c392aa02 |
| disjoint_held_out | PASS | 0 held-out arena(s) overlap a Teacher's curriculum (must be 0 — no training-on-the-test) |
| enough_replicates | PASS | 3 independent curriculum replicate(s) (need >1 for a curriculum-level CI) |

## Curriculum replicates

| replicate | trained win | base win | draw | paired adv | overlap |
|---|---|---|---|---|---|
| 0 | 0.290 | 0.340 | 0.370 | -0.0500 | 0 |
| 1 | 0.190 | 0.380 | 0.430 | -0.1900 | 0 |
| 2 | 0.690 | 0.170 | 0.140 | +0.5200 | 0 |

## Run

- game: `target_knockback`  backend: `modal`
- base handle: `modal:base` -> `modal:Qwen/Qwen3-4B`
- trained handle: `modal:runB/update4` -> `modal:Qwen/Qwen3-4B+runB/update4`
- config fingerprint: `f1475dd5b095` (replicates=3, curriculum_arenas=4, match_seeds/arena/side=10, ppo_episodes=2000)

## Secondary transfer diagnostic (fixed-bot before→after)

> Evidence only — a fresh PPO Player trains on each Teacher's arena and is scored on a fixed held-out reference set. Does NOT set the headline.

- trained beats base on fixed-bot transfer: **False** (delta -0.0578)
- secondary anti-gaming passed: **False**

| model | mean transfer | std | 95% CI | n |
|---|---|---|---|---|
| base | +0.2837 | 0.1389 | [+0.2263, +0.3410] | 25 |
| trained | +0.2259 | 0.1490 | [+0.1644, +0.2874] | 25 |

## Charts

- head_to_head_winrate: `chart_head_to_head_winrate.png`
- per_arena_advantage: `chart_per_arena_advantage.png`
- side_bias: `chart_side_bias.png`
- secondary_transfer: `chart_base_vs_trained.png`
- secondary_seed_variance: `chart_seed_variance.png`

## Replays

- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_target_knockback_trained_win_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_target_knockback_base_win_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_target_knockback_draw_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_target_knockback_base_win_trained_as_P1.json` 
