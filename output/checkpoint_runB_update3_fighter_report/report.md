# Stage 6 — Base Teacher's Student VS Trained Teacher's Student

**Primary claim under test:** the Student trained on the TRAINED Teacher's curricula beats the Student trained on the BASE Teacher's curricula on HELD-OUT arenas, with a curriculum-level 95% CI lower bound above zero.

## Headline verdict: FAIL

- trained-Student win rate: **0.3500**  vs base-Student **0.2767**  (draw 0.3733)
- mean paired advantage: **+0.0733**
- curriculum-level 95% CI: **[-0.2887, +0.4353]** over 3 independent replicates (lower bound must be > 0)
- effect size (Cohen's d): **0.5033**
- anti-circularity checks passed: **False**
- OVERALL HEADLINE (advantage>0 AND CI lower>0 AND anti-circularity): **False**

> The fixed-bot before→after number is SECONDARY evidence only — it does NOT set this headline.

## Side-bias & per-arena-family

- trained-Student win rate as P0: **0.3133**  as P1: **0.3867** (a real edge is side-symmetric)

| arena family | trained paired advantage |
|---|---|
| narrow_ledge | -0.0000 |
| wide_arena | -0.0333 |
| floaty_lowg | +0.1667 |
| heavy_knock | +0.1667 |
| high_grav | +0.0667 |

## Anti-circularity checks (primary)

| check | result | detail |
|---|---|---|
| ci_lower_above_zero | FAIL | paired-advantage mean +0.0733, 95% CI lower bound -0.2887 <= 0 |
| no_side_bias | PASS | trained win-rate P0=0.3133 P1=0.3867 (gap 0.0733, limit 0.25) |
| multi_arena_benefit | PASS | trained Student ahead on 3/5 arena families (need >1) |
| distinct_students | PASS | base and trained Students have distinct policy weights: base=e422d86f47c0a01c trained=32f6d4f8b1b39977 |
| disjoint_held_out | PASS | 0 held-out arena(s) overlap a Teacher's curriculum (must be 0 — no training-on-the-test) |
| enough_replicates | PASS | 3 independent curriculum replicate(s) (need >1 for a curriculum-level CI) |

## Curriculum replicates

| replicate | trained win | base win | draw | paired adv | overlap |
|---|---|---|---|---|---|
| 0 | 0.440 | 0.230 | 0.330 | +0.2100 | 0 |
| 1 | 0.230 | 0.140 | 0.630 | +0.0900 | 0 |
| 2 | 0.380 | 0.460 | 0.160 | -0.0800 | 0 |

## Run

- game: `fighter`  backend: `modal`
- base handle: `modal:base` -> `modal:Qwen/Qwen3-4B`
- trained handle: `modal:runB/update3` -> `modal:Qwen/Qwen3-4B+runB/update3`
- config fingerprint: `2ebb08e8a691` (replicates=3, curriculum_arenas=4, match_seeds/arena/side=10, ppo_episodes=1500)

## Charts

- head_to_head_winrate: `chart_head_to_head_winrate.png`
- per_arena_advantage: `chart_per_arena_advantage.png`
- side_bias: `chart_side_bias.png`
- secondary_transfer: `chart_base_vs_trained.png`
- secondary_seed_variance: `chart_seed_variance.png`

## Replays

- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_trained_win_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_base_win_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_draw_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_trained_win_trained_as_P1.json` 
