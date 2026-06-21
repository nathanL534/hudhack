# Stage 6 — Base Teacher's Student VS Trained Teacher's Student

**Primary claim under test:** the Student trained on the TRAINED Teacher's curricula beats the Student trained on the BASE Teacher's curricula on HELD-OUT arenas, with a curriculum-level 95% CI lower bound above zero.

## Headline verdict: FAIL

- trained-Student win rate: **0.2267**  vs base-Student **0.2167**  (draw 0.5567)
- mean paired advantage: **+0.0100**
- curriculum-level 95% CI: **[-0.5117, +0.5317]** over 3 independent replicates (lower bound must be > 0)
- effect size (Cohen's d): **0.0476**
- anti-circularity checks passed: **False**
- OVERALL HEADLINE (advantage>0 AND CI lower>0 AND anti-circularity): **False**

> The fixed-bot before→after number is SECONDARY evidence only — it does NOT set this headline.

## Side-bias & per-arena-family

- trained-Student win rate as P0: **0.2867**  as P1: **0.1667** (a real edge is side-symmetric)

| arena family | trained paired advantage |
|---|---|
| narrow_ledge | +0.2167 |
| wide_arena | -0.1000 |
| floaty_lowg | -0.0167 |
| heavy_knock | -0.0167 |
| high_grav | -0.0333 |

## Anti-circularity checks (primary)

| check | result | detail |
|---|---|---|
| ci_lower_above_zero | FAIL | paired-advantage mean +0.0100, 95% CI lower bound -0.5117 <= 0 |
| no_side_bias | PASS | trained win-rate P0=0.2867 P1=0.1667 (gap 0.1200, limit 0.25) |
| multi_arena_benefit | FAIL | trained Student ahead on 1/5 arena families (need >1) |
| distinct_students | PASS | base and trained Students have distinct policy weights: base=e422d86f47c0a01c trained=6a00e0f6b90762ac |
| disjoint_held_out | PASS | 0 held-out arena(s) overlap a Teacher's curriculum (must be 0 — no training-on-the-test) |
| enough_replicates | PASS | 3 independent curriculum replicate(s) (need >1 for a curriculum-level CI) |

## Curriculum replicates

| replicate | trained win | base win | draw | paired adv | overlap |
|---|---|---|---|---|---|
| 0 | 0.470 | 0.250 | 0.280 | +0.2200 | 0 |
| 1 | 0.120 | 0.110 | 0.770 | +0.0100 | 0 |
| 2 | 0.090 | 0.290 | 0.620 | -0.2000 | 0 |

## Run

- game: `fighter`  backend: `modal`
- base handle: `modal:base` -> `modal:Qwen/Qwen3-4B`
- trained handle: `modal:runB/update4` -> `modal:Qwen/Qwen3-4B+runB/update4`
- config fingerprint: `234a2fd78cc1` (replicates=3, curriculum_arenas=4, match_seeds/arena/side=10, ppo_episodes=1500)

## Secondary transfer diagnostic (fixed-bot before→after)

> Evidence only — a fresh PPO Player trains on each Teacher's arena and is scored on a fixed held-out reference set. Does NOT set the headline.

- trained beats base on fixed-bot transfer: **True** (delta +0.1435)
- secondary anti-gaming passed: **False**

| model | mean transfer | std | 95% CI | n |
|---|---|---|---|---|
| base | +0.1925 | 0.1393 | [+0.1350, +0.2500] | 25 |
| trained | +0.3360 | 0.1273 | [+0.2835, +0.3885] | 25 |

## Charts

- head_to_head_winrate: `chart_head_to_head_winrate.png`
- per_arena_advantage: `chart_per_arena_advantage.png`
- side_bias: `chart_side_bias.png`
- secondary_transfer: `chart_base_vs_trained.png`
- secondary_seed_variance: `chart_seed_variance.png`

## Replays

- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_trained_win_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_draw_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/replays/h2h_fighter_base_win_trained_as_P1.json` 
