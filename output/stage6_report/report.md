# Stage 6 — Base Teacher's Student VS Trained Teacher's Student

**Primary claim under test:** the Student trained on the TRAINED Teacher's curricula beats the Student trained on the BASE Teacher's curricula on HELD-OUT arenas, with a curriculum-level 95% CI lower bound above zero.

## Headline verdict: FAIL

- trained-Student win rate: **0.0333**  vs base-Student **0.0417**  (draw 0.9250)
- mean paired advantage: **-0.0083**
- curriculum-level 95% CI: **[-0.5377, +0.5211]** over 2 independent replicates (lower bound must be > 0)
- effect size (Cohen's d): **-0.1414**
- anti-circularity checks passed: **False**
- OVERALL HEADLINE (advantage>0 AND CI lower>0 AND anti-circularity): **False**

> The fixed-bot before→after number is SECONDARY evidence only — it does NOT set this headline.

## Side-bias & per-arena-family

- trained-Student win rate as P0: **0.0667**  as P1: **0.0000** (a real edge is side-symmetric)

| arena family | trained paired advantage |
|---|---|
| narrow_ledge | +0.0000 |
| wide_arena | +0.0000 |
| floaty_lowg | +0.0000 |
| heavy_knock | -0.0417 |
| high_grav | +0.0000 |

## Anti-circularity checks (primary)

| check | result | detail |
|---|---|---|
| ci_lower_above_zero | FAIL | paired-advantage mean -0.0083, 95% CI lower bound -0.5377 <= 0 |
| no_side_bias | FAIL | trained win-rate P0=0.0667 P1=0.0000 (gap 0.0667, limit 0.25) — side-position dependency detected |
| multi_arena_benefit | FAIL | trained Student ahead on 0/5 arena families (need >1) |
| distinct_students | PASS | smoke mode: identical-checksum Students tolerated (c4dd37c39ce2b12b) |
| disjoint_held_out | PASS | 0 held-out arena(s) overlap a Teacher's curriculum (must be 0 — no training-on-the-test) |
| enough_replicates | PASS | 2 independent curriculum replicate(s) (smoke: single replicate is a smoke, not a verdict) |

## Curriculum replicates

| replicate | trained win | base win | draw | paired adv | overlap |
|---|---|---|---|---|---|
| 0 | 0.017 | 0.067 | 0.917 | -0.0500 | 0 |
| 1 | 0.050 | 0.017 | 0.933 | +0.0333 | 0 |

## Run

- game: `fighter`  backend: `local`
- base handle: `offline:{"difficulty":0.3,"platform_width":10.0,"gravity":0.6,"knockback":2.0,"spawn_gap":3.5}` -> `offline:base`
- trained handle: `offline:{"difficulty":0.6,"platform_width":14.0,"gravity":0.7,"knockback":3.5,"spawn_gap":5.0}` -> `offline:base(smoke)`
- config fingerprint: `694b5b4f7ffa` (replicates=2, curriculum_arenas=2, match_seeds/arena/side=6, ppo_episodes=150)

## Cross-game generalization (KOTH — final hidden test)

- mapping: fighter-schema -> KOTH zone geometry (koth_adapter mapping)
- FRESH KOTH Students (obs_dim 16) trained from each Teacher's KOTH-mapped curricula, fought head-to-head (NOT forced fighter weights)
- trained-KOTH-Student win rate: **0.4167** vs base **0.4375**
- mean paired advantage: **-0.0208**  95% CI [-0.2855, +0.2439]

## Charts

- head_to_head_winrate: `chart_head_to_head_winrate.png`
- per_arena_advantage: `chart_per_arena_advantage.png`
- side_bias: `chart_side_bias.png`
- secondary_transfer: `chart_base_vs_trained.png`
- secondary_seed_variance: `chart_seed_variance.png`

## Replays

- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/.worktrees/stage6-final-verifier/replays/h2h_fighter_trained_win_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/.worktrees/stage6-final-verifier/replays/h2h_fighter_base_win_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/.worktrees/stage6-final-verifier/replays/h2h_fighter_draw_trained_as_P0.json` 
- `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/.worktrees/stage6-final-verifier/replays/h2h_fighter_draw_trained_as_P1.json` 
