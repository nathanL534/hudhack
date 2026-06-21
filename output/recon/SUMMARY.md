# Recon Summary — hudhack (Crucible)

**Scanned:** /Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack
**Date:** 2026-06-20

## TL;DR
Crucible trains an LLM **Teacher** (Qwen3-4B on Fireworks RFT) to emit JSON parameters for a 2D
platform-fighter RL environment; a small SB3 **PPO Player** trains in the generated arena, and the
strong-vs-weak score gap (band-gated by `p(1-p)>0.2`) is the Teacher's reward. "The harness is the
product" — Teacher/games/Players/Modal are adapters around frozen Pydantic contracts. Built by ~15
parallel AI agents under time pressure, so the spine is clean but there is real duplication/drift.

## Quick Map
- **Frozen contracts:** `contracts.py` (Pydantic: CurriculumSpec, ArenaSpec, ScoringResult, PlayerConfig, ExperimentResult, RewardMode)
- **The one scorer:** `harness/scoring.py::score_curriculum` (gap_proxy = strong−weak if p(1-p)>0.2 else 0)
- **Orchestrator:** `harness/runner.py::DefaultExperimentRunner`
- **Real game:** `games/fighter.py` (+ `harness/fighter_adapter.py`, `harness/ppo_trainer.py`)
- **Teacher/RFT/HUD bridge:** `training/` (ep_remote_server, modal_ep_bridge, hud_teacher_env, fireworks_teacher)
- **Correlation gate (REAL):** `output/modal_player_sweep.py::run_gate` (NOT the documented `eval/proxy_sweep.py`)
- **Run tests:** `.venv/bin/python -m pytest -q` → 45 passed (NEVER `python3`; NEVER `.venv/bin/pytest` — stale shebang)

## Key Insights (decision-relevant)
1. **HUD is decorative, not load-bearing.** Only `from hud import Environment` (one symbol); no HUD API
   call anywhere; `HUD_API_KEY` never read in code; the reward is computed by plain Python
   (`score_teacher_answer`) and shipped to **Fireworks** tracing under the name `hud_reward`. The HUD
   account/CLI authenticates, but Crucible does not route anything through HUD's service. **#1 strategic
   gap for the HUD prize.**
2. **Correlation gate RAN (22:18) and is MARGINAL.** `output/correlation_gate_results.json`:
   Pearson r=0.085 (~zero linear), Spearman ρ=0.249 (weak), n=20 (10 difficulties × 2 seeds),
   agent-labeled "WEAK-PASS". This is NOT a clean GO for Teacher RFT — the cheap proxy only weakly
   ranks curricula the way real PPO learning does.
3. **The trained PPO Player is degenerate.** `inspect_policy_results.json`: win_rate 0.0167 (1/60),
   punch 86% of frames. "Reactive" (29 distinct sequences, entropy 1.55) but a losing punch-spammer.
   This likely poisons the "real learning" side of the correlation gate (noisy/weak learning signal).
4. **The difficulty curve is the strongest real result.** `difficulty_sweep_results.json`: smooth
   monotonic strong-winrate 1.0→0.01, 2 difficulties in the learnable band (0.85, 0.9).
5. **Duplication/drift from 15 agents:** two contract systems (`contracts.py` Pydantic vs `schemas.py`
   TypedDict), three copies of the learnability formula (`harness/scoring.py`, `training/reward.py`,
   `engine/score_env.py`), three Modal "workers" (`modal_player.py` real PPO, `harness/modal_fanout.py`
   dispatcher, `training/modal_ep_bridge.py` placeholder-scorer bridge), and a dead gridworld lineage
   (`engine/`) the live fighter doesn't use.
6. **Modal real PPO worker is live in code** (`modal_player.py::train_player` runs genuine SB3 PPO) —
   HANDOFF.md's claim that it's "fake scores" is STALE.
7. **Only secret read at runtime is `FIREWORKS_API_KEY`.** Anthropic + `fireworks-ai` SDK installed but
   never imported. `gpt-oss-20b` appears nowhere in code.

## Where to read more
The 6 specialist reports were returned inline by read-only Explore agents (no file write). Key sources
on disk: `contracts.py`, `harness/scoring.py`, `output/modal_player_sweep.py`, `training/hud_teacher_env.py`,
`output/correlation_gate_results.json`, `inspect_policy_results.json`, `difficulty_sweep_results.json`.

## Open Questions
- Is the marginal gate (ρ=0.25) good enough to start RFT, or do we strengthen it (more seeds / a Player
  that actually learns) first?
- HUD: make it load-bearing (package fighter as a real HUD env so HUD computes/records the reward) before
  the demo?
- Branch fragmentation: tonight's work is on `sidecar-transport-verify` (7 ahead of origin/main), nothing
  pushed → rag26 is blind. Merge to main?
- Cleanup debt: collapse the duplicate contracts/reward-formulas/Modal-workers, delete dead `engine/` + `schemas.py`?
