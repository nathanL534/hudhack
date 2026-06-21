# Dojo

**An LLM Teacher learns to generate RL curricula that make a fresh student fighter
better — and we can show the student it trains beats the student a base model
trains.**

This is **nested reinforcement learning**. A Qwen3-4B Teacher (LoRA, GRPO-trained)
emits the parameters of a **Ring-Out** fighting arena (platform width, gravity,
knockback, spawn gap, opponent difficulty). For each arena we train a fresh **PPO
student** from scratch and reward the Teacher by how much that student *improves on
a fixed held-out reference* — i.e. the Teacher is graded on the **learnability** of
the worlds it designs, not on any single match.

The **harness is the product**: Teacher, games, and Players are adapters around
frozen, seed-addressable contracts, so any piece can be swapped (fake ↔ real,
local ↔ Modal) without touching the rest.

---

## What actually runs (real, not fakes)

- **Real Teacher training on Modal** — Qwen3-4B + LoRA, GRPO over a group of
  generated arenas, group-relative advantages, one combined update
  (`training/train_teacher_modal.py`). Adapters are namespaced per run on a Modal
  volume.
- **Real PPO students** — Stable-Baselines3 / gymnasium, trained per arena and in
  head-to-head, on Modal GPU/CPU workers (`modal_player.py`).
- **Real reward instrument** — held-out improvement on fixed reference
  difficulties, behind a **behavior gate** (the student must actually move, punch,
  and win by real ring-out — not camp or time out) and a **decisive-reward** rule
  (`output/nested_reward.py`, `games/`).
- **Dojo replay viewer** — canvas renderer with real sprites over a painted dojo
  background; plays the head-to-head traces in `replays/` (`viewer/viewer.html`).

---

## Headline result

Trained Teacher's student **beats** the base Teacher's student, head-to-head on a
focused-league population, **replicated across two checkpoints**:

| metric | result |
|---|---|
| Student-vs-student advantage | **+0.124** (replicated at update3 *and* update5) |
| Draw rate | **14%** — down from 48–80% before the league + decisive-reward fix |
| Evaluator bias | **none** — base-vs-base null = **0.000**, side-gap 0.145 |
| Student behavior | moves 29–83% of frames, real ring-outs 51–80% |

**Honesty:** this is a **consistent directional win**, not a significance-passing
one — at the fast eval budget the 95% CI lower bound still crosses zero. The claim
we stand behind is "trained beats base across independent checks under a
provably-unbiased evaluator," which the data supports.

### The Teacher reward-hack we caught and fixed

Left alone, the GRPO Teacher learned to **collapse its curriculum** — emit one
arena many times — because identical high-scoring arenas survive GRPO's reward
normalization. A naive `×0.5` penalty does nothing (normalization erases it). The
fix in `training/anti_collapse.py`:

1. **oversample** 16 candidate arenas → **select 8 by max parameter distance**
   (≥4 unique or skip the update),
2. **zero the reward** of exact duplicates (this *survives* normalization),
3. add a **novelty bonus** proportional to mean pairwise distance,
4. gate every reward on **real fighting behavior**.

A fresh-from-base retrain with this mechanism produces **7 distinct arenas with
students that fight** (vs the collapsed single point-mass).

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Watch a fight** — open `viewer/viewer.html` in a browser. It plays the
trained-win traces from `replays/` (`replays/demo_locked/` holds verified backups).

**Reproduce the head-to-head** (trained student vs base student, focused league):

```bash
PYTHONPATH=. python output/eval_base_vs_trained.py \
  --game fighter --base modal:base --trained modal:leagueB128/update5 \
  --student-cfg-preset focused256 --replicates 5 --curriculum-arenas 5 \
  --episodes 1500 --match-seeds 10 --out output/h2h.json
```

**Train the Teacher** (Modal, real Qwen3-4B GRPO):

```bash
python training/train_teacher_modal.py --games ring_out \
  --student-cfg-preset focused256 --anti-collapse --updates 5
```

No secrets are committed — `.env`, `.mcp.json`, and `config/` are gitignored;
Modal/Fireworks credentials come from your own environment.

---

## Layout

```
games/                 Ring-Out + KOTH simulators, opponent league, reward shaping
training/              Teacher GRPO trainer (Modal), anti-collapse mechanism
modal_player.py        PPO student workers (inner reward + head-to-head)
output/                eval CLIs, Stage-6 head-to-head, reports
replays/               head-to-head traces (+ demo_locked/ verified backups)
viewer/                dojo replay viewer
contracts.py           frozen, seed-addressable Pydantic contracts
```

## Status & next

Proven: real nested-RL training loop, unbiased head-to-head, draw-camping fixed,
curriculum-collapse diagnosed and fixed. Next: tighten the CI with a larger eval
budget, and run the **cross-game KOTH** transfer gate on the fixed checkpoints
(the held-out second game is wired but its quantitative gate is not yet re-run).
