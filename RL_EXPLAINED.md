# Crucible, explained — the dojo story

How the RL works, ground-up, mapped to the real components. For defending it to a
judge and debugging it later.

## The cast
- **The Ring** = the fighter game (`games/fighter.py`). Knock the other off the platform = win (a *ring-out*). The Ring decides wins by its own physics — no opinions. It is the ground truth.
- **The Student** = a small PPO neural net (`harness/ppo_trainer.py`). Learns to fight. ~50k params, *not* an LLM.
- **The Sensei** = Qwen3-4B, the LLM Teacher. Never fights — it **designs training arenas** (the *curriculum*).
- **The Headmaster** = Fireworks RFT. Improves the Sensei over time.

Two nested learners: the Student learns to fight; the Sensei learns to teach.

## 1. How the Student learns (RL, from scratch)
RL = learning by trial and error with a score. Four pieces:
1. **Observation** — positions, velocities, facing (numbers, not pixels).
2. **Action** — one of 5: idle/left/right/jump/punch.
3. **Policy** — the brain: `state → action`. Starts random.
4. **Reward** — won the ring-out? +1 / 0.

The Student plays full matches; the algorithm nudges its weights so button-presses that led toward winning get more likely. Thousands of matches → a fighter. The algorithm is **PPO**; *"proximal"* = small steps per update so it doesn't forget what already works.

## 2. Why you need a Sensei (the learnable band)
How fast the Student learns depends on its sparring partner:
- god-tier opponent → always loses → reward always 0 → no signal.
- brick → always wins → reward always 1 → no signal.
- **fair fight (≈50/50)** → wins *sometimes* → most informative → fastest learning.

That sweet spot is the **learnable band**, quantified as `p·(1−p)` (win-rate p; gate `>0.2` in `harness/scoring.py`), biggest at p=0.5. Empirically the band is around **difficulty ≈ 0.25–0.5** (improvement 0.27–0.42); base Qwen lazily proposes **d=0.7** (improvement only ~0.166). The Sensei's job: design arenas in the band.

## 3. The second learner: teaching the Sensei (the nested loop)
The Sensei is *also* an RL agent, one level up:
- **Observation** — "here's the game + parameter schema, propose an arena."
- **Action** — emit a curriculum: `{"difficulty":0.7,"platform_width":15,...}`.
- **Reward** — **how much a real Student improved** after training in that arena, measured on **held-out arenas the Sensei didn't design** (the `transfer` number).
- **Update** — Fireworks nudges Qwen toward curricula that produced more improvement.

The principle to say out loud: **the writer is not the grader.** The Sensei writes the curriculum; the *world* (the Ring + a real PPO run) grades it. The reward bottoms out in game physics, not an LLM's opinion → **grounded (RLVR)**. The opposite — a model grading itself — is the self-referential trap judges probe for. (Caveat: RLVR removes *that* class of hacking; the residual risk is gaming the *environment* itself — see failure modes.)

## 4. How Fireworks updates Qwen (RFT / GRPO)
An LLM is a **policy over tokens**. Turning one reward into a weight change:
1. Qwen generates a **group** of ~4 curricula for the same prompt (temperature → they differ).
2. Each is scored by the full nested reward (train a Student → test on held-out).
3. **Advantage** = each one's reward minus the group mean (÷ std). Above-average → positive, below → negative.
4. Push token-probabilities **up** for the above-average curricula, **down** for the rest — clipped (small steps) + KL-leashed (stay near original Qwen).

This is **GRPO** (same family as PPO; the "group mean" replaces PPO's critic — a tournament: reward what beats the field). It's RL not SFT because there's no labeled "correct" arena — Qwen discovers it from scores, generating its own training data (the self-improvement loop).

## 5. Why the infrastructure exists (the nervous system)
Each Sensei reward is expensive + distributed:
- **Fireworks** holds Qwen + runs GRPO → calls out via
- **the Eval-Protocol bridge** (`training/ep_remote_server.py`, `modal_ep_bridge.py`) → which fans out to
- **Modal** — trains several PPO Students in parallel (one reward = a whole training run × seeds) →
- **HUD** — records the reward + a trace (the ground-truth ledger) →
- reward flows back to Fireworks → GRPO update → repeat.
Organs: Fireworks = brain being trained, Modal = muscles, HUD = scorekeeper, Ring = physics. Each rollout carries a `rollout_id` so concurrent rewards never cross-contaminate.

## 6. Where we are (honest)
- ✅ Ring, Student, PPO, smooth difficulty curve — proven.
- ✅ Baseline: base Qwen = **0.166**, stuck at d=0.7, low diversity. The "before."
- ✅ Infra: HUD load-bearing, Modal fan-out, bridge serves the real nested reward, Qwen deployable; one real `/init` returned a genuine reward end-to-end.
- ⚠️ The real fine-tune checkpoint isn't produced yet (the `base(smoke)` eval used base-as-trained — that 0.18→0.25 is run-to-run noise, not training).
- ⚠️ Correlation gate weak (ρ≈0.25), partly because the Student is degenerate (punch-spam, ~1.7% win) → noisy learning signal.

## 7. Defense + debug cheat-sheet
**Judge attacks → answers:**
- *"Teacher grades itself?"* → No. Reward = a real PPO Student's improvement on **held-out** arenas in the game. Writer ≠ grader.
- *"Could it game the reward?"* → That's **exploit environments**, the real risk. Defense: reward *transfer to held-out arenas it didn't pick* + watch proxy-reward and true-transfer **co-rise**; if reward climbs while transfer flatlines → gaming → stop.
- *"Proxy correlation is weak."* → Honest: ρ≈0.25, traced to a degenerate Student; the difficulty-curve evidence (d≈0.25 wins) is independent and solid.
- *"What's the RL?"* → Two loops: **PPO** (Student) + **GRPO** (Teacher), each with explicit state/action/reward.

**Debug order when a run misbehaves:**
1. **Diversity** — is Qwen only emitting d=0.7? (mode collapse → flat group → zero GRPO signal).
2. **Validity** — % curricula failing JSON/clamp (>10% = broken Teacher).
3. **rollout_id** — rewards landing on the right curriculum?
4. **Co-rise** — proxy reward ↑ *and* held-out transfer ↑ together? (divergence = gaming).
5. **Student sanity** — real strategy or punch-spam? (degenerate = noisy reward upstream).

**One-breath pitch:** *We train an AI to design training environments for other AIs — graded only by whether the AIs it trained actually got better in the real game, not by anyone's opinion.*
