# Crucible — Current Session Handoff

**Last updated:** June 20, 2026  
**Repository:** `hudhack`  
**Branch:** `main`

This is the source of truth for continuing the current hackathon build in a new
Claude/Codex session. Read this file before the older planning documents.

## Important: stale documents

The parent-directory files `PLAN.md`, `THE_IDEA.md`, `DECISION.md`, `REPITCH.md`,
`BUILD_PLAN.md`, and parts of `BUILD_ORDER.md` describe earlier ideas
(code verification, office-work observation, outreach, or gridworld-only plans).
They are useful history, but they are **not the current implementation spec**.

The current project is **Crucible**, using the fighter simulator in this repo.

## Current idea

Crucible trains an LLM **Teacher** to generate useful RL curricula for another
agent.

1. The Teacher (Qwen3-4B on Fireworks) receives a game description and a fixed
   parameter schema.
2. It emits **JSON parameters**, never arbitrary environment or reward code.
3. Those parameters configure a fighter arena and opponent.
4. A small neural-network **Player** trains in the generated arena using PPO.
5. The arena is valuable only if it produces measurable Player learning that
   transfers beyond one exploitable opponent/configuration.
6. That measured value becomes the Teacher's RFT reward.
7. After Teacher RFT, compare curricula from the trained Teacher with curricula
   from the identical untrained base model on held-out evaluation.

The intended demo is a visual platform-fighting game: two agents fight live in a
Super Smash Bros/Brawlhalla-like arena. The current Stage-1 action set is:

- idle
- left
- right
- jump
- punch

Kick, dodge, block, and duck were discussed but are not implemented yet.

## Component ownership

| Component | Responsibility | Runtime |
|---|---|---|
| Teacher | Generate bounded curriculum/arena JSON | Qwen3-4B on Fireworks |
| HUD environment | Define the Teacher task and authoritative reward boundary | HUD |
| Player | State-vector policy trained with PPO | local now; Modal later |
| Fighter simulator | Physics, observations, actions, win/loss state | Python/NumPy/Gymnasium |
| Modal | Public rollout bridge and later parallel PPO fan-out | Modal |
| Eval Protocol | Connect Fireworks RFT to our remote HUD reward | Fireworks + Modal endpoint |

The Player is **not an LLM**. It is currently a small MLP policy trained using
Stable-Baselines3 PPO. That is intentional: game control needs fast frame-level
actions, not token generation.

## Where HUD actually enters

The live training path is:

```text
Fireworks RFT
    |
    | POST /init with current Qwen checkpoint URL + Teacher prompt
    v
Modal-hosted Eval Protocol bridge
    |
    | calls current Teacher checkpoint
    v
Teacher emits curriculum JSON
    |
    | strict parse, schema validation, parameter clamping
    v
HUD Teacher task/grader
    |
    | invokes cheap proxy or real PPO/Modal validator
    v
numeric HUD reward
    |
    | Status.rollout_finished + hud_reward in Fireworks tracing
    v
Eval Protocol returns score to Fireworks RFT
```

HUD owns the task and reward semantics. Modal executes/hosts work. Fireworks
updates the Teacher weights.

## What is implemented

### Fighter and Player

- `games/fighter.py`
  - deterministic headless 2D platform fighter;
  - seeded spawn jitter;
  - Stage-1 action space;
  - scripted, random, and difficulty-parametric opponents.
- `harness/fighter_adapter.py`
  - real `GameAdapter`;
  - converts frozen `ArenaSpec` values into fighter settings;
  - evaluates policies over fixed seeds.
- `harness/ppo_trainer.py`
  - real local Stable-Baselines3 PPO Player training.
- `prove_ppo_learns.py`
  - proved a PPO Player can improve from 0% to 100% on one configuration;
  - showed a second configuration remains at 0% under the same budget.
- `inspect_policy.py` / `inspect_policy_results.json`
  - found the first successful Player was degenerate/open-loop:
    it repeated one fixed jump/right/punch sequence across all 80 matches.
- `replay.py`, `record_replay.py`, `viewer/viewer.html`
  - replay format, recorder, and visual browser viewer.

### Teacher, HUD, Fireworks, and Modal boundary

- `training/fireworks_teacher.py`
  - normal Fireworks Teacher client;
  - strict JSON parsing and schema clamping;
  - key-free offline transport for tests.
- `training/hud_teacher_env.py`
  - HUD `Environment("crucible-teacher")`;
  - Teacher task and shared reward-grading core.
- `training/ep_remote_server.py`
  - real Eval Protocol `/init` request shape;
  - calls the current Fireworks checkpoint via `request.model_base_url`;
  - preserves rollout IDs;
  - validates Teacher output;
  - logs terminal reward to Fireworks tracing.
- `training/fireworks_rft_eval.py`
  - converts `hud_reward` from remote rollout metadata into `EvaluateResult`.
- `training/modal_ep_bridge.py`
  - Modal ASGI deployment scaffold for the Fireworks-to-HUD bridge;
  - currently contains a **temporary deterministic scorer** only.
- `training/reward_service.py`
  - older internal/debug async scorer;
  - its custom `/result` route is **not** the real Fireworks training transport.
- `harness/modal_fanout.py` and `modal_player.py`
  - Modal fan-out contract and successful smoke worker;
  - `modal_player.py` still returns deterministic fake scores, not real PPO.

### Verification completed

The local protocol test covers:

```text
fake current Teacher checkpoint
  -> real Eval Protocol InitRequest
  -> /init background rollout
  -> strict JSON/clamping
  -> HUD grading core
  -> hud_reward
  -> Eval Protocol evaluation result
```

Current full test result:

```text
42 passed
```

Run with:

```bash
.venv/bin/python -m pytest -q
```

The direct `.venv/bin/pytest` launcher has a stale absolute shebang from when
the repo lived at `/Users/nathaniellee/hudhack`; use `python -m pytest`.

Modal authentication was tested successfully for the `njlee007` workspace, and
a minimal remote function returned `modal-ok`.

## Work currently in progress

`games/fighter.py` has uncommitted difficulty-opponent changes. The old fixed
opponent produced a sharp 0%/100% difficulty cliff. The new design uses:

- low-difficulty self-sabotage (`epsilon` self-edging);
- a high-difficulty defensive `jump_turtle`;
- seeded per-match dodge reliability;
- overlap between the two mechanisms to create a smoother learnability band.

Do not discard or overwrite these uncommitted changes.

The latest full suite passes with these changes, but the actual difficulty curve
and proxy/PPO relationship still need empirical validation.

## Why Teacher RFT must not start yet

Transport can be tested immediately, but real Teacher optimization must wait
until the reward is validated.

If the cheap reward proxy does not predict real PPO improvement, Qwen will learn
to maximize a broken signal. Likely failure modes include:

- arenas that exploit the scripted/random reference agents;
- difficulty settings that score well but teach no transferable behavior;
- reward shaping that produces punch/position farming instead of winning;
- environments tailored to one opponent policy;
- the same open-loop exploit found in the first PPO Player.

Therefore:

```text
Fireworks/HUD transport smoke: can run now
Teacher RFT against placeholder reward: forbidden
Teacher RFT against fighter proxy: only after correlation gate passes
```

## Immediate next steps, in order

### 1. Finish and measure the fighter difficulty curve

Run enough seeds across a difficulty grid, for example:

```text
difficulty = 0.0, 0.1, ..., 1.0
```

Measure weak/random, scripted/strong, and trained-Player win rates. Confirm there
is a useful middle band rather than another cliff.

### 2. Run the real proxy-versus-PPO correlation gate

Use `eval/proxy_sweep.py`, but replace its test lambdas with:

- `proxy_fn`: the proposed cheap hot-loop score;
- `learning_fn`: actual PPO before/after improvement over multiple seeds.

Do not proceed merely because the proxy varies. It must predict real Player
learning. Record Pearson/Spearman correlation and inspect outliers.

### 3. Test one live Fireworks-to-HUD transport rollout

This can happen in parallel with steps 1–2. It proves credentials and network
wiring, not reward validity.

Required secret:

```bash
modal secret create crucible-fireworks FIREWORKS_API_KEY=...
```

Deploy:

```bash
modal deploy training/modal_ep_bridge.py
```

Then point an Eval Protocol `RemoteRolloutProcessor` at the returned Modal URL
and run exactly one Teacher rollout. Verify:

- `/init` returns quickly;
- Qwen emits valid JSON;
- the rollout ID survives end to end;
- `hud_reward` appears in Fireworks tracing;
- Eval Protocol receives the same score.

The temporary scorer in `training/modal_ep_bridge.py` exists only for this
transport test.

### 4. Replace the Modal placeholder scorer

After the correlation gate passes:

- connect the proven fighter proxy to `training/modal_ep_bridge.py`;
- move real PPO into `modal_player.py` for expensive validation/fan-out;
- keep the same serialized result contracts.

### 5. Run tiny Teacher RFT

Start with 5–10 updates, not a long run. Monitor:

- training reward;
- invalid JSON/output rate;
- curriculum parameter diversity;
- held-out PPO improvement;
- proxy versus real-PPO divergence.

Stop immediately if proxy reward rises while real PPO improvement remains flat
or falls.

## Git/worktree state at handoff

At the time this document was written:

- `main` was one commit ahead of `origin/main`;
- there were uncommitted fighter difficulty changes;
- the new HUD/Eval Protocol bridge files were untracked/uncommitted;
- no secrets were committed.

Always run `git status --short --branch` before editing. Preserve unrelated
changes and do not reset the worktree.

## Environment and credentials

- Modal authentication: configured in `~/.modal.toml`, profile `n`.
- Fireworks API key: expected in `.env` or Modal secret
  `crucible-fireworks`; never print or commit it.
- HUD API key: needed for hosted HUD deployment/traces, but not for the local
  HUD task smoke.
- Installed in `.venv`:
  - `hud-python==0.6.6`
  - `eval-protocol==0.3.31`
  - `openai==2.43.0`
  - Modal, FastAPI, Stable-Baselines3, Gymnasium, Torch.

## Rules for the next session

1. Do not return to the older outreach/code-verifier ideas.
2. Do not let the Teacher write arbitrary environment or reward code.
3. Do not train the Teacher against the placeholder Modal scorer.
4. Do not mistake a Player win-rate increase for genuine strategy; inspect
   action usage, seed robustness, opponent diversity, and held-out transfer.
5. Keep the real game outcome (win/loss) separate from shaped intermediate
   signals.
6. Use populations/held-out opponents for evaluation, but understand they reduce
   rather than eliminate reward hacking.
7. Keep the demo visual, but prioritize the causal result:
   trained-Teacher curricula must produce better held-out Players than
   base-Teacher curricula.

## Suggested prompt for a new session

```text
Read HANDOFF.md completely, then inspect git status and the current fighter
difficulty changes. Continue Crucible from the immediate next steps. Preserve
all uncommitted work. The current priority is validating the smooth difficulty
band and the real proxy-vs-PPO correlation gate. In parallel, the
Fireworks->Modal->HUD transport may be smoke-tested, but do not launch Teacher
RFT against the placeholder scorer.
```
