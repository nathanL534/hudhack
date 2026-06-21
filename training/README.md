# Stages 2–4 — Teacher / Reward Service / Player Fan-out

Credential-free scaffolding for the RFT loop. Everything here runs and is tested
**without** a Fireworks key or Modal creds; the live hooks are documented below.

## What's here

| File | Role | Live dependency |
|------|------|-----------------|
| `training/fireworks_teacher.py` | Fireworks Qwen3-4B Teacher; offline mock; `CurriculumSpec` bridge | `FIREWORKS_API_KEY` |
| `training/reward_service.py` | Async `/init` reward service (the Eval-Protocol seam) | eval-protocol hookup |
| `eval/proxy_sweep.py` | Proxy-vs-PPO-gain correlation sweep | (none; fakes) |
| `harness/modal_fanout.py` | Local/Modal seed fan-out + `PlayerTrainer` bridge + `ExperimentResult` aggregation | Modal creds (Modal mode only) |
| `modal_player.py` | Deployable Modal worker + `local_smoke_worker` fallback | Modal creds + real PPO body |

All of these import the frozen types from `contracts.py` and never redefine them.

## The `/init` correctness fix (the silent-corruption guard)

`reward_service.py`'s `/init` handler is `async def`, reads `rollout_id` from the
top level **or** the `metadata` envelope (modeled by the frozen `ScoringContext`),
validates any incoming `curriculum` against the frozen `CurriculumSpec` at parse
time (422 on a malformed one, **before** scheduling work), returns a fast `202`,
and runs the CPU-bound scorer in a per-rollout background task via
`loop.run_in_executor`. Every background log line is tagged with that rollout's
`rollout_id`, so concurrent `n=4` rollouts can never cross-tag rewards (A's reward
filed under B's id). Finished rewards are collectable at `GET /result/{rollout_id}`.

## Run the tests (no keys needed)

```bash
. .venv/bin/activate
python -m pytest test_stage2_4.py -v        # 13 passing
```

---

## GO-LIVE CHECKLIST

What's still required to make each component LIVE. Items marked **(key)** need a
credential; the rest is wiring.

### 1. Fireworks Teacher — make it real

- [ ] **(key)** Set `FIREWORKS_API_KEY` in `.env` (see `.env.example`).
- [ ] Confirm the model id `accounts/fireworks/models/qwen3-4b` is the one you
      provisioned (override via `FireworksTeacher(model=...)`).
- [ ] Swap `default_teacher()` (in `training/teacher.py`, owned by the Teacher
      dev) to return `FireworksTeacher()` instead of `RandomTeacher`, OR call
      `FireworksTeacher()` directly from `training/rft_run.py`. Until then the
      pipeline runs with `FireworksTeacher.offline(...)` / `RandomTeacher`.
- [ ] Smoke a single live call: `FireworksTeacher().generate(GAME_1)` should
      return a strict-validated, schema-clamped params dict. A response with no
      known keys now raises (it no longer silently defaults).

### 2. Reward service ↔ Eval Protocol — wire the hot loop

- [ ] **(key)** Install the live scorer: pass `create_fastapi_app(RewardService(scorer))`
      a `scorer` that calls `harness.scoring.score_curriculum(spec, game, mode="gap_proxy")`
      rather than a test lambda. (The scorer is real today; only the GameAdapter
      it runs is being finalized by the fighter dev.)
- [ ] Run the service: `uvicorn` an app from `create_fastapi_app(...)` (needs
      `fastapi` + `uvicorn`, already in `requirements.txt`).
- [ ] Point the Fireworks RFT **RemoteRolloutProcessor** at `POST /init`, sending
      `{"params": {...}, "metadata": {"rollout_id": "<id>"}}` (or `curriculum`),
      then collecting `GET /result/{rollout_id}`.
- [ ] Decide the persistence story for `app.state.results`: it's an in-process
      dict (fine for one worker / the hackathon). For multi-worker, back it with
      Redis or run a single uvicorn worker.

### 3. Modal Player fan-out — make it real PPO

- [ ] **(key)** Authenticate Modal: `modal token new` (or set `MODAL_TOKEN_ID` /
      `MODAL_TOKEN_SECRET`).
- [ ] **Replace the smoke body.** `modal_player.py::_smoke_result` returns a
      deterministic placeholder. Swap it for the real PPO training call (the
      fighter's PPO adapter, `harness/ppo_trainer.py`) and return the SAME fields
      (`seed`, `curriculum_id`, `arena_scores`, `mean_score`). The bridge in
      `modal_fanout._row_to_match_result` consumes those into a `MatchResult`
      unchanged.
- [ ] Deploy: `modal deploy modal_player.py` (app `crucible-player`, function
      `train_player`).
- [ ] Flip the backend: set `PlayerConfig.modal_parallel=True` — `FanoutPlayerTrainer`
      routes to Modal; everything else (submit → collect → aggregate) is identical
      to the local path. No runner changes.
- [ ] Until Modal is live, `FanoutPlayerTrainer(local_smoke_worker)` runs the
      full fan-out path locally with zero creds.

### 4. Known non-blocker (NOT this component)

- `test_fighter.py::test_adapter_evaluate_through_real_scorer` currently fails
  (`random_policy` scores ~0.9 vs an expected ≤0.2). This is the in-progress
  fighter difficulty work, owned by the **fighter agent** — out of scope here.
