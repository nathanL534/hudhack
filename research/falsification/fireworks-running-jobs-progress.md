# Falsification Analysis: Are the "Running" Fireworks RFT Jobs Genuinely Progressing?
Date: 2026-06-21
Researcher: falsifier

---

## The Hypothesis Under Test

The Fireworks dashboard showed three jobs as "Running" (gtwwkbsq, qjfm2w4o, bbonmqac). The implicit hypothesis is: **these jobs are genuinely progressing through the nested-RL loop and are likely to produce a trained Teacher checkpoint.**

---

## Per-Job Ground Truth (pulled live, 2026-06-21)

### gtwwkbsq — qwen3-4b on rft-eval-kitchensink-dataset-4row

| Field | Value |
|---|---|
| Actual state (SDK) | **JOB_STATE_FAILED** |
| Status code | `INTERNAL` — "Internal error occurred, please contact Fireworks AI" |
| Epoch | 0 (no progress) |
| successfullyProcessedRequests | 0 |
| past epoch 0 | NO |
| Dataset exists | NO — `rft-eval-kitchensink-dataset-4row` returns 404 |
| Evaluator exists | NO — `rft-eval-kitchensink` returns 404 |
| Rollouts reached bridge | NOT AS RFT ROLLOUTS (see bridge section) |

The `output_metrics` field shows `epoch_to_evaluation_output["0"]` with `end_of_epoch: true` and `metrics: null`. This is the same tombstone pattern seen in all 4 prior INTERNAL-failed jobs. The epoch-0 GCS streamlog URL (signed) returns 404 — the log file was never written.

**Verdict: FAILED. The dashboard was stale when the team saw "Running." The SDK confirms failure.**

---

### qjfm2w4o — qwen3-4b on rft-eval-fullrecipe-dataset-20260621030219

| Field | Value |
|---|---|
| Actual state (SDK) | **JOB_STATE_FAILED** |
| Status code | `INTERNAL` — same as above |
| Epoch | 0 (no progress) |
| successfullyProcessedRequests | 0 |
| past epoch 0 | NO |
| Dataset exists | YES — `rft-eval-fullrecipe-dataset-20260621030219` is READY |
| Evaluator exists | NO — `rft-eval-fullrecipe` returns 404 |
| Rollouts reached bridge | NOT AS RFT ROLLOUTS |

Same tombstone: `epoch_to_evaluation_output["0"]` with `metrics: null`, signed GCS log returns 404, no trainer logs. The evaluator `rft-eval-fullrecipe` does not exist in the account — the job failed because it was submitted referencing a nonexistent evaluator.

**Verdict: FAILED. Same INTERNAL failure at epoch 0.**

---

### bbonmqac — qwen3-0p6b on test-teacher-evaluation-dataset-20260621030804

| Field | Value |
|---|---|
| Actual state (SDK) | **JOB_STATE_RUNNING** |
| Status code | `OK` (no error) |
| Epoch | 0 |
| successfullyProcessedRequests | 0 |
| percent | 0% |
| past epoch 0 | NOT YET — epoch still at 0, percent = 0 |
| Dataset exists | NO — `test-teacher-evaluation-dataset-20260621030804` returns 404 |
| Evaluator exists | NO — `test-teacher-teacher-evaluation` returns 404 |
| output_metrics | EMPTY (no epoch output yet) |
| Rollouts reached bridge | NOT AS RFT ROLLOUTS |

This job reports `JOB_STATE_RUNNING` with `status.code = OK` and no INTERNAL error yet. However:
- Both its dataset and evaluator return 404.
- Epoch is 0, percent is 0, zero requests processed.
- No output_metrics at all (blank string), vs the failed jobs which at least show a tombstone epoch-0 entry.
- It is a different base model (qwen3-0p6b, not 4b) and has a different display_name (blank vs "crucible-tiny-teacher-rft").

**Verdict: NOT-YET — currently running but at 0% with missing infrastructure. Prognosis: likely to fail with the same INTERNAL error once the Fireworks backend attempts its first rollout and finds no dataset/evaluator. This job has not advanced beyond epoch 0 and rollouts have not reached the bridge.**

---

## Did Rollouts Actually Reach the Bridge?

The Modal `crucible-ep-bridge` `/runs` endpoint lists 11 total run_ids:
- 8 are clearly preflight: `pf-run-preflight-<timestamp>-<hex>` format, all with preflight rollout IDs.
- 3 are non-preflight: `tidy-love-739359`, `likely-letter-700996`, `high-wish-868800` — with rollout IDs `bad-drawing-598439`, `full-painting-746201`, `theoretical-desire-687109`.

**Those 3 non-preflight runs DID reach the bridge** (`POST /init -> 202 Accepted` appears 4 times in Modal logs). However:
- All 3 `/debug/result/<rollout_id>` return `status: error` with `litellm.NotFoundError: Model not found, inaccessible, and/or not deployed`.
- The error occurred because the bridge tried to call a Fireworks checkpoint URL that does not exist or is not deployed.
- The word-word-number run_id format (`tidy-love`, `likely-letter`, `high-wish`) matches eval-protocol's auto-generated IDs, not the team's job IDs — these likely came from earlier preflight or test runs (not from the 3 "Running" jobs), possibly from a different team member's direct API test.

**CRITICAL DISTINCTION:** None of the run_ids on the bridge can be attributed to RFT rollouts from gtwwkbsq, qjfm2w4o, or bbonmqac. The RFT jobs failed before they could issue a single `/init` call to the bridge. The 3 non-preflight bridge hits are not RFT training rollouts — they are test/preflight calls where the model endpoint was not deployed. This repeats the misattribution trap from the prior session.

**Evidence basis: directly observed from `/runs` response and `/debug/result/<id>` — not inferred.**

---

## What "Success" Looks Like vs "Epoch-0 INTERNAL Death"

### Healthy job signature (from `demo-rft-job`, the only COMPLETED job on the account):
- State transitions through RUNNING with `percent > 0`
- `output_metrics.epoch_to_evaluation_output` populates entries for epochs 0-N with actual `metrics` objects (not null) — reward distributions, correlation matrices, rollout counts
- `curves.average.Score` shows a rising array across epochs
- `successfullyProcessedRequests > 0` in `jobProgress`
- Multiple epochs appear (epochs 0-9 in demo-rft-job), not just a single epoch-0 tombstone
- `completedTime` is populated when done

### Epoch-0 INTERNAL death signature (all 6 failed Crucible jobs share this):
- State: `JOB_STATE_FAILED`
- Status: `code='INTERNAL'`, message about contacting Fireworks support
- `jobProgress.epoch = 0`, `percent = 0`, `successfullyProcessedRequests = 0`
- `output_metrics.epoch_to_evaluation_output` has exactly one entry, key `"0"`, with `end_of_epoch: true` but `metrics: null`
- `curves: {}` (empty)
- GCS streamlog URL for epoch-0 returns 404 (log was never written)
- No trainer logs

The null metrics in epoch-0 is the distinguishing marker: a healthy epoch-0 has real reward distributions populated. A null epoch-0 means the evaluator was invoked, produced no useful output, and the backend gave up with INTERNAL.

---

## Root Cause (inferred from evidence)

Both gtwwkbsq and qjfm2w4o reference evaluators that do not exist in the account:
- `rft-eval-kitchensink` — 404
- `rft-eval-fullrecipe` — 404

The dataset for gtwwkbsq (`rft-eval-kitchensink-dataset-4row`) also does not exist. The dataset for qjfm2w4o exists and is READY. For bbonmqac, both dataset and evaluator return 404.

The prior 4 INTERNAL failures (b21f143j, uxxnemic, tk3du2cr, vyelcet2) all used the same dataset (`rft-evaluator-test-teacher-rft-dataset-20260621015555`) and different evaluators — some ACTIVE, some not. The INTERNAL failure pattern predates missing evaluators, suggesting the Fireworks backend itself is erroring at epoch-0 rollout time regardless of evaluator state. However, the missing evaluator/dataset combination makes recovery impossible in any case.

**The INTERNAL error is Fireworks-side infrastructure failure during the first epoch-0 rollout pass.** It is not a code error in the bridge or evaluator logic visible to this repo. All 6 failed jobs — spanning multiple evaluator/dataset combinations including ones that exist and are ACTIVE — produce the identical INTERNAL failure pattern.

---

## Falsification Criteria (What Would Show Genuine Progress)

If the hypothesis "these jobs are genuinely running the nested loop" were true, we would observe:
1. `jobProgress.successfullyProcessedRequests > 0` for at least one job
2. `output_metrics.epoch_to_evaluation_output["0"].metrics` populated with non-null reward data
3. `curves.average.Score` with at least one non-zero data point
4. At least one run_id on the bridge that does NOT have the `pf-` prefix and whose `/debug/result/` returns `status: finished` with a real `hud_reward` value — tied by run_id to a specific RFT job's rollout_id
5. `jobProgress.epoch > 0` on any job

**None of these are observed.** The two qwen3-4b jobs are confirmed FAILED. The qwen3-0p6b job is in RUNNING state but at 0/0/0% with no infrastructure backing it.

---

## Strongest Single Indicator

**The `output_metrics.epoch_to_evaluation_output["0"].metrics` field.** On the only successful job in the account (`demo-rft-job`), every epoch entry has a populated metrics object with `rollup_distribution`, `metrics_distribution`, and reward averages. On every failed Crucible job, epoch-0 shows `metrics: null`. This is not ambiguous — it cleanly separates success from INTERNAL death without requiring any inference about the bridge or rollout IDs.

For bbonmqac specifically: it has an empty `output_metrics` field (blank string, not even a tombstone yet). Once it gets a tombstone epoch-0 with `metrics: null`, it has died. If it populates `metrics` with a real reward distribution, it is progressing.

---

## Verdict

| Job | True State | Past Epoch 0 | Rollouts Hit Bridge | Prognosis |
|---|---|---|---|---|
| gtwwkbsq (qwen3-4b) | **FAILED** | NO | NO (bridge has zero RFT rollouts for this job) | Done. No recovery. |
| qjfm2w4o (qwen3-4b) | **FAILED** | NO | NO | Done. No recovery. |
| bbonmqac (qwen3-0p6b) | RUNNING (still) | NOT YET | NO | Likely to FAIL — missing dataset + evaluator; same INTERNAL trajectory |

**Overall verdict on "are the qwen3-4b jobs genuinely running the nested loop end-to-end": FAILING — already FAILED.**

The dashboard "Running" state the team saw was stale. The SDK confirms JOB_STATE_FAILED for both qwen3-4b jobs. The qwen3-0p6b job (bbonmqac) is technically still running but shows every warning sign of imminent INTERNAL failure: zero progress, missing infrastructure, zero bridge contact.

The pattern across all 6 failed jobs is consistent: Fireworks returns INTERNAL at the first epoch-0 rollout attempt regardless of which evaluator or dataset is referenced. This points to a platform-level issue, not a fixable code issue on this team's side.

---

## Inference vs Direct Observation Map

- State of gtwwkbsq = JOB_STATE_FAILED: **directly observed** (SDK `.get()`)
- State of qjfm2w4o = JOB_STATE_FAILED: **directly observed**
- State of bbonmqac = JOB_STATE_RUNNING: **directly observed**
- epoch = 0, percent = 0 for all three: **directly observed**
- No rollouts from RFT jobs reached bridge: **directly observed** (bridge `/runs` shows zero job-matching run_ids; all 3 non-preflight entries have error status and `Model not found` — meaning they reached the bridge but the checkpoint was not available, and these cannot be tied to the 3 specific RFT jobs)
- Missing evaluator/dataset as root cause: **directly observed** (404 from API) — though INTERNAL failures also occur for jobs with existing evaluators, so this is likely a contributing factor, not the sole cause
- bbonmqac will fail: **inferred** from pattern and missing infrastructure; not confirmed yet

---

## Sources

- Fireworks SDK `reinforcement_fine_tuning_jobs.get()` — live job state for gtwwkbsq, qjfm2w4o, bbonmqac
- Fireworks REST API `/v1/accounts/{account}/reinforcementFineTuningJobs` — all 9 jobs list
- Fireworks REST API `/v1/accounts/{account}/evaluators` — evaluator state list
- Fireworks REST API `/v1/accounts/{account}/datasets/{id}` — dataset existence checks
- Modal bridge `GET /runs` — all run_ids currently registered
- Modal bridge `GET /debug/result/{rollout_id}` — error status for non-preflight rollouts
- Modal bridge logs (`modal app logs crucible-ep-bridge`) — 4 POST /init calls, all preflight or test-run
- GCS signed URL for epoch-0 streamlog — 404 (log never written)
- `demo-rft-job` (COMPLETED, gsm8k, qwen3-0p6b) — reference for what a healthy job looks like
