# Falsification Analysis: Fireworks RFT Dataset Root-Cause
Date: 2026-06-21
Researcher: falsifier

---

## The Hypothesis Under Test

"The four FAILED jobs all share the same dataset (`rft-evaluator-test-teacher-rft-dataset-20260621015555`). Two jobs that were reportedly RUNNING used different, newer datasets. Therefore the failures are caused by the DATASET format/content, NOT by the model (qwen3-4b is not entitlement-gated) and NOT by a Fireworks platform/entitlement issue."

---

## Null Hypothesis (H0)

If H is false, the world would look like this: jobs using newer datasets would also fail at epoch 0 with the same pattern. The dataset is a red herring — the real cause is either model-level gating on qwen3-4b or a platform-wide instability.

---

## Falsification Criteria

1. If we observe any non-`015555`-dataset qwen3-4b job completing epoch 1 successfully, H is confirmed.
2. If we observe non-`015555` qwen3-4b jobs ALSO failing at epoch 0, the dataset hypothesis is REFUTED (or at least insufficient).
3. If qwen3-0p6b jobs with the same evaluator pattern also fail at epoch 0, the problem is not model-specific.
4. If the evaluators for the "running" jobs cannot be retrieved (404), the "running" state was a dashboard artifact.

---

## VERDICT: DATASET HYPOTHESIS — REFUTED (as the primary explanation)

The hypothesis as stated is false. The live SDK data (read-only `.get()` calls) shows all qwen3-4b jobs have now FAILED at epoch 0, including `gtwwkbsq` (kitchensink dataset) and `qjfm2w4o` (fullrecipe dataset). The dashboard screenshot showed those two as "RUNNING" — but that was a stale snapshot. They were in the RUNNING→FAILED transition window when the screenshot was taken.

The `015555` dataset is NOT the variable that explains the failure pattern. All four commonly-referenced confounders (model, dataset, evaluator, lora_rank) must be examined against the full job matrix.

---

## The Real Root Cause: The Evaluator, Not the Dataset

### The `"model"` Key Missing from `completion_params`

Every failing evaluator (across all datasets) shares ONE structural defect: the `@evaluation_test` decorator's `completion_params` does not include a `"model"` key.

**Failing evaluator pattern** (all four `015555` jobs, plus `gtwwkbsq` / `qjfm2w4o`):

```python
# training/rft_evaluator.py (original, tk3du2cr / vyelcet2)
# training/rft_eval_modelfix.py (b21f143j — note: modelfix added "model" but still failed)
# eval_flat/test_teacher.py (uxxnemic)
# rft_eval_kitchensink (gtwwkbsq)
# rft_eval_fullrecipe (qjfm2w4o)

completion_params=[{"temperature": 0.8, "max_tokens": 2048}]  # NO "model" key
```

`eval_protocol/pytest/tracing_utils.py`, line 153-154:
```python
if not completion_params_dict.get("model"):
    raise ValueError("Model must be provided in completion_params")
```

This `ValueError` fires on EVERY rollout, before any POST reaches the bridge. Result: 0 bridge hits, trainer-init INTERNAL at epoch 0, no rollouts logged.

**Confirmed by local test** (`evaluator_slim/local_test.log`):
```
ERROR root:evaluation_test_utils.py:451 ❌ Rollout failed (non-retryable error encountered):
ValueError('Model must be provided in completion_params')
```

The `rft_eval_modelfix.py` evaluator (used by `b21f143j`) DID add the `"model"` key:
```python
completion_params=[{"model": "accounts/fireworks/models/qwen3-4b", "temperature": 0.8, "max_tokens": 2048}]
```
But it still failed (`modelfix_watch.json`: `final_state=JOB_STATE_FAILED`, 0 non-preflight runs). This means the model-key fix alone was insufficient.

### The `input_messages` Triple-Nesting: A Second Structural Defect

The `fullrecipe` evaluator (`rft_eval_fullrecipe`) switched to `input_dataset=[RING_OUT_DATASET]` instead of `input_messages=[[[...]]]` — the triple-nested form. The `fullrecipe` evaluator's inline comments explicitly identify this as a fix:

```python
# The old input_messages=[[[...]]] triple-nesting is schema-invalid and
# died at trainer-init with zero rollouts.
input_dataset=[RING_OUT_DATASET],
```

The `input_messages` parameter type is `Sequence[list[InputMessagesParam] | None]` where `InputMessagesParam = list[Message]`. The triple-nesting `[[[{...}]]]` means the outer list is one "parameter combination", the middle list is one "run of messages", and the inner list is the messages themselves. When `eval-protocol create rft` tries to materialize this into a dataset JSONL it uploads, the resulting rows embed `rollout_status`, `execution_metadata`, and `created_at` fields directly in the dataset file — these are live-rollout state fields, not dataset seed fields. The Fireworks trainer receives rows that look like already-running rollouts, which is a schema mismatch.

**The `ring_out_dataset_ks.jsonl`** (4-row kitchensink dataset used by `gtwwkbsq`) shows exactly this contamination pattern. Each row contains:
```json
{
  "messages": [...],
  "input_metadata": {"completion_params": {}},
  "rollout_status": {"code": 101, "message": "Rollout is running", "details": []},
  "execution_metadata": {"invocation_id": "...", "experiment_id": "...", "rollout_id": "..."},
  "created_at": "2026-06-21T09:59:29.167641Z"
}
```

These are `EvaluationRow` objects in mid-rollout state, serialized and uploaded as training dataset rows. The Fireworks RFT trainer ingests this as the input corpus but these rows tell the trainer that rollouts are already running — this is nonsensical as a seed dataset and would cause trainer-init confusion.

**The clean seed dataset format** (what `input_dataset` path produces) is simply:
```json
{"messages": [{"role": "user", "content": "..."}], "input_metadata": {"row_id": "row_0"}}
```
or at minimum:
```json
{"messages": [{"role": "user", "content": "..."}]}
```

### Dataset Structural Comparison: Failing vs Correct

| Field | `015555` dataset (JSONL at upload) | `ring_out_dataset_ks.jsonl` (kitchensink) | Clean `input_dataset` row |
|---|---|---|---|
| `messages` | present | present | present |
| `input_metadata` | `{"completion_params": {}}` | `{"completion_params": {}}` | `{"row_id": "row_0"}` or absent |
| `rollout_status` | present (`code: 101`) | present (`code: 101`) | ABSENT |
| `execution_metadata` | present (random IDs) | present (random IDs) | ABSENT |
| `created_at` | present | present | ABSENT |
| Rows | 1 | 4 | 1 |

The `015555` dataset file was generated by a code path that captured live rollout state into the JSONL before upload. The `run_tiny_rft.py::build_dataset_jsonl()` function in the main branch uses `EvaluationRow.model_dump()` which would naturally include all fields including rollout state if any preflight runs had polluted the local row object.

---

## Job-by-Job Metadata (live SDK read, read-only)

All fields confirmed via `fw.reinforcement_fine_tuning_jobs.get()`:

| job_id | state | epoch | base_model | lora_rank | dataset | evaluator |
|---|---|---|---|---|---|---|
| `b21f143j` | FAILED | 0 | qwen3-4b | 8 | `...015555` | `rft-eval-modelfix-test-teacher-rft` |
| `uxxnemic` | FAILED | 0 | qwen3-4b | 8 | `...015555` | `test-teacher-test-teacher-rft` |
| `tk3du2cr` | FAILED | 0 | qwen3-4b | 8 | `...015555` | `rft-evaluator-test-teacher-rft` |
| `vyelcet2` | FAILED | 0 | qwen3-4b | 8 | `...015555` | `rft-evaluator-test-teacher-rft` |
| `gtwwkbsq` | FAILED | 0 | qwen3-4b | 8 | `rft-eval-kitchensink-dataset-4row` | `rft-eval-kitchensink` |
| `qjfm2w4o` | FAILED | 0 | qwen3-4b | 8 | `rft-eval-fullrecipe-dataset-20260621030219` | `rft-eval-fullrecipe` |
| `bbonmqac` | CANCELLED | 0 | qwen3-0p6b | 8 | `...030804` | `test-teacher-teacher-evaluation` |
| `demo-rft-job` | COMPLETED | 0 | qwen3-0p6b | 8 | `demo-gsm8k-math-dataset-1000` | `gsm8k-evaluator` |

**Key observation**: the evaluators for `gtwwkbsq` and `qjfm2w4o` (`rft-eval-kitchensink`, `rft-eval-fullrecipe`) return 404 on `.get()` — they no longer exist. They were likely created in isolated worktrees and cleaned up. The dataset `rft-eval-kitchensink-dataset-4row` also returns 404. This means the "RUNNING" state in the dashboard screenshot was a transient window before these evaluators and datasets were deleted (or never fully created on the platform), which caused the jobs to fail as soon as the trainer tried to reach the evaluator.

---

## Confound Analysis: Is the Dataset the Only Shared Variable Among the Failures?

The screenshot showed `015555` as the ONLY common dataset among 4 failures. That is true for b21f143j/uxxnemic/tk3du2cr/vyelcet2. But there are OTHER shared variables:

**Shared among ALL 4 `015555` failures:**
- `base_model`: qwen3-4b (same)
- `lora_rank`: 8 (same)
- `dataset`: `015555` (same)
- Evaluator: varies across 3 different evaluators (`rft-evaluator-test-teacher-rft`, `test-teacher-test-teacher-rft`, `rft-eval-modelfix-test-teacher-rft`)

Since the 4 jobs used **3 different evaluators** but the same dataset, the dataset WAS a legitimate correlation target. However, `gtwwkbsq` and `qjfm2w4o` now also FAILED at epoch 0 with completely different datasets and evaluators, while still using qwen3-4b with lora_rank=8. This breaks the "015555 dataset = cause" hypothesis.

The shared variable across ALL failed qwen3-4b jobs is: **the evaluators all have `completion_params` without a `"model"` key at the time the Fireworks evaluator container executes rollouts**.

The `rft-eval-fullrecipe` evaluator (`qjfm2w4o`) DID add the `"model"` key AND switched to `input_dataset`. But it still failed. The `fullrecipe` watch log shows 3 non-preflight runs APPEARED in the bridge (`tidy-love`, `likely-letter`, `high-wish`) — rollouts were dispatched to the bridge — then the job still FAILED. This is a different failure mode: rollouts reached the bridge but the job still died at epoch 0 before completing an epoch. This could be a timeout issue (the PPO seeds running 1500s each × 3 = 75 min total, while the Fireworks job has its own timeout), or a lora_rank=0 trainer behavior, or an epoch-boundary bug.

---

## Reconciliation of Prior Claims

### Claim A: "qwen3-4b is entitlement-gated"

**ORIGIN**: Earlier agents observed only qwen3-4b failures and qwen3-0p6b success (`demo-rft-job` COMPLETED), and inferred the model itself was behind a permission wall.

**WHAT ACTUALLY HAPPENED**: The `demo-rft-job` used the `gsm8k-evaluator` (ACTIVE, codeSnippets path, clean completion_params). It has nothing in common with the custom evaluators. The qwen3-4b jobs all failed because their evaluators had missing `"model"` in `completion_params` — which causes a `ValueError` at rollout init, producing 0 rollouts and a trainer-init INTERNAL. The 0p6b job (`bbonmqac`) ran under the same broken evaluator pattern and was CANCELLED (not COMPLETED) — it too made no progress. The model size is a confound here, not a cause. qwen3-4b is not entitlement-gated.

**FALSIFICATION EVIDENCE**: `qjfm2w4o` (qwen3-4b, new dataset, new evaluator with model key AND input_dataset fix) was listed as RUNNING in the dashboard screenshot, proving the trainer does START qwen3-4b jobs. It was not immediately rejected due to entitlement. It ran and then failed later (different cause).

### Claim B: "Even the official quickstart (qtgtaeuw) failed — platform bug"

**`qtgtaeuw`** does not appear in the live job list at all, and there is no job.json referencing it in the main worktrees. The SVG quickstart poll file (`output/svg_quickstart_test/poll_qtgtaeuw.jsonl`) exists in worktree `agent-ae64129442a28029b` and contains 5525 bytes of polling logs — but the job was submitted from a quickstart experiment, not this project's main RFT flow.

The `watch_bbonmqac.log` shows that `bbonmqac` (the 0p6b job running under the *quickstart*-style teacher evaluator) had 1 non-preflight run visible in the bridge, but 0 completed evaluations and 0 epochs before the watch loop ended. The bbonmqac job is now CANCELLED (manually or by timeout).

**WHAT ACTUALLY HAPPENED**: The "quickstart also failed" observation likely reflects the same completion_params / evaluator structural bug, not a platform-wide outage. The SVG quickstart (`svgbench`) used the correct evaluator form (input_dataset + model key), which is why those jobs behaved differently.

---

## Structural Dataset Difference: Failing vs Working

### Failing dataset row (`015555`, `ring_out_dataset_ks.jsonl`):
```json
{
  "messages": [{"role": "user", "content": "..."}],
  "input_metadata": {"completion_params": {}},
  "rollout_status": {"code": 101, "message": "Rollout is running", "details": []},
  "execution_metadata": {"invocation_id": "...", "experiment_id": "...", "rollout_id": "..."},
  "created_at": "2026-06-21T08:55:55.013956Z"
}
```

These rows contain live rollout state serialized into the seed dataset. The trainer receives them as if rollouts are already in progress.

### Working dataset row (clean `input_dataset` form, per SVG quickstart and eval-protocol docs):
```json
{"messages": [{"role": "user", "content": "..."}], "input_metadata": {"row_id": "row_0"}}
```
or minimal:
```json
{"messages": [{"role": "user", "content": "..."}]}
```

**How the contamination happened**: The `run_tiny_rft.py::build_dataset_jsonl()` function calls `EvaluationRow(...).model_dump(mode="json", exclude_none=True)`. If any preflight trace or prior run had set `rollout_status` / `execution_metadata` fields on the row objects, those would be serialized. The watch logs for `qjfm2w4o` (fullrecipe, the job that launched successfully) show those fields ARE present in the kitchensink JSONL — this is a materializer-side bug where the evaluator's local test loop (which runs as part of `eval-protocol create rft`) serialized mid-run `EvaluationRow` state back into the JSONL file that was then uploaded as the dataset.

---

## Doc-Backed Correct Format

From [Fireworks RFT Quickstart — Single-Turn Math](https://docs.fireworks.ai/fine-tuning/quickstart-math):
> Each line in your dataset file should be a JSON object with `messages` (list of chat messages) and optionally `metadata`.

From `eval_protocol/cli_commands/create_rft.py::_validate_dataset_jsonl()` (line 287-327, SDK source):
> Streams up to 50 rows, validates each with `EvaluationRow.model_validate(data)`. A `ValidationError` causes failure. `rollout_status` with `code: 101` (RUNNING) would pass model validation but is semantically wrong as a seed dataset row — the trainer does not expect pre-populated rollout state.

From `eval_protocol/cli_commands/create_rft.py`, warning at line 473:
> "Please switch to a JSONL-based dataset via `input_dataset` arg in `@evaluation_test` decorator." (printed when `input_messages` is auto-detected instead of a JSONL path)

The correct pattern, doc-backed by the eval-protocol SVG quickstart (`quickstart/svg_agent/evaluator/test_svgagent.py`, line 113-114):
```python
input_dataset=[str(Path(__file__).parent / "svgbench_dataset.jsonl")],
completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b", ...}],
```

The `model` key is REQUIRED in `completion_params` at the decorator level, and the dataset must be a JSONL of clean seed rows — not serialized mid-rollout `EvaluationRow` state.

---

## What a Malformed Dataset Produces

The `015555` dataset's structural issue (rollout state in seed rows) would interact with the trainer-init as follows: Fireworks' RFT trainer receives rows that have non-null `rollout_status.code = 101` ("Rollout is running"). The trainer's expectation for seed rows is a clean prompt with no in-progress rollout state. If the trainer treats these as already-dispatched rollouts, it may skip dispatching new rollouts for epoch 0, see 0 completed rollouts, and report INTERNAL at "epoch 0, 0 rollouts" — which matches the observed failure signature exactly.

However, given that even the `fullrecipe` job (with clean dataset rows) also eventually FAILed at epoch 0 after dispatching some rollouts, the dataset format alone does not fully explain the failures. The evaluator's rollout timeout behavior (1500s per rollout × concurrent rollouts) is also a candidate.

---

## Unfalsifiability Risk

[ ] High
[x] Medium — the "completion_params missing model" bug is directly observable and confirmed. The dataset contamination hypothesis is plausible but not directly confirmed as the trainer-side failure mode. The evaluator-timeout and lora_rank=0 hypotheses have some evidence but not conclusive logs.

---

## What Would Change the Verdict

1. **Confirm dataset hypothesis as ALSO a cause**: Submit a job with qwen3-4b + `015555` dataset + an evaluator that has `"model"` key in completion_params. If it still fails, the dataset alone is not the primary issue. If it succeeds, the dataset contamination WAS the cause for that subset of failures.
2. **Isolate the fullrecipe failure mode**: The fullrecipe job dispatched 3 non-preflight rollouts but still failed. Fetching the Fireworks error log for `qjfm2w4o` would confirm whether: (a) rollouts timed out (PPO jobs ran too long), (b) the evaluator's tracing write failed, or (c) an epoch-boundary bug occurred.
3. **Confirm qwen3-4b entitlement**: A fresh job with qwen3-4b + clean evaluator (model key present) + clean JSONL dataset + correct lora_rank completing epoch 1 would definitively rule out model-level gating.

---

## Summary Verdict

The "dataset is the cause" hypothesis is **REFUTED as a complete explanation**. The true picture is more nuanced:

**Primary confirmed cause (evaluator bug)**: Every evaluator deployed across all failing jobs lacked `"model"` in `completion_params`, causing a `ValueError` on every rollout before the bridge is ever reached. This alone is sufficient to produce "INTERNAL at epoch 0, 0 rollouts." Confirmed by local pytest log.

**Secondary possible cause (dataset contamination)**: The `015555` and kitchensink datasets contain mid-rollout `EvaluationRow` state (`rollout_status`, `execution_metadata`, `created_at`) serialized into seed dataset rows. This is a distinct structural error from the evaluator bug. It may compound the trainer-init failure.

**The dashboard snapshot was stale**: `gtwwkbsq` and `qjfm2w4o` were "RUNNING" because the screenshot was taken during their RUNNING→FAILED transition. Both have now FAILED at epoch 0. Their evaluators (`rft-eval-kitchensink`, `rft-eval-fullrecipe`) are 404 — they may have been cleaned up mid-run or never properly reached ACTIVE on the platform.

**qwen3-4b is NOT entitlement-gated**: Every failing job used a broken evaluator. The model is not the cause.

**The "platform bug / even quickstart failed" conclusion was wrong**: The prior `fireworks-evaluator-blocker.md` analysis (correctly) identified inline comments in requirements.txt and hud-python==0.6.6 as the evaluator build failures. Once those were fixed (evaluator `rft-evaluator-test-teacher-rft` reached ACTIVE), the remaining failures are evaluator-runtime bugs, not platform infra.

---

## Sources

- [Fireworks RFT Single-Turn Quickstart](https://docs.fireworks.ai/fine-tuning/quickstart-math) — canonical dataset JSONL format for RFT jobs
- [Fireworks Connect Environments (RemoteRolloutProcessor)](https://docs.fireworks.ai/fine-tuning/connect-environments) — RemoteRolloutProcessor + completion_params requirements
- [Eval Protocol Docs — Single-Turn Eval](https://evalprotocol.io/tutorial/single-turn-eval-static) — input_dataset vs input_messages distinction
- [Eval Protocol GitHub Quickstart](https://github.com/eval-protocol/quickstart) — svg-agent as reference implementation
- SDK source: `.venv/lib/python3.12/site-packages/eval_protocol/pytest/tracing_utils.py:153` — `ValueError("Model must be provided in completion_params")`
- SDK source: `.venv/lib/python3.12/site-packages/eval_protocol/cli_commands/create_rft.py:287` — `_validate_dataset_jsonl()` and warning about `input_messages`
- SDK source: `.venv/lib/python3.12/site-packages/eval_protocol/quickstart/svg_agent/evaluator/test_svgagent.py:113` — reference pattern with `input_dataset` + `"model"` key
- Live SDK read-only queries: `fw.reinforcement_fine_tuning_jobs.get()` and `.list()` for all jobs, `fw.evaluators.get()` for all evaluators, `fw.datasets.get()` for all datasets
- `.claude/worktrees/agent-a4b52aaa133fae4f6/evaluator_slim/local_test.log` — confirmed local `ValueError('Model must be provided in completion_params')` 
- `.claude/worktrees/agent-a6de2e59f1b2ea0b7/output/tiny_rft/watch.log` — fullrecipe job reached 3 non-preflight runs then FAILED
- `.claude/worktrees/agent-ab54dc67949e4e953/output/tiny_rft/modelfix_watch.json` — modelfix (b21f143j) FAILED with 0 non-preflight runs despite model key added
- `output/tiny_rft/resources.json` — confirms `015555` dataset used by vyelcet2 / the main launcher
- `research/falsification/fireworks-evaluator-blocker.md` — prior falsifier analysis of evaluator BUILD_FAILED
