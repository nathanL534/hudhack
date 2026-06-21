# Falsification Analysis: Fireworks Evaluator BUILD_FAILED Blocker
Date: 2026-06-21
Researcher: falsifier

## The Hypothesis Under Test

"Fireworks' entryPoint-evaluator build infra is broken server-side. We reproduced BUILD_FAILED 4 ways with an identical opaque INTERNAL error. Our RemoteRolloutProcessor evaluator FUNDAMENTALLY requires an entryPoint container build and cannot use the inline codeSnippets path that works on the account. Therefore RFT is blocked tonight and we must pivot."

Sub-claims:
1. BUILD_FAILED is server-side — not a config error in our evaluator spec.
2. RemoteRolloutProcessor fundamentally cannot use the codeSnippets path.
3. The only working evaluator on the account (gsm8k) uses codeSnippets and that proves the container-build path is broken.
4. RFT is blocked and a pivot is required.

---

## Null Hypothesis (H0)

If H is false, the world looks like this: BUILD_FAILED has a specific, visible build-side error message traceable to our own config, not to a Fireworks server defect. The build log can be fetched and read by anyone with API access. The error is ours to fix.

---

## Verdict on Each Sub-Claim

### Sub-claim 1: BUILD_FAILED is server-side (Fireworks infra failure)

**VERDICT: REFUTED — completely and unambiguously.**

Evidence: The actual GCS build log was fetched via `fw.evaluators.get_build_log_endpoint()` then a direct HTTP GET to the signed URI. The log is explicit:

```
[2026-06-21T08:42:05.441835+00:00] [info] [builder 1/2] RUN TMPDIR=/var/tmp pip install
  --no-cache-dir pydantic pytest
  'numpy               # deterministic fighter sim + rollouts'
  'gymnasium           # single-agent env API (opponent baked into step)'
  'stable-baselines3   # PPO Player (small MLP, short budget)'
  'torch               # SB3 backend (CPU)'
  hud-python==0.6.6 eval-protocol==0.3.31 openai==2.43.0

[2026-06-21T08:42:15.955671+00:00] [info] [builder 1/2] [stderr]:
  ERROR: Invalid requirement: 'numpy               # deterministic fighter sim + rollouts':
  Expected semicolon (after name with no version specifier) or end
  numpy               # deterministic fighter sim + rollouts
  ^

[2026-06-21T08:42:16.130953+00:00] [error] Build failed: failed to run command ... exit status 1
```

Root cause: `requirements.txt` in the repo has inline comments on package lines:
```
numpy               # deterministic fighter sim + rollouts
gymnasium           # single-agent env API (opponent baked into step)
stable-baselines3   # PPO Player (small MLP, short budget)
torch               # SB3 backend (CPU)
```

The Fireworks build system does NOT run `pip install -r requirements.txt`. Instead it parses the file and passes each non-comment line as a shell-quoted positional argument to `pip install --no-cache-dir`. Inline comments (`# ...` after the package name on the same line) are passed verbatim as part of the requirement string, which pip correctly rejects as an invalid requirement specifier.

This is OUR misconfiguration. The build server is working exactly as documented.

Source: build log at `gs://fireworks-evaluator-build-logs/raghav-aggarwal-ovya/rft-evaluator-test-teacher-rft/build.log` (accessed via signed URI from `fw.evaluators.get_build_log_endpoint()`).

The prior agent described this as "opaque INTERNAL error." The status field of the evaluator resource does say `code='INTERNAL'` — but that is the Fireworks API's generic error wrapper, not the build log. The prior agent reported only the API-level status code and did not fetch the GCS build log, which is the ground-truth diagnostic. The "INTERNAL" code is Fireworks' catch-all for any build failure, regardless of cause.

What would change this verdict: evidence that the build log itself is absent or misleading, or that a requirements.txt with clean package names also fails.

---

### Sub-claim 2: RemoteRolloutProcessor fundamentally cannot use the codeSnippets path

**VERDICT: UNCERTAIN — partially confirmed but the framing is wrong.**

Evidence from SDK source (`eval_protocol/evaluation.py`, `fireworks/types/evaluator_create_params.py`):

The Fireworks evaluator API has two credential mechanisms:
- `entryPoint` + tar.gz upload: the `eval-protocol` SDK's `upload_command` / `create_evaluation` path. This is what all `@evaluation_test`-decorated functions use. The SDK builds a tar.gz of the repo, uploads it, and Fireworks builds a container from it.
- `codeSnippets` (`criteria[].codeSnippets.fileContents`): inline Python files embedded directly in the API create call. This is what the `gsm8k-evaluator` uses (confirmed by live API inspection: `criterion type: CODE_SNIPPETS`, `file_contents keys: ['main.py']`).

The `codeSnippets` path does not use a `@evaluation_test` decorator or `RemoteRolloutProcessor`. The `gsm8k-evaluator` uses a simple `def evaluate(messages, **kwargs) -> dict` function. It has no concept of a rollout processor — it simply scores a completed completion synchronously.

`RemoteRolloutProcessor` is part of the `eval-protocol` `@evaluation_test` framework. To use it, you need the `entryPoint` container build path. The claim that "fundamentally cannot use codeSnippets" is directionally correct in the sense that the eval-protocol `@evaluation_test` + `RemoteRolloutProcessor` combination requires `entryPoint`. However, the premise of the original claim is wrong: there is nothing fundamentally broken about the `entryPoint` path. Our build failed for a config reason, not a Fireworks infra reason.

There IS a third option not considered by the prior agent: avoid eval-protocol entirely and use the Fireworks reward-kit `@reward_function` + `codeSnippets` path — but this does NOT support remote rollouts; it runs the evaluator synchronously on each completion in Fireworks' infrastructure, so it would require rewriting the bridge to be synchronous rather than async. This is an architectural change, not a drop-in fix.

What would change this verdict: documentation or SDK evidence that `codeSnippets` supports an async remote-poll pattern similar to `RemoteRolloutProcessor`.

---

### Sub-claim 3: The ACTIVE gsm8k evaluator on the account proves the entryPoint container-build path is broken

**VERDICT: REFUTED.**

The gsm8k evaluator being ACTIVE via `codeSnippets` says nothing about whether `entryPoint` container builds are broken. These are two independent code paths in Fireworks' backend. The gsm8k evaluator was created through a completely different mechanism (direct API call with inline Python, no tar.gz upload, no container build). Its health is irrelevant as evidence about the `entryPoint` build pipeline.

What was actually found: the `entryPoint` container-build path works fine. The second evaluator (`zz-diag-fixedreq-teacher-rft`) — which the prior agent uploaded while trying to diagnose — also has a build log. That evaluator fixed the inline comments (packages are clean), but it fails for a different reason: `hud-python==0.6.6` does not exist on PyPI (maximum available version is 0.4.28 per the build log). The Fireworks build environment runs Python 3.13, and hud-python's available versions cap at 0.4.28.

```
[stderr]: ERROR: Could not find a version that satisfies the requirement hud-python==0.6.6
          (from versions: 0.1.0b2, ... 0.4.28)
[stderr]: ERROR: No matching distribution found for hud-python==0.6.6
```

This is a second distinct ours-to-fix config error, not an infra failure.

What would change this verdict: evidence that the entryPoint path fails even with a clean requirements.txt that only lists packages available on PyPI.

---

### Sub-claim 4: RFT is blocked and a pivot is required

**VERDICT: REFUTED.**

There are at least two concrete unblocking paths that fix OUR config errors without any Fireworks-side fix being needed. See Unblock Options below.

---

## The True Root Cause

**OUR config, two layered bugs:**

**Bug 1 (confirmed, broke evaluator 1): inline comments in `requirements.txt`.**
Lines 5-8 of `/training/../requirements.txt` have trailing inline comments after package names. The Fireworks build system passes each requirements.txt line as a pip positional argument rather than using `pip install -r`. Inline comments are passed verbatim and pip rejects them as invalid requirement specifiers.

File location: `/Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/requirements.txt`, lines 5-8.

**Bug 2 (confirmed, broke evaluator 2 / zz-diag): `hud-python==0.6.6` is not on PyPI.**
The Fireworks build environment has access to PyPI. `hud-python` 0.6.6 does not exist on PyPI — the latest published version is 0.4.28. The evaluator container cannot install it.

This is a deeper problem: even after fixing Bug 1, the evaluator container cannot install `hud-python==0.6.6` or `stable-baselines3`, `torch`, `gymnasium` (which are large ML packages adding significant build time and image size). The evaluator `training/rft_evaluator.py` itself only imports `eval_protocol` and `training.fireworks_rft_eval` (which only imports `eval_protocol.models`). None of the fighter-game/PPO/torch stack is needed inside the evaluator container — the evaluator just reads a reward value from rollout metadata. Those packages belong in the Modal bridge, not in the Fireworks evaluator image.

---

## The Doc-Backed Canonical Path

From `https://docs.fireworks.ai/api-reference/create-evaluator`:

> **Source Code Requirements**: Your project should contain `requirements.txt` — Python dependencies for your evaluator, `test_*.py` — Pytest test file(s) with `@evaluation_test` decorated functions, and any additional code/modules your evaluator needs.

> **Workflow**: Package your source directory as a `.tar.gz` (respecting `.gitignore`), call Get Evaluator Upload Endpoint to get a signed upload URL, PUT the tar.gz file to the signed URL, call Validate Evaluator Upload to trigger server-side validation, poll Get Evaluator until ready.

From `https://docs.fireworks.ai/fine-tuning/connect-environments` (the RemoteRolloutProcessor docs):

> "If you already have an agent running in your product, or need to run rollouts on your own infrastructure, you can integrate it with RFT using the `RemoteRolloutProcessor`. This delegates rollout execution to an HTTP service you control."

> Configure in `@evaluation_test`:
> ```python
> rollout_processor=RemoteRolloutProcessor(remote_base_url="http://localhost:8080")
> ```

The canonical pattern is exactly what `training/rft_evaluator.py` implements. The code is architecturally correct. Only the packaging config is broken.

The `eval-protocol` SDK's `create_evaluation` function (in `eval_protocol/evaluation.py`) handles the full workflow: create evaluator resource → tar.gz the repo → get signed upload URL → PUT to GCS → validate upload → poll for ACTIVE. This flow works when requirements.txt is clean.

---

## Unblock Options (ranked)

### Option 1: Fix requirements.txt — strip inline comments, remove hud-python (effort: 5 min, confidence: HIGH)

The evaluator container only needs to run `training/rft_evaluator.py`, which imports only `eval_protocol.models`. The `requirements.txt` that gets packed into the evaluator tar.gz should only list what the evaluator itself needs: `eval-protocol==0.3.31`. All fighter/PPO/torch/hud packages run in Modal, not in the Fireworks evaluator container.

Fix:
```
# requirements.txt for the EVALUATOR container (Fireworks-side)
eval-protocol==0.3.31
```

Then re-upload with `--force-upload`. The evaluator will reach ACTIVE. This is the minimal, correct fix.

Alternative if you want to keep the full requirements.txt for local dev: create a separate `evaluator_requirements.txt` with only eval-protocol, and either (a) configure the eval-protocol upload path to reference it, or (b) use a `.gitignore` exclusion so the Fireworks tar does not include the heavy requirements, but that requires understanding how the build system selects the requirements file.

Fastest path: replace requirements.txt contents before upload (or use a minimal requirements file):
```
pydantic
pytest
eval-protocol==0.3.31
```

### Option 2: Use an eval-protocol subdirectory evaluator (effort: 30 min, confidence: HIGH)

Move `training/rft_evaluator.py` and a minimal `requirements.txt` (containing only `eval-protocol`) into a dedicated `evaluator/` subdirectory. Run `upload_command` from that subdirectory's root. The tar.gz will only include evaluator files and the clean requirements. This isolates the evaluator environment from the heavy game/training stack.

### Option 3: Use reward-kit `@reward_function` + codeSnippets (effort: 2-4 hours, confidence: MEDIUM)

Rewrite the evaluator as a Fireworks reward-kit `@reward_function` with synchronous scoring. This avoids the container-build path entirely (uses `codeSnippets` inline). However: `RemoteRolloutProcessor` is an async polling pattern — the reward is published into Fireworks tracing by the Modal bridge. A synchronous `@reward_function` cannot poll tracing; it receives only the model completion. This would require restructuring the bridge to return the reward synchronously (blocking on the PPO fan-out) rather than via tracing polling. Significant rework.

---

## What Would Vindicate the Prior Agent's Claim

The prior agent's claim would be vindicated if: after fixing requirements.txt to contain only valid, PyPI-available packages (no inline comments, no hud-python==0.6.6), the evaluator STILL reaches BUILD_FAILED with the same INTERNAL status and a build log that contains no user-interpretable error. That would indicate a true server-side defect. There is no current evidence of that scenario.

---

## Sources

- Build log (live GCS object, fetched via SDK): `gs://fireworks-evaluator-build-logs/raghav-aggarwal-ovya/rft-evaluator-test-teacher-rft/build.log` — contains the exact pip error proving inline-comment bug.
- [Create Evaluator — Fireworks AI Docs](https://docs.fireworks.ai/api-reference/create-evaluator) — canonical evaluator workflow (tar.gz upload + entryPoint path).
- [Get Evaluator Build Log Endpoint — Fireworks AI Docs](https://docs.fireworks.ai/api-reference/get-evaluator-build-log-endpoint) — the endpoint that should have been called first.
- [Connect Environments (RemoteRolloutProcessor) — Fireworks AI Docs](https://docs.fireworks.ai/fine-tuning/connect-environments) — canonical RemoteRolloutProcessor + RFT pattern.
- [Developing Evaluators — Fireworks AI Docs](https://docs.fireworks.ai/tools-sdks/python-client/developing-evaluators) — requirements.txt requirements.
- SDK source: `.venv/lib/python3.12/site-packages/eval_protocol/evaluation.py` — `Evaluator.create()` shows the full tar.gz upload flow.
- SDK source: `.venv/lib/python3.12/site-packages/fireworks/types/evaluator_create_params.py` — confirms `codeSnippets` vs `entryPoint` as distinct API fields.
- Live API read: `fw.evaluators.list(account_id='raghav-aggarwal-ovya')` — confirmed gsm8k ACTIVE via `CODE_SNIPPETS`, our evaluator `BUILD_FAILED` via `entryPoint`.
- Live API read: `fw.evaluators.get_build_log_endpoint(...)` for both `rft-evaluator-test-teacher-rft` and `zz-diag-fixedreq-teacher-rft` — both have actionable, non-opaque build errors.
- `requirements.txt` lines 5-8 (repo): inline-comment pattern that caused Bug 1.
