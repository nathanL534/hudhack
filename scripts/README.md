# Qwen3-4B dedicated Fireworks deployment

Replicable scripts to stand up a **dedicated** Fireworks endpoint for Qwen3-4B,
prove it serves inference, and tear it down to stop billing.

This account has **no serverless access** to Qwen3-4B, so any inference (and the
RFT serving round-trip) needs a dedicated on-demand deployment.

> Note on RFT: Reinforcement Fine-Tuning runs the model **internally** on
> Fireworks' side. This dedicated deployment is for **serving / inference + the
> round-trip smoke test** — it is **not strictly required** to run RFT itself.

## Files

| File | What it does |
|------|--------------|
| `deploy_qwen.sh`   | Creates a dedicated Qwen3-4B deployment, waits for READY, prints the deployment id + inference handle, runs one tiny completion to prove it serves real tokens. |
| `teardown_qwen.sh` | Deletes the deployment to **stop billing**. |
| `README.md`        | This file. |

## Tooling

`firectl` is **not installed** in this environment. These scripts drive the
Fireworks **control-plane REST API** through the installed `fireworks` Python SDK
(`.venv/bin/python`) — the documented fallback when `firectl` isn't available. If
you install `firectl` later, the same deployment shows up under
`firectl list deployments`.

The `FIREWORKS_API_KEY` is read from the repo-root `.env` via `python-dotenv`
inside the scripts and is **never printed**.

## Run it

From the repo root (`hudhack/`):

```bash
# Deploy the default base model (accounts/fireworks/models/qwen3-4b):
./scripts/deploy_qwen.sh

# Or deploy the instruct variant:
./scripts/deploy_qwen.sh qwen3-4b-instruct-2507

# Tear down when done testing (STOPS billing):
./scripts/teardown_qwen.sh
```

### Optional knobs (env vars on `deploy_qwen.sh`)

| Var | Default | Meaning |
|-----|---------|---------|
| `MODEL_SLUG` | `qwen3-4b` | base model slug under `accounts/fireworks/models` |
| `DEPLOYMENT_ID` | `qwen3-4b-dedicated` | fixed id; makes re-runs idempotent (won't double-deploy) |
| `ACCELERATOR_TYPE` | *(unset)* | e.g. `NVIDIA_H100_80GB`; unset = Fireworks default shape |
| `MIN_REPLICAS` | `1` | keep one replica warm; set `0` to allow scale-to-zero |
| `WAIT_TIMEOUT_SECS` | `900` | how long to wait for READY |

`deploy_qwen.sh` is **idempotent-ish**: if a deployment for this base model is
already live (or matches `DEPLOYMENT_ID`), it reuses it instead of creating a
second billable endpoint.

### Inference handle

Once live, point OpenAI-compatible inference at:

```
base_url = https://api.fireworks.ai/inference/v1
model    = accounts/fireworks/models/<slug>#accounts/<your-account>/deployments/<deployment-id>
```

`deploy_qwen.sh` prints the exact handle for your deployment.

## ⚠️ COST — read this

A dedicated Qwen3-4B deployment **bills for as long as it is up (~$7/hr** on a
1×H100-class accelerator). It does **NOT** stop when `deploy_qwen.sh` exits.

**Always run `./scripts/teardown_qwen.sh` the moment you're done testing.**

`teardown_qwen.sh` lists what remains after deleting, so you get a clear
"no active deployments remain — billing stopped" confirmation.

## Prerequisite: payment method on the Fireworks account

Fireworks requires a **card on file** to create a dedicated deployment, **even
when the account has prepaid credits**. With credits but no card, the control
plane rejects `deployments.create` with:

```
400 - {'code': 9, 'message': 'payment method is required'}
```

If you hit that error, the account owner must add a payment method here:
**Fireworks dashboard → Settings → Billing → add a card**, then re-run
`./scripts/deploy_qwen.sh`. The credits still cover the actual usage; the card is
just the required guarantee Fireworks demands before letting you spin up
dedicated hardware.
