#!/usr/bin/env python3
"""deploy_qwen_test.py — Prove the Crucible Teacher (Qwen3-4B) can serve and emit
VALID curriculum JSON, via a fully replicable deploy -> test -> teardown cycle.

WHY THIS EXISTS
---------------
This account's Qwen3-4B is NOT serverless: hitting the serverless inference path
404s. To run the real RFT Teacher you therefore need a *dedicated* Fireworks
deployment (~$7/hr while up). This script:

  1. Loads FIREWORKS_API_KEY from .env (never prints it).
  2. Deploys accounts/fireworks/models/qwen3-4b to a dedicated endpoint and polls
     until READY (with a timeout).
  3. Runs ONE test inference using the *real* Crucible curriculum prompt and the
     *real* fighter parameter schema, by driving the production FireworksTeacher
     client (training/fireworks_teacher.py) at the deployed endpoint. That means
     the exact same prompt, the same response_format={"type":"json_object"}, and
     the same _strict_validate_params + _clamp_to_schema logic RFT will use.
  4. Reports CLEARLY whether Qwen emitted valid, in-schema curriculum JSON.
  5. TEARS DOWN the deployment in a finally: block so it never bills idle.

GUARANTEED TEARDOWN
-------------------
Teardown runs in `finally:` and is reached on success, on any exception, and on
KeyboardInterrupt. The only way the deployment survives is the explicit
`--no-teardown` flag.

COST WARNING
------------
A dedicated Qwen3-4B deployment bills (~$7/hr) for as long as it is UP. By
default this script deletes it before exiting. If you pass --no-teardown, YOU
are responsible for running:  python scripts/deploy_qwen_test.py --teardown-only <id>

Tooling note: firectl is NOT installed in this environment. The Fireworks
control plane is driven via the installed `fireworks` Python SDK (.venv), which
is the same path the repo's deploy_qwen.sh / teardown_qwen.sh already use. The
created deployment is visible to `firectl list deployments` if firectl is later
installed.

Usage:
  .venv/bin/python scripts/deploy_qwen_test.py                 # deploy, test, teardown
  .venv/bin/python scripts/deploy_qwen_test.py --no-teardown   # keep it up afterward
  .venv/bin/python scripts/deploy_qwen_test.py --teardown-only qwen3-4b-test
  .venv/bin/python scripts/deploy_qwen_test.py --model qwen3-4b-instruct-2507

Options:
  --model SLUG          base model slug under accounts/fireworks/models
                        (default: qwen3-4b)
  --deployment-id ID    fixed id for idempotency / reuse (default: qwen3-4b-test)
  --timeout SECS        max seconds to wait for READY (default: 600)
  --min-replicas N      min replica count, 1 keeps it warm (default: 1)
  --accelerator TYPE    e.g. NVIDIA_H100_80GB (default: Fireworks default shape)
  --no-teardown         do NOT delete the deployment at the end (it keeps billing)
  --teardown-only ID    delete this deployment id and exit; deploy/test nothing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- Load .env (FIREWORKS_API_KEY) without ever printing the value -----------
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - venv should have it
    print(
        "ERROR: python-dotenv is not installed in this interpreter. "
        "Run this with .venv/bin/python (deps live in .venv).",
        file=sys.stderr,
    )
    raise SystemExit(2)

load_dotenv(ROOT / ".env")

import os  # noqa: E402  (after load_dotenv so os.getenv sees .env)

# Reuse the REAL Teacher client + curriculum prompt + validation, so this test
# exercises the exact code path RFT uses (same prompt, same json_object request,
# same _strict_validate_params + _clamp_to_schema).
from engine.games import Game  # noqa: E402
from training.fireworks_teacher import FireworksTeacher, _strict_validate_params  # noqa: E402
from training.teacher import _clamp_to_schema  # noqa: E402

BASE_MODEL_TMPL = "accounts/fireworks/models/{slug}"

# The fighter parameter schema + safe ranges — copied verbatim from the bounds
# the Modal EP bridge advertises (training/modal_ep_bridge.py) and the arena
# defaults from games/fighter.py. The Game dataclass needs a param_schema and
# defaults; both _strict_validate_params and _clamp_to_schema read only those.
FIGHTER_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "gravity": (0.2, 1.2),
    "knockback": (0.5, 6.0),
    "spawn_gap": (1.0, 12.0),
}
FIGHTER_DEFAULTS: dict[str, float] = {
    "difficulty": 0.5,
    "platform_width": 10.0,
    "gravity": 0.6,
    "knockback": 2.5,
    "spawn_gap": 4.0,
}


def _fighter_game() -> Game:
    """Construct the fighter Game so FireworksTeacher.generate(game) emits the
    real curriculum prompt for the fighter schema.

    to_gridworld_params is unused on this code path (generate() only reads
    name/param_schema, and validation only reads param_schema/defaults), so it is
    an identity passthrough purely to satisfy the frozen dataclass.
    """
    return Game(
        name="fighter",
        param_schema=FIGHTER_BOUNDS,
        to_gridworld_params=lambda params: dict(params),
        description="Platform-fighter arena (Crucible Teacher target).",
        defaults=FIGHTER_DEFAULTS,
    )


# -----------------------------------------------------------------------------
# Fireworks control-plane helpers (via the `fireworks` Python SDK).
# -----------------------------------------------------------------------------

def _require_api_key() -> str:
    key = (os.getenv("FIREWORKS_API_KEY") or "").strip()
    if not key:
        print(
            "ERROR: FIREWORKS_API_KEY is missing/empty. Set it in .env at the repo "
            "root (see .env.example). It is read from .env and never printed.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return key


def _import_fireworks():
    try:
        from fireworks import Fireworks  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: the 'fireworks' Python SDK is not installed in this interpreter. "
            "Install it with: .venv/bin/python -m pip install fireworks",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return Fireworks


def _resolve_account(Fireworks, api_key: str):
    """Resolve the account id from the API key (no hardcoding) and return an
    account-scoped client plus the account id."""
    client = Fireworks(api_key=api_key)
    page = client.accounts.list()
    items = getattr(page, "accounts", None) or list(page)
    if not items:
        print("ERROR: could not resolve any account from the API key.", file=sys.stderr)
        raise SystemExit(2)
    acct_name = getattr(items[0], "name", None) or str(items[0])
    account_id = acct_name.split("/")[-1]  # accounts/<id> -> <id>
    return Fireworks(api_key=api_key, account_id=account_id), account_id


def _deployment_handle(base_model: str, account_id: str, dep_id: str) -> str:
    """OpenAI-compatible model string that routes inference to this dedicated
    deployment (Fireworks' `<base>#accounts/<acct>/deployments/<id>` form)."""
    return f"{base_model}#accounts/{account_id}/deployments/{dep_id}"


def _find_existing(fw, base_model: str, deployment_id: str):
    """Idempotency: reuse a live deployment for this base model if one exists."""
    for d in fw.deployments.list():
        if getattr(d, "base_model", None) != base_model:
            continue
        short = (d.name or "").split("/")[-1]
        if short == deployment_id or getattr(d, "state", None) in ("READY", "CREATING", "UPDATING"):
            return d
    return None


def _delete_deployment(fw, deployment_id: str) -> bool:
    """Delete one deployment id. Returns True if a delete call succeeded."""
    deps = list(fw.deployments.list())
    targets = [d for d in deps if (d.name or "").split("/")[-1] == deployment_id]
    if not targets:
        print(f"==> Teardown: no deployment matched id '{deployment_id}'. Nothing to delete.")
        if deps:
            print("    Deployments currently on the account:")
            for d in deps:
                print(
                    f"      - {(d.name or '').split('/')[-1]} "
                    f"| base={d.base_model} | state={d.state}"
                )
        return False
    ok = False
    for d in targets:
        dep_id = (d.name or "").split("/")[-1]
        print(f"==> Teardown: deleting deployment '{dep_id}' "
              f"(base={d.base_model}, state={d.state}) ...")
        try:
            # ignore_checks=True forces deletion even though we JUST sent a test
            # inference; without it Fireworks rejects the delete for an hour with
            # "deployment has received inference requests in the last hour", which
            # would leave the deployment UP and billing (~$7/hr). Teardown must
            # always win over that guard.
            fw.deployments.delete(dep_id, ignore_checks=True)
            print(f"    deleted: {dep_id}")
            ok = True
        except Exception as e:  # noqa: BLE001 - report honestly, keep going
            print(f"    ERROR deleting {dep_id}: {type(e).__name__}: {e}", file=sys.stderr)
    return ok


def _verify_gone(fw, deployment_id: str) -> None:
    """Confirm the deployment is gone (or DELETING) — the firectl-equivalent check."""
    remaining = [
        (d.name or "").split("/")[-1]
        for d in fw.deployments.list()
        if (d.name or "").split("/")[-1] == deployment_id
        and getattr(d, "state", None) not in ("DELETED", "DELETING")
    ]
    if remaining:
        print(
            f"==> WARNING: deployment '{deployment_id}' still present after delete "
            f"(state not DELETED/DELETING). It may still be billing. Re-run "
            f"--teardown-only {deployment_id}.",
            file=sys.stderr,
        )
    else:
        print(f"==> Verified: deployment '{deployment_id}' is gone "
              f"(absent or DELETING). Dedicated serving billing has stopped.")


# -----------------------------------------------------------------------------
# Deploy + wait
# -----------------------------------------------------------------------------

def _deploy_and_wait(fw, args, base_model: str) -> tuple[str, float]:
    """Create-or-reuse the dedicated deployment and poll until READY.

    Returns (deployment_id, seconds_to_ready). Raises on timeout/terminal state.
    """
    existing = _find_existing(fw, base_model, args.deployment_id)
    if existing is not None:
        dep = existing
        dep_id = (dep.name or "").split("/")[-1]
        print(f"==> Reusing existing deployment '{dep_id}' (state={dep.state}); "
              f"not creating a new one.")
    else:
        print(f"==> Creating dedicated deployment '{args.deployment_id}' for {base_model} ...")
        kwargs = dict(
            base_model=base_model,
            deployment_id=args.deployment_id,
            min_replica_count=args.min_replicas,
            max_replica_count=max(args.min_replicas, 1),
            display_name=f"qwen3-4b test ({args.model})",
            description="Crucible Teacher test deployment (HUD hackathon). Bills while up.",
        )
        if args.accelerator:
            kwargs["accelerator_type"] = args.accelerator
        dep = fw.deployments.create(**kwargs)
        dep_id = (dep.name or "").split("/")[-1]
        print(f"==> Create accepted: {dep.name} (state={dep.state})")

    print(f"==> Waiting up to {args.timeout}s for deployment to become READY ...")
    start = time.time()
    deadline = start + args.timeout
    last_state = None
    while time.time() < deadline:
        cur = fw.deployments.get(dep_id)
        st = getattr(cur, "state", None)
        if st != last_state:
            elapsed = int(time.time() - start)
            print(f"    [{elapsed:>4}s] state={st} "
                  f"replicas={getattr(cur, 'replica_count', '?')}")
            last_state = st
        if st == "READY":
            return dep_id, time.time() - start
        if st in ("FAILED", "DELETING", "DELETED"):
            raise RuntimeError(f"deployment entered terminal state {st} before READY")
        time.sleep(10)
    raise TimeoutError(
        f"timed out after {args.timeout}s waiting for READY (last state={last_state}). "
        f"The deployment may still be coming up — teardown will run to avoid charges."
    )


# -----------------------------------------------------------------------------
# Test inference + validation (real Teacher code path)
# -----------------------------------------------------------------------------

def _run_inference_and_validate(handle: str, api_key: str) -> bool:
    """Drive the production FireworksTeacher at the deployed endpoint and validate
    its output against the fighter schema. Returns True iff valid in-schema JSON.

    We capture the RAW completion via a thin transport shim that wraps the real
    HTTP transport, so we can print exactly what Qwen returned AND the parsed/
    clamped params — while still running the genuine generate() prompt + retry +
    _strict_validate_params + _clamp_to_schema path.
    """
    game = _fighter_game()
    captured: dict[str, object] = {}

    teacher = FireworksTeacher(model=handle, api_key=api_key)
    real_transport = teacher._http_transport  # the genuine Fireworks HTTP call

    def _capturing_transport(payload: dict) -> dict:
        # First call captures the exact prompt payload we send.
        captured.setdefault("request_payload", payload)
        response = real_transport(payload)
        # Capture the raw completion content for printing.
        try:
            captured["raw_completion"] = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            captured["raw_completion"] = repr(response)
        return response

    teacher._transport = _capturing_transport

    print("==> Sending the REAL Crucible curriculum prompt (game='fighter', "
          "fighter schema, response_format=json_object) to the deployed endpoint ...")

    # Show the exact user-message prompt that crosses the wire (no secrets in it).
    # We mirror what generate() builds so the operator can see it even if the call
    # raises before capture.
    preview_prompt = {
        "game": game.name,
        "parameter_schema": {
            key: {"minimum": low, "maximum": high}
            for key, (low, high) in game.param_schema.items()
        },
        "instruction": (
            "Return one JSON object containing only parameter keys. "
            "Choose a potentially learnable environment."
        ),
    }
    print("---- curriculum prompt (user message) ----")
    print(json.dumps(preview_prompt, indent=2, sort_keys=True))
    print("------------------------------------------")

    try:
        validated = teacher.generate(game)
    except Exception as e:  # noqa: BLE001
        raw = captured.get("raw_completion")
        print()
        print("============================================================")
        print("RESULT: Qwen did NOT emit valid curriculum JSON.")
        print("============================================================")
        print(f"  error: {type(e).__name__}: {e}")
        if raw is not None:
            print("  raw completion returned by the model:")
            print(f"    {raw!r}")
        else:
            print("  (no completion captured — the request itself failed; "
                  "see error above)")
        return False

    raw = captured.get("raw_completion")

    # Re-derive the parsed + clamped view explicitly so we print BOTH the raw and
    # the post-validation params (generate() already returned the clamped dict via
    # _strict_validate_params -> _clamp_to_schema; we recompute for transparency).
    parsed_obj = None
    clamped = validated
    if isinstance(raw, str):
        try:
            parsed_obj = json.loads(raw)
            # Same two-step the real path runs:
            clamped = _strict_validate_params(parsed_obj, game)
            clamped = _clamp_to_schema(clamped, game)
        except Exception as e:  # noqa: BLE001 - validated already succeeded; informational
            parsed_obj = f"<re-parse failed: {type(e).__name__}: {e}>"

    print()
    print("============================================================")
    print("RESULT: Qwen emitted VALID, in-schema curriculum JSON.")
    print("============================================================")
    print("  raw completion (verbatim from the model):")
    print(f"    {raw!r}")
    print()
    print("  parsed JSON object:")
    print(json.dumps(parsed_obj, indent=4, sort_keys=True)
          if isinstance(parsed_obj, (dict, list)) else f"    {parsed_obj!r}")
    print()
    print("  validated + clamped params (what RFT would consume):")
    print(json.dumps(clamped, indent=4, sort_keys=True))
    print()
    print("  schema bounds enforced:")
    for key, (low, high) in FIGHTER_BOUNDS.items():
        val = clamped.get(key)
        in_range = isinstance(val, (int, float)) and low <= val <= high
        flag = "OK" if in_range else "OUT-OF-RANGE"
        print(f"    {key:<15} = {val!r:<22} in [{low}, {high}]  -> {flag}")
    print("============================================================")
    return True


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deploy Qwen3-4B to a dedicated Fireworks endpoint, test that "
                    "it emits valid fighter curriculum JSON, then tear it down.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="qwen3-4b",
                   help="base model slug under accounts/fireworks/models (default: qwen3-4b)")
    p.add_argument("--deployment-id", default="qwen3-4b-test",
                   help="fixed deployment id for idempotency (default: qwen3-4b-test)")
    p.add_argument("--timeout", type=int, default=600,
                   help="max seconds to wait for READY (default: 600)")
    p.add_argument("--min-replicas", type=int, default=1,
                   help="min replica count; 1 keeps it warm (default: 1)")
    p.add_argument("--accelerator", default="NVIDIA_H100_80GB",
                   help="accelerator type (default: NVIDIA_H100_80GB). Fireworks now "
                        "REQUIRES accelerator_type for non-embeddings engines, so an "
                        "empty value is rejected at create time; pass '' to force the "
                        "(now-rejected) Fireworks default shape.")
    p.add_argument("--no-teardown", action="store_true",
                   help="do NOT delete the deployment at the end (it keeps billing)")
    p.add_argument("--teardown-only", metavar="ID", default=None,
                   help="delete this deployment id and exit; deploy/test nothing")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    api_key = _require_api_key()
    Fireworks = _import_fireworks()
    fw, account_id = _resolve_account(Fireworks, api_key)

    # --- escape hatch: teardown only ----------------------------------------
    if args.teardown_only:
        print(f"==> --teardown-only: deleting deployment '{args.teardown_only}' and exiting.")
        _delete_deployment(fw, args.teardown_only)
        _verify_gone(fw, args.teardown_only)
        return 0

    base_model = BASE_MODEL_TMPL.format(slug=args.model)

    # Confirm the base model is resolvable before spending money.
    try:
        m = fw.models.get(args.model, account_id="fireworks")
        state = getattr(m, "state", None)
        if state not in (None, "READY"):
            print(f"==> WARNING: base model {base_model} state={state} (not READY).",
                  file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: base model {base_model} not resolvable: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    print("============================================================")
    print("Crucible Teacher deploy -> test -> teardown")
    print("============================================================")
    print(f"  base model     : {base_model}")
    print(f"  account        : {account_id}")
    print(f"  deployment id  : {args.deployment_id}")
    print(f"  min replicas   : {args.min_replicas}")
    print(f"  ready timeout  : {args.timeout}s")
    print(f"  teardown after : {'NO (--no-teardown)' if args.no_teardown else 'YES'}")
    if args.accelerator:
        print(f"  accelerator    : {args.accelerator}")
    print("  COST WARNING   : a dedicated deployment bills ~$7/hr while UP.")
    print("============================================================")
    print()

    deployment_id = args.deployment_id
    inference_ok = False
    exit_code = 1
    try:
        deployment_id, secs = _deploy_and_wait(fw, args, base_model)
        handle = _deployment_handle(base_model, account_id, deployment_id)
        print()
        print(f"==> Deployment READY in {secs:.1f}s.")
        print(f"    deployment_id   : {deployment_id}")
        print(f"    inference handle: {handle}")
        print(f"    endpoint base   : https://api.fireworks.ai/inference/v1")
        print()

        inference_ok = _run_inference_and_validate(handle, api_key)
        exit_code = 0 if inference_ok else 1

    except KeyboardInterrupt:
        print("\n==> Interrupted by user. Running teardown so nothing bills idle ...",
              file=sys.stderr)
        exit_code = 130
    except Exception as e:  # noqa: BLE001 - report honestly, then ALWAYS tear down
        print(f"\n==> ERROR during deploy/test: {type(e).__name__}: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        print()
        if args.no_teardown:
            print("============================================================")
            print(f"==> --no-teardown set: leaving deployment '{deployment_id}' UP.")
            print(f"    IT KEEPS BILLING (~$7/hr). Tear it down with:")
            print(f"    .venv/bin/python scripts/deploy_qwen_test.py "
                  f"--teardown-only {deployment_id}")
            print("============================================================")
        else:
            print("============================================================")
            print("==> GUARANTEED TEARDOWN (finally:) — deleting deployment to stop billing.")
            print("============================================================")
            try:
                _delete_deployment(fw, deployment_id)
                _verify_gone(fw, deployment_id)
            except Exception as e:  # noqa: BLE001
                print(f"==> ERROR during teardown: {type(e).__name__}: {e}", file=sys.stderr)
                print(f"==> The deployment '{deployment_id}' MAY still be up and billing. "
                      f"Run: .venv/bin/python scripts/deploy_qwen_test.py "
                      f"--teardown-only {deployment_id}", file=sys.stderr)

    print()
    if args.teardown_only is None:
        print(f"==> Final result: inference {'PASSED' if inference_ok else 'FAILED'}.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
