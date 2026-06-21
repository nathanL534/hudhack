"""output/stage6/handles.py — resolve a Teacher *handle* into a Teacher.

THE CORE ABSTRACTION of Stage 6's flexibility. A "handle" is an opaque string the
operator passes on the CLI for the BASE and TRAINED Teachers. It must resolve to
EITHER of two very different things, because Fireworks gates >=4B RFT on this
account, so the real trained Teacher may be a LOCAL Modal-GRPO LoRA adapter, not a
Fireworks checkpoint:

  1. a Fireworks model id / dedicated-deployment handle
       e.g. ``accounts/fireworks/models/qwen3-4b``
            ``fw:qwen3-4b-rft``  (a deployment id, resolved against the account)
            ``accounts/.../models/...#accounts/.../deployments/...``  (full handle)
  2. a LOCAL LoRA adapter path
       e.g. ``local:/abs/path/to/adapter``  or a bare existing directory path
            containing ``adapter_config.json`` (a PEFT adapter dir).

Design
------
* ``ResolvedTeacher`` is a small value object: a human label, the canonical
  resolved id (what actually gets recorded in the result JSON), the backend kind
  (``fireworks`` / ``local``), and a zero-arg ``build()`` that returns a live
  ``Teacher``. Resolution (account lookup, path checks) happens ONCE, up front, so
  the evaluator can fail fast and record exactly what it ran against.
* The local-adapter path is intentionally LAZY about heavy deps: if ``peft`` /
  ``transformers`` / ``vllm`` are not installed, ``build()`` raises a clear
  "local adapter inference not installed" error instead of crashing obscurely. The
  RESOLVE step still succeeds (the path is recorded), so ``--smoke`` and dry runs
  that never call ``build()`` keep working.

This module knows nothing about games, PPO, or Modal — only "string -> Teacher".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# A Teacher is anything with ``.generate(game) -> dict`` (see training/teacher.py).
TeacherFactory = Callable[[], "object"]

FIREWORKS_DEFAULT_MODEL = "accounts/fireworks/models/qwen3-4b"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


@dataclass(frozen=True)
class ResolvedTeacher:
    """A handle resolved to everything the evaluator needs, recorded verbatim."""

    label: str          # human label, e.g. "base" / "trained"
    handle: str         # the raw handle the operator passed
    resolved_id: str    # canonical id we actually run against (recorded in JSON)
    kind: str           # "fireworks" | "local"
    _build: TeacherFactory

    def build(self) -> object:
        """Construct the live Teacher (may raise for an uninstalled local backend)."""
        return self._build()


def _looks_like_fireworks(handle: str) -> bool:
    """Heuristic: Fireworks ids carry ``accounts/`` or an explicit ``fw:`` prefix."""
    h = handle.strip()
    return (
        h.startswith("fw:")
        or h.startswith("fireworks:")
        or "accounts/" in h
        or h.endswith("#deployment")  # tolerated shorthand
    )


def _looks_like_local_path(handle: str) -> bool:
    """A local adapter handle: explicit ``local:`` prefix OR an existing dir/path."""
    h = handle.strip()
    if h.startswith("local:") or h.startswith("file:"):
        return True
    # Bare path that exists on disk (a PEFT adapter directory or weights file).
    p = Path(os.path.expanduser(h))
    return p.exists()


def _strip_prefix(handle: str, *prefixes: str) -> str:
    h = handle.strip()
    for pfx in prefixes:
        if h.startswith(pfx):
            return h[len(pfx):]
    return h


# ---------------------------------------------------------------------------
# Fireworks resolution
# ---------------------------------------------------------------------------


def _resolve_fireworks_account_and_key() -> tuple[str, str]:
    """Resolve the Fireworks account slug + API key (mirrors the legacy evaluator).

    Kept here so the handle layer owns the only Fireworks-account round-trip.
    """
    from fireworks import Fireworks

    key = (os.getenv("FIREWORKS_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("FIREWORKS_API_KEY missing from environment/.env")
    client = Fireworks(api_key=key)
    page = client.accounts.list()
    items = getattr(page, "accounts", None) or list(page)
    if not items:
        raise RuntimeError("could not resolve a Fireworks account from the API key")
    acct = (getattr(items[0], "name", None) or str(items[0])).split("/")[-1]
    return acct, key


def _fireworks_inference_handle(token: str, account: str, base_model: str) -> str:
    """Turn a deployment-id token into the full Fireworks inference handle.

    Accepts, in order of specificity:
      * a FULL handle already containing ``#accounts/.../deployments/...`` -> as-is
      * a fully-qualified model id (``accounts/.../models/...``)           -> as-is
      * a bare deployment id (``qwen3-4b-rft``) -> ``{base_model}#accounts/{acct}/deployments/{id}``
    """
    t = token.strip()
    if "#accounts/" in t and "/deployments/" in t:
        return t
    if t.startswith("accounts/") and "/models/" in t and "#" not in t:
        return t
    return f"{base_model}#accounts/{account}/deployments/{t}"


def resolve_fireworks(
    label: str,
    handle: str,
    *,
    base_model: str = FIREWORKS_DEFAULT_MODEL,
    base_url: str = FIREWORKS_BASE_URL,
    account: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResolvedTeacher:
    """Resolve a Fireworks handle. Account/key are looked up once unless supplied."""
    token = _strip_prefix(handle, "fw:", "fireworks:")
    if account is None or api_key is None:
        account, api_key = _resolve_fireworks_account_and_key()
    inference_handle = _fireworks_inference_handle(token, account, base_model)

    def _build() -> object:
        from training.fireworks_teacher import FireworksTeacher

        return FireworksTeacher(model=inference_handle, api_key=api_key, base_url=base_url)

    return ResolvedTeacher(
        label=label,
        handle=handle,
        resolved_id=inference_handle,
        kind="fireworks",
        _build=_build,
    )


# ---------------------------------------------------------------------------
# Local LoRA-adapter resolution
# ---------------------------------------------------------------------------


def resolve_local_adapter(
    label: str,
    handle: str,
    *,
    base_model: str = FIREWORKS_DEFAULT_MODEL,
) -> ResolvedTeacher:
    """Resolve a LOCAL LoRA-adapter handle to a Teacher backed by local inference.

    The path is validated NOW (so a typo fails fast and is recorded), but the heavy
    inference stack (transformers/peft/vllm) is imported LAZILY inside ``build()``.
    If those deps are absent, ``build()`` raises a clear, actionable error — the
    real trained Teacher coming from a local Modal-GRPO run plugs in here.
    """
    raw = _strip_prefix(handle, "local:", "file:")
    adapter_path = Path(os.path.expanduser(raw)).resolve()
    if not adapter_path.exists():
        raise RuntimeError(
            f"local adapter path does not exist: {adapter_path} (handle {handle!r})"
        )

    def _build() -> object:
        return LocalAdapterTeacher(str(adapter_path), base_model=base_model)

    return ResolvedTeacher(
        label=label,
        handle=handle,
        resolved_id=f"local:{adapter_path}",
        kind="local",
        _build=_build,
    )


class LocalAdapterTeacher:
    """A Teacher that runs a LOCAL base model + LoRA adapter for inference.

    Conforms to the Teacher contract (``generate(game) -> dict``) using the SAME
    prompt construction as ``FireworksTeacher`` so base/trained prompts stay
    identical regardless of backend. It reuses ``FireworksTeacher``'s prompt +
    strict-validate + clamp machinery by injecting a local "transport".

    Heavy deps (transformers/peft) are imported in ``__init__`` so a misconfigured
    environment fails with a precise message rather than at first ``generate()``.
    """

    def __init__(self, adapter_path: str, *, base_model: str = FIREWORKS_DEFAULT_MODEL):
        self.adapter_path = adapter_path
        self.base_model = base_model
        try:
            import torch  # noqa: F401
            from peft import PeftModel  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "local adapter inference not installed: this handle is a local LoRA "
                "adapter path, which needs `transformers` + `peft` (and a torch "
                "backend) to run. Install them, or point --trained at a Fireworks "
                f"deployment id instead. (underlying import error: {exc})"
            ) from exc
        # Lazy heavy load deferred to first generate() so resolve()/construction is
        # cheap; the import check above already proved the stack is present.
        self._pipe = None  # built on first use

    def _ensure_pipe(self):  # pragma: no cover - requires real weights + GPU/CPU
        if self._pipe is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, self.adapter_path)
        model.eval()
        self._tok, self._model = tok, model
        self._pipe = True

    def generate(self, game) -> dict:  # pragma: no cover - requires real weights
        """Generate one validated+clamped param set, reusing the Fireworks prompt."""
        from training.fireworks_teacher import FireworksTeacher, _strict_validate_params

        self._ensure_pipe()
        # Build the IDENTICAL prompt FireworksTeacher would send, then run it locally.
        proxy = FireworksTeacher(model=self.base_model, transport=self._local_transport)
        return proxy.generate(game)

    def _local_transport(self, payload: dict) -> dict:  # pragma: no cover
        """Run the chat payload through the local model; return Fireworks-shaped JSON."""
        import torch

        messages = payload["messages"]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=float(payload.get("temperature", 0.8)),
                do_sample=True,
            )
        completion = self._tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return {"choices": [{"message": {"content": completion}}]}


# ---------------------------------------------------------------------------
# Offline resolution (credential-free; for the cheap local smoke)
# ---------------------------------------------------------------------------


def resolve_offline(label: str, handle: str = "offline") -> ResolvedTeacher:
    """A credential-free Teacher backed by a canned response (no Fireworks key).

    Lets the documented cheap local smoke (``--smoke --backend local``) run with NO
    network and NO API key, while still exercising the full generate -> clamp ->
    fan-out path. ``offline:{json}`` lets the smoke pin specific params if needed.
    """
    import json as _json

    # A fighter-valid canned set so the common smoke (game=fighter) passes strict
    # validation; ``offline:{json}`` overrides for other games. The base/trained
    # labels get slightly different params so a base-vs-base smoke still produces a
    # non-degenerate (>=2 unique) arena set.
    if label.startswith("base") and not label.startswith("base(smoke)"):
        default_params = {"difficulty": 0.5, "platform_width": 12.0, "gravity": 0.6,
                          "knockback": 2.5, "spawn_gap": 4.0}
    else:
        default_params = {"difficulty": 0.6, "platform_width": 14.0, "gravity": 0.7,
                          "knockback": 3.0, "spawn_gap": 5.0}
    params = default_params
    raw = _strip_prefix(handle, "offline:")
    if raw and raw != "offline":
        try:
            params = _json.loads(raw)
        except Exception:
            params = default_params

    def _build() -> object:
        from training.fireworks_teacher import FireworksTeacher

        return FireworksTeacher.offline(params=params)

    return ResolvedTeacher(
        label=label, handle=handle, resolved_id=f"offline:{label}",
        kind="offline", _build=_build,
    )


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def resolve_handle(
    label: str,
    handle: str,
    *,
    base_model: str = FIREWORKS_DEFAULT_MODEL,
    base_url: str = FIREWORKS_BASE_URL,
    account: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResolvedTeacher:
    """Resolve ANY handle to a ``ResolvedTeacher`` (Fireworks id OR local adapter).

    Dispatch order: an explicit ``local:``/``file:`` prefix or an existing on-disk
    path is treated as a local adapter; anything that looks like a Fireworks id
    (``accounts/`` / ``fw:`` / ``fireworks:``) goes to Fireworks. A bare token that
    is neither an existing path nor obviously Fireworks is treated as a Fireworks
    *deployment id* (the common ``--trained qwen3-4b-rft`` case).
    """
    if handle.strip().startswith("offline:") or handle.strip() == "offline":
        return resolve_offline(label, handle)
    if _looks_like_local_path(handle):
        return resolve_local_adapter(label, handle, base_model=base_model)
    if _looks_like_fireworks(handle):
        return resolve_fireworks(
            label, handle, base_model=base_model, base_url=base_url,
            account=account, api_key=api_key,
        )
    # Bare token -> Fireworks deployment id.
    return resolve_fireworks(
        label, handle, base_model=base_model, base_url=base_url,
        account=account, api_key=api_key,
    )
