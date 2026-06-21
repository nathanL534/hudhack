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
  3. a MODAL Teacher-generation handle (the working path on this account, since
     Fireworks gives a 404 for serverless qwen3-4b inference and the host venv has
     no transformers/peft):
       e.g. ``modal:base``         -> Qwen3-4B base weights, NO adapter, on Modal GPU
            ``modal:update2``      -> base + the LoRA adapter at Volume subdir update2
            ``modal:`` (bare)      -> base, no adapter (same as ``modal:base``)
     Generation runs in the ``crucible-teacher-gen`` Modal app (transformers+peft
     live in its image); the host needs only Modal auth, not a GPU or HF stack.

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
# ``build`` takes an optional ``gen_seed`` so each Stage-6 replicate can request a
# DIFFERENT Teacher-generation seed (so replicates sample different curricula); a
# backend for which a client-side seed is meaningless simply ignores it.
TeacherFactory = Callable[..., "object"]

FIREWORKS_DEFAULT_MODEL = "accounts/fireworks/models/qwen3-4b"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# The HuggingFace base-model id the LOCAL and MODAL backends load. The trained GRPO
# LoRA adapters (Modal Volume ``crucible-teacher-lora``) target THIS id, not the
# Fireworks id, so a ``local:``/``modal:`` handle must default here — passing the
# Fireworks id (``accounts/...``) to transformers/peft would 404 / not resolve.
HF_DEFAULT_MODEL = "Qwen/Qwen3-4B"

# The Modal generation app + LoRA adapter Volume (see training/teacher_gen_modal.py).
MODAL_GEN_APP = "crucible-teacher-gen"
MODAL_GEN_CLS = "TeacherGenerator"

# Stage-6 game names -> games_registry discriminators. The Stage-6 ``engine.games.Game``
# the Teacher is prompted with is named ``"fighter"`` for the Ring-Out duel, but the
# trainer's prompt for that game is registered under ``"ring_out"`` — so the two names
# must be bridged to reach the right training prompt. ``target_knockback`` matches
# verbatim. KOTH's cross-game path prompts the Teacher with the FIGHTER game (its
# ``_MappingTeacher`` calls ``generate(fighter_game)``), so it never reaches here under
# a "koth" name; the fighter mapping below covers it correctly.
_STAGE6_GAME_TO_REGISTRY = {
    "fighter": "ring_out",
    "ring_out": "ring_out",
    "target_knockback": "target_knockback",
}


def _training_prompt_for_game(game) -> str:
    """Return the EXACT training prompt for ``game`` from the single source of truth.

    Sources the prompt from ``games_registry`` (the same module the GRPO trainer
    mirrors + parity-asserts), so Stage-6 generation conditions the Teacher on the
    identical string it was trained on — making the base-vs-trained comparison a
    clean evaluation on the training task, not a prompt-generalization test.
    """
    from games_registry import get as _registry_get

    name = getattr(game, "name", "")
    key = _STAGE6_GAME_TO_REGISTRY.get(name)
    if key is None:
        raise KeyError(
            f"no training prompt registered for Stage-6 game {name!r}; "
            f"known: {sorted(_STAGE6_GAME_TO_REGISTRY)}"
        )
    return _registry_get(key).prompt


@dataclass(frozen=True)
class ResolvedTeacher:
    """A handle resolved to everything the evaluator needs, recorded verbatim."""

    label: str          # human label, e.g. "base" / "trained"
    handle: str         # the raw handle the operator passed
    resolved_id: str    # canonical id we actually run against (recorded in JSON)
    kind: str           # "fireworks" | "local"
    _build: TeacherFactory

    def build(self, *, gen_seed: Optional[int] = None) -> object:
        """Construct the live Teacher (may raise for an uninstalled local backend).

        ``gen_seed`` lets a caller request a SPECIFIC Teacher-generation seed so that
        independent Stage-6 replicates sample DIFFERENT curricula from the same
        Teacher (otherwise every replicate would emit identical arenas, making the
        curriculum-level CI capture only Student-training noise). The seed is applied
        identically to base and trained for a given replicate, so the fairness
        invariant holds; backends without a client-side seed ignore it.
        """
        return self._build(gen_seed=gen_seed)


def _looks_like_fireworks(handle: str) -> bool:
    """Heuristic: Fireworks ids carry ``accounts/`` or an explicit ``fw:`` prefix."""
    h = handle.strip()
    return (
        h.startswith("fw:")
        or h.startswith("fireworks:")
        or "accounts/" in h
        or h.endswith("#deployment")  # tolerated shorthand
    )


def _looks_like_modal(handle: str) -> bool:
    """A Modal Teacher-generation handle carries the explicit ``modal:`` prefix."""
    return handle.strip().startswith("modal:")


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

    def _build(*, gen_seed: Optional[int] = None) -> object:
        # Fireworks generation is server-side stochastic (no client seed knob), so a
        # per-replicate ``gen_seed`` is accepted for interface uniformity but unused.
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

    def _build(*, gen_seed: Optional[int] = None) -> object:
        # Local sampling seed lives inside the lazy transport; accept for interface
        # uniformity. (The local backend is unused on this account; Modal is the path.)
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
# Modal Teacher-generation resolution (the working path on this account)
# ---------------------------------------------------------------------------


def resolve_modal(
    label: str,
    handle: str,
    *,
    base_model: str = HF_DEFAULT_MODEL,
    base_seed: int = 0,
) -> ResolvedTeacher:
    """Resolve a ``modal:`` handle to a Teacher that GENERATES on a Modal GPU.

    The handle's payload selects the adapter:
      * ``modal:base`` / ``modal:`` -> Qwen3-4B base weights, NO adapter.
      * ``modal:<tag>``             -> base + the LoRA adapter at Volume subdir
                                       ``<tag>`` (e.g. ``modal:update2``).

    Resolution is cheap (records the adapter tag + base model); the heavy GPU work
    happens lazily inside ``ModalTeacher.generate`` via the deployed/ephemeral
    ``crucible-teacher-gen`` app. The base/trained fairness invariant holds because
    BOTH use the SAME ``base_model`` + prompt + sampling — only the adapter tag
    differs, which is exactly what we want to attribute the delta to.

    The Teacher-generation seed is per-build: a caller passes ``gen_seed`` to
    ``build()`` (the Stage-6 replicate index) so each replicate samples DIFFERENT
    curricula from the same Teacher. ``base_seed`` shifts the whole sequence; the
    EFFECTIVE seed handed to the GPU is ``base_seed + gen_seed`` and is applied
    identically to base and trained for a given replicate (fairness preserved).
    """
    tag = _strip_prefix(handle, "modal:").strip()
    if tag in ("", "base", "none"):
        tag = ""  # base weights, no adapter
    resolved_id = f"modal:{base_model}" if not tag else f"modal:{base_model}+{tag}"

    def _build(*, gen_seed: Optional[int] = None) -> object:
        seed = base_seed + (gen_seed if gen_seed is not None else 0)
        return ModalTeacher(adapter_tag=tag, base_model=base_model, seed=seed)

    return ResolvedTeacher(
        label=label,
        handle=handle,
        resolved_id=resolved_id,
        kind="modal",
        _build=_build,
    )


class ModalTeacher:
    """A Teacher that generates arena JSONs on a Modal GPU (base or base+LoRA).

    Conforms to the Teacher contract (``generate(game) -> dict``) using the SAME
    prompt construction + strict-validate/clamp machinery as ``FireworksTeacher`` so
    base and trained prompts stay byte-identical regardless of backend. The expensive
    GPU generation is BATCHED: the first ``generate`` call samples a buffer of
    completions in ONE remote round-trip and serves them one per call, refilling in
    batches only if the evaluator asks for more than the buffer. ``transformers`` /
    ``peft`` never load on the host — they live in the Modal image.
    """

    # How many completions to pull per remote round-trip. The evaluator asks for
    # ``arenas_per_model`` arenas (typically 2-5), so one batch usually suffices.
    _BATCH = 8

    def __init__(self, *, adapter_tag: str = "", base_model: str = HF_DEFAULT_MODEL,
                 temperature: float = 0.8, top_p: float = 0.95, max_new_tokens: int = 512,
                 seed: int = 0, batch: int | None = None):
        self.adapter_tag = adapter_tag or ""
        self.base_model = base_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._batch = batch or self._BATCH
        self._buffer: list[str] = []      # decoded completions awaiting parse
        self._refills = 0                  # how many remote batches we've pulled
        self._cls = None                   # resolved Modal class (lazy)

    def _ensure_cls(self):
        if self._cls is not None:
            return
        import modal

        self._cls = modal.Cls.from_name(MODAL_GEN_APP, MODAL_GEN_CLS)

    @staticmethod
    def _prompt_messages(game) -> tuple[str, str]:
        """Build the EXACT (system, user) the Teacher was TRAINED with for this game.

        Stage 6 is a clean base-vs-trained comparison on the TRAINING task, not a
        prompt-generalization test — so generation MUST use the same prompt the GRPO
        trainer conditioned on. That prompt is the per-game ``prompt`` string in
        ``games_registry`` (the single source of truth the trainer mirrors and
        asserts parity against in ``train_teacher_modal._assert_registry_parity``).

        The trainer renders the chat template from ``[{"role": "user", ...}]`` with
        NO system message (``train_teacher_modal.TeacherGenerator._render_prompt``),
        and the Modal generator skips the system turn when ``system`` is empty
        (``teacher_gen_modal.TeacherGenerator._render_prompt``). So returning
        ``("", training_prompt)`` makes Stage-6 generation byte-identical to training.

        Base and trained both flow through here, so the fairness invariant still
        holds: identical prompt + sampling, only the LoRA adapter differs.
        """
        return "", _training_prompt_for_game(game)

    def _refill(self, game) -> None:
        self._ensure_cls()
        system, user = self._prompt_messages(game)
        inst = self._cls(model_name=self.base_model, adapter_tag=self.adapter_tag)
        # Per-refill seed is the replicate seed shifted into its OWN block (×1000) so a
        # replicate's later refills can never collide with the NEXT replicate's base
        # seed — distinct ``self.seed`` per replicate => disjoint seed sequences =>
        # different curricula across replicates, while a 2nd batch within a replicate
        # still varies (not an exact replay).
        refill_seed = self.seed * 1000 + self._refills
        comps = inst.generate.remote(
            n=self._batch, system=system, user=user,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=self.max_new_tokens, seed=refill_seed,
        )
        self._refills += 1
        self._buffer.extend(comps)

    def generate(self, game) -> dict:
        """Return ONE validated+clamped param set, served from the batched buffer."""
        from training.fireworks_teacher import _strict_validate_params

        # Pull a fresh batch if the buffer is empty.
        if not self._buffer:
            self._refill(game)
        last_error: Exception | None = None
        # Drain completions until one validates (mirrors FireworksTeacher's retry),
        # refilling once if the whole buffer is exhausted by invalid outputs.
        attempts = 0
        while attempts < (self._batch * 2):
            if not self._buffer:
                self._refill(game)
            raw_text = self._buffer.pop(0)
            attempts += 1
            try:
                parsed = _extract_json_object(raw_text)
                return _strict_validate_params(parsed, game)
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc
                continue
        raise ValueError(f"Modal Teacher returned no valid JSON after {attempts} completions: {last_error}")


def _extract_json_object(text: str) -> object:
    """Parse the first balanced JSON object out of a (possibly reasoning) completion.

    ``enable_thinking=False`` makes Qwen3 emit JSON-first, but we still defensively
    strip a ``<think>`` block / ```` ```json ```` fence and extract the first
    brace-balanced object so a trailing prose sentence does not break parsing. A
    completion with no object falls through to ``json.loads`` and is rejected as a
    non-object by ``_strict_validate_params`` (keeps the strict contract intact).
    """
    import json as _json
    import re

    s = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    if start == -1:
        return _json.loads(s)  # no object -> let the strict validator reject it
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _json.loads(s[start : i + 1])
    return _json.loads(s[start:])  # unbalanced -> json.loads raises a clear error


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

    def _build(*, gen_seed: Optional[int] = None) -> object:
        # Offline returns a fixed canned set; the seed is accepted (interface
        # uniformity) but does not vary the deterministic response.
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
    """Resolve ANY handle to a ``ResolvedTeacher`` (Modal / Fireworks id / local).

    Dispatch order: an explicit ``modal:`` prefix -> Modal GPU generation (the
    working path on this account); an explicit ``local:``/``file:`` prefix or an
    existing on-disk path -> a local adapter; anything that looks like a Fireworks id
    (``accounts/`` / ``fw:`` / ``fireworks:``) -> Fireworks. A bare token that is
    neither an existing path nor obviously Fireworks is treated as a Fireworks
    *deployment id* (the common ``--trained qwen3-4b-rft`` case).

    For the Modal/local HF backends, a caller that left ``base_model`` at the
    Fireworks default gets the HF id (``Qwen/Qwen3-4B``) instead — the
    transformers/peft stack cannot load an ``accounts/...`` Fireworks id, and the
    trained adapters target the HF id.
    """
    if handle.strip().startswith("offline:") or handle.strip() == "offline":
        return resolve_offline(label, handle)
    if _looks_like_modal(handle):
        hf_base = HF_DEFAULT_MODEL if base_model == FIREWORKS_DEFAULT_MODEL else base_model
        return resolve_modal(label, handle, base_model=hf_base)
    if _looks_like_local_path(handle):
        hf_base = HF_DEFAULT_MODEL if base_model == FIREWORKS_DEFAULT_MODEL else base_model
        return resolve_local_adapter(label, handle, base_model=hf_base)
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
