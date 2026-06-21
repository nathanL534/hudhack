"""training/teacher_gen_modal.py — Modal Teacher-GENERATION backend for Stage-6 eval.

Stage-6's base-vs-trained decider needs to SAMPLE arenas from two Qwen3-4B Teachers:
a BASE Teacher (no adapter) and a TRAINED Teacher (base + a GRPO LoRA adapter). The
host venv has NO ``transformers`` / ``peft`` and the account has NO Fireworks
serverless inference for ``qwen3-4b`` (404 NOT_FOUND), so neither the ``local:`` nor
the ``fireworks`` handle can actually generate. This file closes that gap: it runs
Qwen3-4B generation on a Modal GPU (where transformers+peft install cleanly in the
image), for BOTH base-no-adapter AND base+LoRA-from-Volume.

It is a SEPARATE Modal app (``crucible-teacher-gen``) — it does NOT touch the
deployed ``crucible-player`` PPO worker, the ``crucible-teacher-trainer``, the EP
bridge, or the OWA daemon. The Stage-6 evaluator dispatches its slow PPO fan-out to
``crucible-player`` exactly as before; this app only produces the arena JSONs.

Design (mirrors ``training/train_teacher_modal.py`` so generation is byte-identical):
  * Same image (torch + transformers==4.51.3 + peft + accelerate), same HF cache and
    adapter Volumes, same chat-template with ``enable_thinking=False`` so completions
    are JSON-first (not a ``<think>`` wall).
  * Same sampling defaults (temperature/top_p) and the same adapter Volume layout
    (``crucible-teacher-lora`` with ``update1``/``update2`` dirs).
  * The GPU class loads the base ONCE and lazily attaches a LoRA adapter on first use
    when ``adapter_tag`` is set; a base-no-adapter instance never imports peft.

The host-side ``ModalTeacher`` adapter (in ``output/stage6/handles.py``) conforms to
the Teacher contract (``generate(game) -> dict``) by pre-generating a batch of
completions in ONE remote call and serving them one at a time, so the evaluator's
existing per-arena ``generate`` loop triggers exactly one GPU round-trip per model.
"""

# NOTE: deliberately NO ``from __future__ import annotations`` — Modal's class
# parameter serde (``modal.parameter``) introspects annotations BY TYPE; PEP-563
# stringized annotations break it ("No class parameter encoder implemented for str").

import os
import time

import modal

# ---------------------------------------------------------------------------
# Constants — kept identical to training/train_teacher_modal.py so the base and
# trained Teachers generate from the SAME weights/prompt/sampling, only the
# adapter differing (the Stage-6 fairness invariant).
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_GPU = "A100-40GB"

# The trained LoRA adapters live here (written by train_teacher_modal.py):
#   crucible-teacher-lora/{update1,update2,...}
VOLUME_NAME = "crucible-teacher-lora"
ADAPTER_DIR = "/adapters"

app = modal.App("crucible-teacher-gen")

adapter_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
# Reuse the trainer's HF cache Volume so the ~8GB Qwen3-4B weights are already warm.
hf_cache_volume = modal.Volume.from_name("crucible-hf-cache", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",   # Qwen3 + enable_thinking chat-template kwarg
        "peft==0.14.0",
        "accelerate==1.4.0",
        "safetensors",
        "sentencepiece",
        "huggingface_hub",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
)


@app.cls(
    image=gpu_image,
    gpu=DEFAULT_GPU,
    volumes={ADAPTER_DIR: adapter_volume, "/root/.cache/huggingface": hf_cache_volume},
    timeout=60 * 30,
    min_containers=0,
    max_containers=1,
    # A single warm container holds the base model (and, for a trained instance, the
    # LoRA-wrapped model) across calls. The eval makes one generate() call per model,
    # so a short scaledown keeps GPU cost minimal between base and trained.
    scaledown_window=60 * 5,
)
class TeacherGenerator:
    """Warm GPU container: load Qwen3-4B once, optionally attach a LoRA adapter, sample.

    ``adapter_tag`` selects the backend:
      * ``""``            -> BASE weights, no adapter (peft never imported).
      * ``"update2"`` etc -> base + the LoRA adapter at ``{ADAPTER_DIR}/{tag}``.
    """

    model_name: str = modal.parameter(default=DEFAULT_MODEL)
    adapter_tag: str = modal.parameter(default="")  # "" => base, else Volume subdir

    @modal.enter()
    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        t0 = time.time()
        tag = (self.adapter_tag or "").strip()
        print(f"[gen] loading {self.model_name} (bf16) adapter={tag or '<base>'} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        if tag:
            from peft import PeftModel

            adapter_path = os.path.join(ADAPTER_DIR, tag)
            if not os.path.isdir(adapter_path):
                raise RuntimeError(
                    f"adapter tag {tag!r} not found at {adapter_path} on Volume "
                    f"{VOLUME_NAME!r}; available: {sorted(os.listdir(ADAPTER_DIR))}"
                )
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"[gen] attached LoRA adapter from {adapter_path}", flush=True)
        model.eval()
        self.model = model
        print(f"[gen] ready in {time.time()-t0:.1f}s", flush=True)

    def _render_prompt(self, system: str, user: str) -> str:
        """Chat-template render with thinking suppressed (JSON-first completions)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Tokenizer without enable_thinking — still works; host parser strips
            # any <think> block anyway.
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    @modal.method()
    def generate(
        self,
        n: int,
        system: str,
        user: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        seed: int = 0,
    ) -> list[str]:
        """Sample ``n`` completions for ONE (system,user) prompt; return decoded text.

        ``num_return_sequences=n`` so all arenas for a model are generated in ONE
        forward pass (one remote round-trip). The same prompt + sampling are used for
        base and trained — only the loaded adapter differs.
        """
        torch = self.torch
        if seed is not None:
            torch.manual_seed(int(seed))
        prompt_text = self._render_prompt(system, user)
        enc = self.tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
        prompt_ids = enc.input_ids
        prompt_len = prompt_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                prompt_ids,
                attention_mask=enc.attention_mask,  # pad==eos: pass mask explicitly
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(max_new_tokens),
                num_return_sequences=int(n),
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        completions = []
        for row in out.sequences:
            comp_ids = row[prompt_len:]
            completions.append(self.tokenizer.decode(comp_ids, skip_special_tokens=True))
        return completions


# ---------------------------------------------------------------------------
# Standalone smoke (no Stage-6 wiring): generate a few arenas base + trained.
# ---------------------------------------------------------------------------

# The Teacher prompt — char-for-char the RING_OUT_PROMPT used by the trainer, so the
# smoke exercises the exact generation surface Stage-6 will use for the fighter game.
_SMOKE_SYSTEM = (
    "You design RL training environments. Output strict JSON only; never output "
    "code or alter the true objective."
)
_SMOKE_USER = (
    "You design RL training curricula for a Stage-1 2D platform fighter (the Ring-Out "
    "duel: two fighters, knock the opponent off the platform). Return ONE JSON object "
    "with EXACTLY these keys, each a number in range: difficulty [0.0,1.0], "
    "platform_width [8.0,30.0], gravity [0.2,1.2], knockback [0.5,6.0], "
    "spawn_gap [1.0,12.0]. Choose a LEARNABLE arena whose trained Player transfers "
    "broadly. Output strict JSON only; no prose, no code."
)


@app.local_entrypoint()
def smoke(n: int = 3, adapter_tag: str = "update2"):
    """`modal run training/teacher_gen_modal.py::smoke` — base + trained gen check."""
    import json

    for tag in ("", adapter_tag):
        gen = TeacherGenerator(model_name=DEFAULT_MODEL, adapter_tag=tag)
        comps = gen.generate.remote(
            n=n, system=_SMOKE_SYSTEM, user=_SMOKE_USER, temperature=0.8, seed=0
        )
        label = "BASE" if not tag else f"TRAINED({tag})"
        print(f"\n=== {label} — {n} completions ===")
        for i, c in enumerate(comps):
            print(f"[{label} {i}] {c[:200]!r}")
            try:
                # crude object-extraction parity with the host parser
                start = c.find("{")
                end = c.rfind("}")
                obj = json.loads(c[start : end + 1]) if start != -1 and end != -1 else None
                print(f"        parsed: {obj}")
            except Exception as exc:
                print(f"        PARSE FAILED: {exc}")
