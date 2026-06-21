"""training/train_teacher_modal.py — CONTINGENCY: self-contained Modal Teacher-RFT trainer.

Fireworks RFT is entitlement-gated for >=4B models on this account (only qwen3-0p6b
trains there). This file is the fallback: it does the ACTUAL LoRA weight updates on
Qwen ourselves, on a Modal GPU, using the EXACT same nested-RL Teacher reward the
Fireworks path uses — so we can train a real 4B Teacher even though Fireworks won't.

MULTI-GAME (this version): each GRPO step trains on BOTH Crucible games at once —
Ring-Out (the platform-fighter) AND Target Knockback — combined into ONE unbiased
LoRA update. NOT alternating game-by-step; both games are in every step's gradient.

What it reuses VERBATIM (does NOT rebuild):
  * Ring-Out reward: ``output.nested_reward.teacher_reward`` (3-PPO-seed held-out
    transfer reward) fanned on the deployed ``crucible-player`` Modal app. Same broad
    15-arena held-out grid as ``modal_ep_bridge``'s ``nested_reward_scorer``.
  * Target-Knockback reward: ``output.nested_reward_tk.tk_teacher_reward`` (3-PPO-seed
    held-out improvement, ``status=="ppo_tk"``) fanned on the SEPARATE deployed
    ``crucible-player-tk`` Modal app. The two reward paths never touch (each names its
    own app/function), so a live Ring-Out RFT and a TK RFT can coexist.
  * The parse/validate path: ``training.hud_teacher_env.parse_teacher_params`` (strips
    ``<think>``, strict JSON, clamps to a per-game ``bounds`` map). Each game validates
    its completions against ITS OWN schema (Ring-Out's 5 keys vs TK's 7 keys, both from
    ``games_registry``). Invalid JSON -> reward 0 (teaches format too).
  * The Teacher prompts + schemas: the two ``games_registry`` GameSpecs
    (``RING_OUT`` / ``TARGET_KNOCKBACK``) — single source of truth for prompt + bounds
    (asserted equal at host import, below).
  * The base-vs-trained yardstick: each game's own held-out transfer reward, measured
    on the SAME policy before and after the updates.

THE UNBIASED MULTI-GAME ALGORITHM (this is the point — see step 4):

  Each GRPO step:
  1. Sample G completions from the Ring-Out prompt AND G from the TK prompt
     (separate prompts, same shared Qwen+LoRA policy).
  2. Parse+validate each completion against ITS game's schema (invalid -> reward 0).
  3. Score Ring-Out samples via ``teacher_reward`` (crucible-player) and TK samples via
     ``tk_teacher_reward`` (crucible-player-tk) — all 2G rewards fanned out in PARALLEL.
  4. **Advantages are computed PER GAME** — group-relative WITHIN the Ring-Out group
     and WITHIN the TK group SEPARATELY: ``A_i = (r_i - mean_game) / (std_game + eps)``.
     This is the critical anti-bias step: the two games have different reward SCALES
     (TK held-out improvement vs Ring-Out held-out improvement), so normalizing within
     each game — NOT pooling rewards across games — keeps the update from being biased
     toward whichever game happens to have the larger raw reward magnitude. Each game's
     advantages are mean~0 / std~1 by construction, so both games contribute equally.
  5. Concatenate the per-game-normalized advantages and the per-game cached completion
     sequences, then take ONE GRPO gradient step over ALL 2G completions
     (loss = ``-(A_i * sum_logprob(completion_i)).mean()`` over the concatenated set);
     backprop into the LoRA only. ONE LoRA update per step, informed by both games.
  6. Save the adapter (Modal Volume) after each step.

ARCHITECTURE (why split host <-> GPU):
  * The GPU work (load Qwen, sample per-game, recompute logprobs, LoRA gradient step)
    runs in a warm Modal ``@app.cls`` GPU container holding the model + optimizer +
    BOTH games' last-sampled sequences across calls, persisting the adapter to a Volume.
  * The slow REMOTE rewards (``teacher_reward`` -> crucible-player and
    ``tk_teacher_reward`` -> crucible-player-tk) are dispatched FROM THE HOST — the
    proven path. The host computes the PER-GAME advantages (the anti-bias logic lives
    where it is auditable) and hands them to the GPU's ``update_multi``; the GPU step is
    purely mechanical (concat logprobs, one backward). No nested Modal-from-Modal map.

RIGOR (this version):
  * SIGNED reward: TK's per-game reward is the RAW SIGNED held-out improvement (can be
    negative); we do NOT re-clamp it to [0,1] before GRPO. Per-game advantage norm
    ``(r-mean)/(std+eps)`` is shift-invariant, so a signed reward is strictly more
    informative — a "this arena hurt transfer" sample still pushes the Teacher away
    instead of collapsing to the 0.0 floor.
  * ROTATED training seeds + RESERVED validation seeds: the PPO reward seeds VARY per
    update (distinct block each step, anti-overfit) and a SEPARATE fixed validation pool
    (disjoint from every training block) is used ONLY for the final-adapter scoring pass.
  * FINAL-ADAPTER scoring pass: after the last update, the SAVED final adapter is
    sampled + scored on the validation seeds and recorded in summary.json as the TRUE
    final-adapter reward (the per-update logs are PRE-update by construction).

RUN (from repo root; .env sourced; ~/.modal.toml profile njlee007):

    set -a && source .env && set +a
    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    # rank-32 rerun (Nathan's default now): both games, signed reward, rotated seeds
    .venv/bin/python3 training/train_teacher_modal.py --updates 15 --group-size 5 --temperature 1.1
    # (defaults are already --lora-r 32 --lora-alpha 64; pass them explicitly to be sure)
    .venv/bin/python3 training/train_teacher_modal.py --updates 15 --group-size 5 --lora-r 32 --lora-alpha 64
    # scale to a fuller run: just raise --updates (e.g. 4-5)
    .venv/bin/python3 training/train_teacher_modal.py --updates 5 --group-size 5 --temperature 1.1
    # single-game (legacy): just Ring-Out
    .venv/bin/python3 training/train_teacher_modal.py --games ring_out --updates 2 --group-size 8
    # disable seed rotation (legacy: reuse update-1's block every step)
    .venv/bin/python3 training/train_teacher_modal.py --updates 5 --no-seed-rotation
    .venv/bin/python3 training/train_teacher_modal.py --gpu-smoke  # GPU-only: load+sample+fake-update, no reward
    .venv/bin/python3 training/train_teacher_modal.py --model Qwen/Qwen3-1.7B  # smaller/faster

This file is contingency infra: it does NOT touch the Fireworks launcher, the deployed
``crucible-ep-bridge`` / ``crucible-ep-bridge-tk``, or the OWA daemon. It only IMPORTS
the existing rewards and CALLS the already-deployed ``crucible-player`` /
``crucible-player-tk`` workers (normal shared use).
"""

# NOTE: deliberately NO ``from __future__ import annotations`` — Modal's class
# parameter serde (``modal.parameter``) introspects the class annotations BY TYPE
# (``str``/``float``/``int``), which PEP-563 stringized annotations break with
# "No class parameter encoder implemented for type `str`".

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env (FIREWORKS_API_KEY, HUD_API_KEY) and drop the EMPTY Modal token
# placeholders so Modal uses the active ~/.modal.toml profile (mirrors dry_loop.py
# / eval_base_vs_trained.py). Must happen before any `import modal`.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except Exception:  # pragma: no cover
    pass
for _k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    if os.environ.get(_k, "").strip() == "":
        os.environ.pop(_k, None)

import modal  # noqa: E402

# ---------------------------------------------------------------------------
# Per-game prompts + schemas (single source of truth = games_registry)
# ---------------------------------------------------------------------------
#
# Both games' prompt + bounds come from games_registry (the shared schema/router the
# HUD/EP/reward paths all read). They are mirrored into plain module constants here so
# the GPU container can be handed just the strings + bounds at call time — its image
# never imports games_registry / hud / eval-protocol. We ASSERT the mirror equals the
# registry at host import (see _assert_registry_parity) so the two can never drift.

# Ring-Out (game #1): the Stage-1 2D platform fighter. 5-key schema.
RING_OUT_PROMPT = (
    "You design RL training curricula for a Stage-1 2D platform fighter "
    "(the Ring-Out duel: two fighters, knock the opponent off the platform). "
    "Return ONE JSON object with EXACTLY these keys, each a number in range: "
    "difficulty [0.0,1.0], platform_width [8.0,30.0], gravity [0.2,1.2], "
    "knockback [0.5,6.0], spawn_gap [1.0,12.0]. Choose a LEARNABLE arena whose "
    "trained Player transfers broadly. Output strict JSON only; no prose, no code."
)
RING_OUT_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "gravity": (0.2, 1.2),
    "knockback": (0.5, 6.0),
    "spawn_gap": (1.0, 12.0),
}

# Target Knockback (game #2): same physics, score for keeping the opponent in a marked
# target zone. 7-key schema (the 5 fighter knobs + the two TK zone dials).
TK_PROMPT = (
    "You design RL training curricula for a 2D platform game called Target "
    "Knockback: two fighters on a platform with a marked TARGET ZONE; you score "
    "for every tick your OPPONENT is knocked into the zone (punch them in, they "
    "jump to escape). Return ONE JSON object with EXACTLY these keys, each a "
    "number in range: difficulty [0.0,1.0], platform_width [8.0,30.0], "
    "gravity [0.2,1.2], knockback [0.5,6.0], spawn_gap [1.0,12.0], "
    "zone_half [0.6,4.0], zone_center_frac [0.3,0.7]. Choose a LEARNABLE arena "
    "whose trained Player transfers broadly. Output strict JSON only; no prose, "
    "no code."
)
TK_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "gravity": (0.2, 1.2),
    "knockback": (0.5, 6.0),
    "spawn_gap": (1.0, 12.0),
    "zone_half": (0.6, 4.0),
    "zone_center_frac": (0.3, 0.7),
}

# The two game keys, in a STABLE order. Every cross-host/GPU list (prompts, sequence
# groups, advantage groups) is keyed by this order so the GPU concatenates the same
# completions the host scored — no game/sample misalignment.
GAME_KEYS = ("ring_out", "target_knockback")

# Prompts handed to the GPU container at load (it tokenizes both, one prompt_len each).
GAME_PROMPTS: dict[str, str] = {
    "ring_out": RING_OUT_PROMPT,
    "target_knockback": TK_PROMPT,
}
GAME_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "ring_out": RING_OUT_BOUNDS,
    "target_knockback": TK_BOUNDS,
}


def _assert_registry_parity() -> None:
    """Fail fast if the mirrored prompt/bounds drift from the games_registry source.

    The registry (games_registry.RING_OUT / TARGET_KNOCKBACK) is the single source of
    truth the HUD/EP/reward paths read. This trainer re-declares the strings so the GPU
    image stays registry-free, so we verify equality at host import — a registry edit
    that is not mirrored here is a loud error, never a silent mis-train.
    """
    try:
        from games_registry import RING_OUT, TARGET_KNOCKBACK
    except Exception as exc:  # pragma: no cover - registry must import on host
        raise RuntimeError(f"games_registry import failed: {exc}") from exc

    assert RING_OUT.prompt == RING_OUT_PROMPT, "ring_out prompt drifted from registry"
    assert TARGET_KNOCKBACK.prompt == TK_PROMPT, "tk prompt drifted from registry"
    assert dict(RING_OUT.bounds) == RING_OUT_BOUNDS, "ring_out bounds drifted from registry"
    assert dict(TARGET_KNOCKBACK.bounds) == TK_BOUNDS, "tk bounds drifted from registry"


# Default Teacher base = the model size Fireworks is gating. Fall back to a smaller
# model with --model if the GPU is too tight/slow (the point is to prove the LOOP).
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_GPU = "A100-40GB"

# ---------------------------------------------------------------------------
# PER-RUN ISOLATION (env-driven, read at IMPORT time) — the launch blocker.
# ---------------------------------------------------------------------------
#
# Four variant runs (varA..varD) launch CONCURRENTLY against the SAME deployed
# crucible-player / crucible-player-tk reward workers. Without isolation they would
# (1) all create the same Modal App ``crucible-teacher-trainer`` -> share ONE warm GPU
# container -> corrupt each other's in-memory model/optimizer/cached sequences;
# (2) all write ``update1``/``update2`` into the SAME volume path -> overwrite each
# other's checkpoints; and (3) all write the SAME output dir -> clobber summary.json.
#
# A SINGLE env var ``CRUCIBLE_RUN_ID`` drives all three. It is read HERE, at import,
# because the Modal App name must be fixed BEFORE ``modal.App(...)`` is constructed
# (which is before CLI args are parsed). When set (e.g. ``CRUCIBLE_RUN_ID=varA``):
#   * App name      -> ``crucible-teacher-trainer-varA``  (own GPU container)
#   * Adapter path  -> ``/adapters/varA/update1`` ...      (namespaced on the SAME volume)
#   * Output dir    -> ``output/contingency_varA/``        (own summary.json/history.json)
# When UNSET, every name falls back to the original default (back-compat).
RUN_ID = (os.environ.get("CRUCIBLE_RUN_ID") or "").strip()

# Modal App name: ``-<run_id>`` suffix when isolated, so each run gets its OWN warm
# GPU container (critical — the container holds the model + optimizer + per-game
# cached sequences in memory across calls; a shared container would corrupt them).
APP_NAME = f"crucible-teacher-trainer-{RUN_ID}" if RUN_ID else "crucible-teacher-trainer"

# GPU (also import-time: it pins the ``@app.cls(gpu=...)`` decorator, which runs before
# CLI parse). ``CRUCIBLE_GPU`` lets a launcher set the trainer's GPU without a code edit;
# the ``--gpu`` flag defaults to this same value (and main() asserts they agree, since the
# decorator can't be re-pinned post-import). Default A100-40GB (fits LoRA r=32 on Qwen3-4B).
GPU_SPEC = (os.environ.get("CRUCIBLE_GPU") or "").strip() or DEFAULT_GPU

# Where the GPU container persists the LoRA adapter across calls + runs. The VOLUME is
# shared (one ``crucible-teacher-lora`` volume), but each run namespaces its checkpoints
# under a ``<run_id>/`` SUBDIR so concurrent runs never overwrite one another's adapters.
VOLUME_NAME = "crucible-teacher-lora"
ADAPTER_DIR = "/adapters"
# Subdir under ADAPTER_DIR that THIS run writes/reads checkpoints in. Isolated runs get
# ``/adapters/<run_id>``; the unset (legacy) case keeps writing flat ``/adapters``.
ADAPTER_RUN_DIR = os.path.join(ADAPTER_DIR, RUN_ID) if RUN_ID else ADAPTER_DIR

app = modal.App(APP_NAME)

# The adapter Volume survives between the warm container's calls AND across runs, so
# a checkpoint is recoverable even if the container is recycled.
adapter_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# A persistent HF cache Volume so the ~8GB Qwen3-4B weights download ONCE, not every
# cold start (keeps GPU cost minimal across runs).
hf_cache_volume = modal.Volume.from_name("crucible-hf-cache", create_if_missing=True)

# GPU image: transformers/peft/accelerate + torch (the doc-verified API). No TRL —
# the loop is hand-rolled (cleaner with a slow REMOTE reward: the GPU side never
# blocks on the reward; the host orchestrates the sample -> score -> update cycle).
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",   # Qwen3 supported; enable_thinking chat-template kwarg
        "peft==0.14.0",
        "accelerate==1.4.0",
        "safetensors",
        "sentencepiece",
        "huggingface_hub",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
    # Propagate the run-id INTO the container. ADAPTER_RUN_DIR is computed at import
    # from CRUCIBLE_RUN_ID, but `save_adapter` is a @modal.method that runs CONTAINER-
    # side, where the host's shell env does NOT reach. Without this line the container
    # imports with CRUCIBLE_RUN_ID unset -> RUN_ID="" -> every run writes flat
    # /adapters/<tag>, so concurrent variants CLOBBER each other's checkpoints. Each
    # variant is its own Modal App -> own container -> own image-env -> own
    # /adapters/<run_id> subdir on the shared Volume. (Caught by the collision test.)
    .env({"CRUCIBLE_RUN_ID": RUN_ID})
    # Reduce CUDA allocator fragmentation: the G=12 logprob forward materialises a large
    # transient (B,L,V) logits tensor per update, and a fragmented heap OOM'd a 40GB A100
    # by only ~116MB. expandable_segments lets the allocator grow segments instead of
    # failing on fragmentation (the fix the OOM error itself recommends).
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)


@app.cls(
    image=gpu_image,
    gpu=GPU_SPEC,
    volumes={ADAPTER_DIR: adapter_volume, "/root/.cache/huggingface": hf_cache_volume},
    timeout=60 * 60,
    # One warm container holds the model + optimizer + LoRA state + BOTH games'
    # last-sampled sequences across every sample/update call of a run. Single
    # container = single coherent policy. IMPORTANT: ``update_multi`` recomputes
    # logprobs over the sequences ``sample`` cached on the instance, and the host's
    # REMOTE reward scoring (BOTH games, fanned in parallel) sits BETWEEN those calls
    # (~3-6 min). The scaledown window must comfortably exceed that gap or the
    # container recycles and the cached sequences are lost.
    min_containers=0,
    max_containers=1,
    scaledown_window=60 * 30,
)
class TeacherTrainer:
    """Warm GPU container: load Qwen+LoRA once, then sample / GRPO-update on demand.

    State persists across method calls in ONE container: the LoRA-wrapped model, the
    tokenizer, the Adam optimizer over the LoRA params, and the LAST sampled token
    sequences PER GAME (``self._last_sequences[game]``). ``sample`` returns G decoded
    completions for ONE game; the host scores both games' completions, computes the
    per-game advantages, and calls ``update_multi`` with the per-game advantages to
    take ONE GRPO step over both games' cached sequences. ``save_adapter`` writes the
    adapter to the Volume.
    """

    # modal.parameter only supports int/str/bytes/bool in modal 1.5 — lr is carried
    # as a string and float()'d in _load (it is needed when the optimizer is built).
    model_name: str = modal.parameter(default=DEFAULT_MODEL)
    lr_str: str = modal.parameter(default="1e-4")
    lora_r: int = modal.parameter(default=16)
    lora_alpha: int = modal.parameter(default=32)

    @modal.enter()
    def _load(self) -> None:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        t0 = time.time()
        print(f"[gpu] loading {self.model_name} (bf16) ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        # Standard Qwen attention + MLP projections for LoRA.
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.0,  # deterministic logprob recompute (no dropout noise)
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.train()
        # get_peft_model freezes the base; only LoRA params have requires_grad. Adam
        # over exactly those params => only the adapter moves. (No kbit prep / no
        # enable_input_require_grads needed for full-bf16 LoRA.)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        # Remember the BASE lr so update_multi can scale it per step for an optional
        # linear-decay schedule (host passes a per-update lr_scale in [0,1]; default
        # 1.0 => constant lr, the legacy behavior). The schedule is applied on the
        # optimizer param-groups right before step(), so it never touches Adam's
        # moment state — only the effective step size shrinks toward ~0.
        self._base_lr = float(self.lr_str)
        self.optimizer = torch.optim.Adam(trainable, lr=self._base_lr)

        # Build BOTH games' prompts once: chat template, thinking suppressed (Qwen3
        # supports enable_thinking=False so completions are JSON-first, not a <think>
        # wall). Each game has its own prompt_len; sample()/update_multi() index by it.
        self._prompt_ids: dict = {}
        self._prompt_len: dict = {}
        self._last_sequences: dict = {}
        for game, prompt in GAME_PROMPTS.items():
            text = self._render_prompt(prompt)
            ids = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda:0")
            self._prompt_ids[game] = ids
            self._prompt_len[game] = ids.shape[1]
        print(
            f"[gpu] ready in {time.time()-t0:.1f}s | trainable LoRA params "
            f"{n_train:,} / {n_total:,} ({100*n_train/n_total:.3f}%) | "
            f"prompt_len ring_out={self._prompt_len['ring_out']} "
            f"tk={self._prompt_len['target_knockback']}",
            flush=True,
        )

    def _render_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Tokenizer without enable_thinking — still works; the host parser strips
            # any <think> block anyway.
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _completion_logprobs(self, sequences, prompt_len: int):
        """Sum of per-token logprobs of the COMPLETION tokens under the current model.

        Keeps the autograd graph (fresh forward, NOT generate's scores). Shift: a
        forward over ``sequences`` gives logits[i] = distribution for token i+1, so
        we align logits[:, :-1] with targets sequences[:, 1:]. After the left-shift,
        target position j corresponds to original position j+1, so the completion
        region (original index >= prompt_len) starts at shifted index prompt_len-1.
        ``prompt_len`` is passed per game (the two games' prompts differ in length).
        """
        torch = self.torch
        logits = self.model(sequences).logits[:, :-1, :]               # (B, L-1, V)
        targets = sequences[:, 1:]                                     # (B, L-1)
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        # Completion mask in the shifted frame: keep targets that are completion
        # tokens (original index >= prompt_len) AND not padding.
        idx = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
        comp_mask = idx >= (prompt_len - 1)
        pad_mask = targets != self.tokenizer.pad_token_id
        mask = (comp_mask & pad_mask).to(tok_logp.dtype)
        return (tok_logp * mask).sum(dim=1)                            # (B,)

    @modal.method()
    def sample(self, game: str, group_size: int, temperature: float = 0.9,
               top_p: float = 0.95, max_new_tokens: int = 512,
               seed: int | None = None) -> list[str]:
        """Sample ``group_size`` completions for ONE game from the CURRENT policy.

        ``game`` selects which prompt to condition on (``ring_out`` / ``target_knockback``).
        The host parses/validates/scores them and hands the per-game advantages back to
        ``update_multi``. We cache the sampled token sequences on the instance KEYED BY
        GAME so ``update_multi`` recomputes logprobs over the EXACT same sequences for
        each game (no resample drift between the action that earned the reward and the
        gradient).
        """
        assert game in GAME_PROMPTS, f"unknown game {game!r}"
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        prompt_ids = self._prompt_ids[game]
        prompt_len = self._prompt_len[game]
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                prompt_ids,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=group_size,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        self.model.train()
        self._last_sequences[game] = out.sequences  # (G, prompt_len + new)
        completions = []
        for row in out.sequences:
            comp_ids = row[prompt_len:]
            completions.append(self.tokenizer.decode(comp_ids, skip_special_tokens=True))
        return completions

    @modal.method()
    def update_multi(self, advantages_by_game: dict, lr_scale: float = 1.0) -> dict:
        """ONE GRPO step over BOTH games' last sampled groups, given PER-GAME advantages.

        The host has already computed each game's group-relative advantages
        ``A_i = (r_i - mean_game)/(std_game + eps)`` WITHIN that game (the anti-bias
        normalization — NO cross-game pooling). This method just recomputes the
        completion logprobs for each game's cached sequences, CONCATENATES the
        per-game (advantage, logprob) pairs across both games, and takes a SINGLE
        gradient step:

            loss = -(A_concat * logprob_concat).mean()

        Both games' completions sit in the same backward, so one LoRA update is informed
        by both games equally (each game's advantages are already ~zero-mean/unit-std).

        ``advantages_by_game`` maps game -> list[float]; only games with a cached
        sample group are used (so a single-game run passes one key and this reduces to
        the original single-game GRPO step exactly).

        ``lr_scale`` (default 1.0) multiplies the base lr for THIS step only — the host
        passes a value in [0,1] to drive an optional LINEAR-DECAY schedule (1.0 on the
        first update, decaying toward ~0 on the last). At the default 1.0 the effective
        lr is exactly ``self._base_lr`` every step (constant lr = legacy behavior).
        """
        torch = self.torch
        # Apply the (optional) per-step lr schedule to every param group BEFORE step().
        # Default lr_scale=1.0 => effective_lr == base_lr (constant; byte-identical to
        # the legacy path). A linear-decay run passes a shrinking scale each update.
        effective_lr = self._base_lr * float(lr_scale)
        for pg in self.optimizer.param_groups:
            pg["lr"] = effective_lr
        per_game_logp_mean: dict = {}
        used_games = []
        # GRADIENT ACCUMULATION over games (memory fix for G>=12 on a 40GB GPU). We
        # compute logprobs + backward ONE GAME AT A TIME so peak VRAM holds only one
        # game's (G, L, V) logits, not both concatenated (which OOM'd a 40GB A100 at
        # G=12). Each game's loss is scaled by 1/N_total (the size of the concatenated
        # batch the old path averaged over), so the SUM of the per-game gradients is
        # mathematically IDENTICAL to a single .mean() backward over the concatenation.
        n_total = sum(
            len(advantages_by_game[g]) for g in GAME_KEYS if g in advantages_by_game
        )
        assert n_total > 0, "update_multi called with no game groups"

        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        n_completions = 0
        for game in GAME_KEYS:
            if game not in advantages_by_game:
                continue
            assert game in self._last_sequences, f"sample({game!r}) not called before update_multi"
            seqs = self._last_sequences[game]
            game_adv = advantages_by_game[game]
            assert len(game_adv) == seqs.shape[0], (
                f"{game}: {len(game_adv)} advantages vs {seqs.shape[0]} samples"
            )
            lp = self._completion_logprobs(seqs, self._prompt_len[game])  # (G,) with grad
            adv = torch.tensor(game_adv, dtype=torch.float32, device="cuda:0")
            # .sum()/n_total (NOT .mean()) so the per-game grads SUM to the old single
            # .mean() over the concatenated batch. backward() frees this game's logits
            # graph before the next game's forward -> peak memory is one game, not both.
            game_loss = -(adv * lp).sum() / n_total
            game_loss.backward()
            total_loss += float(game_loss.detach().cpu())
            per_game_logp_mean[game] = float(lp.detach().mean().cpu())
            n_completions += int(lp.shape[0])
            used_games.append(game)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        return {
            "loss": total_loss,
            "grad_norm": float(grad_norm),
            "games": used_games,
            "n_completions": n_completions,
            "lr_scale": round(float(lr_scale), 6),
            "effective_lr": effective_lr,
            "mean_completion_logprob_by_game": {
                g: round(v, 4) for g, v in per_game_logp_mean.items()
            },
        }

    @modal.method()
    def save_adapter(self, tag: str) -> str:
        """Persist the current LoRA adapter to the Volume; return its path.

        Writes under ``ADAPTER_RUN_DIR`` (``/adapters/<run_id>`` when ``CRUCIBLE_RUN_ID``
        is set, else flat ``/adapters``), so four concurrent runs namespace their
        ``update1``/``update2``/... checkpoints and never overwrite each other. The path
        is os.makedirs'd because save_pretrained needs the parent run subdir to exist.
        """
        path = os.path.join(ADAPTER_RUN_DIR, tag)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        adapter_volume.commit()
        print(f"[gpu] saved adapter -> {path}", flush=True)
        return path

    @modal.method()
    def gpu_smoke(self, group_size: int = 2) -> dict:
        """Self-contained GPU check: load + sample BOTH games + ONE fake-reward GRPO
        step over both + save. NO reward compute.

        Proves the GPU half of the MULTI-GAME path end to end (load Qwen, generate for
        each game, recompute logprobs per game, concat, backward, optimizer.step, save)
        WITHOUT spending any reward compute. Fake advantages are the centered sample
        index, so the step is non-degenerate and real.
        """
        previews = {}
        adv_by_game = {}
        for game in GAME_KEYS:
            comps = self.sample.local(game=game, group_size=group_size,
                                      max_new_tokens=128, seed=0)
            previews[game] = [c[:160] for c in comps]
            # Fake per-game advantages (already "normalized" stand-ins: centered).
            adv_by_game[game] = [float(i) - (group_size - 1) / 2.0 for i in range(group_size)]
        stats = self.update_multi.local(adv_by_game)
        path = self.save_adapter.local("gpu_smoke")
        return {
            "completions_preview": previews,
            "update_stats": stats,
            "adapter_path": path,
        }


# ---------------------------------------------------------------------------
# HOST side: parse/validate + REMOTE reward dispatch (the proven path), per game
# ---------------------------------------------------------------------------


def _score_ring_out(answer: str, *, seeds, episodes, eval_seeds,
                    held_out, student_cfg: dict | None = None) -> tuple[float, dict | None, str]:
    """Score ONE Ring-Out completion via the EXISTING fighter nested reward.

    parse (strip <think> + strict JSON + clamp to RING_OUT_BOUNDS + FIGHTER-schema
    map) -> ``teacher_reward`` (3 PPO seeds on crucible-player). Invalid -> reward 0.

    ``student_cfg`` (default None): an optional inner-Student population config (arch,
    opponent league, entropy coef, ...) threaded down into ``teacher_reward`` so the
    nested Players train with the new population. It is only forwarded when NON-None —
    the default path passes NO ``student_cfg`` kwarg at all, so behavior is byte-
    identical and stays compatible with a ``teacher_reward`` that doesn't yet accept it
    (the reward-path teammate consumes it).
    """
    from output.nested_reward import teacher_reward
    from training.hud_teacher_env import (
        fighter_geometry_override,
        params_to_curriculum,
        parse_teacher_params,
    )

    try:
        params = parse_teacher_params(answer, RING_OUT_BOUNDS)
    except Exception as exc:
        return 0.0, None, f"invalid:{type(exc).__name__}"

    spec = params_to_curriculum(params, curriculum_id="contingency-rft")
    reward_kwargs = dict(
        backend="modal",
        seeds=seeds,
        episodes=episodes,
        eval_seeds=eval_seeds,
        held_out_arenas=held_out,
        geometry_override=fighter_geometry_override(params),
    )
    # Forward student_cfg ONLY when set — keeps the default call identical to the
    # legacy signature (so no break before the reward teammate adds the parameter).
    if student_cfg is not None:
        reward_kwargs["student_cfg"] = student_cfg
    reward = teacher_reward(spec, **reward_kwargs)
    return float(max(0.0, min(1.0, reward))), params, "ok"


def _score_tk(answer: str, *, seeds, episodes, eval_seeds) -> tuple[float, dict | None, str]:
    """Score ONE Target-Knockback completion via the EXISTING TK nested reward.

    parse (strip <think> + strict JSON + clamp to TK_BOUNDS, the 7-key TK schema) ->
    ``tk_teacher_reward`` (3 PPO seeds on crucible-player-tk; reads ``improvement``,
    asserts ``status=="ppo_tk"``). Invalid -> reward 0. TK does NOT use the fighter's
    broad held-out grid: the TK worker builds its own held-out reference, so no
    held_out_arenas is passed (that is the TK reward's load-bearing yardstick).

    RIGOR FIX (signed reward): GRPO normalizes per-game advantages as
    ``(r - mean) / (std + eps)``, which is SHIFT-invariant — so feeding the RAW SIGNED
    held-out improvement (which CAN be negative when an arena hurts transfer) is
    strictly more informative than the [0,1]-clamped reward. A clamp collapses every
    "this arena made the Player WORSE" sample to the same 0.0 floor, erasing the
    gradient that should push the Teacher AWAY from those arenas. We therefore read the
    SIGNED ``mean_improvement`` and do NOT re-clamp it here.

    Source of the signed value: ``tk_teacher_reward`` exposes the raw signed mean via
    its ``_detail_sink["mean_improvement"]`` regardless of whether the return value is
    clamped (a parallel change is removing the return-side clamp in
    ``output/nested_reward_tk.py``). We read the detail sink so this is correct under
    BOTH states — clamped or unclamped return — and never re-clamps negatives away.
    """
    from output.nested_reward_tk import tk_teacher_reward
    from training.hud_teacher_env import parse_teacher_params

    try:
        params = parse_teacher_params(answer, TK_BOUNDS)
    except Exception as exc:
        return 0.0, None, f"invalid:{type(exc).__name__}"

    detail: dict = {}
    reward_ret = tk_teacher_reward(
        params,
        backend="modal",
        seeds=tuple(seeds),
        episodes=episodes,
        eval_seeds=eval_seeds,
        curriculum_id="contingency-rft-tk",
        _detail_sink=detail,
    )
    # Prefer the RAW SIGNED held-out improvement (can be negative). The detail sink's
    # ``mean_improvement`` is the unclamped signed mean today; if the parallel change
    # makes the return value itself signed, both agree. Fall back to the return value
    # only if the sink is somehow absent. NO re-clamp on negatives.
    signed = detail.get("mean_improvement")
    reward = float(signed) if signed is not None else float(reward_ret)
    return reward, params, "ok"


def _score_all_games(
    completions_by_game: dict,
    *,
    ring_seeds, ring_episodes, ring_eval_seeds, ring_held_out,
    tk_seeds, tk_episodes, tk_eval_seeds,
    max_workers: int,
    ring_student_cfg: dict | None = None,
) -> dict:
    """Score ALL games' completions, fanning every (slow ~2min) reward out in parallel.

    Each completion is one task; Ring-Out tasks fan 3 PPO seeds onto crucible-player and
    TK tasks fan 3 PPO seeds onto crucible-player-tk. Running all 2G across both games
    concurrently (threads — the work is a remote Modal map + network wait) means the
    whole step's reward wall-clock ~= one completion, not 2G of them. Results are
    returned keyed by game, in the SAME order as the input completions (so advantages
    align with the GPU's cached sequences).
    """
    from concurrent.futures import ThreadPoolExecutor

    # Flatten to (game, index) tasks so both games share one pool (max parallelism).
    tasks: list = []
    for game in GAME_KEYS:
        for i in range(len(completions_by_game.get(game, []))):
            tasks.append((game, i))

    results: dict = {
        game: [None] * len(completions_by_game.get(game, [])) for game in GAME_KEYS
    }

    def _one(task):
        game, i = task
        answer = completions_by_game[game][i]
        if game == "ring_out":
            scored = _score_ring_out(
                answer, seeds=ring_seeds, episodes=ring_episodes,
                eval_seeds=ring_eval_seeds, held_out=ring_held_out,
                student_cfg=ring_student_cfg,
            )
        else:
            scored = _score_tk(
                answer, seeds=tk_seeds, episodes=tk_episodes, eval_seeds=tk_eval_seeds,
            )
        return game, i, scored

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for game, i, scored in ex.map(_one, tasks):
            results[game][i] = scored
    return results


def _build_held_out() -> list[dict]:
    """The BROAD 15-arena held-out population — the Ring-Out reward's yardstick."""
    from output.broad_eval_set import build_broad_eval_arenas, payload_arenas

    return payload_arenas(build_broad_eval_arenas(grid="full"))


# ---------------------------------------------------------------------------
# Seed rotation + reserved validation seeds (anti-overfit)
# ---------------------------------------------------------------------------
#
# Original behavior scored EVERY update against the SAME 3 PPO seeds (1,2,3), so the
# Teacher could overfit those specific PPO populations rather than learning arenas that
# transfer. The fix has two halves:
#   1. TRAINING seeds rotate per update — each update consumes a DISTINCT contiguous
#      block in a HIGH range derived from ``seed_base`` + a per-game offset + the
#      update index, so no PPO population is ever scored against twice.
#   2. VALIDATION seeds are a SEPARATE fixed low-range pool, never used to compute a
#      training reward — reserved purely for the final-adapter scoring pass.
# The two pools are asserted DISJOINT before the run starts.

# Per-game offset into the high training-seed range, so ring_out and target_knockback
# rotate through NON-overlapping sub-ranges (they index by GAME_KEYS order).
_GAME_SEED_OFFSET = {game: gi * 1_000_000 for gi, game in enumerate(GAME_KEYS)}


def _train_seeds_for(game: str, update_index: int, n_seeds: int, seed_base: int) -> list[int]:
    """The rotated TRAINING PPO seeds for ``game`` at ``update_index`` (1-based).

    Each update gets a fresh, distinct, contiguous block of ``n_seeds`` integers:
        block_start = seed_base + game_offset + (update_index - 1) * n_seeds
    Distinct (game, update) -> distinct block, so the Teacher never re-scores against a
    PPO population it has already seen, and ring/tk blocks never collide (per-game
    offset). All blocks sit in the HIGH range (>= seed_base), disjoint from the
    low-range validation pools.
    """
    start = seed_base + _GAME_SEED_OFFSET[game] + (update_index - 1) * n_seeds
    return [start + j for j in range(n_seeds)]


def _all_training_seeds(game: str, updates: int, n_seeds: int, seed_base: int,
                        rotation: bool) -> set[int]:
    """Every training seed ``game`` will EVER use across the whole run.

    With rotation off, all updates reuse the first block (legacy behavior), so the union
    is just that one block. Used to assert the train/validation pools never intersect.
    """
    if updates <= 0:
        return set()
    span = updates if rotation else 1
    seeds: set[int] = set()
    for u in range(1, span + 1):
        seeds.update(_train_seeds_for(game, u, n_seeds, seed_base))
    return seeds


def _assert_seed_pools_disjoint(game: str, train_seeds: set[int], val_seeds) -> None:
    """Fail fast if a game's rotated training seeds overlap its validation seeds.

    Disjoint train/validation pools are the whole point of fix #2 — a leaked seed would
    let the Teacher train on a population it is later 'validated' on. We verify rather
    than trust the ranges (a low --seed-base or hand-set --val-seeds could collide).
    """
    overlap = train_seeds & set(val_seeds)
    assert not overlap, (
        f"{game}: training seeds overlap validation seeds {sorted(overlap)} — raise "
        f"--seed-base or change --{'tk-' if game == 'target_knockback' else ''}val-seeds "
        f"so the pools stay disjoint"
    )


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _per_game_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Group-relative advantages WITHIN one game: A_i = (r_i - mean)/(std + eps).

    This is THE anti-bias step. It is applied to each game's rewards SEPARATELY (never
    across the pooled 2G set), so each game's advantages are ~zero-mean/unit-std and the
    combined gradient weights both games equally regardless of their reward SCALES.
    Mirrors the GPU's original single-game ``r.std(unbiased=False)`` (population std).
    """
    if not rewards:
        return []
    m = _mean(rewards)
    s = _std(rewards)
    return [(r - m) / (s + eps) for r in rewards]


def _arena_diversity(params_list: list) -> dict:
    """A per-game arena-diversity signal: spread of the emitted knobs (collapse check).

    Collapse toward one arena shows up as ~0 std across the group's difficulties (and
    other knobs). We report difficulty mean/std + unique-rounded count as the headline,
    plus per-knob std, computed over the VALID params only.
    """
    valid = [p for p in params_list if p]
    if not valid:
        return {"n_valid": 0}
    keys = sorted({k for p in valid for k in p})
    knob_std = {k: round(_std([float(p[k]) for p in valid if k in p]), 4) for k in keys}
    diffs = [float(p["difficulty"]) for p in valid if "difficulty" in p]
    return {
        "n_valid": len(valid),
        "difficulty_mean": round(_mean(diffs), 4),
        "difficulty_std": round(_std(diffs), 4),
        "difficulty_unique": len({round(d, 2) for d in diffs}),
        "knob_std": knob_std,
    }


# ---------------------------------------------------------------------------
# Optional inner-Student population config (gated; default off)
# ---------------------------------------------------------------------------
#
# Some focused runs want the nested Ring-Out Players trained with a NEW population
# (a small MLP arch, an opponent league, decisive-reward + entropy bonus) instead of
# the reward path's built-in default. ``--student-cfg-preset`` builds one of these dicts
# and the host threads it into ``_score_ring_out`` -> ``teacher_reward(student_cfg=...)``
# (the reward teammate consumes it). With NO preset the dict is None and nothing changes.
STUDENT_CFG_PRESETS: dict[str, dict] = {
    # focused128: the leagueB1 population — 2x128 MLP, an opponent league
    # (aggressive 0.7 / prior_student 0.3), decisive reward, entropy coef 0.03.
    "focused128": {
        "arch": [128, 128],
        "opponent_league_enabled": True,
        "opponent_league": [
            {"id": "aggressive", "weight": 0.7},
            {"id": "prior_student", "weight": 0.3},
        ],
        "decisive_reward": True,
        "ent_coef": 0.03,
        "prior_student_path": "/root/prior_student.json",
    },
    # focused256: IDENTICAL focused league/reward to focused128, only a wider 2x256 MLP
    # (~135k params vs ~37k). The capacity-comparison arm — run leagueB256 in parallel
    # with leagueB128 to test whether bigger Students make the Sensei teach better fighting.
    "focused256": {
        "arch": [256, 256],
        "opponent_league_enabled": True,
        "opponent_league": [
            {"id": "aggressive", "weight": 0.7},
            {"id": "prior_student", "weight": 0.3},
        ],
        "decisive_reward": True,
        "ent_coef": 0.03,
        "prior_student_path": "/root/prior_student.json",
    },
}


def _build_student_cfg(preset: str | None) -> dict | None:
    """Resolve ``--student-cfg-preset`` to a Student population dict (or None).

    None (no flag) => unchanged behavior (no ``student_cfg`` reaches the reward path).
    A fresh ``dict(...)`` copy is returned so callers can't mutate the module constant.
    """
    if not preset:
        return None
    if preset not in STUDENT_CFG_PRESETS:
        raise SystemExit(
            f"unknown --student-cfg-preset {preset!r}; choices: {sorted(STUDENT_CFG_PRESETS)}"
        )
    return json.loads(json.dumps(STUDENT_CFG_PRESETS[preset]))


def _lr_scale_for(update_index: int, updates: int, decay: bool) -> float:
    """The per-update lr multiplier in [0,1] for a LINEAR-DECAY schedule.

    ``update_index`` is 1-based (update1..updateN). With ``decay`` off this is always
    1.0 (constant lr = legacy behavior). With decay on, the scale runs from 1.0 on the
    first update DOWN toward ~0 on the last, linearly:
        scale = 1 - (update_index - 1) / updates
    so for updates=5 the scales are 1.0, 0.8, 0.6, 0.4, 0.2 (never exactly 0, but the
    last step's effective lr is base_lr/updates — a small, near-zero final step).
    """
    if not decay or updates <= 1:
        return 1.0
    return max(0.0, 1.0 - (update_index - 1) / float(updates))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_gpu_smoke(args) -> int:
    """GPU-only smoke: no reward compute. Proves load+sample(both games)+update+save."""
    print("=== GPU SMOKE (no reward, both games) ===", flush=True)
    with app.run():
        trainer = TeacherTrainer(model_name=args.model, lr_str=str(args.lr),
                                 lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        out = trainer.gpu_smoke.remote(group_size=args.group_size)
    print(json.dumps(out, indent=2))
    return 0


def run_train(args) -> int:
    games = list(args.games)
    assert all(g in GAME_KEYS for g in games), f"unknown game in {games}"
    _assert_registry_parity()
    multi = len(games) > 1

    # --- Seed pools: rotated TRAINING seeds (per update) + reserved VALIDATION seeds ---
    # Training-reward seeds rotate per update (distinct PPO populations each step ->
    # anti-overfit). Validation seeds are a SEPARATE fixed pool used ONLY by the final
    # scoring pass. We assert the two pools are disjoint up front so a leak is a loud
    # error, never a silent train-on-your-validation-set.
    n_seeds = {"ring_out": args.ring_n_seeds, "target_knockback": args.tk_n_seeds}
    val_seeds = {"ring_out": list(args.val_seeds), "target_knockback": list(args.tk_val_seeds)}
    for game in games:
        train_union = _all_training_seeds(
            game, args.updates, n_seeds[game], args.seed_base, args.seed_rotation
        )
        _assert_seed_pools_disjoint(game, train_union, val_seeds[game])

    ring_held_out = _build_held_out() if "ring_out" in games else []
    # Optional inner-Student population config (gated). None unless --student-cfg-preset
    # is passed; only ever applied to the Ring-Out reward (TK does not consume it).
    ring_student_cfg = _build_student_cfg(getattr(args, "student_cfg_preset", None))
    # Optional LINEAR LR DECAY across the updates (gated; default off => constant lr).
    lr_decay = bool(getattr(args, "lr_decay", False))
    # Output dir is per-run: ``output/contingency_<run_id>/`` when isolated, else the
    # legacy ``output/contingency_rft_multigame/``. Four concurrent runs each own their
    # summary.json / history.json.
    out_subdir = f"contingency_{RUN_ID}" if RUN_ID else "contingency_rft_multigame"
    out_dir = _REPO_ROOT / "output" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== CONTINGENCY Teacher-RFT (Modal LoRA GRPO) — MULTI-GAME ===", flush=True)
    print(f"    run_id={RUN_ID or '(none / legacy default)'}  app={APP_NAME}", flush=True)
    print(f"    adapter_volume_path={ADAPTER_RUN_DIR}  output_dir={out_dir}", flush=True)
    print(f"    model={args.model}  gpu={GPU_SPEC}", flush=True)
    print(f"    games={games}  (one COMBINED GRPO update per step over all games)", flush=True)
    print(f"    updates={args.updates}  group_size={args.group_size}/game  "
          f"temperature={args.temperature}  lr={args.lr}", flush=True)
    print(f"    seed_rotation={args.seed_rotation}  seed_base={args.seed_base}  "
          f"(training seeds VARY per update; validation seeds are reserved + disjoint)", flush=True)
    if lr_decay:
        _lr_scales = [_lr_scale_for(u, args.updates, True) for u in range(1, args.updates + 1)]
        print(f"    lr_schedule=LINEAR DECAY  base_lr={args.lr}  per-update lr_scale="
              f"{[round(s, 3) for s in _lr_scales]}  "
              f"(effective_lr decays {args.lr} -> ~{args.lr * (_lr_scales[-1] if _lr_scales else 1.0):.2e})",
              flush=True)
    else:
        print(f"    lr_schedule=CONSTANT (legacy)  lr={args.lr}  "
              f"(pass --lr-decay for a linear decay over the updates)", flush=True)
    if ring_student_cfg is not None:
        print(f"    student_cfg_preset={args.student_cfg_preset} -> ring_out reward "
              f"trains inner Students with: {json.dumps(ring_student_cfg)}", flush=True)
    else:
        print("    student_cfg=(none / default Student population)", flush=True)
    if "ring_out" in games:
        ro_blocks = [_train_seeds_for("ring_out", u, n_seeds["ring_out"], args.seed_base)
                     for u in range(1, args.updates + 1)] if args.seed_rotation else \
                    [_train_seeds_for("ring_out", 1, n_seeds["ring_out"], args.seed_base)]
        print(f"    ring_out reward: teacher_reward (crucible-player) | "
              f"train_seed_blocks={ro_blocks} val_seeds={val_seeds['ring_out']} "
              f"episodes={args.episodes} eval_seeds={args.eval_seeds} | "
              f"held_out={len(ring_held_out)} arenas", flush=True)
    if "target_knockback" in games:
        tk_blocks = [_train_seeds_for("target_knockback", u, n_seeds["target_knockback"], args.seed_base)
                     for u in range(1, args.updates + 1)] if args.seed_rotation else \
                    [_train_seeds_for("target_knockback", 1, n_seeds["target_knockback"], args.seed_base)]
        print(f"    tk reward:       tk_teacher_reward (crucible-player-tk) | "
              f"train_seed_blocks={tk_blocks} val_seeds={val_seeds['target_knockback']} "
              f"episodes={args.tk_episodes} eval_seeds={args.tk_eval_seeds} | "
              f"worker-built held-out", flush=True)
    print("    ADVANTAGES ARE PER-GAME (no cross-game reward pooling) -> unbiased combined step", flush=True)
    print("    REWARDS ARE SIGNED (TK held-out improvement can be negative; GRPO norm is shift-invariant)", flush=True)
    print(flush=True)

    history: list = []
    t_run = time.time()

    with app.run():
        trainer = TeacherTrainer(model_name=args.model, lr_str=str(args.lr),
                                 lora_r=args.lora_r, lora_alpha=args.lora_alpha)

        # Steps 0..N. Step 0 is the pure "before" measurement of the BASE policy (no
        # gradient): both games sampled + scored before any update. Steps 1..N each
        # sample BOTH games, score both, compute per-game advantages, take ONE combined
        # GRPO step, and save the adapter.
        for step in range(args.updates + 1):
            tag = "base" if step == 0 else f"update{step}"

            # --- 1) sample G completions PER GAME from the shared policy ---
            completions_by_game: dict = {}
            for gi, game in enumerate(games):
                # Distinct seed per (step, game) so the two games' samples are
                # independent draws, not the same RNG stream.
                gseed = 1000 + step * 10 + gi
                print(f"[{tag}] sampling {args.group_size} {game} completions (seed={gseed}) ...",
                      flush=True)
                completions_by_game[game] = trainer.sample.remote(
                    game=game, group_size=args.group_size, temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens, seed=gseed,
                )

            # --- 2+3) validate + score ALL games' completions in ONE parallel fan-out ---
            # ROTATED TRAINING SEEDS: each step uses a DISTINCT PPO seed block (anti-
            # overfit). step 0 (base) and step k both index by max(step,1) so the base
            # measurement reuses update-1's block; updates 1..N each get their own. With
            # --no-seed-rotation every step falls back to the update-1 block (legacy).
            seed_update_idx = step if (args.seed_rotation and step >= 1) else 1
            step_ring_seeds = tuple(_train_seeds_for(
                "ring_out", seed_update_idx, n_seeds["ring_out"], args.seed_base))
            step_tk_seeds = tuple(_train_seeds_for(
                "target_knockback", seed_update_idx, n_seeds["target_knockback"], args.seed_base))
            t_score = time.time()
            print(f"[{tag}] scoring {sum(len(c) for c in completions_by_game.values())} "
                  f"completions across {len(games)} game(s) (parallel) | "
                  f"train seeds ring={list(step_ring_seeds)} tk={list(step_tk_seeds)} ...", flush=True)
            scored_by_game = _score_all_games(
                completions_by_game,
                ring_seeds=step_ring_seeds, ring_episodes=args.episodes,
                ring_eval_seeds=args.eval_seeds, ring_held_out=ring_held_out,
                tk_seeds=step_tk_seeds, tk_episodes=args.tk_episodes,
                tk_eval_seeds=args.tk_eval_seeds,
                max_workers=args.group_size * len(games),
                ring_student_cfg=ring_student_cfg,
            )
            score_wall = time.time() - t_score

            # --- 4) PER-GAME advantages (anti-bias: normalize WITHIN each game) ---
            per_game: dict = {}
            advantages_by_game: dict = {}
            for game in games:
                scored = scored_by_game[game]
                rewards = [s[0] for s in scored]
                statuses = [s[2] for s in scored]
                params = [s[1] for s in scored]
                n_valid = sum(1 for st in statuses if st == "ok")
                adv = _per_game_advantages(rewards)
                advantages_by_game[game] = adv
                per_game[game] = {
                    "rewards": [round(r, 4) for r in rewards],
                    "statuses": statuses,
                    "n_valid": n_valid,
                    "reward_mean": round(_mean(rewards), 4),
                    "reward_std": round(_std(rewards), 4),
                    "advantages": [round(a, 4) for a in adv],
                    "params": params,
                    "completions_preview": [c[:200] for c in completions_by_game[game]],
                    "arena_diversity": _arena_diversity(params),
                }
                print(f"[{tag}] {game}: rewards={[round(r,4) for r in rewards]}  "
                      f"valid={n_valid}/{len(rewards)}  mean={_mean(rewards):.4f}  "
                      f"std={_std(rewards):.4f}", flush=True)
            print(f"[{tag}] all games scored in {score_wall:.0f}s", flush=True)

            rec = {
                "step": step, "tag": tag,
                "per_game": per_game,
                "score_wall_s": round(score_wall, 1),
                "train_seeds": {
                    "ring_out": list(step_ring_seeds),
                    "target_knockback": list(step_tk_seeds),
                },
            }

            if step == 0:
                print(f"[{tag}] (no update — this is the before-training baseline)\n", flush=True)
            else:
                # --- 5) ONE combined GRPO step over both games' cached sequences ---
                # Per-update lr scale for the optional linear-decay schedule (1.0 when
                # --lr-decay is off => constant lr). step is 1-based over updates here.
                lr_scale = _lr_scale_for(step, args.updates, lr_decay)
                stats = trainer.update_multi.remote(advantages_by_game, lr_scale=lr_scale)
                rec["update_stats"] = stats
                adapter_path = trainer.save_adapter.remote(tag)
                rec["adapter_path"] = adapter_path
                print(f"[{tag}] COMBINED GRPO step over {stats['n_completions']} completions "
                      f"({'+'.join(stats['games'])}): loss={stats['loss']:.4f} "
                      f"grad_norm={stats['grad_norm']:.3f}  "
                      f"lr_scale={stats.get('lr_scale')} effective_lr={stats.get('effective_lr'):.2e} "
                      f"-> saved {adapter_path}\n", flush=True)

            history.append(rec)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # ---------------------------------------------------------------------
        # FINAL-ADAPTER SCORING PASS (fix #3) — score the SAVED final adapter.
        # ---------------------------------------------------------------------
        # The per-update logged reward is the PRE-update policy (we sample+score BEFORE
        # applying that update), so the SAVED update-N adapter — the one a rerun ships —
        # was never itself scored. After the last update_multi above, the warm
        # container's IN-MEMORY policy IS exactly the post-update-N weights that
        # save_adapter wrote as ``update{N}`` (same model object, no reload needed), so
        # we sample + score it here. This pass uses the RESERVED VALIDATION seeds (never
        # touched during training-reward computation) and DISTINCT sample seeds, and is
        # recorded SEPARATELY as the TRUE final-adapter reward — not mixed into the
        # per-update logs.
        final_adapter_eval: dict | None = None
        if args.updates >= 1:
            final_tag = f"update{args.updates}"
            print(f"=== FINAL-ADAPTER SCORING PASS ({final_tag}, validation seeds) ===", flush=True)
            final_completions: dict = {}
            for gi, game in enumerate(games):
                vseed = 9000 + gi  # distinct from the 1000-range training sample seeds
                print(f"[final/{final_tag}] sampling {args.group_size} {game} completions "
                      f"from the FINAL adapter (seed={vseed}) ...", flush=True)
                final_completions[game] = trainer.sample.remote(
                    game=game, group_size=args.group_size, temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens, seed=vseed,
                )

            t_val = time.time()
            print(f"[final/{final_tag}] scoring on VALIDATION seeds "
                  f"ring={val_seeds['ring_out']} tk={val_seeds['target_knockback']} (parallel) ...",
                  flush=True)
            val_scored = _score_all_games(
                final_completions,
                ring_seeds=tuple(val_seeds["ring_out"]), ring_episodes=args.episodes,
                ring_eval_seeds=args.eval_seeds, ring_held_out=ring_held_out,
                tk_seeds=tuple(val_seeds["target_knockback"]), tk_episodes=args.tk_episodes,
                tk_eval_seeds=args.tk_eval_seeds,
                max_workers=args.group_size * len(games),
                ring_student_cfg=ring_student_cfg,
            )
            val_wall = time.time() - t_val

            final_per_game: dict = {}
            for game in games:
                scored = val_scored[game]
                rewards = [s[0] for s in scored]
                statuses = [s[2] for s in scored]
                params = [s[1] for s in scored]
                n_valid = sum(1 for st in statuses if st == "ok")
                final_per_game[game] = {
                    "rewards": [round(r, 4) for r in rewards],
                    "statuses": statuses,
                    "n_valid": n_valid,
                    "reward_mean": round(_mean(rewards), 4),
                    "reward_std": round(_std(rewards), 4),
                    "params": params,
                    "completions_preview": [c[:200] for c in final_completions[game]],
                    "arena_diversity": _arena_diversity(params),
                }
                print(f"[final/{final_tag}] {game}: rewards={[round(r,4) for r in rewards]}  "
                      f"valid={n_valid}/{len(rewards)}  mean={_mean(rewards):.4f}  "
                      f"std={_std(rewards):.4f}", flush=True)
            print(f"[final/{final_tag}] final-adapter scored in {val_wall:.0f}s\n", flush=True)

            final_adapter_eval = {
                "label": "TRUE final-adapter reward (post-update policy on RESERVED "
                         "validation seeds; distinct from the per-update PRE-update logs)",
                "adapter_tag": final_tag,
                "validation_seeds": {
                    "ring_out": list(val_seeds["ring_out"]),
                    "target_knockback": list(val_seeds["target_knockback"]),
                },
                "per_game": final_per_game,
                "val_wall_s": round(val_wall, 1),
            }

    # --- final verdict: per-game before (step 0) vs after (last step) ---
    base_rec = history[0]
    final_rec = history[-1]
    per_game_summary: dict = {}
    for game in games:
        b = base_rec["per_game"][game]["reward_mean"]
        a = final_rec["per_game"][game]["reward_mean"]
        per_game_summary[game] = {
            "before_reward_mean": b,
            "after_reward_mean": a,
            "delta_before_after": round(a - b, 4),
            "before_reward_std": base_rec["per_game"][game]["reward_std"],
            "after_reward_std": final_rec["per_game"][game]["reward_std"],
            "before_valid": base_rec["per_game"][game]["n_valid"],
            "after_valid": final_rec["per_game"][game]["n_valid"],
            "after_arena_diversity": final_rec["per_game"][game]["arena_diversity"],
        }

    # Training seed BLOCKS actually used per update (for the record / audit).
    def _blocks(game: str) -> list[list[int]]:
        span = args.updates if args.seed_rotation else 1
        return [_train_seeds_for(game, max(u, 1), n_seeds[game], args.seed_base)
                for u in range(1, max(span, 1) + 1)]

    # --- HEADLINE = the TRUE final-adapter score (post-update policy on RESERVED
    # validation seeds), NOT a per-update PRE-update logged reward. The per-game
    # ``per_game_summary[*].after_reward_mean`` above is a PRE-update training-seed
    # number (the policy is sampled+scored BEFORE that step's gradient), so it does NOT
    # describe the adapter a rerun actually ships. The headline reads final_adapter_eval.
    headline: dict | None = None
    if final_adapter_eval is not None:
        fa_per_game = {
            game: final_adapter_eval["per_game"][game]["reward_mean"] for game in games
        }
        headline = {
            "source": "final_adapter_eval",
            "label": "TRUE final-adapter reward (post-update policy on RESERVED "
                     "validation seeds) — NOT a per-update PRE-update logged reward",
            "adapter_tag": final_adapter_eval["adapter_tag"],
            "reward_mean_by_game": fa_per_game,
            "reward_mean_overall": round(_mean(list(fa_per_game.values())), 4),
        }

    summary = {
        "experiment": "contingency_teacher_rft_modal_lora_grpo_multigame",
        "run_id": RUN_ID or None,
        "app_name": APP_NAME,
        "adapter_volume_path": ADAPTER_RUN_DIR,
        "output_dir": str(out_dir),
        # HEADLINE first: the reported top-line reward is the final-adapter validation
        # score, not the per-update logs in per_game/history below.
        "headline": headline,
        "model": args.model,
        "gpu": GPU_SPEC,
        "games": games,
        "combined_update": multi,
        "advantage_normalization": "per_game (no cross-game reward pooling)",
        "reward_sign": "signed (TK held-out improvement NOT re-clamped; GRPO norm is shift-invariant)",
        "updates": args.updates,
        "group_size_per_game": args.group_size,
        "temperature": args.temperature,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "lr_schedule": {
            "type": "linear_decay" if lr_decay else "constant",
            "base_lr": args.lr,
            "per_update_lr_scale": (
                [round(_lr_scale_for(u, args.updates, lr_decay), 6)
                 for u in range(1, args.updates + 1)]
                if args.updates >= 1 else []
            ),
        },
        "student_cfg_preset": getattr(args, "student_cfg_preset", None),
        "student_cfg": ring_student_cfg,
        "seed_scheme": {
            "seed_rotation": args.seed_rotation,
            "seed_base": args.seed_base,
            "description": "training PPO reward seeds rotate per update (distinct block "
                           "each step); validation seeds are a SEPARATE fixed pool, "
                           "disjoint from every training block, used only by the "
                           "final-adapter scoring pass.",
            "train_seed_blocks": {game: _blocks(game) for game in games},
            "validation_seeds": {game: list(val_seeds[game]) for game in games},
        },
        "ring_out": {
            "n_reward_seeds": args.ring_n_seeds, "train_seed_blocks": _blocks("ring_out"),
            "validation_seeds": list(val_seeds["ring_out"]),
            "ppo_episodes": args.episodes,
            "eval_seeds": args.eval_seeds, "n_held_out_arenas": len(ring_held_out),
        } if "ring_out" in games else None,
        "target_knockback": {
            "n_reward_seeds": args.tk_n_seeds, "train_seed_blocks": _blocks("target_knockback"),
            "validation_seeds": list(val_seeds["target_knockback"]),
            "ppo_episodes": args.tk_episodes,
            "eval_seeds": args.tk_eval_seeds,
        } if "target_knockback" in games else None,
        "per_game": per_game_summary,
        "final_adapter_eval": final_adapter_eval,
        "grad_norms": [
            h.get("update_stats", {}).get("grad_norm") for h in history if "update_stats" in h
        ],
        "total_wall_s": round(time.time() - t_run, 1),
        "adapter_volume": VOLUME_NAME,
        "history": history,
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("    === BEFORE vs AFTER (per game; PRE-update logged rewards, rotated train seeds) ===", flush=True)
    for game in games:
        s = per_game_summary[game]
        print(f"    [{game}] before={s['before_reward_mean']:+.4f}  "
              f"after={s['after_reward_mean']:+.4f}  "
              f"delta={s['delta_before_after']:+.4f}  "
              f"(std {s['before_reward_std']:.4f}->{s['after_reward_std']:.4f}, "
              f"valid {s['before_valid']}->{s['after_valid']})", flush=True)
    if final_adapter_eval is not None:
        print(f"    === TRUE FINAL-ADAPTER REWARD ({final_adapter_eval['adapter_tag']}, "
              f"validation seeds; distinct from the per-update logs above) ===", flush=True)
        for game in games:
            fp = final_adapter_eval["per_game"][game]
            print(f"    [{game}] final_adapter_reward={fp['reward_mean']:+.4f}  "
                  f"(std {fp['reward_std']:.4f}, valid {fp['n_valid']}/{args.group_size}, "
                  f"val_seeds={final_adapter_eval['validation_seeds'][game]})", flush=True)
    if headline is not None:
        print(f"    >>> HEADLINE (final_adapter_eval, the reported number): "
              f"overall={headline['reward_mean_overall']:+.4f}  "
              f"by_game={ {g: round(v, 4) for g, v in headline['reward_mean_by_game'].items()} }",
              flush=True)
    print(f"    grad_norms: {[round(g,3) for g in summary['grad_norms'] if g is not None]}", flush=True)
    print(f"\n    wrote {out_path}", flush=True)
    print(f"    adapters on Modal Volume '{VOLUME_NAME}' under '{ADAPTER_RUN_DIR}' "
          f"(update1..update{args.updates}); fetch with: "
          f"modal volume get {VOLUME_NAME} {(RUN_ID + '/') if RUN_ID else ''}<tag>", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Contingency Modal Teacher-RFT (LoRA GRPO on Qwen), multi-game.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="base Teacher model (default Qwen/Qwen3-4B; fall back to "
                        "Qwen/Qwen3-1.7B or Qwen/Qwen2.5-1.5B-Instruct if too heavy)")
    p.add_argument("--games", nargs="+", default=list(GAME_KEYS),
                   choices=list(GAME_KEYS),
                   help="games trained in each COMBINED step (default: both). "
                        "One COMBINED GRPO update per step over all listed games.")
    p.add_argument("--updates", type=int, default=2, help="number of combined GRPO updates (scale to 4-5 for a fuller run)")
    p.add_argument("--group-size", dest="group_size", type=int, default=5,
                   help="G completions sampled PER GAME per update (4-6 keeps cost ~like single-game G=8)")
    p.add_argument("--lr", type=float, default=1e-4)
    # LINEAR LR DECAY (gated; default OFF => constant lr, legacy behavior). When set, the
    # effective lr decays linearly from --lr on update1 toward ~0 on the last update
    # (scale = 1 - (i-1)/updates), applied on the optimizer param-groups each step.
    p.add_argument("--lr-decay", dest="lr_decay", action="store_true", default=False,
                   help="linearly decay the lr from --lr (update1) toward ~0 (last update); "
                        "default OFF = constant lr")
    # Default LoRA rank 32 / alpha 64 (Nathan's choice for the rerun: more adapter
    # capacity than the original r=16/alpha=32). alpha=2*r keeps the LoRA scaling
    # (alpha/r) at 2.0, matching the original ratio, so the effective update magnitude
    # is unchanged — only the rank (expressiveness) grows.
    p.add_argument("--lora-r", dest="lora_r", type=int, default=32)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.1,
                   help="sampling temperature (1.1 was the variance-fixed single-game setting)")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=512)
    # Optional inner-Student population config (gated; default None => unchanged). A
    # preset builds a student_cfg dict threaded into the Ring-Out reward so the nested
    # Players train with the new population (the reward-path teammate consumes it).
    p.add_argument("--student-cfg-preset", dest="student_cfg_preset", default=None,
                   choices=sorted(STUDENT_CFG_PRESETS),
                   help="inner-Student population preset for the Ring-Out reward "
                        "(e.g. focused128: 2x128 MLP + opponent league + decisive reward + "
                        "ent_coef 0.03). Default: none = unchanged Student population.")
    # Ring-Out reward config — defaults mirror the dry-loop / bridge nested reward.
    # NOTE: --seeds / --tk-seeds are now the COUNT of PPO seeds per reward (an int N),
    # NOT a literal seed list, because the training seeds ROTATE per update (see the
    # seed-rotation block below). Validation uses the FIXED --val-seeds / --tk-val-seeds
    # pools, which are kept DISJOINT from every rotated training block.
    p.add_argument("--seeds", dest="ring_n_seeds", type=int, default=3,
                   help="Ring-Out: number of PPO seeds per reward (rotated per update during training)")
    p.add_argument("--episodes", type=int, default=1000, help="Ring-Out PPO episodes per seed")
    p.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=50)
    # Target-Knockback reward config — defaults mirror nested_reward_tk.
    p.add_argument("--tk-seeds", dest="tk_n_seeds", type=int, default=3,
                   help="Target-Knockback: number of PPO seeds per reward (rotated per update during training)")
    p.add_argument("--tk-episodes", dest="tk_episodes", type=int, default=2000,
                   help="TK PPO episodes per seed (TK's validated budget)")
    p.add_argument("--tk-eval-seeds", dest="tk_eval_seeds", type=int, default=50)
    # --- Seed rotation + reserved validation seeds (anti-overfit) ---------------
    # Training-reward seeds VARY per update: each update consumes a distinct contiguous
    # block in a HIGH range (--seed-base + per-game offset + update_index*block), so the
    # Teacher never re-scores against the same PPO populations twice and cannot overfit
    # one fixed seed set. Validation uses a SEPARATE fixed low-range pool, never seen
    # during training-reward computation. The two pools are asserted disjoint at runtime.
    p.add_argument("--seed-rotation", dest="seed_rotation", action="store_true", default=True,
                   help="rotate training PPO reward seeds per update (default ON; anti-overfit)")
    p.add_argument("--no-seed-rotation", dest="seed_rotation", action="store_false",
                   help="disable rotation: reuse the first training block every update (legacy behavior)")
    p.add_argument("--seed-base", dest="seed_base", type=int, default=10000,
                   help="base of the HIGH-range training-seed space (rotated blocks live here; "
                        "kept disjoint from the low-range validation pools)")
    p.add_argument("--val-seeds", dest="val_seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="Ring-Out FIXED validation seeds (held out of training; final-adapter scoring only)")
    p.add_argument("--tk-val-seeds", dest="tk_val_seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="Target-Knockback FIXED validation seeds (held out of training; final-adapter scoring only)")
    # --- Per-run isolation + GPU (these PIN import-time globals; the flags exist for
    # ergonomics + the audit record, but the Modal App name and @app.cls(gpu=) were
    # ALREADY fixed at import from CRUCIBLE_RUN_ID / CRUCIBLE_GPU, before argparse ran.
    # So the flags must AGREE with the env — we assert that below rather than silently
    # let a --run-id that differs from CRUCIBLE_RUN_ID write to the wrong app/volume.) ---
    p.add_argument("--run-id", dest="run_id", default=RUN_ID or None,
                   help="per-run isolation id (app name + volume subdir + output dir). "
                        "Must equal CRUCIBLE_RUN_ID (read at import to name the Modal App "
                        "BEFORE args parse). Set CRUCIBLE_RUN_ID=<id> when launching.")
    p.add_argument("--gpu", dest="gpu", default=GPU_SPEC,
                   help="Modal GPU for the trainer @app.cls (default A100-40GB; fits LoRA "
                        "r=32 on Qwen3-4B). Must equal CRUCIBLE_GPU if set (the decorator "
                        "is pinned at import). Set CRUCIBLE_GPU=<gpu> to change it.")
    p.add_argument("--gpu-smoke", dest="gpu_smoke", action="store_true",
                   help="GPU-only check: load+sample(both games)+fake-update+save, NO reward compute")
    args = p.parse_args(argv)

    # The Modal App name + GPU decorator are import-time globals (they MUST be, to name
    # the App before CLI parse). If the user passes a --run-id / --gpu that disagrees
    # with what was locked at import, fail LOUDLY — silently honoring the flag here would
    # write to the wrong app/volume/output and re-introduce the collision this fixes.
    flag_run_id = (args.run_id or "").strip()
    if flag_run_id != RUN_ID:
        print(
            f"FATAL: --run-id={flag_run_id!r} != CRUCIBLE_RUN_ID={RUN_ID!r}. The Modal App "
            f"name ({APP_NAME!r}) and adapter volume path ({ADAPTER_RUN_DIR!r}) were FIXED "
            f"at import from CRUCIBLE_RUN_ID. Launch with: "
            f"CRUCIBLE_RUN_ID={flag_run_id or '<id>'} ... --run-id {flag_run_id or '<id>'}",
            file=sys.stderr,
        )
        return 2
    if (args.gpu or "").strip() != GPU_SPEC:
        print(
            f"FATAL: --gpu={args.gpu!r} != CRUCIBLE_GPU/effective GPU={GPU_SPEC!r}. The "
            f"@app.cls(gpu=...) decorator was FIXED at import. Launch with: "
            f"CRUCIBLE_GPU={args.gpu} ... --gpu {args.gpu}",
            file=sys.stderr,
        )
        return 2

    if not (os.getenv("FIREWORKS_API_KEY") or "").strip():
        print("NOTE: FIREWORKS_API_KEY not set — the nested reward path does not need it; "
              "source .env to be safe.", file=sys.stderr)

    if args.gpu_smoke:
        return run_gpu_smoke(args)
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
