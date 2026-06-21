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

RUN (from repo root; .env sourced; ~/.modal.toml profile njlee007):

    set -a && source .env && set +a
    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    # multi-game (default): both games every step, G per game
    .venv/bin/python3 training/train_teacher_modal.py --updates 2 --group-size 5 --temperature 1.1
    # scale to a fuller run: just raise --updates (e.g. 4-5)
    .venv/bin/python3 training/train_teacher_modal.py --updates 5 --group-size 5 --temperature 1.1
    # single-game (legacy): just Ring-Out
    .venv/bin/python3 training/train_teacher_modal.py --games ring_out --updates 2 --group-size 8
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

# Where the GPU container persists the LoRA adapter across calls + runs.
VOLUME_NAME = "crucible-teacher-lora"
ADAPTER_DIR = "/adapters"

app = modal.App("crucible-teacher-trainer")

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
)


@app.cls(
    image=gpu_image,
    gpu=DEFAULT_GPU,
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
        self.optimizer = torch.optim.Adam(trainable, lr=float(self.lr_str))

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
    def update_multi(self, advantages_by_game: dict) -> dict:
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
        """
        torch = self.torch
        comp_logps = []
        advs = []
        per_game_logp_mean: dict = {}
        used_games = []
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
            comp_logps.append(lp)
            advs.append(torch.tensor(game_adv, dtype=torch.float32, device="cuda:0"))
            per_game_logp_mean[game] = float(lp.detach().mean().cpu())
            used_games.append(game)

        assert comp_logps, "update_multi called with no game groups"
        logp_concat = torch.cat(comp_logps, dim=0)   # (sum_G,)
        adv_concat = torch.cat(advs, dim=0)          # (sum_G,)

        self.model.train()
        self.optimizer.zero_grad()
        loss = -(adv_concat * logp_concat).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm),
            "games": used_games,
            "n_completions": int(logp_concat.shape[0]),
            "mean_completion_logprob_by_game": {
                g: round(v, 4) for g, v in per_game_logp_mean.items()
            },
        }

    @modal.method()
    def save_adapter(self, tag: str) -> str:
        """Persist the current LoRA adapter to the Volume; return its path."""
        path = os.path.join(ADAPTER_DIR, tag)
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
                    held_out) -> tuple[float, dict | None, str]:
    """Score ONE Ring-Out completion via the EXISTING fighter nested reward.

    parse (strip <think> + strict JSON + clamp to RING_OUT_BOUNDS + FIGHTER-schema
    map) -> ``teacher_reward`` (3 PPO seeds on crucible-player). Invalid -> reward 0.
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
    reward = teacher_reward(
        spec,
        backend="modal",
        seeds=seeds,
        episodes=episodes,
        eval_seeds=eval_seeds,
        held_out_arenas=held_out,
        geometry_override=fighter_geometry_override(params),
    )
    return float(max(0.0, min(1.0, reward))), params, "ok"


def _score_tk(answer: str, *, seeds, episodes, eval_seeds) -> tuple[float, dict | None, str]:
    """Score ONE Target-Knockback completion via the EXISTING TK nested reward.

    parse (strip <think> + strict JSON + clamp to TK_BOUNDS, the 7-key TK schema) ->
    ``tk_teacher_reward`` (3 PPO seeds on crucible-player-tk; reads ``improvement``,
    asserts ``status=="ppo_tk"``). Invalid -> reward 0. TK does NOT use the fighter's
    broad held-out grid: the TK worker builds its own held-out reference, so no
    held_out_arenas is passed (that is the TK reward's load-bearing yardstick).
    """
    from output.nested_reward_tk import tk_teacher_reward
    from training.hud_teacher_env import parse_teacher_params

    try:
        params = parse_teacher_params(answer, TK_BOUNDS)
    except Exception as exc:
        return 0.0, None, f"invalid:{type(exc).__name__}"

    reward = tk_teacher_reward(
        params,
        backend="modal",
        seeds=tuple(seeds),
        episodes=episodes,
        eval_seeds=eval_seeds,
        curriculum_id="contingency-rft-tk",
    )
    return float(max(0.0, min(1.0, reward))), params, "ok"


def _score_all_games(
    completions_by_game: dict,
    *,
    ring_seeds, ring_episodes, ring_eval_seeds, ring_held_out,
    tk_seeds, tk_episodes, tk_eval_seeds,
    max_workers: int,
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

    ring_seeds = tuple(args.seeds)
    tk_seeds = tuple(args.tk_seeds)
    ring_held_out = _build_held_out() if "ring_out" in games else []
    out_dir = _REPO_ROOT / "output" / "contingency_rft_multigame"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== CONTINGENCY Teacher-RFT (Modal LoRA GRPO) — MULTI-GAME ===", flush=True)
    print(f"    model={args.model}  gpu={DEFAULT_GPU}", flush=True)
    print(f"    games={games}  (one COMBINED GRPO update per step over all games)", flush=True)
    print(f"    updates={args.updates}  group_size={args.group_size}/game  "
          f"temperature={args.temperature}  lr={args.lr}", flush=True)
    if "ring_out" in games:
        print(f"    ring_out reward: teacher_reward (crucible-player) | seeds={ring_seeds} "
              f"episodes={args.episodes} eval_seeds={args.eval_seeds} | "
              f"held_out={len(ring_held_out)} arenas", flush=True)
    if "target_knockback" in games:
        print(f"    tk reward:       tk_teacher_reward (crucible-player-tk) | seeds={tk_seeds} "
              f"episodes={args.tk_episodes} eval_seeds={args.tk_eval_seeds} | "
              f"worker-built held-out", flush=True)
    print("    ADVANTAGES ARE PER-GAME (no cross-game reward pooling) -> unbiased combined step", flush=True)
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
            t_score = time.time()
            print(f"[{tag}] scoring {sum(len(c) for c in completions_by_game.values())} "
                  f"completions across {len(games)} game(s) (parallel) ...", flush=True)
            scored_by_game = _score_all_games(
                completions_by_game,
                ring_seeds=ring_seeds, ring_episodes=args.episodes,
                ring_eval_seeds=args.eval_seeds, ring_held_out=ring_held_out,
                tk_seeds=tk_seeds, tk_episodes=args.tk_episodes,
                tk_eval_seeds=args.tk_eval_seeds,
                max_workers=args.group_size * len(games),
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
            }

            if step == 0:
                print(f"[{tag}] (no update — this is the before-training baseline)\n", flush=True)
            else:
                # --- 5) ONE combined GRPO step over both games' cached sequences ---
                stats = trainer.update_multi.remote(advantages_by_game)
                rec["update_stats"] = stats
                adapter_path = trainer.save_adapter.remote(tag)
                rec["adapter_path"] = adapter_path
                print(f"[{tag}] COMBINED GRPO step over {stats['n_completions']} completions "
                      f"({'+'.join(stats['games'])}): loss={stats['loss']:.4f} "
                      f"grad_norm={stats['grad_norm']:.3f} -> saved {adapter_path}\n", flush=True)

            history.append(rec)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

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

    summary = {
        "experiment": "contingency_teacher_rft_modal_lora_grpo_multigame",
        "model": args.model,
        "gpu": DEFAULT_GPU,
        "games": games,
        "combined_update": multi,
        "advantage_normalization": "per_game (no cross-game reward pooling)",
        "updates": args.updates,
        "group_size_per_game": args.group_size,
        "temperature": args.temperature,
        "ring_out": {
            "reward_seeds": list(ring_seeds), "ppo_episodes": args.episodes,
            "eval_seeds": args.eval_seeds, "n_held_out_arenas": len(ring_held_out),
        } if "ring_out" in games else None,
        "target_knockback": {
            "reward_seeds": list(tk_seeds), "ppo_episodes": args.tk_episodes,
            "eval_seeds": args.tk_eval_seeds,
        } if "target_knockback" in games else None,
        "per_game": per_game_summary,
        "grad_norms": [
            h.get("update_stats", {}).get("grad_norm") for h in history if "update_stats" in h
        ],
        "total_wall_s": round(time.time() - t_run, 1),
        "adapter_volume": VOLUME_NAME,
        "history": history,
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("    === BEFORE vs AFTER (per game; this trainer's own policy) ===", flush=True)
    for game in games:
        s = per_game_summary[game]
        print(f"    [{game}] before={s['before_reward_mean']:+.4f}  "
              f"after={s['after_reward_mean']:+.4f}  "
              f"delta={s['delta_before_after']:+.4f}  "
              f"(std {s['before_reward_std']:.4f}->{s['after_reward_std']:.4f}, "
              f"valid {s['before_valid']}->{s['after_valid']})", flush=True)
    print(f"    grad_norms: {[round(g,3) for g in summary['grad_norms'] if g is not None]}", flush=True)
    print(f"\n    wrote {out_path}", flush=True)
    print(f"    adapters on Modal Volume '{VOLUME_NAME}' "
          f"(update1..update{args.updates}); fetch with: modal volume get {VOLUME_NAME} <tag>", flush=True)
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
    p.add_argument("--lora-r", dest="lora_r", type=int, default=16)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.1,
                   help="sampling temperature (1.1 was the variance-fixed single-game setting)")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=512)
    # Ring-Out reward config — defaults mirror the dry-loop / bridge nested reward.
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="Ring-Out PPO seeds per reward")
    p.add_argument("--episodes", type=int, default=1000, help="Ring-Out PPO episodes per seed")
    p.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=50)
    # Target-Knockback reward config — defaults mirror nested_reward_tk.
    p.add_argument("--tk-seeds", dest="tk_seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="Target-Knockback PPO seeds per reward")
    p.add_argument("--tk-episodes", dest="tk_episodes", type=int, default=2000,
                   help="TK PPO episodes per seed (TK's validated budget)")
    p.add_argument("--tk-eval-seeds", dest="tk_eval_seeds", type=int, default=50)
    p.add_argument("--gpu-smoke", dest="gpu_smoke", action="store_true",
                   help="GPU-only check: load+sample(both games)+fake-update+save, NO reward compute")
    args = p.parse_args(argv)

    if not (os.getenv("FIREWORKS_API_KEY") or "").strip():
        print("NOTE: FIREWORKS_API_KEY not set — the nested reward path does not need it; "
              "source .env to be safe.", file=sys.stderr)

    if args.gpu_smoke:
        return run_gpu_smoke(args)
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
