"""training/train_teacher_modal.py — CONTINGENCY: self-contained Modal Teacher-RFT trainer.

Fireworks RFT is entitlement-gated for >=4B models on this account (only qwen3-0p6b
trains there). This file is the fallback: it does the ACTUAL LoRA weight updates on
Qwen ourselves, on a Modal GPU, using the EXACT same nested-RL Teacher reward the
Fireworks path uses — so we can train a real 4B Teacher even though Fireworks won't.

What it reuses VERBATIM (does NOT rebuild):
  * The reward: ``output.nested_reward.teacher_reward`` (the 3-PPO-seed held-out
    transfer reward) fanned out on the deployed ``crucible-player`` Modal app. Same
    formula, same 15-arena broad held-out grid, same config as ``modal_ep_bridge``'s
    ``nested_reward_scorer``.
  * The parse/validate path: ``training.hud_teacher_env.parse_teacher_params`` /
    ``params_to_curriculum`` / ``fighter_geometry_override`` (strips ``<think>``,
    strict JSON, clamps to ``FIGHTER_BOUNDS``, maps to the FIGHTER schema via
    ``contracts``). Invalid JSON -> reward 0 (teaches format too).
  * The Teacher prompt: ``training.rft_evaluator.RING_OUT_PROMPT``.
  * The base-vs-trained yardstick: the broad held-out population, scored with the
    same ``teacher_reward``. Base Qwen baseline ~= 0.166 (output/baseline_qwen.json).

The algorithm is NOT multi-step PPO on Qwen. The reward is ONE scalar per generated
arena-JSON, so the Teacher update is a single-step bandit policy gradient — GRPO /
REINFORCE, exactly what Fireworks RFT does in its outer loop:

  1. Sample G completions from the current Qwen+LoRA for ``RING_OUT_PROMPT``.
  2. Parse+validate each (invalid -> reward 0).
  3. Score each VALID completion with the existing Modal nested reward, fanned out
     in parallel across the G samples (each reward is ~2 min).
  4. GRPO: advantages A_i = (r_i - mean(r)) / (std(r)+eps); loss
     -(A_i * sum_logprob(completion_i)).mean(); backprop into LoRA params only.
  5. Save the LoRA adapter (Modal Volume) after each update.
  6. Repeat for 1-2 updates (cost stays tiny).
  7. Compare the policy's reward BEFORE vs AFTER the updates, vs the 0.166 baseline.

ARCHITECTURE (why split host <-> GPU):
  * The GPU work (load Qwen, sample, recompute logprobs, LoRA gradient step) runs in
    a warm Modal ``@app.cls`` GPU container that holds the model + optimizer across
    ``sample``/``update`` calls and persists the adapter to a Modal Volume.
  * The slow REMOTE reward (``teacher_reward`` -> ``crucible-player.train_player_transfer.map``)
    is dispatched FROM THE HOST — the exact proven path ``dry_loop.py`` /
    ``eval_base_vs_trained.py`` already use. The host has Modal auth + the local
    source; it does NOT need a GPU or transformers/peft. The GPU container does NOT
    re-dispatch PPO. Clean separation, no nested Modal-from-Modal map.

RUN (from repo root; .env sourced; ~/.modal.toml profile njlee007):

    source .env
    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    .venv/bin/python3 training/train_teacher_modal.py            # full: 1 update, G=4
    .venv/bin/python3 training/train_teacher_modal.py --updates 2 --group-size 6
    .venv/bin/python3 training/train_teacher_modal.py --gpu-smoke # GPU-only: load+sample+fake-update, no reward
    .venv/bin/python3 training/train_teacher_modal.py --model Qwen/Qwen3-1.7B  # smaller/faster

This file is contingency infra: it does NOT touch the Fireworks launcher, the deployed
``crucible-ep-bridge``, or the OWA daemon. It only IMPORTS the existing reward and
CALLS the already-deployed ``crucible-player`` worker (normal shared use).
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
# Shared constants (mirror the existing reward/eval path — do NOT diverge)
# ---------------------------------------------------------------------------

# The Teacher prompt + FIGHTER schema bounds. Re-declared here as the single source
# the GPU container is handed at call time (it only needs the strings — its image
# never imports hud/eval-protocol). Kept char-for-char identical to
# training/rft_evaluator.RING_OUT_PROMPT and training/hud_teacher_env.FIGHTER_BOUNDS.
RING_OUT_PROMPT = (
    "You design RL training curricula for a Stage-1 2D platform fighter "
    "(the Ring-Out duel: two fighters, knock the opponent off the platform). "
    "Return ONE JSON object with EXACTLY these keys, each a number in range: "
    "difficulty [0.0,1.0], platform_width [8.0,30.0], gravity [0.2,1.2], "
    "knockback [0.5,6.0], spawn_gap [1.0,12.0]. Choose a LEARNABLE arena whose "
    "trained Player transfers broadly. Output strict JSON only; no prose, no code."
)

FIGHTER_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "gravity": (0.2, 1.2),
    "knockback": (0.5, 6.0),
    "spawn_gap": (1.0, 12.0),
}

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
    # One warm container holds the model + optimizer + LoRA state across every
    # sample/update call of a run. Single container = single coherent policy.
    # IMPORTANT: ``update`` recomputes logprobs over the sequences ``sample`` cached
    # on the instance, and the host's REMOTE reward scoring sits BETWEEN those two
    # calls (~3-5 min for a parallel group). The scaledown window must comfortably
    # exceed that gap or the container recycles and ``_last_sequences`` is lost, so
    # it is set well above the worst-case scoring wall.
    min_containers=0,
    max_containers=1,
    scaledown_window=60 * 30,
)
class TeacherTrainer:
    """Warm GPU container: load Qwen+LoRA once, then sample / GRPO-update on demand.

    State persists across method calls in ONE container: the LoRA-wrapped model, the
    tokenizer, and the Adam optimizer over the LoRA params. ``sample`` returns G
    decoded completions to the host; the host scores them with the remote reward and
    calls ``update`` with the rewards to take ONE GRPO step. ``save_adapter`` writes
    the adapter to the Volume.
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
        # Build the prompt once: chat template, thinking suppressed (Qwen3 supports
        # enable_thinking=False so completions are JSON-first, not a <think> wall).
        self.prompt_text = self._render_prompt()
        self.prompt_ids = self.tokenizer(self.prompt_text, return_tensors="pt").input_ids.to("cuda:0")
        self.prompt_len = self.prompt_ids.shape[1]
        print(
            f"[gpu] ready in {time.time()-t0:.1f}s | trainable LoRA params "
            f"{n_train:,} / {n_total:,} ({100*n_train/n_total:.3f}%) | "
            f"prompt_len={self.prompt_len}",
            flush=True,
        )

    def _render_prompt(self) -> str:
        messages = [{"role": "user", "content": RING_OUT_PROMPT}]
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

    def _completion_logprobs(self, sequences):
        """Sum of per-token logprobs of the COMPLETION tokens under the current model.

        Keeps the autograd graph (fresh forward, NOT generate's scores). Shift: a
        forward over ``sequences`` gives logits[i] = distribution for token i+1, so
        we align logits[:, :-1] with targets sequences[:, 1:]. After the left-shift,
        target position j corresponds to original position j+1, so the completion
        region (original index >= prompt_len) starts at shifted index prompt_len-1.
        """
        torch = self.torch
        logits = self.model(sequences).logits[:, :-1, :]               # (B, L-1, V)
        targets = sequences[:, 1:]                                     # (B, L-1)
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        # Completion mask in the shifted frame: keep targets that are completion
        # tokens (original index >= prompt_len) AND not padding.
        idx = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
        comp_mask = idx >= (self.prompt_len - 1)
        pad_mask = targets != self.tokenizer.pad_token_id
        mask = (comp_mask & pad_mask).to(tok_logp.dtype)
        return (tok_logp * mask).sum(dim=1)                            # (B,)

    @modal.method()
    def sample(self, group_size: int, temperature: float = 0.9, top_p: float = 0.95,
               max_new_tokens: int = 512, seed: int | None = None) -> list[str]:
        """Sample ``group_size`` completions from the CURRENT policy; return decoded text.

        The host parses/validates/scores them and hands the rewards back to ``update``.
        We cache the sampled token sequences on the instance so ``update`` recomputes
        logprobs over the EXACT same sequences (no resample drift between the action
        that earned the reward and the gradient).
        """
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                self.prompt_ids,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=group_size,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        self.model.train()
        self._last_sequences = out.sequences  # (G, prompt_len + new)
        completions = []
        for row in out.sequences:
            comp_ids = row[self.prompt_len:]
            completions.append(self.tokenizer.decode(comp_ids, skip_special_tokens=True))
        return completions

    @modal.method()
    def update(self, rewards: list[float]) -> dict:
        """One GRPO step over the LAST sampled group, given its per-sample rewards.

        advantages = (r - mean) / (std + eps); loss = -(A * sum_logprob).mean();
        backprop into LoRA params only. Returns scalars for the host's run log.
        """
        torch = self.torch
        assert hasattr(self, "_last_sequences"), "call sample() before update()"
        seqs = self._last_sequences
        assert len(rewards) == seqs.shape[0], f"{len(rewards)} rewards vs {seqs.shape[0]} samples"
        r = torch.tensor(rewards, dtype=torch.float32, device="cuda:0")
        adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)

        self.model.train()
        self.optimizer.zero_grad()
        comp_logp = self._completion_logprobs(seqs)                   # (G,) with grad
        loss = -(adv * comp_logp).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm),
            "reward_mean": float(r.mean().cpu()),
            "reward_std": float(r.std(unbiased=False).cpu()),
            "adv": [round(float(a), 4) for a in adv.cpu().tolist()],
            "mean_completion_logprob": float(comp_logp.detach().mean().cpu()),
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
        """Self-contained GPU check: load + sample + ONE fake-reward GRPO step + save.

        Proves the GPU half end to end (load Qwen, generate, recompute logprobs,
        backward, optimizer.step, save) WITHOUT spending any reward compute. Fake
        rewards are the sample index, so advantages are non-degenerate and the step
        is real.
        """
        comps = self.sample.local(group_size=group_size, max_new_tokens=128, seed=0)
        fake_rewards = [float(i) for i in range(group_size)]
        stats = self.update.local(fake_rewards)
        path = self.save_adapter.local("gpu_smoke")
        return {
            "completions_preview": [c[:160] for c in comps],
            "update_stats": stats,
            "adapter_path": path,
        }


# ---------------------------------------------------------------------------
# HOST side: parse/validate + REMOTE reward dispatch (the proven path)
# ---------------------------------------------------------------------------


def _score_completion(answer: str, *, seeds: tuple[int, ...], episodes: int,
                      eval_seeds: int, held_out: list[dict]) -> tuple[float, dict | None, str]:
    """Score ONE Teacher completion with the EXISTING nested reward path.

    Reuses the bridge's grading semantics: parse (<think>-strip + strict JSON +
    clamp + FIGHTER-schema map), then ``teacher_reward`` (3 PPO seeds fanned out on
    the deployed crucible-player). Invalid JSON / schema -> reward 0.0 (teaches
    format). Returns (reward, params_or_None, status).
    """
    from output.nested_reward import teacher_reward
    from training.hud_teacher_env import (
        fighter_geometry_override,
        params_to_curriculum,
        parse_teacher_params,
    )

    try:
        params = parse_teacher_params(answer, FIGHTER_BOUNDS)
    except Exception as exc:  # invalid JSON / missing key / non-object
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


def _score_group(completions: list[str], *, seeds, episodes, eval_seeds, held_out,
                 max_workers: int) -> list[tuple[float, dict | None, str]]:
    """Score a whole group, fanning the (slow, ~2min each) rewards out in parallel.

    Each ``_score_completion`` itself fans 3 PPO seeds onto crucible-player; running
    the G completions concurrently (threads, since the work is a remote Modal map +
    network wait) means the group's wall-clock ~= one completion, not G of them.
    """
    from concurrent.futures import ThreadPoolExecutor

    results: list = [None] * len(completions)

    def _one(i: int):
        return i, _score_completion(
            completions[i], seeds=seeds, episodes=episodes,
            eval_seeds=eval_seeds, held_out=held_out,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in ex.map(_one, range(len(completions))):
            results[i] = res
    return results


def _build_held_out() -> list[dict]:
    """The BROAD 15-arena held-out population — the same yardstick the reward uses."""
    from output.broad_eval_set import build_broad_eval_arenas, payload_arenas

    return payload_arenas(build_broad_eval_arenas(grid="full"))


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_gpu_smoke(args) -> int:
    """GPU-only smoke: no reward compute. Proves load+sample+update+save on the GPU."""
    print("=== GPU SMOKE (no reward) ===", flush=True)
    with app.run():
        trainer = TeacherTrainer(model_name=args.model, lr_str=str(args.lr),
                                 lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        out = trainer.gpu_smoke.remote(group_size=args.group_size)
    print(json.dumps(out, indent=2))
    return 0


def run_train(args) -> int:
    seeds = tuple(args.seeds)
    held_out = _build_held_out()
    out_dir = _REPO_ROOT / "output" / "contingency_rft"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== CONTINGENCY Teacher-RFT (Modal LoRA GRPO) ===", flush=True)
    print(f"    model={args.model}  gpu={DEFAULT_GPU}", flush=True)
    print(f"    updates={args.updates}  group_size={args.group_size}  lr={args.lr}", flush=True)
    print(f"    reward: nested teacher_reward | seeds={seeds} episodes={args.episodes} "
          f"eval_seeds={args.eval_seeds} | held_out={len(held_out)} arenas", flush=True)
    print("    base Qwen baseline (output/baseline_qwen.json) ~= 0.166", flush=True)
    print(flush=True)

    history: list[dict] = []
    t_run = time.time()

    with app.run():
        trainer = TeacherTrainer(model_name=args.model, lr_str=str(args.lr),
                                 lora_r=args.lora_r, lora_alpha=args.lora_alpha)

        # Steps 0..N. Step 0 is the pure "before" measurement of the BASE policy (no
        # gradient): the same kind of group, scored before any update. Steps 1..N each
        # sample, score, take ONE GRPO step, and save the adapter.
        for step in range(args.updates + 1):
            tag = "base" if step == 0 else f"update{step}"
            print(f"[{tag}] sampling {args.group_size} completions ...", flush=True)
            completions = trainer.sample.remote(
                group_size=args.group_size, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens, seed=1000 + step,
            )
            t_score = time.time()
            print(f"[{tag}] scoring group via nested reward (parallel) ...", flush=True)
            scored = _score_group(
                completions, seeds=seeds, episodes=args.episodes,
                eval_seeds=args.eval_seeds, held_out=held_out,
                max_workers=args.group_size,
            )
            rewards = [s[0] for s in scored]
            statuses = [s[2] for s in scored]
            n_valid = sum(1 for st in statuses if st == "ok")
            score_wall = time.time() - t_score
            print(f"[{tag}] rewards={[round(r,4) for r in rewards]}  "
                  f"valid={n_valid}/{len(rewards)}  mean={_mean(rewards):.4f}  "
                  f"({score_wall:.0f}s)", flush=True)

            rec = {
                "step": step, "tag": tag,
                "rewards": [round(r, 4) for r in rewards],
                "statuses": statuses,
                "n_valid": n_valid,
                "reward_mean": round(_mean(rewards), 4),
                "params": [s[1] for s in scored],
                "completions_preview": [c[:200] for c in completions],
                "score_wall_s": round(score_wall, 1),
            }

            if step == 0:
                print(f"[{tag}] (no update — this is the before-training baseline)\n", flush=True)
            else:
                stats = trainer.update.remote(rewards)
                rec["update_stats"] = stats
                adapter_path = trainer.save_adapter.remote(tag)
                rec["adapter_path"] = adapter_path
                print(f"[{tag}] GRPO step: loss={stats['loss']:.4f} "
                      f"grad_norm={stats['grad_norm']:.3f} adv={stats['adv']} "
                      f"-> saved {adapter_path}\n", flush=True)

            history.append(rec)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # --- final verdict: before (update 0 group) vs after (last update group) ---
    base_mean = history[0]["reward_mean"]
    final_mean = history[-1]["reward_mean"]
    summary = {
        "experiment": "contingency_teacher_rft_modal_lora_grpo",
        "model": args.model,
        "gpu": DEFAULT_GPU,
        "updates": args.updates,
        "group_size": args.group_size,
        "reward_seeds": list(seeds),
        "ppo_episodes": args.episodes,
        "eval_seeds": args.eval_seeds,
        "n_held_out_arenas": len(held_out),
        "published_base_baseline": 0.166,
        "before_policy_reward_mean": base_mean,
        "after_policy_reward_mean": final_mean,
        "delta_before_after": round(final_mean - base_mean, 4),
        "delta_vs_published_baseline": round(final_mean - 0.166, 4),
        "total_wall_s": round(time.time() - t_run, 1),
        "adapter_volume": VOLUME_NAME,
        "history": history,
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("    === BEFORE vs AFTER (this trainer's own policy) ===", flush=True)
    print(f"    before (base policy group) reward mean = {base_mean:+.4f}", flush=True)
    print(f"    after  (update {args.updates} group)  reward mean = {final_mean:+.4f}", flush=True)
    print(f"    delta before->after                  = {final_mean - base_mean:+.4f}", flush=True)
    print(f"    after vs published 0.166 baseline    = {final_mean - 0.166:+.4f}", flush=True)
    print(f"\n    wrote {out_path}", flush=True)
    print(f"    adapters on Modal Volume '{VOLUME_NAME}' "
          f"(update1..update{args.updates}); fetch with: modal volume get {VOLUME_NAME} <tag>", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Contingency Modal Teacher-RFT (LoRA GRPO on Qwen).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="base Teacher model (default Qwen/Qwen3-4B; fall back to "
                        "Qwen/Qwen3-1.7B or Qwen/Qwen2.5-1.5B-Instruct if too heavy)")
    p.add_argument("--updates", type=int, default=1, help="number of GRPO updates (keep 1-2; cost)")
    p.add_argument("--group-size", dest="group_size", type=int, default=4,
                   help="G completions sampled per update (4-8)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", dest="lora_r", type=int, default=16)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=512)
    # Reward config — defaults mirror the dry-loop / bridge nested reward.
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="PPO seeds per reward (the nested reward's inner seeds)")
    p.add_argument("--episodes", type=int, default=1000, help="PPO episodes per seed")
    p.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=50)
    p.add_argument("--gpu-smoke", dest="gpu_smoke", action="store_true",
                   help="GPU-only check: load+sample+fake-update+save, NO reward compute")
    args = p.parse_args(argv)

    if not (os.getenv("FIREWORKS_API_KEY") or "").strip():
        print("NOTE: FIREWORKS_API_KEY not set — the nested reward path does not need it; "
              "source .env to be safe.", file=sys.stderr)

    if args.gpu_smoke:
        return run_gpu_smoke(args)
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
