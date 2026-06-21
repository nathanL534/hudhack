# Sensei (Teacher) Training + Stage-6 Eval — Operational RUNBOOK

> Goal: a NEW collaborator trains their own Teacher (Sensei) GRPO runs on THEIR
> own Modal account using our exact workflow, then runs the Stage-6 head-to-head
> decider — so we can **pool training-run data across people**.
>
> Every command, flag, default, env var, and path below is grounded in the actual
> code. Key sources:
> `training/train_teacher_modal.py`, `modal_player.py`, `modal_player_tk.py`,
> `training/teacher_gen_modal.py`, `output/eval_base_vs_trained.py`,
> `output/nested_reward.py`, `output/nested_reward_tk.py`,
> `output/stage6/handles.py`, `output/stage6/config.py`.

---

## 1. What this is

This is **nested RL**. An LLM **Teacher** (`Qwen/Qwen3-4B` + a LoRA adapter) is
**GRPO-trained to generate RL arena curricula** for 2 games: **Ring-Out fighter**
(5-key schema) and **Target-Knockback** (7-key schema). The **reward** for a
generated arena = how much a *fresh PPO Student* improves on a **FIXED held-out
reference set** after training only on that Teacher's arena (held-out *transfer*,
not in-distribution — trivial arenas score low). The **Stage-6 verdict** is a
direct head-to-head: a Student trained by the **trained**-Teacher must beat a
Student trained by the **base**-Teacher, side-swapped, CI lower bound > 0, plus a
hidden **KOTH cross-game transfer probe** (anti-overfit gate).

Two anti-bias details that matter:
- **Per-game advantages.** Each GRPO step samples G completions from BOTH game
  prompts off the SAME Qwen+LoRA policy, scores them, and normalizes advantages
  *within each game* (`A_i = (r_i - mean_game)/(std_game + eps)`) — never pooling
  rewards across games (different reward scales). One combined LoRA update per step.
- **Signed reward.** TK's per-game reward is the RAW SIGNED held-out improvement
  (can be negative); GRPO's advantage norm is shift-invariant, so a negative
  "this arena hurt transfer" sample still pushes the Teacher away.

---

## 2. Architecture map

Four distinct Modal pieces. The trainer is **ephemeral** (`with app.run()`); the
reward workers and the generator are **deployed** (long-lived apps you call into).

```
                         YOUR HOST (repo root, .venv, ~/.modal.toml)
                         python3 training/train_teacher_modal.py ...
                                   |
           sample / update_multi   |   (per-game advantages computed HOST-side)
           save_adapter            v
   +-------------------------------------------------+
   |  EPHEMERAL TRAINER APP  (1x A100 GPU)           |   app: crucible-teacher-trainer[-<run_id>]
   |  class TeacherTrainer  @app.cls(gpu=...)        |   started by `with app.run():` each launch
   |  - loads Qwen3-4B + LoRA, holds optimizer +     |   max_containers=1 (ONE warm GPU/run)
   |    both games' last-sampled sequences in memory |   writes adapter -> Volume crucible-teacher-lora
   +-------------------------------------------------+        /adapters/<run_id>/update{1..N}
            |  HOST fans the slow rewards out in parallel (threads)  |
            v                                                        v
   +-------------------------+                      +-----------------------------+
   |  crucible-player        |  (DEPLOYED)          |  crucible-player-tk         | (DEPLOYED)
   |  train_player_transfer  |  Ring-Out reward     |  train_tk_player            |  TK held-out reward
   |  (3 PPO seeds -> held-  |  CPU PPO containers  |  status=="ppo_tk",          |  CPU PPO containers
   |   out improvement)      |                      |  signed mean_improvement    |
   +-------------------------+                      +-----------------------------+

   STAGE-6 EVAL (separate, after training):
   python3 output/eval_base_vs_trained.py --base modal:base --trained modal:<run_id>/<updateN> ...
            |  generates arenas from BASE + TRAINED teachers              |  PPO students + matches
            v                                                            v
   +-------------------------+                      +-----------------------------+
   |  crucible-teacher-gen   |  (DEPLOYED)          |  crucible-player            | (DEPLOYED, reused)
   |  class TeacherGenerator |  base & base+LoRA    |  train_student_policy       |  fresh Students
   |  adapter_tag -> /adapters/<tag>               |  head_to_head_match         |  side-swapped fights
   +-------------------------+                      +-----------------------------+
```

Who talks to whom:
- **Trainer → reward workers.** `train_teacher_modal.py` imports
  `output.nested_reward.teacher_reward` (which does `modal.Function.from_name(
  "crucible-player","train_player_transfer").map(...)`) and
  `output.nested_reward_tk.tk_teacher_reward` (→ `crucible-player-tk`/`train_tk_player`).
  The trainer NEVER deploys these — it calls the already-deployed apps.
- **Trainer → Volume.** Adapters persist to Volume `crucible-teacher-lora` under
  `/adapters/<run_id>/updateN`.
- **Eval → crucible-teacher-gen.** Stage-6 resolves `modal:<tag>` via
  `modal.Cls.from_name("crucible-teacher-gen","TeacherGenerator")` — so
  `crucible-teacher-gen` MUST be deployed.
- **Eval → crucible-player.** Student training + head-to-head matches fan out to
  the same deployed `crucible-player` app (`train_student_policy`,
  `head_to_head_match`).

---

## 3. One-time setup

```bash
# 0) From repo root, with the project venv. .env is gitignored; copy the template.
cp .env.example .env        # fill nothing mandatory for training (see note below)

# 1) Modal account + auth (writes ~/.modal.toml). Browser SSO.
pip install modal
modal token new             # or: modal setup

# 2) Python deps (host side). The GPU/reward IMAGES are built by Modal from the
#    pip lists inside each file; the HOST just needs modal + dotenv + the repo.
pip install -r requirements.txt        # includes modal, python-dotenv, sb3, etc.
#    (.venv/bin/python is the project interpreter in the code comments.)
```

> **.env / Modal token note (grounded in code).** Both `train_teacher_modal.py`
> and `eval_base_vs_trained.py` load `.env`, then **delete EMPTY**
> `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` so Modal falls back to your
> `~/.modal.toml` profile. So either leave those two blank in `.env` (use the
> toml profile) **or** set real `ak-...`/`as-...` values. `FIREWORKS_API_KEY` is
> **NOT needed** for the nested-reward path (the trainer prints a NOTE if missing
> and continues).

**Deploy the three long-lived apps once (and after any code change to them):**

```bash
# Ring-Out reward worker (app name: crucible-player)
modal deploy modal_player.py

# Target-Knockback reward worker (app name: crucible-player-tk)
modal deploy modal_player_tk.py

# Teacher generator for Stage-6 (app name: crucible-teacher-gen)
#   REQUIRED for eval — Stage-6 does modal.Cls.from_name("crucible-teacher-gen", ...)
modal deploy training/teacher_gen_modal.py
```

> The **trainer app itself is NOT deployed**. `train_teacher_modal.py` runs it
> ephemerally with `with app.run():` on every launch. The persistent **Volumes**
> (`crucible-teacher-lora` for adapters, `crucible-hf-cache` for the ~8GB Qwen
> weights) are auto-created (`create_if_missing=True`) on first trainer run.

Verify: `modal app list` should show `crucible-player`, `crucible-player-tk`,
`crucible-teacher-gen` deployed.

---

## 4. QUICKSTART — launch ONE variant end-to-end

```bash
# from repo root
set -a && source .env && set +a
unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET     # use ~/.modal.toml profile

CRUCIBLE_RUN_ID=myrun1 CRUCIBLE_GPU=A100-80GB PYTHONPATH=. \
python3 training/train_teacher_modal.py \
  --run-id myrun1 \
  --gpu A100-80GB \
  --updates 14 \
  --group-size 12 \
  --lora-r 32 \
  --lora-alpha 64 \
  --temperature 1.4 \
  --seed-base 101
```

You MUST pass `--run-id` equal to `CRUCIBLE_RUN_ID` and `--gpu` equal to
`CRUCIBLE_GPU` — the code asserts equality and exits 2 otherwise (see §6).

### Every flag (from the argparse + defaults)

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-4B` | Base Teacher. Fall back to `Qwen/Qwen3-1.7B` if GPU is tight. |
| `--games` | `ring_out target_knockback` (both) | Games trained per COMBINED step. One combined GRPO update over all listed games. |
| `--updates` | `2` | Number of combined GRPO updates. Adapter `update1..updateN` saved. (Use 14 for a full run.) |
| `--group-size` | `5` | G completions sampled **per game** per update. **G=12 needs A100-80GB** (see §6). |
| `--lr` | `1e-4` | Adam LR over LoRA params. |
| `--lora-r` | `32` | LoRA rank (adapter capacity). |
| `--lora-alpha` | `64` | LoRA alpha. `alpha=2*r` keeps scaling at 2.0. |
| `--temperature` | `1.1` | Sampling temperature for completions. Higher = more arena diversity. |
| `--max-new-tokens` | `512` | Completion length cap. |
| `--seeds` | `3` | Ring-Out: **count** of PPO seeds per reward (rotated per update). |
| `--episodes` | `1000` | Ring-Out PPO episodes per seed. |
| `--eval-seeds` | `50` | Ring-Out held-out eval seeds. |
| `--tk-seeds` | `3` | TK: count of PPO seeds per reward (rotated). |
| `--tk-episodes` | `2000` | TK PPO episodes per seed (TK's validated budget). |
| `--tk-eval-seeds` | `50` | TK held-out eval seeds. |
| `--seed-rotation` / `--no-seed-rotation` | rotation **ON** | Rotate training PPO seeds per update (anti-overfit). `--no-seed-rotation` reuses update-1's block (legacy). |
| `--seed-base` | `10000` | Base of the HIGH-range training-seed space (rotated blocks live here; kept disjoint from validation pools). |
| `--val-seeds` | `1 2 3` | Ring-Out FIXED validation seeds (final-adapter scoring only). |
| `--tk-val-seeds` | `1 2 3` | TK FIXED validation seeds. |
| `--run-id` | `$CRUCIBLE_RUN_ID` | Per-run isolation id. **Must equal** `CRUCIBLE_RUN_ID`. |
| `--gpu` | `$CRUCIBLE_GPU` or `A100-40GB` | Trainer GPU. **Must equal** `CRUCIBLE_GPU`. |
| `--gpu-smoke` | off | GPU-only: load+sample(both games)+fake-update+save, NO reward compute. |

> `--seed-base` in the QUICKSTART is `101` (matches our live runs). With seed
> rotation ON, the validation pool defaults (`1 2 3`) stay disjoint from training
> blocks starting at 101 — the code asserts this disjointness up front and aborts
> loudly if it ever overlaps.

**What a run produces:**
- Adapters on Volume `crucible-teacher-lora` under `/adapters/<run_id>/update{1..N}`.
- `output/contingency_<run_id>/history.json` (per-update logs, written every step).
- `output/contingency_<run_id>/summary.json` (headline = TRUE final-adapter reward
  on reserved validation seeds).

---

## 5. The variant-sweep idea (how we "get more data")

**Everyone runs DIFFERENT configs under DIFFERENT `CRUCIBLE_RUN_ID`s, and results
pool automatically** because each run writes to its OWN:
- adapter subdir `/adapters/<run_id>/updateN` (shared Volume, namespaced),
- output dir `output/contingency_<run_id>/`,
- Modal app `crucible-teacher-trainer-<run_id>` (own warm GPU container).

So N people (or N concurrent variants) never collide. Vary `--lora-r`,
`--temperature`, `--seed-base` across runs to cover the config space.

**Example A/B/C/D matrix** (the kind of sweep we ran — same `Qwen3-4B`, both
games, A100-80GB, `--updates 14 --group-size 12`):

| run_id | `--lora-r` | `--lora-alpha` | `--temperature` | `--seed-base` |
|---|---|---|---|---|
| `runA` | 32 | 64 | 1.1 | 101 |
| `runB` | 32 | 64 | 1.4 | 101 |
| `runC` | 16 | 32 | 1.1 | 201 |
| `runD` | 64 | 128 | 1.4 | 301 |

Each launched as its own line, e.g.:

```bash
CRUCIBLE_RUN_ID=runB CRUCIBLE_GPU=A100-80GB PYTHONPATH=. \
python3 training/train_teacher_modal.py --run-id runB --gpu A100-80GB \
  --updates 14 --group-size 12 --lora-r 32 --lora-alpha 64 \
  --temperature 1.4 --seed-base 101
```

> Keep `--lora-alpha = 2 * --lora-r` to hold the LoRA scaling (`alpha/r`) at 2.0
> across variants — then you're varying *capacity*, not effective update magnitude.
> Use a DISTINCT `--seed-base` per variant when you want fully disjoint PPO
> populations across variants (cleaner pooled comparison).

---

## 6. CRITICAL GOTCHAS — Lessons / Don't get burned

These are real failures we already paid for. Read them before launching.

### GPU: G=12 OOMs on A100-40GB → use A100-80GB
The `group_size=12` logprob forward materializes a large transient `(B, L, V)`
logits tensor per update; a fragmented 40GB heap OOM'd by ~116MB. Mitigations
already in the code: `update_multi` does **per-game gradient accumulation**
(backward one game at a time, peak VRAM = one game not both) and the image sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Even with those, **run
G=12 on `A100-80GB`** (`CRUCIBLE_GPU=A100-80GB --gpu A100-80GB`). The trainer's
own default is `A100-40GB`, which only fits smaller G.

### Container limit: ~73 concurrent containers PER variant
Each variant fans rewards out massively: **12 arenas × 3 PPO seeds × 2 games = 72
reward jobs in parallel + 1 trainer ≈ 73 containers**. N concurrent variants need
**~N × 73** workspace containers (e.g. 4 variants → ~300). **Raise your Modal
workspace container limit accordingly.** GPUs are NOT the bottleneck — the
**CPU reward containers** (`crucible-player` / `crucible-player-tk` PPO jobs) are.
If you cap too low, reward jobs queue and a "step" stalls.

### Run isolation: set BOTH `CRUCIBLE_RUN_ID` AND `--run-id` (they must be equal)
The Modal App name and `@app.cls(gpu=...)` are fixed at **import time** from the
env vars `CRUCIBLE_RUN_ID` / `CRUCIBLE_GPU` — *before* argparse runs. The flags
exist for the audit record, so the code **asserts `--run-id == CRUCIBLE_RUN_ID`
(and `--gpu == CRUCIBLE_GPU`) and exits 2** if they disagree. Critically, the env
var is also baked into the GPU image (`.env({"CRUCIBLE_RUN_ID": RUN_ID})`) so the
namespaced `/adapters/<run_id>` path reaches **inside the container** where
`save_adapter` runs — without it, an earlier bug had every run write flat
`/adapters/<tag>` and **clobber each other's checkpoints**. Always launch with the
env var prefix.

### Checkpoints: every update saves + commits, so a run is cut-short-safe
Every update calls `save_adapter("updateN")` → `os.makedirs(/adapters/<run_id>/
updateN)` → `save_pretrained` → `adapter_volume.commit()`. So you can **kill a run
early** and any `updateN` already written is fully eval-able. The per-update
`history.json` is also rewritten every step.

---

## 7. Monitoring a run

```bash
# Per-update logs (rewritten every step): reward mean/std, advantages,
# arena_diversity, grad_norm (via update_stats), completions_preview, train_seeds.
cat output/contingency_<run_id>/history.json | python3 -m json.tool | less

# Quick pulse: per-game reward_mean per step
python3 -c "import json;h=json.load(open('output/contingency_<run_id>/history.json'));[print(r['tag'],{g:r['per_game'][g]['reward_mean'] for g in r['per_game']}) for r in h]"

# Checkpoints written so far for this run (each updateN is eval-able)
modal volume ls crucible-teacher-lora/<run_id>

# Is the trainer app live? (look for crucible-teacher-trainer-<run_id>)
modal app list
```

`history.json` record shape (grounded): each record has
`step, tag, per_game{<game>:{rewards, statuses, n_valid, reward_mean, reward_std,
advantages, params, completions_preview, arena_diversity}}, score_wall_s,
train_seeds`. Update steps additionally carry `update_stats{loss, grad_norm,
games, n_completions, mean_completion_logprob_by_game}`. Watch:
- `reward_mean` trending up across `update1..updateN`,
- `grad_norm` non-zero and not exploding (clipped at 1.0),
- `arena_diversity.difficulty_std > 0` (collapse check — ~0 means the Teacher
  collapsed to one arena).

---

## 8. Selecting the best checkpoint

**Pick by the reserved-VALIDATION score, NOT by the Stage-6 test** (that would
contaminate the test). The trainer's `summary.json` `headline` /
`final_adapter_eval` is the TRUE final-adapter reward measured on the **reserved
validation seeds** (`--val-seeds` / `--tk-val-seeds`, default `1 2 3`), which are
asserted disjoint from every rotated training block.

Selection rule (noise-robust):
- Use a **moving average of validation reward across late updates**, not a raw
  argmax of a single noisy step. The per-update `history.json` rewards are
  *PRE-update* training-seed numbers (the policy is scored BEFORE that step's
  gradient) — do not treat them as the adapter's score.
- The seed pools that must stay **disjoint**: training-reward seeds (high range,
  rotated, from `--seed-base`) vs reserved validation seeds (`--val-seeds`) vs the
  Stage-6 held-out arenas (the eval's own grid). Never let them overlap, or the
  selection/test is circular.
- Once you've picked an `updateN` by validation, that becomes your `--trained`
  handle for Stage-6 (the *test*, run once).

---

## 9. Running the Stage-6 head-to-head

Pick the `updateN` you selected in §8 and run the decider. The `modal:<run_id>/
<updateN>` handle with a **slash just works**: the tag is passed verbatim to
`TeacherGenerator(adapter_tag=...)`, which does `os.path.join("/adapters", tag)`,
so `modal:runA/update14` → `/adapters/runA/update14` on the Volume.

```bash
# Make sure crucible-teacher-gen + crucible-player are deployed (see §3).

# FULL fighter decider: base vs your chosen checkpoint, with KOTH cross-game probe.
PYTHONPATH=. python3 output/eval_base_vs_trained.py \
  --game fighter \
  --backend modal \
  --base modal:base \
  --trained modal:<run_id>/<updateN> \
  --replicates 6 \
  --curriculum-arenas 4 \
  --match-seeds 16 \
  --h2h-grid full \
  --koth-cross-game
```

- **Use `--replicates >= 4`** (default `n_replicates=4`; CI is over replicates).
  We used 6 for the headline.
- `--koth-cross-game` is now effectively a no-op flag (the KOTH cross-game runs
  by default inside `run_full_eval` and feeds the train-game-overfit gate); pass
  it anyway for clarity. Use `--no-koth-cross-game` only for a cheap train-game-only
  run (the anti-gaming gate then SKIPs).
- Verdict = trained-Student beats base-Student head-to-head, **CI lower bound > 0**,
  AND anti-circularity / KOTH cross-game gates pass. The CLI **exits non-zero** if
  the real decider fails, so you can gate scripts on it.

**Always run the NULL test first** (base-vs-base must be ~0, no side bias):

```bash
PYTHONPATH=. python3 output/eval_base_vs_trained.py \
  --null-test --backend modal --base modal:base --no-secondary
# PASS requires |advantage| < 0.10 AND side_gap (|p0 - p1|) < 0.25.
```

If the null test fails, the evaluator favors a side — fix that before trusting any
trained-vs-base result.

**Game coverage caveat (this worktree).** Stage-6's game registry has
`fighter` = installed (headline), `koth` = `role="probe"` (cross-game only, never a
Teacher game), and **`target_knockback` resolves `installed=False`** here — so
`--game target_knockback` reports "not installed" in the eval CLI even though the
*trainer* trains TK. Run the Stage-6 head-to-head on **`--game fighter`** plus the
KOTH cross-game probe.

Outputs: `output/eval_base_vs_trained.json` (or `--out <path>`) + a regenerated
Markdown report/dashboard under `output/stage6_report/` (skip with `--no-report`).

---

## 10. Pooling results across people

To pool, each person shares **three artifacts per run**, keyed by their `run_id`:

1. **`run_id` + launch config** — the exact flags (`--lora-r`, `--lora-alpha`,
   `--temperature`, `--seed-base`, `--updates`, `--group-size`). The config matrix
   row from §5.
2. **`output/contingency_<run_id>/history.json`** (+ `summary.json`) — per-update
   reward curves, arena diversity, grad norms, and the headline final-adapter
   validation reward.
3. **The final Stage-6 JSON** — `output/eval_base_vs_trained.json` from §9 (the
   head-to-head verdict, CI, null-test pass, KOTH cross-game delta).

Compare across people by:
- validation `headline.reward_mean_overall` (from `summary.json`) for checkpoint
  quality, and
- Stage-6 `primary.mean_paired_advantage` + CI lower bound for the real verdict.

Because run_ids namespace the adapter Volume (`/adapters/<run_id>/`) and output
dirs (`output/contingency_<run_id>/`), multiple people's runs coexist without
collision — and any `modal:<run_id>/<updateN>` checkpoint can be re-evaluated by
anyone with the deployed `crucible-teacher-gen` + Volume access.

---

## 11. Collaborator configs (smaller-scale, complementary)

For a collaborator on their OWN Modal account — especially a **container-limited**
one — who wants their runs to **complement** ours, not duplicate them. Same
workflow as §4/§5, just sized down and offset in the config space.

### The container-budget rule (read this first)
Each variant fans out **`group_size × n_seeds × 2_games`** CPU reward containers
**+ 1 trainer** (§6). At OUR defaults (G=12, 3 seeds): `12×3×2 = 72 + 1 ≈ 73`/variant.
Four concurrent variants → `4×73 ≈ 292`, which **overran a 100-container cap**. So:

```
concurrent_variants × per_variant_containers  ≤  your_workspace_container_limit
```

Check YOUR limit in the Modal dashboard under **Workspace metrics → Total
containers / Limit**, then size `concurrent_variants` accordingly. **GPUs are NOT
the bottleneck** — the CPU reward containers (`crucible-player` /
`crucible-player-tk` PPO jobs) are.

### Smaller-scale knobs (grounded in the argparse, §4)
- **`--group-size 8`** (default 12 in our runs): `8×3×2 = 48` reward
  containers/variant (vs ~72). Lower fan-out AND lower VRAM.
- **VRAM:** G=12 is what forced **A100-80GB** — 40GB OOM'd at the GRPO
  `update_multi` even with the per-game gradient-accumulation that's already in
  the code (§6). G=8 has a smaller logits/activation footprint and is **expected**
  to fit **A100-40GB** — but **confirm with a smoke run first**, do not assume:
  ```bash
  CRUCIBLE_RUN_ID=fit CRUCIBLE_GPU=A100-40GB PYTHONPATH=. \
  python3 training/train_teacher_modal.py \
    --run-id fit --gpu A100-40GB --gpu-smoke --group-size 8
  ```
  If that OOMs, fall back to `--gpu A100-80GB` (and `CRUCIBLE_GPU=A100-80GB`).
- **`--seeds 2 --tk-seeds 2`** (default 3 each) to shrink further: `8×2×2 = 32`
  reward containers/variant — at the cost of a **noisier (less seed-averaged)**
  reward signal per arena.

### The complementary config matrix
Our live runs are `runA` (r32 / t1.4 / seed-base 101) and `runB` (r16 / t1.4 /
seed-base 101). Pick **different seeds and temperatures** so curricula don't
overlap, while still covering **both rank 16 and rank 32** for a comparable
ablation. Keep `--lora-alpha = 2 × --lora-r` (§5). Literal copy-paste:

```bash
set -a && source .env && set +a
unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET     # use ~/.modal.toml profile

# rank-16 variant — temp 1.3, seed-base 303
CRUCIBLE_RUN_ID=friend_r16 CRUCIBLE_GPU=A100-40GB PYTHONPATH=. \
python3 training/train_teacher_modal.py \
  --run-id friend_r16 --gpu A100-40GB \
  --updates 14 --group-size 8 \
  --lora-r 16 --lora-alpha 32 \
  --temperature 1.3 --seed-base 303

# rank-32 variant — temp 1.2, seed-base 404
CRUCIBLE_RUN_ID=friend_r32 CRUCIBLE_GPU=A100-40GB PYTHONPATH=. \
python3 training/train_teacher_modal.py \
  --run-id friend_r32 --gpu A100-40GB \
  --updates 14 --group-size 8 \
  --lora-r 32 --lora-alpha 64 \
  --temperature 1.2 --seed-base 404
```

| run_id | `--lora-r` | `--lora-alpha` | `--temperature` | `--seed-base` | `--group-size` |
|---|---|---|---|---|---|
| `friend_r16` | 16 | 32 | 1.3 | 303 | 8 |
| `friend_r32` | 32 | 64 | 1.2 | 404 | 8 |

Notes: `alpha = 2×rank`. Seeds **303 / 404** are disjoint from our 101/202;
temps **1.2 / 1.3** fill the gap between our 1.1 and 1.4. **Run 1–2 at a time**
depending on the container limit (`2 × 49 ≈ 98` fits a 100-cap; if tight, run one).
If the G=8 smoke run OOM'd on 40GB, swap both lines to `A100-80GB` (and
`CRUCIBLE_GPU=A100-80GB`).

### Why this complements ours
A different `--seed-base` → different sampled curricula → **statistically
independent** runs (not a re-roll of ours). Different temperatures → fills the
**curriculum-exploration sweep** between our 1.1 and 1.4. Rank 16 + 32 → **extends
the rank ablation**. Pool everyone's `output/contingency_<run_id>/history.json`
plus each run's best-checkpoint Stage-6 result (§10) and you get a richer
**rank × temp × seed** dataset than any single machine could produce.

---

## Minimum viable run (smoke-test your setup)

GPU-only, no reward compute, no Modal reward containers — proves load + sample
(both games) + fake GRPO update + save on YOUR Modal GPU:

```bash
CRUCIBLE_RUN_ID=smoke PYTHONPATH=. python3 training/train_teacher_modal.py --run-id smoke --gpu-smoke
```

If that prints completions + `update_stats` + an `adapter_path`, your Modal auth,
image, Volume, and GPU all work. For a tiny *real* end-to-end (1 reward-bearing
update, both games — needs `crucible-player` + `crucible-player-tk` deployed):

```bash
CRUCIBLE_RUN_ID=smoke PYTHONPATH=. python3 training/train_teacher_modal.py \
  --run-id smoke --updates 1 --group-size 2
```

---

## 12. Focused opponent-population + Student-capacity experiment

This is a **Stage-6 Student experiment**, not a new Teacher run. Do **not**
change Qwen, retrain the Teacher, or alter the nested Teacher reward for the
first pass. The problem being tested is downstream: Students trained against one
scripted opponent learned to stand still and punch, then drew against other
learned Students.

### Status / prerequisite

The opt-in opponent-league implementation exists on commit `fa5de18` and is not
part of the main branch until explicitly integrated:

```bash
git cherry-pick fa5de18
```

Then wire the Stage-6 Student payload builder:

```python
payload.update(cfg.student_training_options())
```

Redeploy **only** the Student worker after integration:

```bash
modal deploy modal_player.py
```

The feature is OFF by default. The existing single-opponent path and nested
Teacher-reward workers remain unchanged unless Stage 6 explicitly enables it.

### Population V1 — the negative control we already learned from

Do not use equal weights across aggressive + turtle + random. That mix produced
~80% draws: turtle taught survival/passivity, random supplied weak noisy lessons,
and the fixed PPO budget was split across incompatible styles.

Keep that configuration only as a documented negative control.

### Population V2 — focused mix to test now

Use:

- **70% aggressive/parametric scripted opponent**
- **30% frozen prior active Student**
- **0% defensive turtle**
- **0% random**

If no verified active prior Student artifact is available, run 100% aggressive
instead of silently substituting turtle/random.

```python
cfg = EvalConfig(
    fighter_opponent_league=True,
    fighter_opponent_weights=(
        ("parametric", 0.70),
        ("defensive", 0.00),
        ("random", 0.00),
        ("prior", 0.30),
    ),
)
```

The prior Student must be frozen and must pass a behavior check before entering
the population: movement on at least 15% of frames and at least 30% real
ring-outs. Do not use a camper checkpoint as the prior.

### Student network-capacity A/B

Keep the Teacher fixed (`Qwen3-4B`, same chosen LoRA checkpoint). Compare only
the PPO Student architecture:

| variant | policy/value hidden layers | approximate trainable parameters |
|---|---:|---:|
| `student_128` | `[128, 128]` | ~37k |
| `student_256` | `[256, 256]` | ~139k |

The current `[64,64]` Student is ~10k parameters and remains the baseline.

Important: the current code assumes `DEFAULT_NET_ARCH` during policy
serialization/restoration. Before running this A/B, make `student_net_arch`
explicit in the Stage-6 payload, PPO construction, `PolicyArtifact`, and restore
path. Do not train a `[128,128]` policy and restore it as `[64,64]`.

GPU note: these models are tiny and fighter simulation is mostly CPU-bound.
A100/H100 does not materially help. Fan out ordinary Modal CPU containers.

### Shared settings

- PPO episodes: start at `2,000`; use `3,000` only if learning is incomplete.
- Entropy coefficient: `0.03`.
- Randomized spawns: ON.
- Stage-6 anti-camping shaping: ON.
- Decisive timeout: evaluation only.
- Full reward only for a real ring-out.
- Timeout/tiebreak win: zero or small credit.
- Draw: penalty.
- Same curricula, opponent weights, seeds and budget for both architectures.

### Lean test sequence

Do not run a large Stage-6 evaluation first.

1. Unit tests:
   - disabled league reproduces the legacy path;
   - fixed seed gives deterministic opponent sequence;
   - 70/30 sampling is approximately correct;
   - missing prior policy omits/renormalizes safely;
   - policy serialization restores both architectures exactly.
2. One local PPO smoke per architecture.
3. One Modal Student pair per architecture:
   - base Teacher curriculum Student;
   - trained Teacher curriculum Student;
   - 3 fresh Student seeds;
   - diagonal screening arenas only.
4. Record:
   - movement fraction;
   - action histogram;
   - real ring-out rate;
   - timeout-win rate;
   - draw rate;
   - side-swapped trained-vs-base advantage.

### Go / no-go

Promote an architecture only if all hold:

- movement fraction >= 15%;
- at least two actions each exceed 10% usage;
- real ring-out rate >= 30%;
- draw rate < 40%;
- no material side bias;
- behavior repeats across at least 2 of 3 Student seeds.

If `[128,128]` passes, use it and stop—the larger network is unnecessary.
Use `[256,256]` only if it materially improves the behavior gates. If both
architectures still camp, capacity is not the bottleneck; stop scaling the
network and redesign the opponent/reward distribution.

### What to evaluate after a pass

Re-evaluate existing Teacher checkpoints first; do not immediately retrain Qwen:

- `runB/update3`
- `runB/update4`
- best balanced checkpoint selected on reserved validation

Only after the focused population produces active Students should you consider
two new 10-update Qwen3-4B Teacher runs. Changing Qwen to 8B cannot directly fix
a PPO Student that learned to stand still.
