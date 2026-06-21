# Crucible

An LLM **Teacher** learns to generate *learnable* gridworld RL environments.
The Teacher is rewarded by the **strong-vs-weak score gap** on the envs it
produces, trained via **Fireworks RFT**, and judged by a **held-out
head-to-head**. The **harness is the product** — Teacher, games, and Players are
adapters around stable, frozen contracts.

> Status: **PHASE 1 — orchestration pipeline proven end-to-end.** The whole
> flow runs on FAKES (Teacher / GameAdapter / PlayerTrainer) plus the **ONE real
> reward**. No real Fireworks / Modal / PPO / UI yet — those are later phases,
> reserved (not implemented) in the contracts.

---

## Run it

No secrets needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_experiment.py    # acceptance test: prints 5 checkmarks, writes results/*.json, exit 0
pytest                      # the Phase 1 test suite (green)
```

`run_experiment.py` prints exactly:

```
✓ curriculum validated
✓ 2 arenas instantiated
✓ 4 seeded Player jobs completed
✓ Teacher reward: 0.4
✓ experiment result saved as JSON
```

The reward is **computed by the real scorer** (not hardcoded): the fake adapter's
scripted expert scores 0.9, its random policy scores 0.5, `p = 0.5` is inside the
learnable band (`p*(1-p) = 0.25 > 0.2`), so `reward = 0.9 - 0.5 = 0.4`. The
`ExperimentResult` is saved to `results/experiment_<curriculum_id>.json`
(gitignored).

---

## The frozen contracts (`contracts.py`)

Pydantic, serializable, seed-addressable. **Both devs import these before
writing anything else — they must not change without agreement.**

| Contract | Purpose |
|---|---|
| `RewardMode` | `Literal["gap_proxy", "ppo_improvement"]` |
| `ArenaSpec` | One gridworld arena's params (`map_size`, `doors`, `keys`, `hazard_density`, `difficulty`) |
| `CurriculumSpec` | `game_id`, `curriculum_id`, `arenas[]`; `.validate()` + `.model_json_schema()` |
| `ScoringResult` | `reward`, `weak_score`, `strong_score`, `p`, `curriculum_id`, `mode` |
| `TrainingBudget` | `episodes` (+ reserved `steps`) |
| `PlayerConfig` | `architecture`, `num_seeds`, `budget`, `seed`, `modal_parallel` |
| `MatchResult` | per-seed / per-arena scores from a Player job |
| `ExperimentResult` | `per_seed_scores`, `delta_per_arch`, `n_seeds_completed`, `p_value`, `curricula_ids`, `reward` |
| `ScoringContext` | reserves `rollout_id` for the future async `/init` handler (Phase 7/8) |

The shared **JSON Schema** for curricula is `CurriculumSpec.model_json_schema()`,
read by (a) the Teacher prompt, (b) the boundary validator, and (c) the fake —
one schema, no drift.

### The ONE scoring rule (REAL, never faked)

```python
score_curriculum(spec, game, *, mode="gap_proxy", rollout_id=None) -> ScoringResult
#   strong = adapter.scripted_expert  ·  weak = adapter.random_policy
#   p = weak_score
#   reward = (strong - weak) if p*(1-p) > 0.2 else 0.0   # computed INSIDE the scorer
```

The reward is computed **inside** the scorer — never by a caller. This is the
contract that kills the hour-1 divergence between the scorer and the reward
wiring.

---

## Architecture (`harness/`)

```
contracts.py            # THE frozen Pydantic contracts (committed first)
harness/
  interfaces.py         # ABCs: TeacherClient, GameAdapter, PlayerTrainer, TrainingJob, ExperimentRunner
  scoring.py            # score_curriculum() — the ONE real reward
  fakes.py              # FakeTeacher, FakeGameAdapter, FakePlayerTrainer (return FULL frozen types)
  runner.py             # DefaultExperimentRunner — validate -> build -> train N seeds -> score -> aggregate
run_experiment.py       # Phase 1 acceptance test (5 checkmarks, writes results/*.json)
test_phase1.py          # pytest assertions
results/                # generated ExperimentResult JSON (gitignored)
```

Interfaces are the seams where a fake is swapped for the real implementation,
**one component at a time** — right contracts → independent parallel work →
drop-in swaps.

---

## Fake → real swap map

Each fake returns the **full frozen contract type** (a fake that returned a bare
float or `random.random()` would be a *different* interface). Swap is drop-in:

| Phase-1 fake | Real replacement | Owner |
|---|---|---|
| `FakeTeacher` | Fireworks **Qwen3-4B** Teacher (prompt on the param schema, parse + validate JSON) | rag26 |
| `FakeGameAdapter` | real **gridworld** engine + BFS `scripted_expert` + `random_policy` | Nathan |
| `FakePlayerTrainer` | **PPO on Modal** (`submit` → one job / seed, behind `modal_parallel`) | rag26 |
| `score_curriculum` | **already real** — the gap-proxy formula does not change | Nathan |

What stays reserved (NOT implemented in Phase 1):

- `RewardMode="ppo_improvement"` — eval-only real-PPO reward (`score_curriculum`
  raises `NotImplementedError`).
- `PlayerConfig.modal_parallel=True` — Modal dispatch (runner + trainer raise
  `NotImplementedError`; the flag is honored, the code is later).
- `ScoringContext.rollout_id` — threaded by the future async `/init` handler so
  concurrent rollouts don't cross-tag rewards (`# TODO Phase 7/8`).

---

## Role split

| | Owns |
|---|---|
| **Nathan** | `GameAdapter` (gridworld engine, scripted/random policies) + `scoring.py` |
| **rag26** | `TeacherClient` (Fireworks Qwen3-4B) + `PlayerTrainer` / RFT (PPO on Modal) |

Both share `contracts.py` and `harness/runner.py`. Because Modal lives behind
`PlayerConfig.modal_parallel` **inside** `PlayerTrainer` (never in the runner),
local → Modal is one flag flip.

---

## Notes

- **Never commit secrets.** API keys go in `.env` (gitignored).
- Phase 1 deps are just `pydantic` + `pytest`. `numpy` / `matplotlib` /
  `hud-python` / `fireworks-ai` / `modal` are commented in `requirements.txt`
  for when the real engine / Teacher / Players land.
- Built for the 24h HUD Frontier hackathon. Keep it a clean pipeline: freeze the
  contracts, swap fakes for real one at a time, don't over-build ahead of the phase.
```
