"""run_tk_modal_validation.py — small parallel validation of the TK Modal worker.

Fans out 3 seeds at a learnable difficulty (d=0.5) through the deployed
``crucible-player-tk`` / ``train_tk_player`` worker IN PARALLEL via ``fn.map``, prints
per-seed before/after/improvement plus mean/std, and writes ONE captured trained-Player
TK replay to ``replays/tk_modal_trained_d05_seed1.json``.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

# .env ships BLANK MODAL_TOKEN_ID/SECRET that break Modal env-var auth — pop them so the
# njlee007 profile credentials are used instead (same as run_tiny_rft.py).
for _k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    if not os.environ.get(_k):
        os.environ.pop(_k, None)
os.environ.setdefault("MODAL_PROFILE", "njlee007")

import modal

SEEDS = [1, 2, 3]
DIFFICULTY = 0.5  # TK learnable band is 0.45-0.55
REPLAY_PATH = Path(__file__).parent / "replays" / "tk_modal_trained_d05_seed1.json"


def main() -> int:
    fn = modal.Function.from_name("crucible-player-tk", "train_tk_player")

    # seed 1 carries the replay-capture flag; all three share the arena dials.
    payloads = []
    for s in SEEDS:
        p = {
            "difficulty": DIFFICULTY,
            "ppo_episodes": 2000,
            "eval_seeds": 50,
            "curriculum_id": "tk-modal-val-d05",
        }
        if s == SEEDS[0]:
            p["capture_replay_id"] = "tk_modal_trained_d05_seed1"
        payloads.append(p)

    print(f"Fanning out {len(SEEDS)} seeds at d={DIFFICULTY} via crucible-player-tk.train_tk_player (parallel)...")
    rows = list(fn.map(payloads, SEEDS))
    by_seed = {int(r["seed"]): r for r in rows}

    print()
    print(f"  {'seed':>4}  {'before':>7}  {'after':>7}  {'improve':>8}")
    befores, afters, improves = [], [], []
    for s in SEEDS:
        r = by_seed[s]
        b, a, imp = r["before_winrate"], r["after_winrate"], r["improvement"]
        befores.append(b); afters.append(a); improves.append(imp)
        print(f"  {s:>4}  {b:>7.3f}  {a:>7.3f}  {imp:>+8.3f}")

    def _ms(xs):
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

    bm, bs = _ms(befores)
    am, as_ = _ms(afters)
    im, is_ = _ms(improves)
    print()
    print(f"  before  mean={bm:.3f}  std={bs:.3f}")
    print(f"  after   mean={am:.3f}  std={as_:.3f}")
    print(f"  improve mean={im:+.3f}  std={is_:.3f}")

    # Write the captured replay (worker returned the dict; the driver persists it).
    seed1 = by_seed[SEEDS[0]]
    if "replay" in seed1:
        REPLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPLAY_PATH.write_text(json.dumps(seed1["replay"]["data"], indent=2))
        meta = seed1["replay"]["data"]["meta"]
        print()
        print(f"  replay written: {REPLAY_PATH}")
        print(f"    frames={meta['frames']}  winner={meta['winner']}  "
              f"p1={meta['p1_policy']}  p2={meta['p2_policy']}")
    else:
        print("  WARNING: seed-1 row had no replay payload.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
