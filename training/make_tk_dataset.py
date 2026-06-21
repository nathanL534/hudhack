"""training/make_tk_dataset.py — emit the CLEAN Target-Knockback RFT dataset row(s).

Writes ``output/tiny_rft/tk_dataset.jsonl``: one row per Teacher rollout-seed task for
the Target-Knockback RFT. The prompt is sourced from the SHARED registry
(``games_registry.TARGET_KNOCKBACK.prompt``) so the dataset can NEVER drift from the
HUD env / bridge / evaluator schema.

CLEAN rows only — exactly the shape the Fireworks RFT dataset path accepts:

    {"messages": [{"role": "user", "content": "<TK prompt>"}],
     "input_metadata": {"completion_params": {}}}

NO stale ``rollout_status`` / ``execution_metadata`` / ``created_at`` fields (those are
RUNTIME tracing fields that pollute a dataset row and are exactly what made the earlier
fighter dataset rows dirty). The RFT loop generates many rollouts per update from this
single curriculum-design row.

Run::

    .venv/bin/python -m training.make_tk_dataset            # writes the default path
    .venv/bin/python -m training.make_tk_dataset --rows 1   # N identical task rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from games_registry import TARGET_KNOCKBACK

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _REPO_ROOT / "output" / "tiny_rft" / "tk_dataset.jsonl"


def build_rows(n_rows: int = 1) -> list[dict]:
    """Build ``n_rows`` CLEAN TK dataset rows (the single TK curriculum-design task)."""
    row = {
        "messages": [{"role": "user", "content": TARGET_KNOCKBACK.prompt}],
        "input_metadata": {"completion_params": {}},
    }
    return [dict(row) for _ in range(max(1, n_rows))]


def write_dataset(out: Path = DEFAULT_OUT, *, n_rows: int = 1) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(n_rows)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the clean TK RFT dataset.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows", type=int, default=1, help="number of identical task rows")
    args = parser.parse_args(argv)
    path = write_dataset(args.out, n_rows=args.rows)
    print(f"wrote {args.rows} clean TK dataset row(s) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
