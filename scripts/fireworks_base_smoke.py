#!/usr/bin/env python3
"""Spend one small Fireworks inference request to verify the Teacher model.

Loads FIREWORKS_API_KEY from .env without printing it. This does not create a
deployment or start RFT; it only confirms authentication, model availability,
strict JSON generation, and parameter clamping.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from engine.games import GAME_1  # noqa: E402
from training.fireworks_teacher import FireworksTeacher  # noqa: E402


def main() -> int:
    if not os.getenv("FIREWORKS_API_KEY"):
        print("ERROR: FIREWORKS_API_KEY is missing from .env", file=sys.stderr)
        return 2

    model = os.getenv(
        "FIREWORKS_TEACHER_MODEL",
        "accounts/fireworks/models/qwen3-4b",
    )
    teacher = FireworksTeacher(model=model)
    params = teacher.generate(GAME_1)
    print(
        json.dumps(
            {
                "status": "ok",
                "model": model,
                "validated_params": params,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
