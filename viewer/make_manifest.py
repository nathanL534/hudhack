#!/usr/bin/env python3
"""Regenerate replays/manifest.json from replays/*.json.

The viewer (viewer/viewer.html) populates its Example dropdown DYNAMICALLY by
fetching the http.server directory listing of /replays/ and parsing out every
*.json filename. That is the preferred path and needs nothing to keep in sync.

This manifest is only a FALLBACK for environments where directory indexing is
disabled (so `fetch("/replays/")` returns no usable listing). It lists every
replay filename so the dropdown can still be populated.

Re-run this whenever you add/remove replays AND you are relying on the manifest
fallback (you usually are not — directory listing covers it):

    python3 viewer/make_manifest.py

Writes replays/manifest.json as a sorted JSON array of bare filenames, e.g.
    ["modal_trained_v1.json", "scripted_vs_random.json", ...]
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAYS_DIR = REPO_ROOT / "replays"
MANIFEST = REPLAYS_DIR / "manifest.json"


def main() -> None:
    if not REPLAYS_DIR.is_dir():
        raise SystemExit(f"replays dir not found: {REPLAYS_DIR}")

    names = sorted(
        p.name
        for p in REPLAYS_DIR.glob("*.json")
        if p.name != "manifest.json"
    )
    MANIFEST.write_text(json.dumps(names, indent=2) + "\n")
    print(f"wrote {MANIFEST} ({len(names)} replays)")
    for n in names:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
