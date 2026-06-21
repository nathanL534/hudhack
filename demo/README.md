# Crucible judge dashboard

UI-only hackathon dashboard. It reads existing replay and evaluation artifacts;
it does not invoke Fireworks, Modal, HUD, or any training code.

## Open it (no server needed)

The captured head-to-head matches and the real `dashboard.json` are embedded in
`demo/replays.js`, so the page plays straight from disk:

```text
open demo/index.html        # double-click also works — the two canvases animate
```

Both canvases REPLAY real captured matches frame by frame (the trained Student's
match in the right panel, the base Student's match in the left), with a
play/pause button. No `fetch()` is required for the replay or the metrics.

## Optional: served mode

If you'd rather serve it (enables the "Live artifacts" mode to re-fetch fresh
JSON), from the repository root:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/demo/
```

## Regenerating the embedded data

`demo/replays.js` is generated from the real artifacts — never hand-authored:

```bash
python3 - <<'PY'
import json
files={"trained_win_P0":"replays/h2h_fighter_trained_win_trained_as_P0.json",
       "base_win_P0":"replays/h2h_fighter_base_win_trained_as_P0.json",
       "draw_P0":"replays/h2h_fighter_draw_trained_as_P0.json",
       "base_win_P1":"replays/h2h_fighter_base_win_trained_as_P1.json",
       "draw_P1":"replays/h2h_fighter_draw_trained_as_P1.json"}
replays={k:json.load(open(v)) for k,v in files.items()}
dash=json.load(open("output/stage6_report/dashboard.json"))
man=json.load(open("output/stage6_report/replay_manifest.json"))
for m in (dash,man):
    for r in m.get("replays",[]): r.pop("path",None)   # don't leak fs paths
blob={"dashboard":dash,"manifest":man,"replays":replays}
open("demo/replays.js","w").write("window.CRUCIBLE_DATA="+json.dumps(blob,separators=(",",":"))+";\n")
PY
```

The headline is the PRIMARY metric: **Base Teacher's Student VS Trained Teacher's
Student**, fought head-to-head on held-out arenas (side-swapped). There are NO
fabricated numbers — until a real eval has run, every value reads "Awaiting
evaluation".

Modes:

- **Demo replay** (default) reads the embedded real `dashboard.json` and the
  embedded captured replays. Every metric comes from real eval output; any value
  the eval didn't produce reads "Awaiting evaluation" — never a fabricated score.
- **Live artifacts** re-fetches `/output/stage6_report/dashboard.json` (written by
  `report.generate_report`), falling back to the `primary` block of
  `/output/eval_base_vs_trained.json`, then to the embedded copy, then to
  "Awaiting evaluation". (Requires served mode; ignored on `file://`.)

Captured head-to-head replays (`/replays/h2h_fighter_*.json`) are produced by a
real eval run. The dashboard plays the base-win match in the left panel and the
trained-win match in the right panel, resolving which fighter is the trained
Student from each trace's own metadata (the eval side-swaps p1/p2, so the panel
highlights the correct fighter regardless of side).

The existing detailed replay viewer remains at `/viewer/viewer.html`.
