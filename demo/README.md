# Crucible judge dashboard

UI-only hackathon dashboard. It reads existing replay and evaluation artifacts;
it does not invoke Fireworks, Modal, HUD, or any training code.

From the repository root:

```bash
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000/demo/
```

The headline is the PRIMARY metric: **Base Teacher's Student VS Trained Teacher's
Student**, fought head-to-head on held-out arenas (side-swapped). There are NO
fabricated numbers — until a real eval has run, every value reads "Awaiting
evaluation".

Modes:

- **Demo replay** shows the layout with all metrics "Awaiting evaluation"
  (no fabricated numbers) plus any captured head-to-head replays in `/replays/`.
- **Live artifacts** reads `/output/stage6_report/dashboard.json` (written by
  `report.generate_report`), falling back to the `primary` block of
  `/output/eval_base_vs_trained.json`, then to "Awaiting evaluation".

Captured head-to-head replays (`/replays/h2h_fighter_*.json`) are produced by a
real eval run; the dashboard plays the trained-win and base-win matches.

The existing detailed replay viewer remains at `/viewer/viewer.html`.
