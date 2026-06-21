# Replay Viewer

A zero-dependency, single-file browser viewer for **Ring-Out Duel** fighter
replays. It draws the platform, both fighters as colored circles (with a facing
nose), punch flashes, ring-out markers, and a winner banner — with play/pause,
step, scrub, and speed controls.

It's just `viewer.html`. No build step, no npm.

---

## Two ways to open it

### 1. `file://` — works with no server (always shows a fight)

Open the file directly in a browser:

```
file:///Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack/viewer/viewer.html
```

(or just double-click `viewer/viewer.html` in Finder).

A **default replay is inlined as a JS constant** (`DEFAULT_REPLAY` in
`viewer.html`), so the canvas is never blank — it renders a full
scripted-vs-scripted fight immediately, with no network and no CORS issues.

On `file://`, the **Example dropdown is disabled in practice** because browsers
block `fetch()` over `file://`. To load a *different* replay from here, use the
**Load file…** button or **drag & drop** a `replays/*.json` onto the page —
those paths always work.

### 2. `http://` — the Example dropdown (live files in `replays/`)

The dropdown fetches the live JSON in `../replays/`, so you need an HTTP server.

**Serve from the REPO ROOT — not from `viewer/`.** The viewer page lives at
`/viewer/viewer.html` and the replays live at `/replays/*.json` (sibling dirs).
`python -m http.server` refuses `..` path traversal above its document root, so
if you start the server inside `viewer/`, the fetch of `../replays/...` resolves
to `/replays/...` which is **outside** that root and 404s. Serving from the repo
root makes both `/viewer/` and `/replays/` resolve.

```bash
# from the repo root:
cd /Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack
python3 -m http.server 8000
```

Then open:

```
http://localhost:8000/viewer/viewer.html
```

The first example auto-loads; the dropdown switches between
`scripted_vs_random`, `scripted_vs_scripted`, and `trained_vs_scripted`.

#### Start it detached (stays up after your shell closes)

```bash
cd /Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack
nohup python3 -m http.server 8000 > /tmp/hud_viewer.log 2>&1 &
```

Restart (kills any stale server on 8000 first):

```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null; cd /Users/nathaniellee/claude/project_harness/projects/project_ideas/hud_hackathon/hudhack && nohup python3 -m http.server 8000 > /tmp/hud_viewer.log 2>&1 &
```

---

## Controls

- **▶ Play / ⏸ Pause** — `space`
- **⟨ Step / Step ⟩** — `←` / `→`
- **Scrub** — drag the slider to any frame
- **Speed** — 30 / 15 / 10 / 5 fps

## Replay JSON shape

```jsonc
{
  "meta":     { "config": "A", "winner": "p1", "p1_policy": "scripted", "p2_policy": "random", ... },
  "platform": { "x_left": 0.0, "x_right": 10.0, "y": 0.0 },
  "frames": [
    {
      "p1": { "x": 3.01, "y": 0.0, "facing": 1,  "action": "idle" },
      "p2": { "x": 6.97, "y": 0.0, "facing": -1, "action": "left" },
      "events": ["p1_punch", "p2_ringout"]      // optional per-frame events
    }
    // ...
  ]
}
```

A fighter whose `x` falls outside `[x_left, x_right]` is drawn falling into the
void (ring-out). The final frame shows the `meta.winner` banner.

To inline a different default, replace the `DEFAULT_REPLAY` constant in
`viewer.html` with any `replays/*.json` (minify it onto one line).
