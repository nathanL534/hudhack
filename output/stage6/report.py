"""output/stage6/report.py — charts + Markdown report from a Stage-6 result JSON.

Everything here is a PURE function of the saved result dict, so the ``report-only``
command can regenerate the full report WITHOUT rerunning any PPO. Charts use
matplotlib's Agg backend (headless), matching ``eval/plots.py``.

Produces, into a chosen output directory:
  * ``chart_base_vs_trained.png`` — mean held-out transfer per model, with CI bars.
  * ``chart_per_game.png``        — per-game comparison (training game + probes).
  * ``chart_seed_variance.png``   — per-seed improvement scatter with error bars.
  * ``report.md``                 — the concise Markdown writeup (verdict + tables).
  * ``replay_manifest.json``      — replay file paths gathered into a gallery list.

The chart functions degrade gracefully if matplotlib is unavailable (they record a
note and skip, so a report can still be produced text-only).
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL = True
except Exception:  # pragma: no cover - charts optional
    _MPL = False


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _per_model(result: dict) -> list[dict]:
    return result.get("per_model", [])


def chart_base_vs_trained(result: dict, out_dir: Path) -> str | None:
    """Bar chart: mean held-out transfer per model with 95% CI error bars."""
    if not _MPL:
        return None
    models = _per_model(result)
    labels = [m["model"] for m in models]
    means = [m["mean_transfer"] for m in models]
    cis = [m.get("ci", {}).get("half_width", 0.0) for m in models]
    colors = ["#4C72B0", "#55A868"][: len(models)] or ["#4C72B0"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=cis, capsize=6, color=colors, edgecolor="white")
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.set_ylabel("mean held-out transfer (after - before)")
    ax.set_title("Base vs Trained Teacher — held-out transfer")
    for i, (m, c) in enumerate(zip(means, cis)):
        ax.text(i, m, f"{m:+.3f}", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=9)
    fig.tight_layout()
    path = out_dir / "chart_base_vs_trained.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def chart_per_game(result: dict, out_dir: Path) -> str | None:
    """Grouped bars: base vs trained mean transfer per game (training + probes)."""
    if not _MPL:
        return None
    per_game = result.get("per_game", [])
    if not per_game:
        return None
    games = [g["game"] for g in per_game]
    base_vals = [g.get("base_mean", 0.0) for g in per_game]
    trained_vals = [g.get("trained_mean", 0.0) for g in per_game]
    x = range(len(games))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(games)), 4))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="base", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], trained_vals, width, label="trained", color="#55A868")
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(games, rotation=15, ha="right")
    ax.set_ylabel("mean transfer")
    ax.set_title("Per-game comparison (training game + held-out probes)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "chart_per_game.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def chart_seed_variance(result: dict, out_dir: Path) -> str | None:
    """Scatter of per-seed improvements per model, with mean ± std error bars."""
    if not _MPL:
        return None
    models = _per_model(result)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(models):
        imps = [j["held_out_improvement"] for j in m.get("per_job", [])]
        xs = [i + (k - len(imps) / 2) * 0.02 for k in range(len(imps))]
        ax.scatter(xs, imps, alpha=0.6, s=24,
                   color="#4C72B0" if i == 0 else "#55A868")
        ax.errorbar(i, m["mean_transfer"], yerr=m["std_transfer"], fmt="D",
                    color="black", capsize=6, markersize=6, zorder=5)
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m["model"] for m in models])
    ax.set_ylabel("per-seed held-out improvement")
    ax.set_title("Seed variance (scatter) vs mean ± std (diamonds)")
    fig.tight_layout()
    path = out_dir / "chart_seed_variance.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Replay manifest / gallery
# ---------------------------------------------------------------------------


def write_replay_manifest(result: dict, out_dir: Path) -> str | None:
    """Collect replay file paths from the result into a gallery manifest."""
    replays = result.get("replays", [])
    if not replays:
        return None
    manifest = {
        "experiment": result.get("experiment", "stage6"),
        "count": len(replays),
        "replays": replays,
    }
    path = out_dir / "replay_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return str(path)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt_ci(ci: dict) -> str:
    if not ci:
        return "n/a"
    return f"[{ci.get('low', 0):+.4f}, {ci.get('high', 0):+.4f}]"


def render_markdown(result: dict, chart_paths: dict[str, str | None]) -> str:
    models = _per_model(result)
    base = next((m for m in models if m["model"] == "base"), None)
    trained = next((m for m in models if m["model"] != "base"), None)
    verdict = result.get("trained_beats_base")
    gaming = result.get("anti_gaming", {})
    gaming_passed = gaming.get("passed", True)

    lines: list[str] = []
    lines.append("# Stage 6 — Base vs Trained Teacher (held-out transfer decider)\n")
    lines.append(
        "**Claim under test:** the TRAINED Teacher generates environments that "
        "cause fresh PPO Players to improve MORE on held-out tasks than the BASE "
        "Teacher's environments.\n"
    )
    # Headline verdict.
    overall = bool(verdict) and bool(gaming_passed)
    lines.append(f"## Verdict: {'PASS' if overall else 'FAIL'}\n")
    lines.append(
        f"- trained beats base on held-out transfer: **{verdict}** "
        f"(delta {result.get('delta', 0):+.4f})\n"
        f"- anti-gaming checks passed: **{gaming_passed}**\n"
        f"- overall (both required): **{overall}**\n"
    )
    lines.append(
        "> A trained Teacher whose reward rose but that does NOT beat base on "
        "held-out transfer is a FAIL, not a success.\n"
    )

    # Run metadata.
    lines.append("## Run\n")
    lines.append(f"- game: `{result.get('game')}`  backend: `{result.get('backend')}`")
    lines.append(f"- base handle: `{result.get('base_handle')}` -> `{result.get('base_resolved_id')}`")
    lines.append(f"- trained handle: `{result.get('trained_handle')}` -> `{result.get('trained_resolved_id')}`")
    cfg = result.get("config", {})
    lines.append(
        f"- config fingerprint: `{cfg.get('fingerprint')}` "
        f"(arenas/model={cfg.get('arenas_per_model')}, seeds={cfg.get('player_seeds')}, "
        f"ppo_episodes={cfg.get('ppo_episodes')}, eval_seeds={cfg.get('eval_seeds')})"
    )
    lines.append(
        f"- jobs: {result.get('n_jobs')}  wall-clock: {result.get('wall_clock_s')}s  "
        f"est. cost: ${result.get('estimated_cost_usd', 0):.4f}\n"
    )

    # Per-model table.
    lines.append("## Per-model held-out transfer\n")
    lines.append("| model | mean | std | 95% CI | n |")
    lines.append("|---|---|---|---|---|")
    for m in models:
        lines.append(
            f"| {m['model']} | {m['mean_transfer']:+.4f} | {m['std_transfer']:.4f} "
            f"| {_fmt_ci(m.get('ci', {}))} | {m['n_jobs']} |"
        )
    lines.append("")

    # Diversity.
    if trained:
        div = trained.get("diversity", {})
        lines.append("## Parameter diversity (trained Teacher)\n")
        lines.append(
            f"- unique configs: {div.get('n_unique')}/{div.get('n_arenas')} "
            f"(fraction {div.get('unique_fraction')})"
        )
        lines.append(f"- mean per-knob std: {div.get('mean_param_std')}")
        lines.append(f"- collapsed to one config: {div.get('collapsed')}\n")

    # JSON validation / clamping.
    val = result.get("json_validation", {})
    lines.append("## JSON validation / clamping\n")
    lines.append(
        f"- generations: {val.get('total', 0)}  clamped/invalid: "
        f"{val.get('clamped', 0)}  fraction: {val.get('clamp_fraction', 0):.1%}\n"
    )

    # Anti-gaming detail.
    lines.append("## Anti-gaming checks\n")
    lines.append("| check | passed | detail |")
    lines.append("|---|---|---|")
    for c in gaming.get("checks", []):
        mark = "PASS" if c.get("passed") else ("SKIP" if c.get("severity") == "skip" else "FAIL")
        lines.append(f"| {c.get('name')} | {mark} | {c.get('detail')} |")
    lines.append("")

    # Per-game.
    per_game = result.get("per_game", [])
    if per_game:
        lines.append("## Per-game\n")
        lines.append("| game | role | base | trained | delta |")
        lines.append("|---|---|---|---|---|")
        for g in per_game:
            lines.append(
                f"| {g['game']} | {g.get('role','')} | {g.get('base_mean',0):+.4f} "
                f"| {g.get('trained_mean',0):+.4f} | {g.get('delta',0):+.4f} |"
            )
        lines.append("")

    # HUD traces.
    traces = result.get("hud_traces", [])
    if traces:
        lines.append("## HUD traces\n")
        for t in traces:
            url = t.get("url") or "(local trace only)"
            lines.append(f"- `{t.get('trace_id')}` — {url}")
        lines.append("")

    # Charts.
    lines.append("## Charts\n")
    for key, p in chart_paths.items():
        if p:
            lines.append(f"- {key}: `{Path(p).name}`")
    lines.append("")

    # Replays.
    replays = result.get("replays", [])
    if replays:
        lines.append("## Replays\n")
        for r in replays[:20]:
            lines.append(f"- `{r}`")
        lines.append("")

    return "\n".join(lines)


def generate_report(result: dict, out_dir: Path) -> dict:
    """Generate all charts + the Markdown report + replay manifest into ``out_dir``.

    Returns a dict of artifact paths. Pure function of ``result`` — safe to call in
    a report-only flow with no PPO.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = {
        "base_vs_trained": chart_base_vs_trained(result, out_dir),
        "per_game": chart_per_game(result, out_dir),
        "seed_variance": chart_seed_variance(result, out_dir),
    }
    manifest_path = write_replay_manifest(result, out_dir)
    md = render_markdown(result, chart_paths)
    md_path = out_dir / "report.md"
    md_path.write_text(md)

    return {
        "report_md": str(md_path),
        "charts": {k: v for k, v in chart_paths.items() if v},
        "replay_manifest": manifest_path,
        "charts_available": _MPL,
    }
