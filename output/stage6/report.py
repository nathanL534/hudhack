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
# PRIMARY-metric helpers (Student-vs-Student head-to-head)
# ---------------------------------------------------------------------------


def _primary(result: dict) -> dict | None:
    """The primary head-to-head block, whether the result is the full or bare form."""
    if result.get("metric") == "primary_student_vs_student_head_to_head":
        return result
    return result.get("primary")


def _secondary(result: dict) -> dict | None:
    """The secondary transfer diagnostic block (legacy fixed-bot before->after)."""
    if "per_model" in result and "primary" not in result:
        return result  # a bare legacy result IS the secondary
    return result.get("secondary_transfer_diagnostic")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _per_model(result: dict) -> list[dict]:
    """Per-model rows from the SECONDARY block (works for full or bare results)."""
    sec = _secondary(result)
    return (sec or {}).get("per_model", [])


def chart_head_to_head_winrate(result: dict, out_dir: Path) -> str | None:
    """Bar chart: trained-Student vs base-Student win rate, with the CI on the
    paired advantage annotated. THE headline visual."""
    if not _MPL:
        return None
    p = _primary(result)
    if not p:
        return None
    labels = ["Base Student", "Trained Student"]
    rates = [p.get("base_student_win_rate", 0.0), p.get("trained_student_win_rate", 0.0)]
    draw = p.get("draw_rate", 0.0)
    ci = p.get("ci_t", {})

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, rates, color=["#8fa1c8", "#56e5ff"], edgecolor="white")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("head-to-head win rate (held-out arenas)")
    ax.set_title("Base Teacher's Student VS Trained Teacher's Student")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.3f}", ha="center", va="bottom", fontsize=10)
    adv = p.get("mean_paired_advantage", 0.0)
    ax.text(0.5, 0.95,
            f"paired advantage {adv:+.3f}  95% CI [{ci.get('low', 0):+.3f}, {ci.get('high', 0):+.3f}]"
            f"  (draw {draw:.2f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#444")
    fig.tight_layout()
    path = out_dir / "chart_head_to_head_winrate.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def chart_per_arena_advantage(result: dict, out_dir: Path) -> str | None:
    """Bar chart: trained-Student paired advantage per held-out arena FAMILY."""
    if not _MPL:
        return None
    p = _primary(result)
    if not p:
        return None
    fam = p.get("per_arena_family_advantage", {})
    if not fam:
        return None
    names = list(fam.keys())
    vals = [fam[n] for n in names]
    colors = ["#59e391" if v > 0 else "#ff667d" for v in vals]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(names)), 4))
    ax.bar(names, vals, color=colors, edgecolor="white")
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.set_ylabel("trained-Student paired advantage")
    ax.set_title("Per-arena-family advantage (benefit must hold across >1 family)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    fig.tight_layout()
    path = out_dir / "chart_per_arena_advantage.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def chart_side_bias(result: dict, out_dir: Path) -> str | None:
    """Bar chart: trained-Student win rate as P0 vs P1 (side-bias diagnostic)."""
    if not _MPL:
        return None
    p = _primary(result)
    if not p:
        return None
    p0 = p.get("trained_win_rate_as_p0", 0.0)
    p1 = p.get("trained_win_rate_as_p1", 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["as P0", "as P1"], [p0, p1], color=["#6d8cff", "#a77cff"], edgecolor="white")
    ax.set_ylim(0, max(0.1, p0, p1) * 1.3)
    ax.set_ylabel("trained-Student win rate")
    ax.set_title("Side-bias check (P0 ≈ P1 means no position dependency)")
    for b, v in zip(bars, [p0, p1]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = out_dir / "chart_side_bias.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


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
    sec = _secondary(result) or result
    per_game = sec.get("per_game", [])
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
    """Collect replay entries (path + metadata) from the result into a gallery manifest.

    Accepts both the legacy string-path form and the new metadata-dict form
    (``{"path", "label", "outcome", ...}``) the head-to-head capture produces."""
    replays = result.get("replays", []) or (_primary(result) or {}).get("replays", [])
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


def _render_primary(result: dict, lines: list[str]) -> bool:
    """Append the PRIMARY (Student-vs-Student) headline section. Returns True if found."""
    p = _primary(result)
    if not p:
        return False
    gaming = p.get("anti_circularity", {})
    gaming_passed = gaming.get("passed", True)
    adv = p.get("mean_paired_advantage", 0.0)
    ci = p.get("ci_t", {})
    overall = bool(adv > 0) and bool(gaming_passed)

    lines.append("# Stage 6 — Base Teacher's Student VS Trained Teacher's Student\n")
    lines.append(
        "**Primary claim under test:** the Student trained on the TRAINED Teacher's "
        "curricula beats the Student trained on the BASE Teacher's curricula on "
        "HELD-OUT arenas, with a curriculum-level 95% CI lower bound above zero.\n"
    )
    lines.append(f"## Headline verdict: {'PASS' if overall else 'FAIL'}\n")
    lines.append(
        f"- trained-Student win rate: **{p.get('trained_student_win_rate', 0):.4f}**  "
        f"vs base-Student **{p.get('base_student_win_rate', 0):.4f}**  "
        f"(draw {p.get('draw_rate', 0):.4f})\n"
        f"- mean paired advantage: **{adv:+.4f}**\n"
        f"- curriculum-level 95% CI: **[{ci.get('low', 0):+.4f}, {ci.get('high', 0):+.4f}]** "
        f"over {ci.get('n', '?')} independent replicates (lower bound must be > 0)\n"
        f"- effect size (Cohen's d): **{p.get('effect_size_cohens_d')}**\n"
        f"- anti-circularity checks passed: **{gaming_passed}**\n"
        f"- OVERALL HEADLINE (advantage>0 AND CI lower>0 AND anti-circularity): **{overall}**\n"
    )
    lines.append(
        "> The fixed-bot before→after number is SECONDARY evidence only — it does "
        "NOT set this headline.\n"
    )

    # Side-bias + per-arena.
    lines.append("## Side-bias & per-arena-family\n")
    lines.append(
        f"- trained-Student win rate as P0: **{p.get('trained_win_rate_as_p0', 0):.4f}**  "
        f"as P1: **{p.get('trained_win_rate_as_p1', 0):.4f}** "
        f"(a real edge is side-symmetric)\n"
    )
    fam = p.get("per_arena_family_advantage", {})
    if fam:
        lines.append("| arena family | trained paired advantage |")
        lines.append("|---|---|")
        for name, v in fam.items():
            lines.append(f"| {name} | {v:+.4f} |")
        lines.append("")

    # Anti-circularity detail.
    lines.append("## Anti-circularity checks (primary)\n")
    lines.append("| check | result | detail |")
    lines.append("|---|---|---|")
    for c in gaming.get("checks", []):
        mark = "PASS" if c.get("passed") else (
            c.get("severity", "fail").upper() if c.get("severity") in ("skip", "warn") else "FAIL")
        lines.append(f"| {c.get('name')} | {mark} | {c.get('detail')} |")
    lines.append("")

    # Replicate table.
    reps = p.get("replicates", [])
    if reps:
        lines.append("## Curriculum replicates\n")
        lines.append("| replicate | trained win | base win | draw | paired adv | overlap |")
        lines.append("|---|---|---|---|---|---|")
        for r in reps:
            lines.append(
                f"| {r.get('replicate')} | {r.get('trained_win_rate', 0):.3f} "
                f"| {r.get('base_win_rate', 0):.3f} | {r.get('draw_rate', 0):.3f} "
                f"| {r.get('paired_advantage', 0):+.4f} | {r.get('held_out_overlap', 0)} |"
            )
        lines.append("")
    return True


def _render_secondary(result: dict, lines: list[str]) -> None:
    """Append the SECONDARY (fixed-bot transfer diagnostic) section."""
    sec = _secondary(result)
    if not sec:
        return
    models = sec.get("per_model", [])
    gaming = sec.get("anti_gaming", {})
    lines.append("## Secondary transfer diagnostic (fixed-bot before→after)\n")
    lines.append(
        "> Evidence only — a fresh PPO Player trains on each Teacher's arena and is "
        "scored on a fixed held-out reference set. Does NOT set the headline.\n"
    )
    lines.append(
        f"- trained beats base on fixed-bot transfer: **{sec.get('trained_beats_base')}** "
        f"(delta {sec.get('delta', 0):+.4f})\n"
        f"- secondary anti-gaming passed: **{gaming.get('passed', True)}**\n"
    )
    lines.append("| model | mean transfer | std | 95% CI | n |")
    lines.append("|---|---|---|---|---|")
    for m in models:
        lines.append(
            f"| {m['model']} | {m['mean_transfer']:+.4f} | {m['std_transfer']:.4f} "
            f"| {_fmt_ci(m.get('ci', {}))} | {m['n_jobs']} |"
        )
    lines.append("")


def _render_cross_game(result: dict, lines: list[str]) -> None:
    """Append the KOTH cross-game generalization section."""
    cg = result.get("cross_game_koth")
    if not cg:
        return
    lines.append("## Cross-game generalization (KOTH — final hidden test)\n")
    if not cg.get("supported", False):
        lines.append(f"- **UNSUPPORTED**: {cg.get('reason')}")
        lines.append(f"- {cg.get('note', '')}\n")
        return
    ci = cg.get("ci_t", {})
    adv = cg.get("mean_paired_advantage", 0.0)
    lines.append(f"- mapping: {cg.get('mapping')}")
    lines.append(
        f"- FRESH KOTH Students (obs_dim 16) trained from each Teacher's KOTH-mapped "
        f"curricula, fought head-to-head (NOT forced fighter weights)\n"
        f"- trained-KOTH-Student win rate: **{cg.get('trained_student_win_rate', 0):.4f}** "
        f"vs base **{cg.get('base_student_win_rate', 0):.4f}**\n"
        f"- mean paired advantage: **{adv:+.4f}**  95% CI "
        f"[{ci.get('low', 0):+.4f}, {ci.get('high', 0):+.4f}]\n"
    )


def render_markdown(result: dict, chart_paths: dict[str, str | None]) -> str:
    lines: list[str] = []

    has_primary = _render_primary(result, lines)
    if not has_primary:
        # Legacy bare-secondary result: render the old transfer-only report header.
        lines.append("# Stage 6 — fixed-bot transfer diagnostic (secondary metric)\n")
        lines.append(
            "> This result predates the Student-vs-Student primary metric (or is a "
            "secondary-only run). The headline verdict is NOT defined here.\n"
        )

    # Run metadata.
    lines.append("## Run\n")
    lines.append(f"- game: `{result.get('game')}`  backend: `{result.get('backend')}`")
    lines.append(f"- base handle: `{result.get('base_handle')}` -> `{result.get('base_resolved_id')}`")
    lines.append(f"- trained handle: `{result.get('trained_handle')}` -> `{result.get('trained_resolved_id')}`")
    cfg = result.get("config", {})
    lines.append(
        f"- config fingerprint: `{cfg.get('fingerprint')}` "
        f"(replicates={cfg.get('n_replicates')}, curriculum_arenas={cfg.get('curriculum_arenas')}, "
        f"match_seeds/arena/side={cfg.get('match_seeds_per_arena')}, ppo_episodes={cfg.get('ppo_episodes')})\n"
    )

    _render_cross_game(result, lines)
    _render_secondary(result, lines)

    # Charts.
    lines.append("## Charts\n")
    for key, p in chart_paths.items():
        if p:
            lines.append(f"- {key}: `{Path(p).name}`")
    lines.append("")

    # Replays.
    replays = result.get("replays", []) or (_primary(result) or {}).get("replays", [])
    if replays:
        lines.append("## Replays\n")
        for r in replays[:20]:
            label = r.get("label", "") if isinstance(r, dict) else ""
            path = r.get("path", r) if isinstance(r, dict) else r
            lines.append(f"- `{path}` {label}")
        lines.append("")

    return "\n".join(lines)


def write_dashboard_artifact(result: dict, out_dir: Path) -> str | None:
    """Write the compact JSON the demo dashboard consumes (no fabricated numbers).

    Contains ONLY real values present in the result; when a real eval has not run,
    the caller should not write this file at all (the dashboard then shows
    "Awaiting evaluation"). Pure function of the result.
    """
    p = _primary(result)
    if not p:
        return None
    ci = p.get("ci_t", {})
    cg = result.get("cross_game_koth") or {}
    dash = {
        "headline_metric": "base_student_vs_trained_student",
        "trained_student_win_rate": p.get("trained_student_win_rate"),
        "base_student_win_rate": p.get("base_student_win_rate"),
        "draw_rate": p.get("draw_rate"),
        "mean_paired_advantage": p.get("mean_paired_advantage"),
        "ci_low": ci.get("low"),
        "ci_high": ci.get("high"),
        "n_replicates": p.get("n_replicates"),
        "trained_win_rate_as_p0": p.get("trained_win_rate_as_p0"),
        "trained_win_rate_as_p1": p.get("trained_win_rate_as_p1"),
        "primary_pass": p.get("primary_pass"),
        "anti_circularity_passed": p.get("anti_circularity", {}).get("passed"),
        "per_arena_family_advantage": p.get("per_arena_family_advantage", {}),
        "cross_game_koth": {
            "supported": cg.get("supported", False),
            "trained_student_win_rate": cg.get("trained_student_win_rate"),
            "base_student_win_rate": cg.get("base_student_win_rate"),
            "mean_paired_advantage": cg.get("mean_paired_advantage"),
        } if cg else None,
        "replays": p.get("replays", []),
        "is_smoke": result.get("is_smoke", False),
    }
    path = out_dir / "dashboard.json"
    path.write_text(json.dumps(dash, indent=2))
    return str(path)


def generate_report(result: dict, out_dir: Path) -> dict:
    """Generate all charts + the Markdown report + replay manifest + dashboard JSON.

    Returns a dict of artifact paths. Pure function of ``result`` — safe to call in
    a report-only flow with no PPO. Leads with the PRIMARY Student-vs-Student charts;
    keeps the secondary transfer charts (when a secondary block is present).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = {
        # PRIMARY headline charts.
        "head_to_head_winrate": chart_head_to_head_winrate(result, out_dir),
        "per_arena_advantage": chart_per_arena_advantage(result, out_dir),
        "side_bias": chart_side_bias(result, out_dir),
        # SECONDARY transfer charts (present only if a secondary block exists).
        "secondary_transfer": chart_base_vs_trained(result, out_dir),
        "secondary_seed_variance": chart_seed_variance(result, out_dir),
    }
    manifest_path = write_replay_manifest(result, out_dir)
    dashboard_path = write_dashboard_artifact(result, out_dir)
    md = render_markdown(result, chart_paths)
    md_path = out_dir / "report.md"
    md_path.write_text(md)

    return {
        "report_md": str(md_path),
        "charts": {k: v for k, v in chart_paths.items() if v},
        "replay_manifest": manifest_path,
        "dashboard": dashboard_path,
        "charts_available": _MPL,
    }
