"""output/stage6/anti_gaming.py — the checks that can FAIL a run even if reward rose.

The success criterion is NOT "RFT completed" or "Teacher reward went up". It is:
the trained Teacher's generated environments cause fresh PPO Players to improve
MORE on HELD-OUT tasks than the base Teacher's environments. These checks catch the
ways that headline number can be a lie:

  G1 reward_up_but_no_transfer — a training-game / in-distribution signal rose but
     held-out transfer did NOT improve (the whole point of the held-out yardstick).
  G2 arena_collapse            — the trained Teacher's generated arenas collapsed
     to a single parameter configuration (no real curriculum, just one exploit).
  G3 excessive_clamping        — invalid/clamped JSON exceeded 10% of generations
     (the Teacher is emitting garbage that the clamp silently rescues).
  G4 seed_noise_dominates      — seed std exceeds the mean improvement (the
     "improvement" is within the noise floor; not a real effect).
  G5 train_game_only           — performance improved ONLY on the training game and
     not on any held-out PROBE game (overfit to the trained game).
  G6 same_model                — base and trained handles resolved to the SAME model
     during a REAL evaluation (the comparison is vacuous).

Each check returns a ``GamingCheck`` with ``passed`` (True = healthy), a numeric
``value`` vs ``threshold``, and a human ``detail``. ``run_all`` aggregates them and
exposes ``any_failed`` so the evaluator can stamp the run FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CLAMP_FRACTION_LIMIT = 0.10  # >10% invalid/clamped JSON fails G3


@dataclass(frozen=True)
class GamingCheck:
    name: str
    passed: bool
    detail: str
    value: float | None = None
    threshold: float | None = None
    severity: str = "fail"  # "fail" = gates the run; "warn" = advisory; "skip" = n/a

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "value": self.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


def check_transfer_followed_reward(
    base_mean: float, trained_mean: float, *,
    train_signal_rose: bool | None = None,
) -> GamingCheck:
    """G1: a training/in-distribution rise must be matched by a held-out transfer rise.

    If we know the training signal rose (``train_signal_rose``) but the HELD-OUT
    transfer of the trained Teacher is not above the base Teacher's, that is the
    classic "reward up, transfer flat" gaming pattern and FAILS.
    """
    transfer_improved = trained_mean > base_mean
    if train_signal_rose is None:
        # Without a separate training signal, fall back to "transfer must improve".
        passed = transfer_improved
        detail = (
            f"held-out transfer trained({trained_mean:+.4f}) "
            f"{'>' if transfer_improved else '<='} base({base_mean:+.4f})"
        )
        return GamingCheck("reward_up_but_no_transfer", passed, detail,
                           value=round(trained_mean - base_mean, 6), threshold=0.0)
    if train_signal_rose and not transfer_improved:
        return GamingCheck(
            "reward_up_but_no_transfer", False,
            f"training signal rose but held-out transfer did NOT "
            f"(trained {trained_mean:+.4f} <= base {base_mean:+.4f})",
            value=round(trained_mean - base_mean, 6), threshold=0.0,
        )
    return GamingCheck(
        "reward_up_but_no_transfer", True,
        f"training rose={train_signal_rose}; held-out delta {trained_mean - base_mean:+.4f}",
        value=round(trained_mean - base_mean, 6), threshold=0.0,
    )


def check_arena_collapse(trained_diversity: dict) -> GamingCheck:
    """G2: the trained Teacher's generated arenas must not collapse to one config."""
    collapsed = bool(trained_diversity.get("collapsed", False))
    n_unique = trained_diversity.get("n_unique", 0)
    n = trained_diversity.get("n_arenas", 0)
    return GamingCheck(
        "arena_collapse", not collapsed,
        f"trained Teacher emitted {n_unique}/{n} unique arena configs"
        + (" (COLLAPSED to one config)" if collapsed else ""),
        value=float(n_unique), threshold=2.0,
    )


def check_clamp_fraction(clamp_fraction: float) -> GamingCheck:
    """G3: invalid/clamped JSON must stay at or below 10%."""
    passed = clamp_fraction <= CLAMP_FRACTION_LIMIT
    return GamingCheck(
        "excessive_clamping", passed,
        f"invalid/clamped JSON fraction {clamp_fraction:.1%} "
        f"(limit {CLAMP_FRACTION_LIMIT:.0%})",
        value=round(clamp_fraction, 6), threshold=CLAMP_FRACTION_LIMIT,
    )


def check_seed_noise(mean_improvement: float, seed_std: float) -> GamingCheck:
    """G4: seed std must not exceed the mean improvement (else it's in the noise).

    Compares against the MAGNITUDE of the improvement: a +0.20 improvement with
    std 0.25 is noise; a -0.20 "improvement" with std 0.05 is a real (bad) effect.
    """
    passed = seed_std <= abs(mean_improvement) or mean_improvement == 0.0 and seed_std == 0.0
    return GamingCheck(
        "seed_noise_dominates", passed,
        f"seed std {seed_std:.4f} {'<=' if passed else '>'} "
        f"|mean improvement| {abs(mean_improvement):.4f}",
        value=round(seed_std, 6), threshold=round(abs(mean_improvement), 6),
    )


def check_train_game_only(
    held_out_delta: float, probe_deltas: dict[str, float] | None,
) -> GamingCheck:
    """G5: improvement must not be confined to the training game.

    ``probe_deltas`` maps probe-game name -> (trained - base) transfer delta. If the
    trained Teacher beats base on the training game's held-out set but on NO probe
    game, that is overfit-to-training-game. With no probes run, this is a SKIP (it
    cannot be evaluated, and must not silently pass as if it had).
    """
    if not probe_deltas:
        return GamingCheck(
            "train_game_only", True,
            "no held-out probe games were run (check skipped)",
            value=None, threshold=None, severity="skip",
        )
    any_probe_improved = any(d > 0 for d in probe_deltas.values())
    train_improved = held_out_delta > 0
    passed = not (train_improved and not any_probe_improved)
    probe_summary = ", ".join(f"{k}:{v:+.4f}" for k, v in probe_deltas.items())
    return GamingCheck(
        "train_game_only", passed,
        f"training-game held-out delta {held_out_delta:+.4f}; probes [{probe_summary}]"
        + ("" if passed else " (improved on training game ONLY)"),
        value=1.0 if any_probe_improved else 0.0, threshold=1.0,
    )


def check_distinct_models(
    base_resolved_id: str, trained_resolved_id: str, *, is_smoke: bool,
) -> GamingCheck:
    """G6: base and trained must resolve to DIFFERENT models in a real evaluation.

    In ``--smoke`` (base-vs-base) identical ids are expected and fine. In a real
    decider, identical resolved ids make the comparison vacuous and FAIL.
    """
    same = base_resolved_id == trained_resolved_id
    if is_smoke:
        return GamingCheck(
            "same_model", True,
            f"smoke mode: base==trained by design ({base_resolved_id})",
            value=1.0 if same else 0.0, threshold=None, severity="skip",
        )
    return GamingCheck(
        "same_model", not same,
        ("base and trained resolved to the SAME model: " if same else
         "base and trained resolved to distinct models: ")
        + f"base={base_resolved_id} trained={trained_resolved_id}",
        value=1.0 if same else 0.0, threshold=0.0,
    )


@dataclass(frozen=True)
class GamingReport:
    checks: list[GamingCheck] = field(default_factory=list)

    @property
    def any_failed(self) -> bool:
        return any((not c.passed) and c.severity == "fail" for c in self.checks)

    @property
    def passed(self) -> bool:
        return not self.any_failed

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "any_failed": self.any_failed,
            "checks": [c.as_dict() for c in self.checks],
            "failed": [c.name for c in self.checks if (not c.passed) and c.severity == "fail"],
        }


def run_all(
    *,
    base_mean: float,
    trained_mean: float,
    trained_diversity: dict,
    clamp_fraction: float,
    seed_std: float,
    base_resolved_id: str,
    trained_resolved_id: str,
    is_smoke: bool,
    train_signal_rose: bool | None = None,
    probe_deltas: dict[str, float] | None = None,
) -> GamingReport:
    """Run every anti-gaming check and bundle the verdict."""
    held_out_delta = trained_mean - base_mean
    checks = [
        check_transfer_followed_reward(base_mean, trained_mean,
                                       train_signal_rose=train_signal_rose),
        check_arena_collapse(trained_diversity),
        check_clamp_fraction(clamp_fraction),
        check_seed_noise(held_out_delta, seed_std),
        check_train_game_only(held_out_delta, probe_deltas),
        check_distinct_models(base_resolved_id, trained_resolved_id, is_smoke=is_smoke),
    ]
    return GamingReport(checks=checks)
