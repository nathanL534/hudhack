"""training/anti_collapse.py — the STRONG anti-collapse mechanism for the Teacher RFT.

THE PROBLEM. Both prior league runs (leagueB128 / leagueB256) reward-hacked by
COLLAPSING arena diversity: the Teacher learned to emit the SAME high-reward arena
over and over, because a GRPO group of identical arenas still gets a clean gradient
toward "more of that arena". The earlier ×0.5 same-group penalty did nothing — GRPO
normalizes advantages WITHIN the group ((r - mean) / (std + eps)), so a uniform
multiplicative penalty cancels out of the advantage entirely. It is structurally
invisible to the optimizer.

THE FIX (this module). Diversity is enforced at the GROUP-CONSTRUCTION level, not via
a reward shaping term GRPO can normalize away:

  1. OVERSAMPLE: the Teacher generates ``oversample`` (16) candidate arenas per group.
  2. CANONICALIZE: each arena's 5-knob param tuple is rounded to a stable key.
  3. SELECT: greedily pick ``target`` (8) arenas by MAXIMUM pairwise parameter distance
     (farthest-point sampling) on the [0,1]-normalized knobs, REQUIRING >= ``min_unique``
     (4) distinct canonical tuples among the selected. If the candidate pool can't
     supply that, the caller RETRIES generation; after ``max_retries`` it SKIPS the
     optimizer update for that group entirely (better no step than a collapsed one).
  4. DUPLICATE-ZERO: any selected arena that is an exact canonical duplicate of an
     earlier-selected one gets reward 0 (it carries no new information).
  5. NOVELTY BONUS: each selected arena's reward gains
     ``0.10 * normalized_mean_distance`` to the OTHER selected arenas (its mean [0,1]
     knob distance), so the Teacher is pushed toward spreading the group out — a term
     that does NOT cancel under GRPO normalization because it VARIES across the group.
  6. BEHAVIOR GATE: an arena whose trained Student camps / punch-spams / only wins by
     timeout gets ``behavior_gated_improvement = 0`` (the diversity bonus is still
     added on top, but a degenerate Student earns nothing for the transfer itself).

Everything here is a PURE function of its inputs (no Modal, no torch) so it is unit-
testable on the host and auditable in one place. The trainer imports these and threads
them between sampling and the GRPO update.
"""

from __future__ import annotations

from typing import Optional

# The five Ring-Out knobs, in a STABLE order, with their (lo, hi) bounds — mirrored
# from train_teacher_modal.RING_OUT_BOUNDS. Kept here (not imported) so this module is
# import-light and the order is pinned next to the code that uses it. The trainer
# asserts parity against RING_OUT_BOUNDS at load (see _assert_knob_parity there).
KNOB_ORDER: tuple[str, ...] = (
    "difficulty",
    "platform_width",
    "gravity",
    "knockback",
    "spawn_gap",
)
KNOB_BOUNDS: dict[str, tuple[float, float]] = {
    "difficulty": (0.0, 1.0),
    "platform_width": (8.0, 30.0),
    "gravity": (0.2, 1.2),
    "knockback": (0.5, 6.0),
    "spawn_gap": (1.0, 12.0),
}

# Canonicalization rounding: knobs within this many decimals are the SAME arena. 2 dp
# is coarse enough that float jitter ("0.40000001" vs "0.4") collapses to one key but
# fine enough that genuinely different arenas stay distinct.
_CANON_DECIMALS = 2

# Default novelty-bonus weight (the 0.10 in the spec).
DEFAULT_NOVELTY_WEIGHT = 0.10


def canonical_tuple(params: Optional[dict]) -> Optional[tuple]:
    """Round a parsed 5-knob arena to a stable, hashable canonical key.

    Returns ``None`` for an invalid/missing arena (no params, or a missing knob) so the
    caller treats it as a non-candidate. Two arenas with the same canonical tuple are
    EXACT duplicates for the duplicate-zeroing + unique-count logic.
    """
    if not params:
        return None
    out = []
    for k in KNOB_ORDER:
        if k not in params:
            return None
        out.append(round(float(params[k]), _CANON_DECIMALS))
    return tuple(out)


def normalize_knobs(params: dict) -> list[float]:
    """Map a 5-knob arena to its [0,1]-normalized vector (per-knob min-max).

    Each knob is scaled by its (lo, hi) bound and clamped to [0,1], so the pairwise
    distances used for farthest-point selection + the novelty bonus weight every knob
    equally regardless of its raw scale (difficulty 0-1 vs platform_width 8-30).
    """
    vec = []
    for k in KNOB_ORDER:
        lo, hi = KNOB_BOUNDS[k]
        span = hi - lo
        v = (float(params[k]) - lo) / span if span > 0 else 0.0
        vec.append(min(1.0, max(0.0, v)))
    return vec


def _euclid(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def greedy_farthest_select(
    norm_vecs: list[list[float]],
    canon: list[Optional[tuple]],
    *,
    target: int,
    min_unique: int,
) -> tuple[Optional[list[int]], int]:
    """Greedy farthest-point selection of ``target`` candidate indices.

    Operates on the [0,1]-normalized knob vectors. Classic farthest-point sampling:
    seed with the candidate farthest from the centroid (the most extreme arena), then
    repeatedly add the candidate whose MINIMUM distance to the already-selected set is
    largest — maximizing the spread of the chosen group.

    Only VALID candidates (non-None canonical tuple) are eligible. Returns
    ``(selected_indices, n_unique)`` where ``n_unique`` is the count of distinct
    canonical tuples among the selected indices. If fewer than ``min_unique`` distinct
    arenas can be assembled, returns ``(None, n_unique)`` so the caller knows to retry /
    skip. If there are at least ``min_unique`` uniques but fewer than ``target`` total
    valid candidates, it returns as many as it has (still >= min_unique).
    """
    valid_idx = [i for i, c in enumerate(canon) if c is not None]
    distinct = {canon[i] for i in valid_idx}
    if len(distinct) < min_unique:
        # Can't even reach the unique floor from this pool — signal retry/skip.
        return None, len(distinct)

    # Seed: the candidate farthest from the centroid of all valid candidates.
    dim = len(KNOB_ORDER)
    centroid = [0.0] * dim
    for i in valid_idx:
        for d in range(dim):
            centroid[d] += norm_vecs[i][d]
    n = len(valid_idx)
    centroid = [c / n for c in centroid]

    selected: list[int] = [max(valid_idx, key=lambda i: _euclid(norm_vecs[i], centroid))]
    remaining = [i for i in valid_idx if i != selected[0]]

    while len(selected) < target and remaining:
        # Add the remaining candidate whose min-distance to the selected set is largest.
        best = max(
            remaining,
            key=lambda i: min(_euclid(norm_vecs[i], norm_vecs[s]) for s in selected),
        )
        selected.append(best)
        remaining.remove(best)

    n_unique = len({canon[i] for i in selected})
    # Defensive: if greedy happened to under-fill uniques (e.g. target small but many
    # dups), the pool-level distinct check above already guaranteed >= min_unique exist,
    # and farthest-point preferentially spreads, so this should hold. Verify anyway.
    if n_unique < min_unique:
        return None, n_unique
    return selected, n_unique


def novelty_bonuses(norm_vecs_selected: list[list[float]]) -> list[float]:
    """Per-arena novelty bonus signal: each selected arena's MEAN [0,1] knob distance
    to the OTHER selected arenas.

    Returns the raw mean-distance per arena (NOT yet weighted) so the caller multiplies
    by the novelty weight. A lone arena (group of 1) gets 0.0 (no "others" to be novel
    against). This term VARIES across the group, so unlike a uniform multiplicative
    penalty it survives GRPO's per-group advantage normalization.
    """
    m = len(norm_vecs_selected)
    if m <= 1:
        return [0.0] * m
    out = []
    for i in range(m):
        dsum = sum(
            _euclid(norm_vecs_selected[i], norm_vecs_selected[j])
            for j in range(m)
            if j != i
        )
        out.append(dsum / (m - 1))
    return out


def behavior_pass(diag: Optional[dict]) -> bool:
    """The BEHAVIOR GATE predicate over a reward row's behavior_diag.

    Passes only a Student that actually FIGHTS:
        movement_fraction       >= 0.10   (it moves, not camps)
        AND punch_fraction       <= 0.90   (it doesn't only punch-spam)
        AND real_ringout_win_rate >= 0.20  (it wins by REAL ring-outs, not timeout/tiebreak)
    A missing diag (probe didn't run / row invalid) FAILS the gate — we never credit a
    Student whose behavior we couldn't verify.
    """
    if not diag:
        return False
    mv = float(diag.get("movement_fraction", 0.0))
    pf = float(diag.get("punch_fraction", 1.0))
    ro = float(diag.get("real_ringout_win_rate", 0.0))
    return (mv >= 0.10) and (pf <= 0.90) and (ro >= 0.20)


def adjusted_group_rewards(
    selected_params: list[Optional[dict]],
    selected_canon: list[Optional[tuple]],
    selected_improvements: list[float],
    selected_behavior: list[Optional[dict]],
    *,
    novelty_weight: float = DEFAULT_NOVELTY_WEIGHT,
) -> tuple[list[float], list[dict]]:
    """Build the FINAL per-arena GRPO rewards for a selected group.

    For each selected arena i:
        behavior_gated_improvement_i = improvement_i  if behavior_pass(diag_i) else 0
        novelty_i                    = novelty_weight * mean_norm_distance_i
        adjusted_reward_i            = behavior_gated_improvement_i + novelty_i
        ... EXCEPT an exact canonical duplicate of an earlier-selected arena => 0.0
            (a duplicate carries no new information; it earns nothing, not even novelty).

    Returns ``(adjusted_rewards, per_arena_diag)`` where per_arena_diag carries the
    component breakdown for the audit log (so every reward row is explainable).
    """
    # Normalized vectors for the novelty bonus (over the SELECTED group only).
    norm_vecs = [
        normalize_knobs(p) if p else [0.0] * len(KNOB_ORDER) for p in selected_params
    ]
    raw_novelty = novelty_bonuses(norm_vecs)

    seen: set[tuple] = set()
    adjusted: list[float] = []
    diags: list[dict] = []
    for i, (params, canon, improvement, behav) in enumerate(
        zip(selected_params, selected_canon, selected_improvements, selected_behavior)
    ):
        is_dup = canon is not None and canon in seen
        if canon is not None:
            seen.add(canon)

        passed = behavior_pass(behav)
        gated = float(improvement) if passed else 0.0
        novelty = float(novelty_weight) * float(raw_novelty[i])

        if is_dup:
            reward = 0.0  # exact duplicate -> 0 (overrides gate + novelty).
        else:
            reward = gated + novelty

        adjusted.append(reward)
        diags.append(
            {
                "raw_improvement": round(float(improvement), 4),
                "behavior_pass": bool(passed),
                "behavior_gated_improvement": round(gated, 4),
                "novelty_mean_distance": round(float(raw_novelty[i]), 4),
                "novelty_bonus": round(novelty, 4),
                "is_exact_duplicate": bool(is_dup),
                "adjusted_reward": round(reward, 4),
                "behavior_diag": behav,
            }
        )
    return adjusted, diags
