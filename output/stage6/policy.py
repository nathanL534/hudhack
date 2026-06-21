"""output/stage6/policy.py — frozen-policy serialization for Student-vs-Student.

THE primary-metric enabler. The legacy Modal worker trained a fresh PPO Player and
returned only its SCORES (a held-out win-rate vs a fixed bot). The new headline
metric is a DIRECT head-to-head between two Students, so a Student's policy must be
*serialized once, frozen, and reused* across many matches WITHOUT retraining — and
tagged with where it came from (Teacher, curriculum id, training seed) so the
result JSON can prove no circularity.

This module is deliberately tiny and dependency-light:

  * ``PolicyArtifact`` — a value object: the base64-encoded torch ``state_dict`` of
    the SMALL MLP policy (NOT the whole PPO object), an architecture fingerprint,
    a content checksum, and provenance tags (teacher, curriculum_id, seed, game).
  * ``serialize_policy`` / ``restore_policy`` — turn a trained SB3 model's policy
    into a portable artifact and back into a deterministic ``obs -> action``
    callable, by loading the weights into a fresh net of the SAME architecture.
  * ``checksum_of`` — a stable content hash so two artifacts can be compared and a
    "same policy fought itself" circularity can be detected.

The artifact is JSON-serializable (the base64 blob is a plain string), so it rides
the Modal return path with no extra infrastructure: a worker trains a Student and
returns its ``PolicyArtifact`` dict; a second worker (or the local driver) restores
two artifacts and fights them. No filesystem, no Modal Volume, no retraining.

Why state_dict and not ``PPO.save``: ``PPO.save`` pickles the whole algorithm
(optimizer, rollout buffer, env spec) — large and brittle across versions. The
policy ``state_dict`` is just the MLP weights (~45 KB here), restores into a fresh
``PPO("MlpPolicy", net_arch=[64,64])`` deterministically, and is what we actually
fight. Restoring is verified to reproduce identical actions (see test suite).
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# The Student architecture fingerprint. Both Students MUST share this — it is part
# of the fairness invariant (identical architecture). Recorded on every artifact so
# a mismatch is detectable in the result JSON.
DEFAULT_NET_ARCH: tuple[int, ...] = (64, 64)


def _arch_fingerprint(net_arch) -> str:
    return "mlp:" + "-".join(str(int(h)) for h in net_arch)


def _parse_arch(fingerprint: str) -> tuple[int, ...]:
    """Inverse of ``_arch_fingerprint``: ``"mlp:256-256"`` -> ``(256, 256)``.

    Lets ``restore_policy`` rebuild a Student at the EXACT width it was trained at,
    read from the artifact itself — so a [128,128] or [256,256] Student loads into a
    matching net instead of the hardcoded default (a shape mismatch otherwise).
    Falls back to ``DEFAULT_NET_ARCH`` for an empty/malformed tag.
    """
    try:
        body = (fingerprint or "").split(":", 1)[1]
        arch = tuple(int(h) for h in body.split("-") if h != "")
        return arch or DEFAULT_NET_ARCH
    except (IndexError, ValueError):
        return DEFAULT_NET_ARCH


def checksum_of(state_dict_b64: str) -> str:
    """Stable 16-hex content hash of a base64 weight blob."""
    return hashlib.sha256(state_dict_b64.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PolicyArtifact:
    """A frozen, portable Student policy + its provenance.

    Everything here is JSON-serializable so it rides the Modal return path. The
    ``state_dict_b64`` is the ONLY heavy field (~60 KB base64); the rest are tags
    the result JSON records to prove the head-to-head was not circular.
    """

    state_dict_b64: str
    arch: str                       # architecture fingerprint, e.g. "mlp:64-64"
    checksum: str                   # content hash of the weights
    teacher: str                    # "base" | "trained" (which Teacher's curriculum)
    curriculum_id: str              # the replicate/curriculum this Student trained on
    seed: int                       # the PPO training seed
    game: str = "fighter"           # which game the Student plays
    obs_dim: int = 11               # observation dimensionality (sanity guard)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "state_dict_b64": self.state_dict_b64,
            "arch": self.arch,
            "checksum": self.checksum,
            "teacher": self.teacher,
            "curriculum_id": self.curriculum_id,
            "seed": int(self.seed),
            "game": self.game,
            "obs_dim": int(self.obs_dim),
            "extra": dict(self.extra),
        }

    def tag(self) -> dict:
        """The provenance tag WITHOUT the heavy weight blob (for logging/JSON)."""
        d = self.as_dict()
        d.pop("state_dict_b64", None)
        d["bytes"] = len(self.state_dict_b64)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyArtifact":
        return cls(
            state_dict_b64=d["state_dict_b64"],
            arch=d.get("arch", _arch_fingerprint(DEFAULT_NET_ARCH)),
            checksum=d.get("checksum") or checksum_of(d["state_dict_b64"]),
            teacher=d.get("teacher", ""),
            curriculum_id=d.get("curriculum_id", ""),
            seed=int(d.get("seed", 0)),
            game=d.get("game", "fighter"),
            obs_dim=int(d.get("obs_dim", 11)),
            extra=dict(d.get("extra", {})),
        )


def serialize_policy(
    model: Any,
    *,
    teacher: str,
    curriculum_id: str,
    seed: int,
    game: str = "fighter",
    obs_dim: int = 11,
    net_arch=DEFAULT_NET_ARCH,
    extra: dict | None = None,
) -> PolicyArtifact:
    """Serialize a trained SB3 model's POLICY (not the whole algorithm) to an artifact.

    Saves only ``model.policy.state_dict()`` — the MLP weights we fight — so the blob
    is tiny and version-robust. The artifact carries provenance tags so the
    head-to-head can prove which Teacher's curriculum produced this Student.
    """
    import torch

    buf = io.BytesIO()
    torch.save(model.policy.state_dict(), buf)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    return PolicyArtifact(
        state_dict_b64=b64,
        arch=_arch_fingerprint(net_arch),
        checksum=checksum_of(b64),
        teacher=teacher,
        curriculum_id=curriculum_id,
        seed=int(seed),
        game=game,
        obs_dim=int(obs_dim),
        extra=dict(extra or {}),
    )


def restore_policy(
    artifact: "PolicyArtifact | dict",
    env,
    *,
    net_arch=None,
    seed: int = 0,
) -> Callable[[np.ndarray], int]:
    """Restore an artifact into a deterministic ``obs -> action`` callable.

    Loads the weights into a FRESH ``PPO("MlpPolicy", net_arch=...)`` built on
    ``env`` (which only needs to expose the right observation/action spaces), then
    returns the same ``obs -> action`` wrapper the trainer uses. The wrapper is
    deterministic, so a restored policy reproduces the trained policy's actions
    exactly (verified in the test suite).

    ``net_arch`` defaults to the architecture RECORDED ON THE ARTIFACT (so a Student
    trained at [128,128] / [256,256] restores into a matching net, not the [64,64]
    default which would be a shape mismatch). Pass an explicit ``net_arch`` only to
    override the artifact's own record.
    """
    import torch
    from stable_baselines3 import PPO

    art = artifact if isinstance(artifact, PolicyArtifact) else PolicyArtifact.from_dict(artifact)
    if net_arch is None:
        net_arch = _parse_arch(art.arch)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": list(net_arch)},
        device="cpu",
    )
    sd = torch.load(io.BytesIO(base64.b64decode(art.state_dict_b64)), map_location="cpu")
    model.policy.load_state_dict(sd)
    model.policy.set_training_mode(False)

    def act(obs: np.ndarray) -> int:
        action, _ = model.policy.predict(obs, deterministic=True)
        return int(action)

    return act
