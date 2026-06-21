"""training/reward.py — the learnability reward.

This is the scalar the Teacher is optimized against by Fireworks RFT. It is
deliberately trivial and pure so it can be unit-tested and reasoned about:

    reward = (strong - weak)   IF the env sits in the learnable band p*(1-p) > 0.2
             0.0               otherwise

Intuition: we only pay the Teacher for the strong-vs-weak GAP, and only when the
env is genuinely on the knife's edge (not trivial, not impossible). An env where
both players ace it (gap ~0) or both fail it (gap ~0) earns nothing; an env that
a smart agent solves and a dumb one flubs earns the full gap.
"""

from __future__ import annotations

LEARNABLE_GATE = 0.2


def learnability(weak: float, strong: float, p: float) -> float:
    """Return the learnability reward for one scored env."""
    if p * (1.0 - p) > LEARNABLE_GATE:
        return strong - weak
    return 0.0


if __name__ == "__main__":
    # band center p=0.5 -> p(1-p)=0.25 > 0.2 -> pays the gap
    print("center:", learnability(weak=0.2, strong=0.9, p=0.5))   # 0.7
    # trivial p=0.95 -> p(1-p)=0.0475 -> 0
    print("trivial:", learnability(weak=0.95, strong=1.0, p=0.95))  # 0.0
