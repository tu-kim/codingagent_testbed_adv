"""Poisson arrival generator. Same (rate, n, seed) is fully deterministic."""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import AsyncIterator


def arrival_offsets(rate: float, n: int, seed: int) -> list[float]:
    """Return n strictly-monotonic arrival offsets in seconds for a Poisson process at `rate` qps."""
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rng = random.Random(seed)
    offsets: list[float] = []
    t = 0.0
    for _ in range(n):
        u = rng.random()
        # u is in [0, 1); guard against the (vanishingly unlikely) u == 0 case
        # so log(0) doesn't blow up.
        if u <= 0.0:
            u = 1e-12
        t += -math.log(1.0 - u) / rate
        offsets.append(t)
    return offsets


async def arrivals(offsets: list[float], start_monotonic: float | None = None) -> AsyncIterator[int]:
    """Yield each index when its monotonic deadline passes."""
    start = time.monotonic() if start_monotonic is None else start_monotonic
    for i, off in enumerate(offsets):
        deadline = start + off
        residual = deadline - time.monotonic()
        if residual > 0:
            await asyncio.sleep(residual)
        yield i
