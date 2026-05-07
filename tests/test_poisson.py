from __future__ import annotations

import asyncio
import math
import time

import pytest

from testbed.poisson import arrival_offsets, arrivals


def test_arrival_offsets_deterministic():
    a = arrival_offsets(0.5, 100, 42)
    b = arrival_offsets(0.5, 100, 42)
    assert a == b
    c = arrival_offsets(0.5, 100, 43)
    assert a != c


def test_arrival_offsets_strictly_monotonic():
    offsets = arrival_offsets(2.0, 200, 7)
    for prev, nxt in zip(offsets, offsets[1:]):
        assert nxt > prev


def test_arrival_offsets_mean_close_to_inverse_rate():
    rate = 4.0
    n = 10_000
    offsets = arrival_offsets(rate, n, 1)
    inter = [offsets[0]] + [b - a for a, b in zip(offsets, offsets[1:])]
    mean = sum(inter) / len(inter)
    # Within 5% of 1/rate at this n; gives a generous margin while still catching bugs.
    assert math.isclose(mean, 1.0 / rate, rel_tol=0.05)


def test_arrival_offsets_invalid():
    with pytest.raises(ValueError):
        arrival_offsets(0.0, 1, 0)
    with pytest.raises(ValueError):
        arrival_offsets(1.0, -1, 0)


async def test_arrivals_yields_in_order_and_paces():
    offsets = [0.0, 0.05, 0.10]
    start = time.monotonic()
    seen: list[tuple[int, float]] = []
    async for i in arrivals(offsets, start_monotonic=start):
        seen.append((i, time.monotonic() - start))
    assert [i for i, _ in seen] == [0, 1, 2]
    # Pacing check: the 3rd arrival should not fire before its offset.
    assert seen[2][1] >= 0.10 - 0.01  # 10ms slack for scheduler jitter
