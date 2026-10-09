# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interval algebra on (start, end) pairs in one clock domain: unions, intersections, overlap."""

from __future__ import annotations

from typing import Iterable


Interval = tuple[float, float]


def normalize(intervals: Iterable[Interval]) -> list[Interval]:
    """Sorted, merged (touching intervals merge), empty and inverted intervals dropped."""
    items = sorted((float(s), float(e)) for s, e in intervals if e is not None and s is not None and e > s)
    merged: list[Interval] = []
    for start, end in items:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total(intervals: Iterable[Interval]) -> float:
    return float(sum(e - s for s, e in normalize(intervals)))


def union(*groups: Iterable[Interval]) -> list[Interval]:
    return normalize(interval for group in groups for interval in group)


def intersect(a: Iterable[Interval], b: Iterable[Interval]) -> list[Interval]:
    left, right = normalize(a), normalize(b)
    result: list[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def subtract(a: Iterable[Interval], b: Iterable[Interval]) -> list[Interval]:
    """Parts of ``a`` not covered by ``b``."""
    result: list[Interval] = []
    holes = normalize(b)
    for start, end in normalize(a):
        cursor = start
        for h_start, h_end in holes:
            if h_end <= cursor or h_start >= end:
                continue
            if h_start > cursor:
                result.append((cursor, h_start))
            cursor = max(cursor, h_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def clip(intervals: Iterable[Interval], window: Interval) -> list[Interval]:
    return intersect(intervals, [window])


def overlap_time(intervals: Iterable[Interval]) -> float:
    """Total time during which two or more of the raw intervals are active."""
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        if end is not None and start is not None and end > start:
            events.append((float(start), 1))
            events.append((float(end), -1))
    events.sort(key=lambda e: (e[0], e[1]))  # ends before starts at equal times
    active = 0
    last = 0.0
    overlap = 0.0
    for time, delta in events:
        if active >= 2:
            overlap += time - last
        active += delta
        last = time
    return overlap
