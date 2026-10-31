# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supply legacy random draws from the supplied stream instead of global state."""

import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")


@dataclass
class RandomContext:
    """Count draws so legacy guessing and randomized defaults stay observable."""

    rng: random.Random
    draws: int = 0


_CURRENT: ContextVar[RandomContext | None] = ContextVar("voicebench_rule_rng", default=None)


@contextmanager
def using_rng(rng: random.Random) -> Iterator[RandomContext]:
    """Bind the supplied random stream for one synchronous scoring call."""
    context = RandomContext(rng)
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def _current() -> RandomContext:
    context = _CURRENT.get()
    if context is None:
        raise RuntimeError("VoiceBench legacy random draw requires a supplied RNG")
    context.draws += 1
    return context


def choice(values: Sequence[T]) -> T:
    """Use the source parser's uniform choice with the supplied stream."""
    return _current().rng.choice(values)


def randint(lower: int, upper: int) -> int:
    """Retain the source instruction check's inclusive random default."""
    return _current().rng.randint(lower, upper)


def sample(values: Sequence[T], k: int) -> list[T]:
    """Retain sampling without replacement for default instruction keywords."""
    return _current().rng.sample(values, k=k)
