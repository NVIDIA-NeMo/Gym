# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-judge panel support shared by the GDPVal rubric and comparison scorers.

A *panel* is a set of LLM judges (each a distinct upstream model + reasoning
settings, e.g. GPT-5.5 medium, Gemini 3.1 Pro Preview high, Claude Opus 4.8
high). For every individual scoring/comparison we *sample* one panel member —
the per-trial grader is drawn here. Sampling is seeded (via :func:`make_rng`)
so a rerun of the same task lands on the same judges and the result is
reproducible.

This module is intentionally connection-agnostic: a :class:`ResolvedJudge`
carries only the upstream coordinates (base URL / model / api key / create
overrides). The rubric scorers build an ``AsyncOpenAI`` client from these; the
comparison scorer wraps them in its own client-bearing ``Judge`` for the
threaded sync path.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, TypeVar


@dataclass
class ResolvedJudge:
    """A single panel member resolved to concrete upstream coordinates.

    ``create_overrides`` holds provider-specific generation/reasoning knobs
    (e.g. ``reasoning_effort``, ``extra_body``, ``temperature``) that are merged
    into the ``chat.completions.create`` kwargs via :func:`merge_create_kwargs`.
    A ``None`` value there *removes* the default key (so a reasoning model can
    drop ``temperature``).
    """

    name: str
    base_url: str
    model: str
    api_key: str = "dummy"
    create_overrides: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


class _HasWeight(Protocol):
    name: str
    weight: float


_J = TypeVar("_J", bound=_HasWeight)


def make_rng(seed: Optional[int], *parts: str) -> random.Random:
    """Return a ``random.Random`` seeded deterministically from *seed* + *parts*.

    Used to make per-comparison judge sampling reproducible: callers seed with
    a stable identity (e.g. the task id and reference repeat) so the same task
    always draws the same judges across reruns. When *seed* is ``None`` the
    parts alone determine the stream (still reproducible per task); pass no
    parts for a fully fresh stream.
    """
    payload = "|".join([repr(seed), *parts])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def sample_judge(judges: Sequence[_J], rng: random.Random) -> _J:
    """Sample one judge from the panel using each member's ``weight``.

    Duck-typed on ``.weight`` / ``.name`` so it works for both
    :class:`ResolvedJudge` (rubric path) and the comparison scorer's
    client-bearing ``Judge``. Non-positive total weight falls back to a uniform
    choice.
    """
    if not judges:
        raise ValueError("sample_judge requires a non-empty judge panel")
    if len(judges) == 1:
        return judges[0]
    weights = [j.weight if (j.weight and j.weight > 0) else 0.0 for j in judges]
    if sum(weights) <= 0:
        return rng.choice(list(judges))
    return rng.choices(list(judges), weights=weights, k=1)[0]


def merge_create_kwargs(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-judge ``create_overrides`` onto *base* create kwargs.

    A ``None`` override value deletes the key from the result (lets a reasoning
    judge drop ``temperature``); any other value replaces the default.
    """
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def panel_summary(judges: Sequence[_HasWeight]) -> List[Dict[str, Any]]:
    """A small JSON-friendly description of the panel for verify responses."""
    return [{"name": j.name, "weight": j.weight} for j in judges]
