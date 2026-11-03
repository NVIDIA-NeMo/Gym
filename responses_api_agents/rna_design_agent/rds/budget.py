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
"""One budget, metered the same way for every arm. The runner stops a campaign once any dimension is spent."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Limits:
    """Zero means unlimited on that dimension."""

    model_calls: int = 0
    tool_calls: int = 0
    gpu_seconds: float = 0.0
    #: Non-GPU compute, so a model's own sampler in `code.run` is billed like the shared generator.
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    tokens: int = 0
    rounds: int = 0
    #: Candidates admitted, so an arm that generates more per call does not get a larger search for free.
    candidates: int = 0
    #: Caps the reflection rounds the campaign's cadence schedules.
    reflections: int = 0


@dataclass
class Spend:
    model_calls: int = 0
    tool_calls: int = 0
    candidates: int = 0
    #: Charged separately so reflecting does not cost candidates.
    reflections: int = 0
    gpu_seconds: float = 0.0
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    tokens: int = 0
    rounds: int = 0


class Budget:
    """Meters a campaign and decides when it is over."""

    def __init__(self, limits: Limits | None = None, clock=time.monotonic) -> None:
        self.limits = limits or Limits()
        self.spend = Spend()
        self._clock = clock
        self._started = clock()

    def tick_wall(self) -> None:
        self.spend.wall_seconds = self._clock() - self._started

    def charge(
        self,
        *,
        model_calls: int = 0,
        tool_calls: int = 0,
        gpu_seconds: float = 0.0,
        cpu_seconds: float = 0.0,
        tokens: int = 0,
        rounds: int = 0,
        candidates: int = 0,
        reflections: int = 0,
    ) -> None:
        self.spend.model_calls += model_calls
        self.spend.tool_calls += tool_calls
        self.spend.gpu_seconds += gpu_seconds
        self.spend.cpu_seconds += cpu_seconds
        self.spend.tokens += tokens
        self.spend.rounds += rounds
        self.spend.candidates += candidates
        self.spend.reflections += reflections
        self.tick_wall()

    def exhausted(self) -> str | None:
        """The name of the first spent dimension, or None."""
        self.tick_wall()
        for name in (
            "model_calls",
            "tool_calls",
            "gpu_seconds",
            "cpu_seconds",
            "wall_seconds",
            "tokens",
            "rounds",
            "candidates",
        ):
            limit = getattr(self.limits, name)
            if limit and getattr(self.spend, name) >= limit:
                return name
        return None

    def snapshot(self) -> dict:
        self.tick_wall()
        return {"limits": asdict(self.limits), "spend": asdict(self.spend)}
