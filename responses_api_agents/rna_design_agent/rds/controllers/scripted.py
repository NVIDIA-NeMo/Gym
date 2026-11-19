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
"""Non-LLM controllers: the baseline arms, and a stub for testing the loop."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..runner import Action
from ..state import StateView


class ScriptedController:
    """Replays a fixed list of actions. Used to test the loop without a model."""

    name = "scripted"

    def __init__(self, actions: Iterable[Action]) -> None:
        self._actions = list(actions)
        self.seen: list[StateView] = []

    def next_action(self, view: StateView) -> Action:
        self.seen.append(view)
        if not self._actions:
            return Action(Action.STOP, {}, "script exhausted")
        return self._actions.pop(0)


class FixedPipeline:
    """The `fixed` arm: the same cycle every round, blind to the campaign's state.

    Beating it, not a bare generator, is what separates adaptive tool choice from iteration.
    """

    name = "fixed"

    def __init__(self, cycle: list[tuple[str, dict[str, Any]]], seed: int = 101) -> None:
        if not cycle:
            raise ValueError("a fixed pipeline needs at least one step")
        self._cycle = cycle
        self._index = 0
        self._seed = seed

    def next_action(self, view: StateView) -> Action:
        tool, arguments = self._cycle[self._index % len(self._cycle)]
        self._index += 1
        arguments = dict(arguments)
        # A fresh seed per lap, or every repeat is deduplicated away. Offset by the run's seed so
        # replicates differ.
        if "seed" in arguments or tool.startswith("generate"):
            arguments["seed"] = self._seed * 1000 + self._index
        return Action(tool, arguments, rationale=f"fixed cycle step {self._index}")
