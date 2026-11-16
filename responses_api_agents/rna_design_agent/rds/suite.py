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
"""A suite: many targets sharing one objective, budget and disclosure, so a new benchmark is a data change."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .campaign import Campaign, Scoring


@dataclass(frozen=True)
class Suite:
    name: str
    #: The kind of design task, which results are grouped by.
    category: str
    source: str
    scoring: Scoring
    targets: list[dict[str, Any]]
    budget: dict[str, Any] = field(default_factory=dict)
    controller: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    #: Tools the suite withholds from every arm.
    exclude: list[str] = field(default_factory=list)
    #: Disclosure, applied to every target that does not state its own.
    provides: list[str] = field(default_factory=list)
    #: "atomic" or "composite": which packaging the arms are offered.
    exposure: str = "atomic"
    #: "strict" or "leaky": whether a tool must declare what the task supplies.
    disclosure: str = "strict"
    #: "derived" or "handwritten": where the control's cycle comes from.
    control: str = "derived"
    #: The hand-written control, used when control is "handwritten".
    control_cycle: list[Any] = field(default_factory=list)
    #: "metric", "goal" or "hidden": what the agent is told about its scoring.
    objective_visibility: str = "metric"
    #: Rounds between scheduled reflections. 0 leaves them off.
    reflect_every: int = 0
    #: Whether the agent may write and run its own code.
    allow_code: bool = False
    #: The generator the derived control arms draw from, fixed by the suite rather than the run command.
    generator: str = "generate.grnade"
    notes: str = ""

    @staticmethod
    def load(path: str | Path) -> Suite:
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        missing = {"name", "category", "source", "scoring", "targets"} - set(data)
        if missing:
            raise ValueError(f"{path}: suite is missing {sorted(missing)}")
        return Suite(
            name=data["name"],
            category=data["category"],
            source=data["source"],
            scoring=Scoring(**data["scoring"]),
            targets=list(data["targets"]),
            budget=data.get("budget") or {},
            controller=data.get("controller") or {},
            tools=list(data.get("tools") or []),
            exclude=list(data.get("exclude") or []),
            provides=list(data.get("provides") or []),
            exposure=data.get("exposure", "atomic"),
            disclosure=data.get("disclosure", "strict"),
            control=data.get("control", "derived"),
            control_cycle=list(data.get("control_cycle") or []),
            objective_visibility=data.get("objective_visibility", "metric"),
            reflect_every=int(data.get("reflect_every") or 0),
            allow_code=bool(data.get("allow_code", False)),
            generator=data.get("generator", "generate.grnade"),
            notes=data.get("notes", ""),
        )

    def campaigns(self) -> list[Campaign]:
        """One campaign per target, sharing everything else."""
        out = []
        for target in self.targets:
            inline = dict(target)
            inline.setdefault("provides", list(self.provides))
            out.append(
                Campaign(
                    name=f"{self.name}/{target['name']}",
                    target="inline",
                    scoring=self.scoring,
                    budget=dict(self.budget),
                    controller=dict(self.controller),
                    tools=list(self.tools),
                    exclude=list(self.exclude),
                    notes=f"{self.source}, {self.category}",
                    exposure=self.exposure,
                    disclosure=self.disclosure,
                    control=self.control,
                    control_cycle=list(self.control_cycle),
                    objective_visibility=self.objective_visibility,
                    reflect_every=self.reflect_every,
                    allow_code=self.allow_code,
                    inline_target=inline,
                )
            )
        return out
