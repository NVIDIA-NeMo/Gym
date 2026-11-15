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
"""Campaign targets: the fixed facts a design must satisfy, kept as data apart from the tools that check them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """One shape a design has to take, and the condition ("FMN", "+MS2") that produces it.

    `bound` are the 1-indexed positions a ligand holds, forced to pair when folding the bound state.
    """

    structure: str
    condition: str = ""
    bound: tuple[int, ...] = ()
    ligand: str = ""


@dataclass(frozen=True)
class Target:
    name: str
    wild_type: str  # empty when a campaign has no reference sequence
    length: int
    fixed_prefix: str
    conserved: dict[int, str]  # 1-indexed, as the challenge lists them
    min_changes_from_wt: int
    max_free_energy: float | None  # EternaFold dG must be >= this, None disables the gate
    target_structure: str  # extended dot-bracket, [] and {} are pseudoknot layers
    #: What the challenge hands the designer. A tool is offered only when everything it needs is here.
    #: Derived from the other fields unless a campaign states it.
    provides: frozenset[str] = frozenset()
    #: Path to the structure the challenge supplies, when it supplies one.
    structure_path: str = ""
    #: Small molecules present in the task, as CCD codes or SMILES.
    ligands: tuple[str, ...] = ()
    #: Eterna-style count limits ("at least 20 GC pairs", "at most 4 in a row"). None means not asked.
    pair_limits: dict[str, tuple[int | None, int | None]] | None = None
    max_consecutive: int | None = None
    #: The other shapes a switch must take. `target_structure` is the first state.
    states: tuple[State, ...] = ()

    def __post_init__(self) -> None:
        if not self.provides:
            object.__setattr__(self, "provides", self.derived_provides())

    def all_states(self) -> tuple[State, ...]:
        """Every shape this target asks for, the first one included."""
        first = (State(structure=self.target_structure, condition="apo"),)
        return first + tuple(self.states) if self.target_structure else tuple(self.states)

    def derived_provides(self) -> frozenset[str]:
        """Read the inputs off the fields the target already carries."""
        supplied = {"secondary_structure"} if self.target_structure else set()
        if self.wild_type:
            supplied.add("wild_type")
        if self.structure_path:
            supplied.add("backbone_3d")
        if self.ligands or any(s.ligand for s in self.states):
            supplied.add("ligand")
        if self.conserved or self.fixed_prefix:
            supplied.add("sequence_constraints")
        if self.states:
            supplied.add("states")
        return frozenset(supplied)


TARGETS: dict[str, Target] = {}


def register(target: Target) -> Target:
    """Add a target at runtime, so a campaign can carry its own."""
    TARGETS[target.name] = target
    return target


def _states(data: dict) -> tuple[State, ...]:
    """States as a campaign writes them: a list of dicts, or nothing."""
    out = []
    for raw in data.get("states") or ():
        if isinstance(raw, str):
            out.append(State(structure=raw))
            continue
        out.append(
            State(
                structure=raw["structure"],
                condition=raw.get("condition", ""),
                bound=tuple(int(i) for i in raw.get("bound") or ()),
                ligand=raw.get("ligand", ""),
            )
        )
    return tuple(out)


def from_dict(data: dict) -> Target:
    """Build a target from a campaign's `inline_target` block."""
    missing = {"name", "length", "target_structure"} - set(data)
    if missing:
        raise ValueError(f"inline target is missing {sorted(missing)}")
    if len(data["target_structure"]) != data["length"]:
        raise ValueError(f"structure length {len(data['target_structure'])} != declared {data['length']}")
    return Target(
        name=data["name"],
        wild_type=data.get("wild_type", ""),
        length=data["length"],
        fixed_prefix=data.get("fixed_prefix", ""),
        conserved={int(k): v for k, v in (data.get("conserved") or {}).items()},
        min_changes_from_wt=data.get("min_changes_from_wt", 0),
        max_free_energy=data.get("max_free_energy"),
        target_structure=data["target_structure"],
        states=_states(data),
        pair_limits={k: (v.get("min"), v.get("max")) for k, v in (data.get("pair_limits") or {}).items()} or None,
        max_consecutive=data.get("max_consecutive"),
        structure_path=data.get("structure_path", ""),
        ligands=tuple(data.get("ligands") or ()),
        provides=frozenset(data.get("provides") or ()),
    )
