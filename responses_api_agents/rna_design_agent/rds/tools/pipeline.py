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
"""Composite tools: one call where the atomic tools need several in order, with the path handoffs done.

A campaign's `exposure` picks atomic or composite tools, so packaging is an experimental axis.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..registry import dispatch, tool
from ..spec import Cost, field


@dataclass
class DesignAndRankIn:
    target: str = field("Campaign target to design for")
    n: int = field("How many candidates to design", default=100)
    gc_bias: float = field("GC fraction the designer aims for", default=0.5)
    seed: int = field("Random seed; a different seed is a different draw", default=101)
    engine: str = field("Folding engine used to rank; EternaFold is the only one here.", default="eternafold")


@tool(
    name="pipeline.design_and_rank",
    summary=(
        "Design candidates for the target's base pairs and rank them by how "
        "well a thermodynamic engine folds them back to it, in a single call. "
        "Equivalent to generate.inverse_fold then score.fold_match. Cheap and "
        "CPU-only: this is the floor a more expensive route has to beat."
    ),
    cost=Cost.CHEAP,
    model=DesignAndRankIn,
    needs=("secondary_structure",),
    steps=("generate.inverse_fold", "score.fold_match"),
)
def pipeline_design_and_rank(inp: DesignAndRankIn) -> dict:
    designed = dispatch(
        "generate.inverse_fold",
        {
            "target": inp.target,
            "n": inp.n,
            "gc_bias": inp.gc_bias,
            "seed": inp.seed,
        },
    )
    sequences = [row["sequence"] for row in designed["results"]]
    scored = dispatch(
        "score.fold_match",
        {
            "sequences": sequences,
            "target": inp.target,
            "engine": inp.engine,
        },
    )
    return {
        "tool": "score.fold_match",  # the scoring step's name, so the metric key matches the atomic tools
        "composite": "pipeline.design_and_rank",
        "steps": ["generate.inverse_fold", "score.fold_match"],
        "target": inp.target,
        "n": len(sequences),
        "best_f1": scored.get("best_f1"),
        "results": scored["results"],
    }
