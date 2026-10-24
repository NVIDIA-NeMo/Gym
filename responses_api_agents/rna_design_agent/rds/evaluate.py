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
"""The ground-truth protocol, applied after the run from the campaign's own definition.

Nothing the agent measured is trusted, so optimising a different number earns nothing here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from . import registry
from .campaign import Campaign
from .registry import dispatch


def _path_value(row: Any, path: list[str]) -> Any:
    for key in path:
        row = row.get(key) if isinstance(row, dict) else None
    return row


def _tool_of(metric: str) -> tuple[str, list[str]]:
    group, verb, *rest = metric.split(".")
    return f"{group}.{verb}", rest


def protocol_fingerprint(campaign: Campaign) -> str:
    """A short hash of what decided this verdict, so rows scored differently are never averaged together.

    It covers the metric, direction, gates, threshold, target and the scorer functions' source.
    """
    import hashlib
    import inspect

    from . import registry
    from .targets import TARGETS

    scoring = campaign.scoring
    parts = [
        scoring.primary,
        scoring.direction,
        repr(sorted(scoring.gates.items())),
        str(scoring.pass_threshold),
        str(scoring.top_k),
    ]
    target = TARGETS.get(campaign.target_name)
    if target is not None:
        parts += [
            getattr(target, "target_structure", ""),
            getattr(target, "fixed_prefix", ""),
            repr(sorted((getattr(target, "conserved", None) or {}).items())),
            str(getattr(target, "max_free_energy", None)),
        ]
    for metric in [scoring.primary, *sorted(scoring.gates)]:
        tool, _ = _tool_of(metric)
        try:
            parts.append(inspect.getsource(registry.get(tool).fn))
        except (KeyError, OSError, TypeError):
            parts.append(f"{tool}:unavailable")
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]


def _dispatch_scorer(tool: str, sequences: list[str], campaign: Campaign) -> dict:
    """Run a scorer, naming the target only for the ones that take one."""
    arguments = {"sequences": sequences}
    if "target" in {f.name for f in dataclasses.fields(registry.get(tool).model)}:
        arguments["target"] = campaign.target_name
    return dispatch(tool, arguments)


def ground_truth(sequences: list[str], campaign: Campaign) -> dict:
    """Score sequences by the campaign's protocol, computed from scratch."""
    if not sequences:
        return {
            "n": 0,
            "best": None,
            "median": None,
            "solved": False,
            "n_passing_gates": 0,
            "protocol": protocol_fingerprint(campaign),
        }
    metric = campaign.scoring.primary
    tool, path = _tool_of(metric)
    scored = _dispatch_scorer(tool, sequences, campaign)
    rows = []
    for row in scored["results"]:
        value = _path_value(row, path)
        if value is not None:
            rows.append({"sequence": row["sequence"], "value": float(value)})

    gate_values: dict[str, dict[str, Any]] = {}
    for gate in campaign.scoring.gates:
        gate_tool, gate_path = _tool_of(gate)
        result = _dispatch_scorer(gate_tool, sequences, campaign)
        for row in result["results"]:
            gate_values.setdefault(row["sequence"], {})[gate] = _path_value(row, gate_path)

    # The same rule the campaign ranks by: a value, or a {"min", "max"} range, per gate.
    passing = [r for r in rows if campaign.scoring.passes_gates(gate_values.get(r["sequence"], {}))]
    ranked = sorted(passing, key=lambda r: r["value"], reverse=campaign.scoring.direction == "max")
    best = ranked[0]["value"] if ranked else None
    return {
        "metric": metric,
        "protocol": protocol_fingerprint(campaign),
        "n": len(sequences),
        "n_passing_gates": len(passing),
        "best": best,
        "median": ranked[len(ranked) // 2]["value"] if ranked else None,
        "solved": campaign.scoring.solved(best),
        "top": [r["sequence"] for r in ranked[:5]],
    }


def official_result(campaign: Campaign, out_dir: Path, pool: list[str]) -> dict:
    """Score what the run designed, by the campaign's own protocol."""
    return {"designed": ground_truth(pool, campaign)}
