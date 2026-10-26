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
"""The campaign loop, shared by every arm. Arms differ only in the `Controller` that picks the next action."""

from __future__ import annotations

import dataclasses
import subprocess
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

from . import registry
from .budget import Budget
from .spec import Cost, EnvironmentFault
from .state import CampaignState, StateView


@dataclass(frozen=True)
class Action:
    """A tool call the controller wants made, or a request to stop."""

    tool: str
    arguments: dict[str, Any]
    rationale: str = ""
    #: Why the call's arguments could not be read, when they could not.
    error: str = ""

    STOP = "stop"

    @property
    def is_stop(self) -> bool:
        return self.tool == self.STOP


class Controller(Protocol):
    name: str

    def next_action(self, view: StateView) -> Action: ...


def _sequences_from(result: Any) -> list[str]:
    """Pull candidate sequences out of a tool result, if it produced any."""
    if not isinstance(result, dict):
        return []
    rows = result.get("results")
    if not isinstance(rows, list):
        return []
    return [r["sequence"] for r in rows if isinstance(r, dict) and "sequence" in r]


def _scores_from(result: Any) -> dict[str, dict]:
    """Per-sequence scores from a tool result, keyed by sequence."""
    if not isinstance(result, dict):
        return {}
    tool = result.get("tool", "tool")
    scores: dict[str, dict] = {}
    for row in result.get("results") or []:
        if not isinstance(row, dict) or "sequence" not in row:
            continue
        values: dict[str, Any] = {}
        for key, value in row.items():
            if key == "sequence":
                continue
            if isinstance(value, (int, float, bool)):
                values[f"{tool}.{key}"] = value
            elif isinstance(value, dict):
                # Flatten one level, so {"match": {"f1": ...}} becomes `struct.rnet_ss.match.f1`.
                values.update({f"{tool}.{key}.{k}": v for k, v in value.items() if isinstance(v, (int, float, bool))})
        if values:
            scores[row["sequence"]] = values
    return scores


#: Names for candidates already in the state, expanded by the loop before the tool sees them.
SELECTORS = {
    "@all": "every candidate generated so far",
    "@unscored": "candidates the tool you are calling has not scored yet",
    "@top:N": "the N best candidates by the ranked metric",
}


def _expand_selectors(arguments: dict[str, Any], state: CampaignState, rank_by: str, tool: str = "") -> dict[str, Any]:
    """Replace a `sequences` selector with the sequences it names."""
    value = arguments.get("sequences")
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.startswith("@"):
        return arguments
    # Score keys use the dotted registry name, but a model calls the wire name.
    tool = registry.get(tool).name if tool else ""
    return {**arguments, "sequences": state.select(value, rank_by=rank_by, by_tool=tool)}


@contextmanager
def _only(controller: Controller, tool: str):
    """Restrict a controller's menu for one round, then put it back. This is what enforces a reflection round."""
    allowed = getattr(controller, "allowed", None)
    if not hasattr(controller, "allowed"):
        yield
        return
    controller.allowed = {tool}
    try:
        yield
    finally:
        controller.allowed = allowed


#: What a failure caused by the machine looks like. Anything else a tool raises is the request's fault.
ENVIRONMENT_FAULTS = (EnvironmentFault, ImportError, MemoryError, subprocess.SubprocessError)


def apply_tool_call(
    tool: str,
    arguments: dict[str, Any],
    state: CampaignState,
    budget: Budget,
    rank_by: str,
    objective: dict[str, Any] | None,
    failures: list[str],
    offered: set[str] | None = None,
    invalid: str = "",
) -> Any:
    """Run one tool call against a campaign and fold its result into the state. Returns the tool's result.

    Shared by the campaign loop and by harnesses that call tools one at a time. `offered` is set for a call a model
    chose: it must name an offered tool and may not set an operator field. None means an operator-written control.
    `invalid` says why the model's arguments could not be read, and the call fails with it.
    """
    started = time.monotonic()
    length = (objective or {}).get("target_length")
    rna = f"{length} nt of A, C, G and U" if length else "written in A, C, G and U"
    try:
        if invalid:
            raise ValueError(f"{tool}: {invalid}")
        if offered is not None:
            spec = registry.get(tool)
            if spec.name not in offered:
                raise ValueError(f"{tool} is not offered for this task; available: {', '.join(sorted(offered))}")
            fixed = {f.name for f in dataclasses.fields(spec.model) if f.metadata.get("operator")} & set(arguments)
            if fixed:
                raise ValueError(f"{', '.join(sorted(fixed))} is set by the operator, not in a tool call")
        call = _expand_selectors(arguments, state, rank_by, tool)
        typed = call.get("sequences")
        if isinstance(typed, list):
            # One mistyped sequence would fail a scorer's whole call, so it is dropped like a generated one below.
            wrong = [
                s for s in typed if not isinstance(s, str) or (length and len(s) != length) or set(s) - set("ACGU")
            ]
            if wrong:
                failures.append(f"{tool}: dropped {len(wrong)} sequences that are not {rna}")
                call = {**call, "sequences": [s for s in typed if s not in wrong]}
        result = registry.dispatch(tool, call)
    except Exception as exc:  # a failing tool is data, not a crash
        environment = isinstance(exc, ENVIRONMENT_FAULTS)
        failures.append(f"{tool}: {type(exc).__name__}: {exc}")
        budget.charge(tool_calls=1)
        state.record(
            "tool_error",
            tool=tool,
            arguments=arguments,
            error=type(exc).__name__,
            message=str(exc),
            environment=environment,
        )
        raise

    elapsed = time.monotonic() - started
    spec = registry.get(tool)
    # Charge the time to the axis the tool's cost class names. A GPU tool run on CPU still charges gpu_seconds.
    on_gpu = spec.cost is not Cost.CHEAP
    budget.charge(tool_calls=1, gpu_seconds=elapsed if on_gpu else 0.0, cpu_seconds=0.0 if on_gpu else elapsed)
    state.record(
        "tool_result", tool=tool, arguments=arguments, cost=spec.cost.value, seconds=round(elapsed, 3), result=result
    )

    produced = _sequences_from(result)
    # A design for the wrong puzzle, or not written in RNA, is not a design, so it never enters the population.
    wrong = [s for s in produced if (length and len(s) != length) or set(s) - set("ACGU")]
    if wrong:
        produced = [s for s in produced if s not in wrong]
        failures.append(f"{tool}: dropped {len(wrong)} sequences that are not {rna}")
    room = budget.limits.candidates - budget.spend.candidates if budget.limits.candidates else None
    added = state.add_candidates(produced, source=tool, limit=room)
    if room is not None and added == room and len(produced) > added:
        failures.append(f"{tool}: the candidate budget is full, so the rest of this call's designs were not admitted")
    budget.charge(candidates=added)
    scored = state.score(_scores_from(result))
    state.record("state_update", added=added, scored=scored, total=state.count())
    if not added and not scored and spec.name != "report.rules":
        # A call that changed nothing spent budget and reads as success, so the controller repeats it. Reporting
        # rules never changes the pool, so the hint would tell the model its rules were refused.
        failures.append(f"{tool}: changed nothing — same arguments will waste the next round too")
    return result


def run_campaign(
    controller: Controller,
    state: CampaignState,
    budget: Budget,
    rank_by: str,
    max_rounds: int = 50,
    objective: dict[str, Any] | None = None,
    reflect_every: int = 0,
) -> dict:
    """Run one campaign until the controller stops, the budget runs out or `max_rounds`, and record which.

    Every `reflect_every`th round offers only `report.rules`, charged to its own budget axis.
    """
    state.record("campaign_start", controller=controller.name, rank_by=rank_by, budget=budget.snapshot())
    failures: list[str] = []
    reason = "max_rounds"
    #: Rounds whose action carried a rationale.
    explained = 0
    #: Extra tool calls in one message, not run because the rule is one call per round.
    dropped = 0

    for round_number in range(1, max_rounds + 1):
        spent = budget.exhausted()
        if spent:
            reason = f"budget:{spent}"
            break
        state.round = round_number
        # A spent reflection allowance ends the reflections, not the campaign.
        allowed = budget.spend.reflections < (budget.limits.reflections or float("inf"))
        reflecting = bool(reflect_every) and round_number % reflect_every == 0 and allowed
        view = state.view(
            budget.snapshot(), rank_by=rank_by, failures=failures, objective=objective, selectors=SELECTORS
        )
        if reflecting:
            view.objective = {
                **view.objective,
                "this_round": (
                    "Reflection. Write down what this campaign has "
                    "taught you about designing for targets like "
                    "this one, as rules that would hold for a target "
                    "you have not seen. Only report.rules is "
                    "available this round, and it does not cost you "
                    "a tool call."
                ),
            }

        started = time.monotonic()
        with _only(controller, "report.rules") if reflecting else nullcontext():
            action = controller.next_action(view)
            # What this decision was offered. The allowance is restored when the reflection context closes.
            offered = getattr(controller, "allowed", None)
            offered = None if offered is None else set(offered)
        usage = getattr(controller, "last_reply", None)
        spent = getattr(getattr(usage, "usage", None), "input_tokens", 0) + getattr(
            getattr(usage, "usage", None), "output_tokens", 0
        )
        budget.charge(
            model_calls=getattr(controller, "calls_this_round", 1) or 1,
            rounds=1,
            tokens=spent,
            reflections=1 if reflecting else 0,
        )
        dropped += getattr(controller, "dropped_this_round", 0)
        explained += 1 if action.rationale.strip() else 0
        state.record(
            "action",
            tool=action.tool,
            arguments=action.arguments,
            rationale=action.rationale,
            decide_seconds=round(time.monotonic() - started, 3),
        )

        if action.is_stop:
            reason = "controller_stop"
            break

        try:
            apply_tool_call(
                action.tool, action.arguments, state, budget, rank_by, objective, failures, offered, action.error
            )
        except Exception:  # recorded as a failure inside, and the campaign goes on
            continue

    summary = {
        "controller": controller.name,
        "reason": reason,
        "rounds": state.round,
        "candidates": state.count(),
        "explained_rounds": explained,
        "dropped_tool_calls": dropped,
        "budget": budget.snapshot(),
        "failures": failures,
    }
    state.record("campaign_end", **summary)
    return summary
