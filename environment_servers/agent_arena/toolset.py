# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tools of one AgentArena turn, as an ``EpisodeToolset``: the row's tools, each call answered as the
``agentic_turn_simulation`` resources server's ``/simulate_tool_call`` answers it (in code from the replayed workspace,
by the tool simulator LLM, or with an injected obstacle).

A call reaches ``call`` with its tool's name and arguments only, so the simulator's view of the turn so far is rebuilt
from the episode's earlier calls (`TurnToolset.turn_so_far`). It holds the calls and their results, not what the policy
wrote between them.
"""

import json
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from nemo_gym.server_runtime import ToolCallRecord, ToolResult, ToolSpec
from resources_servers.agentic_turn_simulation.schemas import AgenticTurnSimulationRunRequest, SimulateToolCallRequest
from resources_servers.agentic_turn_simulation.turns import TurnSession, TurnSimulation, _row_tools


# The id of the call being answered in the rebuilt turn; earlier calls keep the ids the agent sent with them.
_CURRENT_CALL_ID = "ng_current_call"


class TurnToolset:
    """The row's tools for one episode: the tools a call may name (`TurnSession.tool_names`, the set the scoring's
    nonexistent_tool gate checks). ``bind`` gives it the episode's call records once ``serve_tools`` returns them."""

    def __init__(
        self,
        simulation: TurnSimulation,
        session: TurnSession,
        row: AgenticTurnSimulationRunRequest,
        *,
        call_context: Callable[[], AbstractContextManager[None]] = nullcontext,
    ) -> None:
        self._simulation = simulation
        # Entered around each simulated call: the server's model-call sampling (see AgentArenaEnvironmentServer).
        self._call_context = call_context
        self._session = session
        self._calls: list[ToolCallRecord] = []
        harness = session.harness
        described = {tool.get("name"): tool for tool in harness.described_tools(_row_tools(row), session.prefix)}
        # A name a tool grant refuses, or parameters that are not an object schema, fail here (ToolSpec).
        self._specs = [_spec(name, described.get(name, {})) for name in sorted(session.tool_names, key=str)]

    def bind(self, calls: list[ToolCallRecord]) -> None:
        self._calls = calls

    def specs(self) -> Sequence[ToolSpec]:
        return self._specs

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        request = SimulateToolCallRequest(
            call_id=_CURRENT_CALL_ID,
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
            turn=self.turn_so_far(name, arguments),
        )
        with self._call_context():
            answer = await self._simulation.simulate_tool_call(self._session, request)
        # A failed simulation's output is one the scoring masks; the agent passes it on like any other.
        return ToolResult(output=answer.output, is_error=answer.source == "failed")

    def turn_so_far(self, name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """The turn the simulator sees: each earlier call of the episode and its result, in order, then this call.

        The one place that decides a simulated call's context: a richer view of the policy's turn replaces it.

        Only results the policy saw: a call the server refused or failed (a record it wrote itself, ``{"detail": ...}``:
        no such tool, a failed or timed-out call) is left out, since the agent answers a refusal with its own text and
        retries a failure under the same call id. Of a call id's records, the last is the one the policy saw.
        """
        last = {record.call_id: index for index, record in enumerate(self._calls) if record.call_id is not None}
        items: list[dict[str, Any]] = []
        for index, record in enumerate(self._calls):
            if _server_failure(record) or (record.call_id is not None and last[record.call_id] != index):
                continue
            call_id = record.call_id or f"ng_call_{index}"
            output = record.output if isinstance(record.output, str) else json.dumps(record.output, ensure_ascii=False)
            items += [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": record.name,
                    "arguments": json.dumps(record.arguments, ensure_ascii=False),
                },
                {"type": "function_call_output", "call_id": call_id, "output": output},
            ]
        current = json.dumps(arguments, ensure_ascii=False)
        items.append({"type": "function_call", "call_id": _CURRENT_CALL_ID, "name": name, "arguments": current})
        return items


def _server_failure(record: ToolCallRecord) -> bool:
    """A record the server wrote for a call it refused or that failed; a simulated answer is always text."""
    return record.is_error and isinstance(record.output, dict) and set(record.output) == {"detail"}


def _spec(name: Any, tool: dict[str, Any]) -> ToolSpec:
    """The spec of one tool a call may name. The row declares its tools to the policy itself, so a spec only names a
    tool the server answers: one the row (or, for Cline, its history) does not describe gets an open object schema."""
    parameters = tool.get("parameters")
    description = tool.get("description")
    return ToolSpec(
        name=name,
        description=description if isinstance(description, str) and description.strip() else name,
        parameters=parameters if parameters is not None else {"type": "object"},
    )
