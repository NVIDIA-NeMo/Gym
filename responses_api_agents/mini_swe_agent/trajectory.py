# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert mini-SWE 2.4.6's native trajectory into Gym response output and agent observations.

mini-SWE saves each model response (``object: response``) with its parsed bash actions, then one
``function_call_output`` per action. A submission saves no output: its exit message carries the command's timing.
A response rejected for its format is kept in the next message's ``extra.response``; it counts as a model call but
mini-SWE does not replay it, and its function calls, which never ran, are left out of the response output. MCP calls
that a bash command made through the task-local CLI come from ``mcp_calls.jsonl`` and are placed after the bash call
that made them.
"""

import base64
import binascii
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import TypeAdapter

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputItem,
    NeMoGymResponseUsage,
)
from nemo_gym.trajectory import AgentInvocation, ModelCallRef, ObservationGap, ToolCallObservation


LOG = logging.getLogger(__name__)
_INPUT_ITEMS = TypeAdapter(list[NeMoGymResponseInputItem])


def original_response_id(response_id: str) -> str:
    """Undo LiteLLM's routing envelope so references join Gym's captured response IDs."""
    if response_id.startswith("resp_"):
        try:
            decoded = base64.b64decode(response_id[5:], validate=True).decode()
            if decoded.startswith("litellm:custom_llm_provider:") and ";response_id:" in decoded:
                return decoded.split(";response_id:", 1)[1]
        except (ValueError, UnicodeDecodeError, binascii.Error):
            pass
    return response_id


def mcp_tool_name(server: str, tool: str) -> str:
    """Gym's name for an MCP tool, which verifiers strip back to the tool's own name."""
    return f"mcp__{server}__{tool}"


@dataclass
class NativeToolCall:
    """One bash action and, once mini-SWE saved it, its observation."""

    tool_call_id: str
    model_index: int
    output: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = None
    status: str = "incomplete"
    error_type: str | None = None


@dataclass
class Projection:
    """The model calls and tool calls a saved trajectory records, in order."""

    history: list[dict] = field(default_factory=list)
    responses: list[NeMoGymResponse] = field(default_factory=list)
    tools: list[NativeToolCall] = field(default_factory=list)
    mcp_calls: list[dict] = field(default_factory=list)
    # The 1-based positions in ``responses`` of responses mini-SWE rejected for their format; their calls never ran.
    rejected: set[int] = field(default_factory=set)
    # A saved model response that is not a valid Responses API object; projection stops before it.
    invalid_response: bool = False
    # The model calls mini-SWE started, counted before each call; ``None`` when the trajectory does not say.
    model_calls: int | None = None

    @property
    def usage(self) -> NeMoGymResponseUsage | None:
        """Total usage, only when every model call reported its own; a subtotal would undercount.

        A call that a stop or a failure interrupted is counted by mini-SWE but saved no response, so it has no usage.
        """
        if not self.responses or any(response.usage is None for response in self.responses):
            return None
        if self.model_calls is not None and self.model_calls > len(self.responses):
            return None
        return NeMoGymResponseUsage.sum_from_list([response.usage for response in self.responses])

    def output(self) -> list:
        """Response output: each model response, then for each of its bash calls the MCP calls it made and its result.

        A response mini-SWE rejected for its format keeps its other items but not its function calls, which never ran.
        An MCP call whose bash call is not in the projection, such as one from a background process, comes last.
        """
        owners = {tool.tool_call_id for tool in self.tools}
        items = []
        for index, response in enumerate(self.responses, start=1):
            if index in self.rejected:
                items.extend(item for item in response.output if item.type != "function_call")
                continue
            items.extend(response.output)
            for tool in self.tools:
                if tool.model_index != index:
                    continue
                items.extend(self._mcp_items(lambda call: call.get("tool_call_id") == tool.tool_call_id))
                if tool.output is not None:
                    items.append(
                        NeMoGymFunctionCallOutput(
                            type="function_call_output", call_id=tool.tool_call_id, output=tool.output
                        )
                    )
        items.extend(self._mcp_items(lambda call: call.get("tool_call_id") not in owners))
        return items

    def _mcp_items(self, selected: Callable[[dict], bool]) -> list:
        items = []
        for number, call in enumerate(self.mcp_calls):
            if not selected(call):
                continue
            call_id = mcp_call_id(number, call)
            items.append(
                NeMoGymResponseFunctionToolCall(
                    type="function_call",
                    id=call_id,
                    call_id=call_id,
                    name=mcp_tool_name(call["server"], call["tool"]),
                    arguments=json.dumps(call.get("arguments") or {}),
                    status="completed",
                )
            )
            items.append(
                NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=call["output"])
            )
        return items


def mcp_call_id(number: int, call: dict) -> str:
    """A call ID for the ``number``-th MCP call, derived from the bash call that made it."""
    return f"{call.get('tool_call_id') or 'mcp'}-mcp-{number}"


def project(trajectory: dict | None, mcp_calls: list[dict]) -> Projection:
    """Project saved decisions and observations without inventing timings."""
    model_calls = (((trajectory or {}).get("info") or {}).get("model_stats") or {}).get("api_calls")
    projection = Projection(
        mcp_calls=list(mcp_calls), model_calls=model_calls if isinstance(model_calls, int) else None
    )
    conversation: list[dict] = []
    for message in (trajectory or {}).get("messages", []):
        extra = message.get("extra", {})
        response = message if message.get("object") == "response" else extra.get("response")
        if isinstance(response, dict) and response.get("object") == "response":
            response = dict(response, id=original_response_id(response.get("id", "")))
            try:
                validated = NeMoGymResponse.model_validate({k: v for k, v in response.items() if k != "extra"})
            except Exception:
                LOG.exception("Unable to project a malformed mini-SWE model response")
                projection.invalid_response = True
                break
            projection.responses.append(validated)
            if message.get("object") != "response":
                projection.rejected.add(len(projection.responses))
            projection.history.append(
                {"input": list(conversation), "timestamp": extra.get("timestamp", response.get("created_at", 0))}
            )
            # A rejected response is evidence of a model call, but mini-SWE does not replay it in the next request.
            if message.get("object") == "response":
                conversation.extend(response.get("output", []))
                for action in extra.get("actions", []):
                    projection.tools.append(
                        NativeToolCall(tool_call_id=action["tool_call_id"], model_index=len(projection.responses))
                    )
        if message.get("type") == "function_call_output":
            conversation.append({k: v for k, v in message.items() if k != "extra"})
            tool = _find(projection.tools, message.get("call_id"))
            if tool is not None:
                error = extra.get("exception_type")
                tool.output = message.get("output")
                tool.started_at = extra.get("started_at")
                # The command's own end time; artifacts without it keep the observation's save time.
                tool.completed_at = extra.get("completed_at", extra.get("timestamp"))
                tool.duration_ms = extra.get("duration_ms")
                tool.status = (
                    "timeout" if error == "TimeoutExpired" else "failed" if extra.get("returncode") else "completed"
                )
                tool.error_type = error
        elif message.get("role") in {"system", "user", "assistant"}:
            conversation.append({k: v for k, v in message.items() if k != "extra"})
        elif message.get("role") == "exit" and extra.get("exit_status") == "Submitted" and extra.get("tool_call_id"):
            # mini-SWE submits only after the command exited 0 and saves no observation; the runner's exit message
            # carries the command's output and timing. The model never sees that output: the run ends there.
            tool = _find(projection.tools, extra["tool_call_id"])
            if tool is not None:
                tool.output = extra.get("output")
                tool.started_at = extra.get("started_at")
                tool.completed_at = extra.get("completed_at")
                tool.duration_ms = extra.get("duration_ms")
                tool.status = "completed"
    return projection


def _find(tools: list[NativeToolCall], tool_call_id: object) -> NativeToolCall | None:
    return next((tool for tool in reversed(tools) if tool.tool_call_id == tool_call_id), None)


def observation_records(
    projection: Projection,
    *,
    invocation: AgentInvocation,
    model_server: ModelServerRef,
    sandbox_id: str | None,
    stopped: str | None,
) -> tuple[list, list[ObservationGap]]:
    """Fill ``invocation`` with the projected conversation and model calls; return tool records and gaps.

    ``stopped`` is ``timeout`` or ``cancelled`` when Gym stopped the run, which also ends a tool call mini-SWE had
    started but not saved. An MCP call is part of the bash call that made it, which the model requested, so it is not
    a tool call record of its own; the response carries it for verifiers.
    """
    gaps: list[ObservationGap] = []
    if projection.responses:
        # Each request extends the one before it, so the last request and its response hold the whole conversation.
        invocation.conversation = _INPUT_ITEMS.validate_python(projection.history[-1]["input"])
        invocation.conversation.extend(projection.responses[-1].output)
    for index, response in enumerate(projection.responses, start=1):
        if response.id:
            invocation.model_calls.append(ModelCallRef(model_ref=model_server, response_id=response.id))
        else:
            gaps.append(
                ObservationGap(
                    code="model_call_reference_unavailable",
                    invocation_id=invocation.invocation_id,
                    detail=f"turn:{index}",
                )
            )
    last = len(projection.responses)
    for tool in projection.tools:
        if tool.model_index == last and tool.output is not None:
            invocation.conversation.append(
                NeMoGymFunctionCallOutput(type="function_call_output", call_id=tool.tool_call_id, output=tool.output)
            )
    records: list = []
    for tool in projection.tools:
        status, error_type = tool.status, tool.error_type
        if status == "incomplete" and stopped is not None:
            status, error_type = stopped, f"invocation_{stopped}"
        if tool.duration_ms is None:
            gaps.append(
                ObservationGap(
                    code="tool_timing_unavailable", invocation_id=invocation.invocation_id, detail=tool.tool_call_id
                )
            )
        records.append(
            ToolCallObservation(
                invocation_id=invocation.invocation_id,
                tool_call_id=tool.tool_call_id,
                sandbox_id=sandbox_id,
                tool_name="bash",
                started_at=tool.started_at,
                completed_at=tool.completed_at,
                duration_ms=tool.duration_ms,
                timing_source="artifact",
                status=status,
                error_type=error_type,
            )
        )
    return records, gaps
