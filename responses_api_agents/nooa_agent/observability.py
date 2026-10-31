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

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from time import perf_counter, time
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic_core import to_json

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from responses_api_agents.nooa_agent.gym_llm import GymModelCall, RolloutLLMState
from responses_api_agents.nooa_agent.termination import NOOATerminationReason, policy_termination


if TYPE_CHECKING:
    from nooa.agent import Agent


ATOM_SESSION_ID_ATTR = "nooa.atom.session.id"
ATOM_SESSION_PARENT_ATTR = "nooa.atom.session.parent_id"
TRACE_SESSION_ID_ATTR = "session.id"


def _current_atom_session() -> tuple[str | None, str | None]:
    """(session id, parent session id) of the current trace context.

    Atom stamps its sessions' spans with `nooa.atom.session.*` through NOOA's trace context (the root Atom session
    has no Atom parent). Calls made before any Atom session opens carry only NOOA's trace session id, which stands
    in as their session and as the root Atom session's parent. Without NOOA tracing, (None, None).
    """
    try:
        from nooa.tracing._session import get_span_attributes
        from opentelemetry import context as otel_context
    except ImportError:
        return None, None
    attributes = get_span_attributes()
    trace_session_id = otel_context.get_value(TRACE_SESSION_ID_ATTR)
    session_id = attributes.get(ATOM_SESSION_ID_ATTR)
    if session_id is None:
        session_id = trace_session_id
    parent_id = attributes.get(ATOM_SESSION_PARENT_ATTR)
    if parent_id is None and trace_session_id is not None and str(session_id) != str(trace_session_id):
        # The root Atom session has no Atom parent; the entry calls (solve, send_message) that opened it carry
        # only the trace session id, so they are its parent.
        parent_id = trace_session_id
    return (str(session_id) if session_id is not None else None, str(parent_id) if parent_id is not None else None)


def _json_output(value: Any) -> str:
    return value if isinstance(value, str) else to_json(value, serialize_unknown=True).decode()


def _input_items(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [NeMoGymEasyInputMessage(role="user", content=value)]
    return list(value)


def _history_key(item: Any) -> tuple[Any, ...] | None:
    """Identity of a conversation item across the requests of one harness session, or None if it has none.

    Tool calls and their outputs are identified by call id; user and assistant messages by their content. System
    instructions are sent with every request and are not history.
    """
    call_id = getattr(item, "call_id", None)
    if call_id is not None:
        return (item.type, call_id)
    role = getattr(item, "role", None)
    if role in {"user", "assistant"}:
        return (item.type, role, to_json(getattr(item, "content", None), serialize_unknown=True))
    return None


@dataclass(slots=True)
class _AgentCall:
    invocation_id: str
    started: float
    token: Token[str]


@dataclass(slots=True)
class _ToolCall:
    observation: TrajectoryToolCall
    arguments: dict[str, Any]
    started: float
    token: Token[str | None] | None = None


class GymTraceHooks:
    """Capture NOOA's public lifecycle facts in one per-rollout event sequence.

    Model requests/responses remain Gym-owned. ContextVars attribute overlapping
    calls without installing global exporters or patching agent construction.
    """

    def __init__(self, *, on_progress: Callable[[], None] | None = None) -> None:
        self._current: ContextVar[str] = ContextVar("gym_nooa_invocation", default="root")
        # The tool call executing in this context: a child session opened from inside a code cell was spawned by it.
        self._current_tool: ContextVar[str | None] = ContextVar("gym_nooa_tool_call", default=None)
        self._invocations: dict[str, AgentInvocation] = {}
        self._events: list[GymModelCall | _ToolCall] = []
        self._python_outputs: dict[tuple[str, str], str] = {}
        self._captured_python_outputs: dict[tuple[str, str], str] = {}
        self._python_output_gaps: set[tuple[str, str]] = set()
        self._session_openers: dict[str, str] = {}
        # Tool calls NOOA answered without running them (unknown tool, invalid arguments): recorded once each.
        self._rejected_tools: set[str] = set()
        self._on_progress = on_progress

    def _invocation(self, identity: str) -> AgentInvocation:
        return self._invocations.setdefault(identity, AgentInvocation(invocation_id=identity))

    def on_model_call(self, call: GymModelCall) -> None:
        call.invocation_id = self._current.get()
        self._invocation(call.invocation_id)
        self._events.append(call)
        # A message build can be rejected by the policy budget before capture.
        # Promote feedback only here so an unsent rerender cannot erase it.
        visible_text = {
            item.content
            for item in _input_items(call.request.input)
            if getattr(item, "role", None) == "user" and isinstance(getattr(item, "content", None), str)
        }
        self._captured_python_outputs.update(
            (key, value)
            for key, value in self._python_outputs.items()
            if key[0] == call.invocation_id and value in visible_text
        )

    def before_agent_call(self, *, call_id: str, parent_call_id: str | None, **_: Any) -> _AgentCall:
        # A delegated Atom session runs its own method calls with no traced caller, so NOOA reports no parent
        # call. Atom's session tree (the trace context attributes) still names the parent session; link the
        # invocation to the invocation that opened that session so root and child cells stay separable.
        session_id, parent_session_id = _current_atom_session()
        if parent_call_id is None and parent_session_id is not None:
            parent_call_id = self._session_openers.get(parent_session_id)
        spawned_by = None
        if session_id is not None and session_id not in self._session_openers:
            self._session_openers[session_id] = call_id
            # A session opened from inside the parent's code cell (delegate, send_child) was spawned by that cell.
            spawned_by = self._current_tool.get() if parent_session_id is not None else None
        self._invocations[call_id] = AgentInvocation(
            invocation_id=call_id,
            parent_invocation_id=parent_call_id,
            harness_session_id=session_id,
            spawned_by_tool_call_id=spawned_by,
            status="incomplete",
        )
        return _AgentCall(call_id, perf_counter(), self._current.set(call_id))

    def after_agent_call(
        self, *, context: Any, exception: BaseException | None, agent: Agent | None = None, **_: Any
    ) -> None:
        if not isinstance(context, _AgentCall):
            return
        if agent is not None:
            # A rejection on the call's last turn is followed by no message build; record it here.
            self._record_rejected_tool_calls(
                agent,
                {event.observation.tool_call_id for event in self._events if isinstance(event, _ToolCall)},
            )
        invocation = self._invocations[context.invocation_id]
        invocation.status = (
            "incomplete"
            if policy_termination(exception)[0] is not None
            else "failed"
            if exception is not None
            else "completed"
        )
        invocation.error_type = type(exception).__name__ if exception is not None else None
        invocation.duration_ms = max(0.0, (perf_counter() - context.started) * 1000)
        self._current.reset(context.token)

    @contextmanager
    def activate_agent_call(self, context: Any) -> Iterator[None]:
        token = self._current.set(context.invocation_id)
        try:
            yield
        finally:
            self._current.reset(token)

    def _start_tool(self, name: str, arguments: dict[str, Any], call_id: str) -> _ToolCall:
        invocation_id = self._current.get()
        self._invocation(invocation_id)
        event = _ToolCall(
            TrajectoryToolCall(
                invocation_id=invocation_id,
                tool_call_id=call_id,
                tool_name=name,
                started_at=time(),
                timing_source="harness",
                status="incomplete",
            ),
            arguments,
            perf_counter(),
        )
        event.token = self._current_tool.set(call_id)
        self._events.append(event)
        return event

    def _finish_tool(self, event: _ToolCall, exception: BaseException | None) -> None:
        record = event.observation
        record.completed_at = max(record.started_at, time())
        record.duration_ms = max(0.0, (perf_counter() - event.started) * 1000)
        if exception is not None:
            record.status = "cancelled" if isinstance(exception, asyncio.CancelledError) else "failed"
            record.error_type = type(exception).__name__
            record.output = {"error": str(exception), "error_type": record.error_type}
        elif record.status == "incomplete":
            record.status = "completed"
        if event.token is not None:
            try:
                self._current_tool.reset(event.token)
            except ValueError:
                # Finished from another context (a cancelled cell reported by its awaiter): the call is over either way.
                self._current_tool.set(None)
            event.token = None
        if self._on_progress is not None:
            self._on_progress()

    @contextmanager
    def resource_call(self, name: str, arguments: dict[str, Any]) -> Iterator[TrajectoryToolCall]:
        event = self._start_tool(name, arguments, f"resource-{uuid4().hex}")
        try:
            yield event.observation
        except BaseException as error:
            self._finish_tool(event, error)
            raise
        else:
            self._finish_tool(event, None)

    def before_code_execution(self, *, code: str, execution_id: str, **kwargs: Any) -> _ToolCall:
        return self._start_tool("execute_python", {"code": code}, str(kwargs.get("tool_call_id") or execution_id))

    def after_code_execution(self, *, context: Any, result: Any, exception: BaseException | None, **_: Any) -> None:
        if isinstance(context, _ToolCall):
            # ExecutionResult retains live REPL locals, including imported modules.
            # Snapshot its serializable evidence before locals change or projection
            # deep-copies the observation; Pydantic excludes interpreter-only fields.
            context.observation.output = _json_output(result)
            self._finish_tool(context, exception)

    def before_tool_execution(
        self, *, tool_name: str, arguments: dict[str, Any], execution_id: str, **kwargs: Any
    ) -> _ToolCall | None:
        # Code has its own boundary. return_result is NOOA's value validation,
        # whose hook has no model tool-call ID; it is not an external action.
        if tool_name in {"execute_python", "return_result"}:
            return None
        return self._start_tool(tool_name, arguments, str(kwargs.get("tool_call_id") or execution_id))

    def after_tool_execution(
        self,
        *,
        context: Any,
        result: Any,
        exception: BaseException | None,
        tool_name: str = "",
        **kwargs: Any,
    ) -> None:
        self.after_code_execution(context=context, result=result, exception=exception, **kwargs)
        if tool_name == "return_result" and self._on_progress is not None:
            # Value validation is a completed tool boundary, but its arbitrary
            # Python return value is not an external action or serialized output.
            self._on_progress()

    def before_generation(self, **_: Any) -> None:
        pass

    def after_generation(self, **_: Any) -> None:
        pass

    def before_method_invocation(self, **_: Any) -> None:
        pass

    def after_method_invocation(self, **_: Any) -> None:
        pass

    def on_messages_built(self, *, agent: Agent, messages: list[object], **_: Any) -> None:
        """Join typed Python results to the exact feedback NOOA rendered for the model."""
        from nooa.context_blocks.models import BlockMetadata, ResolvedBlock
        from nooa.context_blocks.renderer import render_context
        from nooa.events import PythonOutput

        owned_tools = {
            (event.observation.invocation_id, event.observation.tool_call_id)
            for event in self._events
            if isinstance(event, _ToolCall)
        }
        visible_text = {
            item["content"]
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str)
        }
        for event in agent.events.query(type="PythonOutput"):
            if not isinstance(event, PythonOutput):
                continue
            invocation_id = event.metadata.get("call_id") or self._current.get()
            key = (invocation_id, event.tool_call_id)
            if key not in owned_tools or invocation_id != self._current.get():
                continue
            tag = event.tag if event.tag is not None else event.id
            # Use NOOA's configured renderer and persisted per-event truncation,
            # then verify the result against the actual messages. Neither repr
            # parsing nor stdout that resembles another event can establish a join.
            try:
                rendered = render_context(
                    [
                        ResolvedBlock(
                            key=f"event_{tag}",
                            content="",
                            role=event.event_role,
                            metadata=BlockMetadata(expr=f'self.events["{tag}"]', tag=tag),
                            event=event,
                        )
                    ],
                    block_formatter=agent.render_config.block_formatter,
                    provider_formatter=agent.render_config.provider_formatter,
                    event_format_resolver=agent.runtime._event_format_for_event,
                ).output
                outputs = [
                    item["content"]
                    for item in rendered
                    if isinstance(item, dict)
                    and item.get("role") == "user"
                    and isinstance(item.get("content"), str)
                    and item["content"] in visible_text
                ]
            except Exception:
                # A custom formatter may require the full context. Observability
                # must retain captured execution evidence when an exact join fails.
                outputs = []
            if len(outputs) == 1:
                self._python_outputs[key] = outputs[0]
                self._python_output_gaps.discard(key)
            elif key not in self._python_outputs:
                self._python_output_gaps.add(key)
        self._record_rejected_tool_calls(agent, {call_id for _, call_id in owned_tools})

    def _record_rejected_tool_calls(self, agent: Agent, owned_call_ids: set[str]) -> None:
        """Observe the tool calls NOOA answered without running them.

        An unknown tool name or invalid arguments get an error ToolResult (the feedback the model reads next turn)
        and no execution hook. Without a record of their own, those calls would show up only through the history
        a later request re-renders. Each becomes a failed tool call whose output is that feedback.
        """
        from nooa.context_blocks.events import ResultStatus, ToolCallEvent

        for event in agent.events.query(type="ToolCallEvent"):
            if not isinstance(event, ToolCallEvent) or event.result is None:
                continue
            call_id = event.tool_call_id
            if event.result.result_status != ResultStatus.ERROR or call_id in owned_call_ids:
                continue
            if call_id in self._rejected_tools:
                continue
            self._rejected_tools.add(call_id)
            rejected = self._start_tool(event.name, dict(event.arguments), call_id)
            rejected.observation.status = "failed"
            rejected.observation.error_type = "ToolCallRejected"
            rejected.observation.output = event.result.content
            self._finish_tool(rejected, None)

    def project(
        self,
        *,
        create_params: NeMoGymResponseCreateParamsNonStreaming,
        state: RolloutLLMState,
        task_id: str,
        rollout_id: str,
        default_model: str = "nooa",
        partial: bool = False,
        root_status: Literal["completed", "failed", "incomplete"] | None = None,
        root_error_type: str | None = None,
    ) -> tuple[AgentEpisode, TrajectoryRecord]:
        """Project evidence, applying the runner's final status only to the first root.

        Partial snapshots omit pending tool outputs and retain each invocation's
        contemporaneous outcome until the runner supplies a final root status.
        """

        invocations = {key: record.model_copy(deep=True) for key, record in self._invocations.items()}
        output: list[Any] = []
        tools: list[TrajectoryToolCall] = []
        turns: list[TrajectoryTurn] = []
        turn_counts: dict[str, int] = {}
        initialized_inputs: set[str] = set()
        # Requests preserve native history; responses cover final calls that have
        # not been replayed yet. Prefer the latest captured request for each call.
        call_items = {}
        for call in state.calls:
            items = _input_items(call.request.input)
            for item in [*(call.response.output if call.response is not None else []), *items]:
                if isinstance(item, NeMoGymResponseFunctionToolCall):
                    call_items[(call.invocation_id, item.call_id)] = item
        # Python tool results acknowledge acceptance; execution feedback arrives
        # separately as a PythonOutput user event. Only use verified feedback that
        # reached a captured request, not a prompt stopped by the policy-call budget.
        python_outputs = self._captured_python_outputs
        visible_outputs = {
            (call.invocation_id, item.call_id): item.output
            for call in state.calls
            for item in _input_items(call.request.input)
            if isinstance(item, NeMoGymFunctionCallOutput)
        }
        visible_outputs.update(python_outputs)
        # A later invocation of a harness session re-sends the session's history in its first request. Those items
        # belong to the invocations that produced them: a session's history is its invocations' conversations in
        # order, so each item is projected once, in the invocation that produced it.
        session_history: dict[str, set[tuple[Any, ...]]] = {}

        def extend_conversation(invocation: AgentInvocation, items: list[Any]) -> None:
            invocation.conversation.extend(items)
            if invocation.harness_session_id is not None:
                seen = session_history.setdefault(invocation.harness_session_id, set())
                seen.update(key for item in items if (key := _history_key(item)) is not None)

        for event in self._events:
            if isinstance(event, GymModelCall):
                invocation = invocations[event.invocation_id]
                if invocation.invocation_id not in initialized_inputs:
                    initialized_inputs.add(invocation.invocation_id)
                    initial = []
                    if event.request.instructions:
                        initial.append(NeMoGymEasyInputMessage(role="system", content=event.request.instructions))
                    # A CodeAct prefill may have emitted tools before the first
                    # model call. Preserve the prompt without duplicating those tools.
                    initial.extend(
                        [NeMoGymEasyInputMessage(role="user", content=event.request.input)]
                        if isinstance(event.request.input, str)
                        else event.request.input
                    )
                    earlier = session_history.get(invocation.harness_session_id or "", set())
                    initial = [item for item in initial if _history_key(item) not in earlier]
                    visible = {(item.type, getattr(item, "call_id", None)) for item in initial}
                    prefill = [
                        item
                        for item in invocation.conversation
                        if (item.type, getattr(item, "call_id", None)) not in visible
                    ]
                    invocation.conversation = []
                    extend_conversation(invocation, initial + prefill)
                if event.response is None:
                    continue
                response = event.response
                reference = ModelCallRef(model_ref=event.model_ref, response_id=response.id)
                invocation.model_calls.append(reference)
                extend_conversation(invocation, list(response.output))
                output.extend(response.output)
                turn_no = turn_counts.get(invocation.invocation_id, 0) + 1
                turn_counts[invocation.invocation_id] = turn_no
                turns.append(
                    TrajectoryTurn(
                        invocation_id=invocation.invocation_id,
                        task_id=task_id,
                        rollout_id=rollout_id,
                        turn_no=turn_no,
                        timestamp=response.created_at,
                        question=event.request.model_dump(mode="json", exclude_none=True),
                        answer=[item.model_dump(mode="json", exclude_none=True) for item in response.output],
                        step_count=turn_no,
                        model_calls=[reference],
                    )
                )
            else:
                # NOOA execution results can retain module objects in their REPL locals.
                # Copy the observation shell; output is replaced with serialized evidence below.
                record = event.observation.model_copy()
                invocation = invocations[record.invocation_id]
                key = (record.invocation_id, record.tool_call_id)
                record.output = visible_outputs.get(key, record.output)
                captured_call = call_items.get(key)
                if captured_call is not None:
                    record.tool_name = captured_call.name
                if not any(
                    isinstance(item, NeMoGymResponseFunctionToolCall) and item.call_id == record.tool_call_id
                    for item in invocation.conversation
                ):
                    call_item = (
                        captured_call.model_copy(deep=True)
                        if captured_call is not None
                        else NeMoGymResponseFunctionToolCall(
                            id=record.tool_call_id,
                            call_id=record.tool_call_id,
                            name=record.tool_name,
                            arguments=_json_output(event.arguments),
                        )
                    )
                    extend_conversation(invocation, [call_item])
                    output.append(call_item)
                if partial and record.completed_at is None:
                    # A sibling tool may still be running at a checkpoint. Keep
                    # its start event without inventing a completed/null output.
                    record.output = None
                    tools.append(record)
                    continue
                result_item = NeMoGymFunctionCallOutput(
                    call_id=record.tool_call_id,
                    output=_json_output(record.output),
                    status="completed" if record.status == "completed" else "incomplete",
                )
                # Persist the same serialized evidence as the conversation, not NOOA runtime objects.
                record.output = result_item.output
                extend_conversation(invocation, [result_item])
                output.append(result_item)
                tools.append(record)

        # A tool call the model emitted that reached no tool boundary and got no output: NOOA stops a response's
        # remaining calls after one fails or completes the call, and tells the model nothing about them.
        if not partial:
            observed = {(tool.invocation_id, tool.tool_call_id) for tool in tools}
            not_executed = sorted(
                (call.invocation_id, item.call_id)
                for call in state.calls
                if call.response is not None
                for item in call.response.output
                if isinstance(item, NeMoGymResponseFunctionToolCall)
                and (call.invocation_id, item.call_id) not in observed
                and (call.invocation_id, item.call_id) not in visible_outputs
            )
        else:
            not_executed = []

        # The first request can replace an earlier prefill result with its native
        # acceptance receipt. Normalize after that merge as collection obtains tool
        # outputs from this conversation. Exact model-call requests stay untouched.
        for invocation in invocations.values():
            invocation.conversation = [
                item.model_copy(update={"output": python_outputs[(invocation.invocation_id, item.call_id)]})
                if isinstance(item, NeMoGymFunctionCallOutput)
                and (invocation.invocation_id, item.call_id) in python_outputs
                else item
                for item in invocation.conversation
            ]

        if not invocations:
            invocations["root"] = AgentInvocation(invocation_id="root", status="completed")
        if root_status is not None:
            # The runner owns final classification after adapter cleanup and fatal
            # transport checks. Other invocations (including detached summaries)
            # and earlier checkpoints keep their own outcomes.
            root = next((item for item in invocations.values() if item.parent_invocation_id is None), None)
            if root is not None:
                root.status = root_status
                root.error_type = root_error_type
        if not initialized_inputs:
            root = next((item for item in invocations.values() if item.parent_invocation_id is None), None)
            if root is not None:
                root.conversation = [*_input_items(create_params.input), *root.conversation]

        gaps = [
            *state.gaps,
            *(
                ObservationGap(
                    code="python_output_projection_unavailable",
                    invocation_id=invocation_id,
                    detail=f"Native execution feedback for {call_id} did not join exactly to the rendered messages.",
                )
                for invocation_id, call_id in sorted(self._python_output_gaps)
            ),
            *(
                ObservationGap(
                    code="tool_call_not_executed",
                    invocation_id=invocation_id,
                    detail=f"The model emitted {call_id} but the harness never ran it and gave it no output.",
                )
                for invocation_id, call_id in not_executed
            ),
        ]
        observations = AgentObservationBundle(
            source="nooa",
            records=[
                *invocations.values(),
                *(ToolCallObservation.model_validate(tool.model_dump(exclude={"output"})) for tool in tools),
            ],
            gaps=gaps,
        )
        # Aggregate only returned responses. A failed attempt has no reported
        # usage of its own; it must not erase known totals or invent per-call zeroes.
        usages = [call.response.usage for call in state.calls if call.response is not None]
        latest_response = next((call.response for call in reversed(state.calls) if call.response is not None), None)
        episode = AgentEpisode(
            response=NeMoGymResponse(
                id=f"nooa-{rollout_id}",
                created_at=time(),
                model=create_params.model or default_model,
                object="response",
                output=output,
                incomplete_details=latest_response.incomplete_details if latest_response is not None else None,
                tools=create_params.tools,
                tool_choice=create_params.tool_choice,
                parallel_tool_calls=create_params.parallel_tool_calls,
                usage=(
                    NeMoGymResponseUsage.sum_from_list(usages) if all(usage is not None for usage in usages) else None
                ),
            ),
            observations=observations,
        )
        trajectory = TrajectoryRecord(
            task_id=task_id,
            rollout_id=rollout_id,
            invocations=list(invocations.values()),
            turns=turns,
            tool_calls=tools,
            gaps=gaps,
        )
        return episode, trajectory


def _ends_with_completed_assistant_message(response: NeMoGymResponse) -> bool:
    if not response.output:
        return False
    last = response.output[-1]
    return isinstance(last, NeMoGymResponseOutputMessage) and last.status == "completed"


def ensure_verifier_final_message(
    response: NeMoGymResponse,
    return_value: Any,
) -> tuple[NeMoGymResponse, list[ObservationGap]]:
    """Add a verifier-facing fallback message without mutating the captured episode."""

    gaps: list[ObservationGap] = []
    if return_value is None:
        return response, gaps
    if _ends_with_completed_assistant_message(response):
        text = "\n".join(part.text for part in response.output[-1].content if part.type == "output_text")
        if text == _json_output(return_value):
            return response, gaps

    # Verifiers commonly grade the last assistant message, so preserve the full trace and append the
    # entrypoint return unless that trace already ends with the same value. A child's
    # last message or a deterministic post-processing step must not override the return.
    output = list(response.output)
    output.append(
        NeMoGymResponseOutputMessage(
            id="nooa_fallback",
            content=[NeMoGymResponseOutputText(annotations=[], text=_json_output(return_value))],
        )
    )
    gaps.append(
        ObservationGap(
            code="non_trainable_terminal_output",
            detail="NOOA returned a value without a final model-authored message.",
        )
    )
    return response.model_copy(update={"output": output}), gaps


def finalize_observation_gaps(
    observations: AgentObservationBundle,
    *,
    extra_gaps: list[ObservationGap] | None = None,
    termination_reason: NOOATerminationReason | None = None,
    termination_error: str | None = None,
) -> AgentObservationBundle:
    """Merge lifecycle-only observation gaps onto the Gym projection."""

    gaps = [*observations.gaps, *(extra_gaps or [])]
    if termination_reason is not None:
        gaps.append(
            ObservationGap(
                code=termination_reason,
                detail=termination_error or f"NOOA execution terminated with {termination_reason}.",
            )
        )
    if not gaps:
        return observations
    return observations.model_copy(update={"gaps": gaps})
