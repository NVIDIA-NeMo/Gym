# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability hooks for the pinned Hermes ``AIAgent`` integration."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from time import monotonic, time
from typing import Any

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ObservationGap,
    ToolCallObservation,
)


_SOURCE = "hermes"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(getattr(part, "text", "")) for part in content
        )
    return "" if content is None else str(content)


def normalize_hermes_messages(messages: Iterable[Any], *, id_prefix: str = "hermes") -> list[NeMoGymResponseInputItem]:
    """Convert a Hermes conversation to ordered Gym Responses items."""
    output: list[NeMoGymResponseInputItem] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), _text(message.get("content"))
        if role in {"system", "user", "developer"}:
            output.append(NeMoGymEasyInputMessage(role=role, content=content))
        elif role == "assistant":
            output.append(
                NeMoGymResponseOutputMessage(
                    id=str(message.get("id") or f"{id_prefix}-message-{index}"),
                    content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                )
            )
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments", "")
                if not isinstance(arguments, str):
                    try:
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    except (TypeError, ValueError):
                        arguments = "{}"
                call_id = str(call.get("id") or "")
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=arguments,
                        call_id=call_id,
                        id=call_id or None,
                        name=str(function.get("name") or ""),
                        status="completed",
                    )
                )
        elif role == "tool":
            output.append(
                NeMoGymFunctionCallOutput(
                    call_id=str(message.get("tool_call_id") or ""),
                    output=content,
                    status="completed",
                )
            )
    return output


class _ObservedChildren(list):
    def __init__(self, values: Iterable[Any], observer: "HermesAgentObserver", parent_id: str):
        super().__init__(values)
        self.observer, self.parent_id = observer, parent_id

    def append(self, child: Any) -> None:
        super().append(child)
        self.observer._child_added(child, self.parent_id)


class HermesAgentObserver:
    """Instrument one Hermes agent tree without modifying global Hermes state."""

    def __init__(self, *, root_invocation_id: str = "root") -> None:
        self._lock = threading.RLock()
        self._current = threading.local()
        self._root_id = root_invocation_id
        self._child_index = 0
        self._agents: set[int] = set()
        self._tools_by_args: dict[int, tuple[str, str]] = {}
        self._tools: dict[tuple[str, str], ToolCallObservation] = {}
        self._started_ticks: dict[tuple[str, str], float] = {}
        self._invocations = {root_invocation_id: AgentInvocation(invocation_id=root_invocation_id)}
        self._compactions: list[ContextCompactionObservation] = []
        self._gaps: list[ObservationGap] = []

    def instrument(self, agent: Any) -> "HermesAgentObserver":
        self._instrument_safely(agent, self._root_id, wrap_conversation=False)
        return self

    def finish(
        self, result: dict[str, Any] | None = None, *, error: BaseException | None = None
    ) -> AgentObservationBundle:
        self._record_conversation(self._root_id, result, error)
        with self._lock:
            for tool in self._tools.values():
                if tool.status == "unknown":
                    tool.status = "incomplete"
                if tool.timing_source is None:
                    self._gap(
                        "tool_execution_boundary_unavailable",
                        tool.invocation_id,
                        f"No matching _invoke_tool execution was observed for {tool.tool_call_id}.",
                    )
            for invocation in self._invocations.values():
                self._gap(
                    "model_call_ownership_unavailable",
                    invocation.invocation_id,
                    "Hermes messages do not expose outer model response IDs.",
                )
            return AgentObservationBundle(
                source=_SOURCE,
                invocations=list(self._invocations.values()),
                tool_calls=list(self._tools.values()),
                compactions=self._compactions,
                gaps=self._gaps,
            )

    def _instrument_safely(self, agent: Any, invocation_id: str, *, wrap_conversation: bool) -> None:
        try:
            self._instrument(agent, invocation_id, wrap_conversation)
        except Exception as exc:
            self._gap("hermes_observer_error", invocation_id, f"instrument: {type(exc).__name__}")

    def _instrument(self, agent: Any, invocation_id: str, wrap_conversation: bool) -> None:
        with self._lock:
            agent_id = id(agent)
            if agent_id in self._agents:
                return
            self._agents.add(agent_id)

        self._chain_callback(agent, "tool_start_callback", self._tool_started, invocation_id)
        self._chain_callback(agent, "tool_complete_callback", self._tool_completed, invocation_id)
        self._wrap_invoke(agent, invocation_id)
        self._wrap_compaction(agent, invocation_id)

        children = getattr(agent, "_active_children", None)
        if isinstance(children, list):
            setattr(agent, "_active_children", _ObservedChildren(children, self, invocation_id))
        else:
            self._gap("hermes_hook_unavailable", invocation_id, "_active_children")

        if wrap_conversation:
            original = getattr(agent, "run_conversation", None)
            if callable(original):

                def run(*args: Any, **kwargs: Any) -> Any:
                    try:
                        child_result = original(*args, **kwargs)
                    except BaseException as exc:
                        self._record_conversation(invocation_id, None, exc)
                        raise
                    self._record_conversation(invocation_id, child_result, None)
                    return child_result

                setattr(agent, "run_conversation", run)
            else:
                self._gap("hermes_hook_unavailable", invocation_id, "run_conversation")

    def _chain_callback(self, agent: Any, name: str, observer: Callable[..., None], invocation_id: str) -> None:
        if not hasattr(agent, name):
            self._gap("hermes_hook_unavailable", invocation_id, name)
            return
        previous = getattr(agent, name)

        def callback(*args: Any, **kwargs: Any) -> None:
            try:
                observer(invocation_id, *args, **kwargs)
            except Exception:
                self._gap("hermes_observer_error", invocation_id, name)
            if callable(previous):
                try:
                    previous(*args, **kwargs)
                except Exception:
                    pass  # Hermes callbacks are explicitly non-fatal.

        setattr(agent, name, callback)

    def _wrap_invoke(self, agent: Any, invocation_id: str) -> None:
        original = getattr(agent, "_invoke_tool", None)
        if not callable(original):
            self._gap("hermes_hook_unavailable", invocation_id, "_invoke_tool")
            return

        def invoke(*args: Any, **kwargs: Any) -> Any:
            key, previous = None, getattr(self._current, "tool", None)
            try:
                name = args[0] if args else kwargs.get("function_name")
                call_args = args[1] if len(args) > 1 else kwargs.get("function_args")
                with self._lock:
                    key = self._tools_by_args.get(id(call_args))
                if key is not None and key[0] == invocation_id:
                    self._current.tool = (*key, name)
                    self._start_execution(key)
                else:
                    key = None
            except Exception:
                self._gap("hermes_observer_error", invocation_id, "_invoke_tool")
            failed = False
            try:
                return original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                if key is not None:
                    self._end_execution(key, failed=failed)
                self._current.tool = previous

        setattr(agent, "_invoke_tool", invoke)

    def _wrap_compaction(self, agent: Any, invocation_id: str) -> None:
        original = getattr(agent, "_compress_context", None)
        if not callable(original):
            self._gap("hermes_hook_unavailable", invocation_id, "_compress_context")
            return

        def compact(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            try:
                before = kwargs.get("approx_tokens")
                after = getattr(getattr(agent, "context_compressor", None), "last_prompt_tokens", None)
                with self._lock:
                    self._compactions.append(
                        ContextCompactionObservation(
                            invocation_id=invocation_id,
                            observed_at=time(),
                            trigger="context_pressure",
                            tokens_before=before if type(before) is int else None,
                            tokens_after=after if type(after) is int else None,
                        )
                    )
            except Exception:
                self._gap("hermes_observer_error", invocation_id, "_compress_context")
            return result

        setattr(agent, "_compress_context", compact)

    def _child_added(self, child: Any, parent_id: str) -> None:
        try:
            with self._lock:
                self._child_index += 1
                invocation_id = f"{parent_id}.child-{self._child_index}"
                current = getattr(self._current, "tool", None)
                call_id = current[1] if current and current[0] == parent_id and current[2] == "delegate_task" else None
                self._invocations[invocation_id] = AgentInvocation(
                    invocation_id=invocation_id,
                    parent_invocation_id=parent_id,
                    spawned_by_tool_call_id=call_id,
                )
                if call_id is None:
                    self._gap("subagent_spawn_unattributed", invocation_id, "No active delegate_task matched.")
            self._instrument_safely(child, invocation_id, wrap_conversation=True)
        except Exception as exc:
            self._gap("hermes_observer_error", parent_id, f"child: {type(exc).__name__}")

    def _tool_started(self, invocation_id: str, call_id: Any, name: Any, args: Any) -> None:
        key = (invocation_id, str(call_id or ""))
        with self._lock:
            if key in self._tools:
                self._gap("duplicate_tool_call", invocation_id, key[1])
            self._tools[key] = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=key[1],
                tool_name=str(name or "") or None,
            )
            if isinstance(args, dict):
                self._tools_by_args[id(args)] = key
        self._current.tool = (*key, str(name or ""))

    def _tool_completed(self, invocation_id: str, call_id: Any, name: Any, args: Any, result: Any) -> None:
        key = (invocation_id, str(call_id or ""))
        with self._lock:
            if key not in self._tools:
                self._tool_started(invocation_id, call_id, name, args)
            tool = self._tools[key]
            failed = self._failed_result(result)
            if tool.timing_source is None:
                tool.status = "failed" if failed else "completed"
            elif failed:
                tool.status = "failed"
            if isinstance(args, dict):
                self._tools_by_args.pop(id(args), None)
        current = getattr(self._current, "tool", None)
        if current and current[:2] == key:
            self._current.tool = None

    def _start_execution(self, key: tuple[str, str]) -> None:
        with self._lock:
            tool = self._tools[key]
            tool.started_at = time()
            tool.clock_id = "unix_epoch"
            tool.timing_source = "executor"
            self._started_ticks[key] = monotonic()

    def _end_execution(self, key: tuple[str, str], *, failed: bool) -> None:
        try:
            with self._lock:
                tool = self._tools[key]
                if tool.completed_at is None:
                    tool.completed_at = time()
                    tool.duration_ms = max(0.0, (monotonic() - self._started_ticks.pop(key)) * 1000)
                    tool.status = "failed" if failed else "completed"
                elif failed:
                    tool.status = "failed"
        except Exception:
            self._gap("hermes_observer_error", key[0], "_invoke_tool")

    def _record_conversation(
        self,
        invocation_id: str,
        result: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        try:
            messages = result.get("messages") if isinstance(result, dict) else []
            conversation = normalize_hermes_messages(messages or [], id_prefix=invocation_id)
            status = "failed" if error or (result and result.get("error")) else "unknown"
            if isinstance(result, dict) and status != "failed":
                if result.get("interrupted"):
                    status = "incomplete"
                elif result.get("completed") or result.get("final_response"):
                    status = "completed"
                elif messages:
                    status = "incomplete"
            with self._lock:
                self._invocations[invocation_id].conversation = conversation
                self._invocations[invocation_id].status = status
        except Exception as exc:
            self._gap("hermes_observer_error", invocation_id, f"conversation: {type(exc).__name__}")

    @staticmethod
    def _failed_result(result: Any) -> bool:
        if not isinstance(result, str):
            return False
        value = result.lstrip()
        if value.lower().startswith("error executing tool"):
            return True
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(payload, dict) and (
            payload.get("status") in {"error", "failed"} or bool(payload.get("error"))
        )

    def _gap(self, code: str, invocation_id: str | None, detail: str | None = None) -> None:
        with self._lock:
            gap = ObservationGap(code=code, source=_SOURCE, invocation_id=invocation_id, detail=detail)
            if gap not in self._gaps:
                self._gaps.append(gap)
