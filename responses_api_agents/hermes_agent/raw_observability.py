# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect JSON-serializable Hermes events without importing Gym dependencies."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from time import monotonic, time
from typing import Any


class _ObservedChildren(list):
    def __init__(self, values: Iterable[Any], observer: "RawHermesObserver", parent_id: str):
        super().__init__(values)
        self.observer, self.parent_id = observer, parent_id

    def append(self, child: Any) -> None:
        super().append(child)
        self.observer._child_added(child, self.parent_id)


class RawHermesObserver:
    """Instrument one Hermes agent tree and return only JSON-compatible data."""

    def __init__(self, *, root_invocation_id: str = "root") -> None:
        self._lock = threading.RLock()
        self._current = threading.local()
        self._root_id = root_invocation_id
        self._child_index = 0
        self._agents: set[int] = set()
        self._invocation_agents: dict[str, Any] = {}
        self._tools_by_args: dict[int, tuple[str, str]] = {}
        self._tools: dict[tuple[str, str], dict[str, Any]] = {}
        self._started_ticks: dict[tuple[str, str], float] = {}
        self._invocations = {root_invocation_id: self._new_invocation(root_invocation_id)}
        self._compactions: list[dict[str, Any]] = []
        self._gaps: list[dict[str, Any]] = []

    @staticmethod
    def _new_invocation(
        invocation_id: str,
        *,
        parent_invocation_id: str | None = None,
        spawned_by_tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "invocation_id": invocation_id,
            "parent_invocation_id": parent_invocation_id,
            "spawned_by_tool_call_id": spawned_by_tool_call_id,
            "status": "unknown",
            "model_response_ids": [],
            "messages": [],
            "system_message": None,
        }

    def instrument(self, agent: Any) -> "RawHermesObserver":
        self._instrument_safely(agent, self._root_id, wrap_conversation=False)
        return self

    def finish(
        self,
        result: dict[str, Any] | None = None,
        *,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        self._record_conversation(
            self._root_id,
            result,
            error,
            system_message=self._system_prompt(self._invocation_agents.get(self._root_id)),
        )
        with self._lock:
            for tool in self._tools.values():
                if tool["status"] == "unknown":
                    tool["status"] = "incomplete"
            for invocation in self._invocations.values():
                if not invocation["model_response_ids"]:
                    self._gap("model_call_ownership_unavailable", invocation["invocation_id"])
            return {
                "source": "hermes",
                "invocations": list(self._invocations.values()),
                "tools": list(self._tools.values()),
                "compactions": list(self._compactions),
                "gaps": list(self._gaps),
            }

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
            self._invocation_agents[invocation_id] = agent

        self._chain_callback(agent, "tool_start_callback", self._tool_started, invocation_id)
        self._chain_callback(agent, "tool_complete_callback", self._tool_completed, invocation_id)
        self._wrap_model_calls(agent, invocation_id)
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
                        self._record_conversation(
                            invocation_id,
                            None,
                            exc,
                            system_message=self._system_prompt(agent),
                        )
                        raise
                    self._record_conversation(
                        invocation_id,
                        child_result,
                        None,
                        system_message=self._system_prompt(agent),
                    )
                    return child_result

                setattr(agent, "run_conversation", run)
            else:
                self._gap("hermes_hook_unavailable", invocation_id, "run_conversation")

    def _wrap_model_calls(self, agent: Any, invocation_id: str) -> None:
        original = getattr(agent, "_interruptible_api_call", None)
        if not callable(original):
            self._gap("hermes_hook_unavailable", invocation_id, "_interruptible_api_call")
            return

        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                response = original(*args, **kwargs)
            except BaseException as exc:
                self._gap("model_response_id_unavailable", invocation_id, type(exc).__name__)
                raise
            response_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
            if not isinstance(response_id, str) or not response_id:
                self._gap("model_response_id_unavailable", invocation_id)
            else:
                with self._lock:
                    response_ids = self._invocations[invocation_id]["model_response_ids"]
                    if response_id not in response_ids:
                        response_ids.append(response_id)
            return response

        setattr(agent, "_interruptible_api_call", call)

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
                    pass

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
            failed = False
            try:
                result = original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                before = kwargs.get("approx_tokens")
                try:
                    after = None
                    if not failed:
                        value = getattr(getattr(agent, "context_compressor", None), "last_prompt_tokens", None)
                        if type(value) is int and value >= 0:
                            after = value
                    with self._lock:
                        self._compactions.append(
                            {
                                "invocation_id": invocation_id,
                                "observed_at": time(),
                                "trigger": "context_pressure",
                                "tokens_before": before if type(before) is int else None,
                                "tokens_after": after,
                                "outcome": "failed" if failed else "completed",
                            }
                        )
                    self._gap("compaction_model_call_boundary_unavailable", invocation_id)
                    self._gap("compaction_summary_unavailable", invocation_id)
                    if after is None:
                        self._gap("compaction_tokens_after_unavailable", invocation_id)
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
                self._invocations[invocation_id] = self._new_invocation(
                    invocation_id,
                    parent_invocation_id=parent_id,
                    spawned_by_tool_call_id=call_id,
                )
                if call_id is None:
                    self._gap("subagent_spawn_unattributed", invocation_id)
            self._instrument_safely(child, invocation_id, wrap_conversation=True)
        except Exception as exc:
            self._gap("hermes_observer_error", parent_id, f"child: {type(exc).__name__}")

    def _tool_started(self, invocation_id: str, call_id: Any, name: Any, args: Any) -> None:
        key = (invocation_id, str(call_id or ""))
        started_at = time()
        started_tick = monotonic()
        with self._lock:
            if key in self._tools:
                self._gap("duplicate_tool_call", invocation_id, key[1])
            self._tools[key] = {
                "invocation_id": invocation_id,
                "tool_call_id": key[1],
                "tool_name": str(name or "") or None,
                "started_at": started_at,
                "completed_at": None,
                "duration_ms": None,
                "timing_source": "harness",
                "status": "unknown",
            }
            self._started_ticks[key] = started_tick
            if isinstance(args, dict):
                self._tools_by_args[id(args)] = key
        self._current.tool = (*key, str(name or ""))

    def _tool_completed(self, invocation_id: str, call_id: Any, name: Any, args: Any, result: Any) -> None:
        key = (invocation_id, str(call_id or ""))
        with self._lock:
            if key not in self._tools:
                self._tool_started(invocation_id, call_id, name, args)
            tool = self._tools[key]
            failed = self._failed_result(name, result)
            if failed:
                tool["status"] = "failed"
            if isinstance(args, dict):
                self._tools_by_args.pop(id(args), None)
        current = getattr(self._current, "tool", None)
        if current and current[:2] == key:
            self._current.tool = None
        self._end_execution(key, failed=failed)

    def _start_execution(self, key: tuple[str, str]) -> None:
        with self._lock:
            tool = self._tools[key]
            tool["started_at"] = time()
            tool["timing_source"] = "executor"
            self._started_ticks[key] = monotonic()

    def _end_execution(self, key: tuple[str, str], *, failed: bool) -> None:
        try:
            with self._lock:
                tool = self._tools[key]
                if tool["completed_at"] is None:
                    tool["completed_at"] = time()
                    started_tick = self._started_ticks.pop(key, None)
                    if started_tick is not None:
                        tool["duration_ms"] = max(0.0, (monotonic() - started_tick) * 1000)
                    tool["status"] = "failed" if failed else "completed"
                elif failed:
                    tool["status"] = "failed"
        except Exception:
            self._gap("hermes_observer_error", key[0], "_invoke_tool")

    def _record_conversation(
        self,
        invocation_id: str,
        result: dict[str, Any] | None,
        error: BaseException | None,
        *,
        system_message: Any = None,
    ) -> None:
        try:
            messages = result.get("messages") if isinstance(result, dict) else []
            if any(isinstance(message, dict) and message.get("reasoning_details") for message in messages or []):
                self._gap("reasoning_details_unavailable", invocation_id)
            status = "failed" if error or (result and result.get("error")) else "unknown"
            if isinstance(result, dict) and status != "failed":
                if result.get("interrupted"):
                    status = "incomplete"
                elif result.get("completed") is True:
                    status = "completed"
                elif result.get("completed") is False:
                    status = "incomplete"
                elif result.get("final_response"):
                    status = "completed"
                elif messages:
                    status = "incomplete"
            with self._lock:
                invocation = self._invocations[invocation_id]
                invocation["messages"] = list(messages or [])
                invocation["system_message"] = system_message if isinstance(system_message, str) else None
                invocation["status"] = status
        except Exception as exc:
            self._gap("hermes_observer_error", invocation_id, f"conversation: {type(exc).__name__}")

    @staticmethod
    def _system_prompt(agent: Any) -> str | None:
        if agent is None:
            return None
        cached = getattr(agent, "_cached_system_prompt", None)
        ephemeral = getattr(agent, "ephemeral_system_prompt", None)
        parts = [value for value in (cached, ephemeral) if isinstance(value, str) and value]
        return "\n\n".join(parts).strip() or None

    @staticmethod
    def _failed_result(tool_name: Any, result: Any) -> bool:
        if not isinstance(result, str):
            return False
        value = result.lstrip()
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            lower = value[:500].lower()
            return value.startswith("Error") or '"error"' in lower or '"failed"' in lower
        if not isinstance(payload, dict):
            return False
        if tool_name == "terminal" and payload.get("exit_code") not in (None, 0):
            return True
        return payload.get("status") in {"error", "failed"} or bool(payload.get("error"))

    def _gap(self, code: str, invocation_id: str | None, detail: str | None = None) -> None:
        with self._lock:
            gap = {"code": code, "invocation_id": invocation_id, "detail": detail}
            if gap not in self._gaps:
                self._gaps.append(gap)
