# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect Hermes events inside a sandbox without importing NeMo Gym."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from contextvars import ContextVar
from copy import deepcopy
from time import monotonic, time
from typing import Any


_SUMMARY_CAPTURE: ContextVar[Any] = ContextVar("gym_hermes_summary_capture", default=None)


def _install_summary_capture() -> None:
    # The pinned compressor calls this module-level function, including for
    # delegated agents. ContextVar keeps overlapping invocations separate.
    import agent.context_compressor as compressor

    original = compressor.call_llm
    if getattr(original, "_gym_observed", False):
        return

    def observed(**kwargs: Any) -> Any:
        response = original(**kwargs)
        capture = _SUMMARY_CAPTURE.get()
        if capture is not None and kwargs.get("task") == "compression":
            capture(response)
        return response

    observed._gym_observed = True
    compressor.call_llm = observed


class _ObservedChildren(list):
    _gym_child_hook_supported = True

    def __init__(self, values: Iterable[Any], observer: "SandboxHermesObserver", parent_id: str):
        super().__init__(values)
        self.observer = observer
        self.parent_id = parent_id
        self._gym_prepare_child = getattr(values, "_gym_prepare_child", None)

    def append(self, child: Any) -> None:
        if self._gym_prepare_child is not None:
            self._gym_prepare_child(child)
        super().append(child)
        self.observer._child_added(child, self.parent_id)


class SandboxHermesObserver:
    """Capture raw event dictionaries for host-side validation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._child_index = 0
        self._started_ticks: dict[tuple[str, str], float] = {}
        self._tools: dict[tuple[str, str], dict[str, Any]] = {}
        self._invocations: dict[str, dict[str, Any]] = {
            "root": {
                "invocation_id": "root",
                "model_response_ids": [],
                "messages": [],
                "status": "unknown",
            }
        }
        self._compactions: list[dict[str, Any]] = []
        self._history: dict[str, tuple[list[dict[str, Any]], int]] = {}
        self._summary_usage_unknown = False
        self._summary_details_unknown = False
        self._gaps: list[dict[str, Any]] = []

    def instrument(self, agent: Any) -> "SandboxHermesObserver":
        _install_summary_capture()
        self._instrument(agent, "root", wrap_conversation=False)
        return self

    def on_iteration_limit_reached(self, *, invocation_id: str) -> None:
        """Record this invocation's budget stop once, without affecting its parent or summary."""
        with self._lock:
            invocation = self._invocations.get(invocation_id)
            if invocation is None:
                self._gap("hermes_observer_error", invocation_id, "iteration_limit: unknown invocation")
                return
            invocation["stop_reason"] = "max_iterations"
            invocation["status"] = "incomplete"

    def finish(
        self,
        result: dict[str, Any] | None,
        error: BaseException | None,
    ) -> dict[str, Any]:
        self._record_conversation("root", result, error)
        with self._lock:
            for tool in self._tools.values():
                if tool["status"] == "unknown":
                    tool["status"] = "incomplete"
            return {
                "invocations": list(self._invocations.values()),
                "tools": list(self._tools.values()),
                "compactions": list(self._compactions),
                "gaps": list(self._gaps),
            }

    def _instrument(self, agent: Any, invocation_id: str, *, wrap_conversation: bool) -> None:
        agent._gym_invocation_id = invocation_id
        agent._gym_on_iteration_limit_reached = self.on_iteration_limit_reached
        self._chain_callback(agent, "tool_start_callback", self._tool_started, invocation_id)
        self._chain_callback(agent, "tool_complete_callback", self._tool_completed, invocation_id)
        self._wrap_model_calls(agent, invocation_id)
        self._wrap_compaction(agent, invocation_id)

        children = getattr(agent, "_active_children", None)
        if isinstance(children, list):
            agent._active_children = _ObservedChildren(children, self, invocation_id)
        else:
            self._gap("hermes_hook_unavailable", invocation_id, "_active_children")

        if wrap_conversation:
            original = getattr(agent, "run_conversation", None)
            if not callable(original):
                self._gap("hermes_hook_unavailable", invocation_id, "run_conversation")
                return

            def run(*args: Any, **kwargs: Any) -> Any:
                try:
                    child_result = original(*args, **kwargs)
                except BaseException as error:
                    self._record_conversation(invocation_id, None, error)
                    raise
                self._record_conversation(invocation_id, child_result, None)
                return child_result

            agent.run_conversation = run

    def _chain_callback(self, agent: Any, name: str, callback: Any, invocation_id: str) -> None:
        if not hasattr(agent, name):
            self._gap("hermes_hook_unavailable", invocation_id, name)
            return
        previous = getattr(agent, name)

        def chained(*args: Any, **kwargs: Any) -> None:
            try:
                callback(invocation_id, *args, **kwargs)
            except Exception as error:
                self._gap("hermes_observer_error", invocation_id, f"{name}:{type(error).__name__}")
            if callable(previous):
                try:
                    previous(*args, **kwargs)
                except Exception:
                    pass

        setattr(agent, name, chained)

    def _wrap_model_calls(self, agent: Any, invocation_id: str) -> None:
        original = getattr(agent, "_interruptible_api_call", None)
        if not callable(original):
            self._gap("hermes_hook_unavailable", invocation_id, "_interruptible_api_call")
            return

        def call(*args: Any, **kwargs: Any) -> Any:
            # Observation failures must not prevent the underlying model request.
            try:
                # The SDK preserves this on HTTP retries, including attempts without a response ID.
                api_kwargs = dict(args[0] if args else kwargs["api_kwargs"])
                api_kwargs["extra_headers"] = {
                    **(api_kwargs.get("extra_headers") or {}),
                    "x-session-id": invocation_id,
                }
                if args:
                    args = (api_kwargs, *args[1:])
                else:
                    kwargs = {**kwargs, "api_kwargs": api_kwargs}
            except Exception as error:
                self._gap("hermes_observer_error", invocation_id, f"model_headers:{type(error).__name__}")
            response = original(*args, **kwargs)
            try:
                response_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
                if isinstance(response_id, str) and response_id:
                    with self._lock:
                        response_ids = self._invocations[invocation_id]["model_response_ids"]
                        if response_id not in response_ids:
                            response_ids.append(response_id)
                else:
                    self._gap("model_response_id_unavailable", invocation_id)
            except Exception as error:
                self._gap("hermes_observer_error", invocation_id, f"model_response:{type(error).__name__}")
            return response

        agent._interruptible_api_call = call

    def _wrap_compaction(self, agent: Any, invocation_id: str) -> None:
        original = getattr(agent, "_compress_context", None)
        if not callable(original):
            self._gap("hermes_hook_unavailable", invocation_id, "_compress_context")
            return

        def compact(*args: Any, **kwargs: Any) -> Any:
            compressor = getattr(agent, "context_compressor", None)
            count_before = getattr(compressor, "compression_count", None)
            response_ids = []
            summaries = []

            def capture_response(response: Any) -> None:
                if response.id:
                    response_ids.append(response.id)
                    self._invocations[invocation_id]["model_response_ids"].append(response.id)
                summaries.append(response.choices[0].message.content)
                usage = response.usage
                if usage is not None:
                    for source, target in (
                        ("prompt_tokens", "session_prompt_tokens"),
                        ("completion_tokens", "session_completion_tokens"),
                    ):
                        value = getattr(usage, source, None)
                        current = getattr(agent, target, None)
                        if type(value) is int and type(current) is int:
                            setattr(agent, target, current + value)
                    for details, key, target in (
                        ("prompt_tokens_details", "cached_tokens", "session_cache_read_tokens"),
                        ("completion_tokens_details", "reasoning_tokens", "session_reasoning_tokens"),
                    ):
                        value = getattr(getattr(usage, details, None), key, None)
                        current = getattr(agent, target, None)
                        if type(value) is int and type(current) is int:
                            setattr(agent, target, current + value)
                        else:
                            self._summary_details_unknown = True
                            self._gap("compaction_usage_details_unavailable", invocation_id)
                else:
                    self._summary_usage_unknown = True
                    self._gap("compaction_usage_unavailable", invocation_id)

            def capture(response: Any) -> None:
                try:
                    capture_response(response)
                except Exception as error:
                    self._summary_usage_unknown = True
                    self._gap("hermes_observer_error", invocation_id, f"compaction_response:{type(error).__name__}")

            token = _SUMMARY_CAPTURE.set(capture)
            failed = False
            before = deepcopy(args[0] if args else kwargs.get("messages", []))
            result = None
            try:
                result = original(*args, **kwargs)
                return result
            except BaseException:
                failed = True
                raise
            finally:
                _SUMMARY_CAPTURE.reset(token)
                count_after = getattr(compressor, "compression_count", None)
                # Hermes invokes compression even when every message is protected.
                # That no-op is not a completed compaction.
                changed = count_before is not None and count_after != count_before
                if changed and isinstance(result, tuple) and isinstance(result[0], list):
                    retained, offset = self._history.get(invocation_id, ([], 0))
                    self._history[invocation_id] = (retained + before[offset:], len(result[0]))
                if failed or changed or response_ids:
                    after = getattr(compressor, "last_prompt_tokens", None)
                    with self._lock:
                        self._compactions.append(
                            {
                                "invocation_id": invocation_id,
                                "observed_at": time(),
                                "trigger": "automatic",
                                "tokens_before": kwargs.get("approx_tokens"),
                                "tokens_after": after if type(after) is int and after >= 0 else None,
                                "outcome": "failed" if failed else "completed",
                                "summary": summaries[-1] if summaries else None,
                                "model_calls": [{"response_id": value} for value in response_ids],
                            }
                        )

        agent._compress_context = compact

    def _child_added(self, child: Any, parent_id: str) -> None:
        with self._lock:
            self._child_index += 1
            invocation_id = f"{parent_id}.child-{self._child_index}"
            self._invocations[invocation_id] = {
                "invocation_id": invocation_id,
                "parent_invocation_id": parent_id,
                "model_response_ids": [],
                "messages": [],
                "status": "unknown",
            }
        self._instrument(child, invocation_id, wrap_conversation=True)

    def _tool_started(self, invocation_id: str, call_id: Any, name: Any, args: Any) -> None:
        key = (invocation_id, str(call_id or ""))
        with self._lock:
            if key in self._tools:
                self._gap("duplicate_tool_call", invocation_id, key[1])
            self._tools[key] = {
                "invocation_id": invocation_id,
                "tool_call_id": key[1],
                "tool_name": str(name or "") or None,
                "started_at": time(),
                "completed_at": None,
                "duration_ms": None,
                "timing_source": "harness",
                "status": "unknown",
                "error_type": None,
            }
            self._started_ticks[key] = monotonic()

    def _tool_completed(self, invocation_id: str, call_id: Any, name: Any, args: Any, result: Any) -> None:
        key = (invocation_id, str(call_id or ""))
        with self._lock:
            if key not in self._tools:
                self._tool_started(invocation_id, call_id, name, args)
            tool = self._tools[key]
            failed = self._failed_result(name, result)
            tool["completed_at"] = time()
            started = self._started_ticks.pop(key, None)
            tool["duration_ms"] = max(0.0, (monotonic() - started) * 1000) if started is not None else None
            tool["status"] = "failed" if failed else "completed"
            tool["error_type"] = "tool_result" if failed else None

    def restore_messages(self, result: dict[str, Any], *, invocation_id: str = "root") -> dict[str, Any]:
        """Export the executed conversation, retaining history removed from model context."""
        result = dict(result)
        if self._summary_usage_unknown:
            result.update(prompt_tokens=None, completion_tokens=None)
        if self._summary_details_unknown or self._summary_usage_unknown:
            result.update(cache_read_tokens=None, reasoning_tokens=None)
        if invocation_id in self._history:
            retained, offset = self._history[invocation_id]
            result["messages"] = [*retained, *(result.get("messages") or [])[offset:]]
        return result

    def _record_conversation(
        self,
        invocation_id: str,
        result: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        messages = (
            self.restore_messages(result, invocation_id=invocation_id).get("messages")
            if isinstance(result, dict)
            else []
        )
        status = "failed" if error or (result and result.get("error")) else "unknown"
        if isinstance(result, dict) and status != "failed":
            if result.get("interrupted") or result.get("completed") is False:
                status = "incomplete"
            elif result.get("completed") is True or result.get("final_response"):
                status = "completed"
            elif messages:
                status = "incomplete"
        with self._lock:
            self._invocations[invocation_id]["messages"] = messages or []
            self._invocations[invocation_id]["status"] = (
                "incomplete" if self._invocations[invocation_id].get("stop_reason") == "max_iterations" else status
            )

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
        gap = {"code": code, "invocation_id": invocation_id, "detail": detail}
        with self._lock:
            if gap not in self._gaps:
                self._gaps.append(gap)
