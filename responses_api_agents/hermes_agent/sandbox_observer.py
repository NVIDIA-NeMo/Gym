# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect Hermes events inside a sandbox without importing NeMo Gym."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from time import monotonic, time
from typing import Any


class _ObservedChildren(list):
    def __init__(self, values: Iterable[Any], observer: "SandboxHermesObserver", parent_id: str):
        super().__init__(values)
        self.observer = observer
        self.parent_id = parent_id

    def append(self, child: Any) -> None:
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
        self._gaps: list[dict[str, Any]] = []

    def instrument(self, agent: Any) -> "SandboxHermesObserver":
        self._instrument(agent, "root", wrap_conversation=False)
        return self

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
            response = original(*args, **kwargs)
            response_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
            if isinstance(response_id, str) and response_id:
                with self._lock:
                    response_ids = self._invocations[invocation_id]["model_response_ids"]
                    if response_id not in response_ids:
                        response_ids.append(response_id)
            else:
                self._gap("model_response_id_unavailable", invocation_id)
            return response

        agent._interruptible_api_call = call

    def _wrap_compaction(self, agent: Any, invocation_id: str) -> None:
        original = getattr(agent, "_compress_context", None)
        if not callable(original):
            self._gap("hermes_hook_unavailable", invocation_id, "_compress_context")
            return

        def compact(*args: Any, **kwargs: Any) -> Any:
            failed = False
            try:
                return original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                after = getattr(getattr(agent, "context_compressor", None), "last_prompt_tokens", None)
                with self._lock:
                    self._compactions.append(
                        {
                            "invocation_id": invocation_id,
                            "observed_at": time(),
                            "trigger": "context_pressure",
                            "tokens_before": kwargs.get("approx_tokens"),
                            "tokens_after": after if isinstance(after, int) and after >= 0 else None,
                            "outcome": "failed" if failed else "completed",
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

    def _record_conversation(
        self,
        invocation_id: str,
        result: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        messages = result.get("messages") if isinstance(result, dict) else []
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
            self._invocations[invocation_id]["status"] = status

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
