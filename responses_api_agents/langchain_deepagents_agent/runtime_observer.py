# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native LangChain callback evidence; this module also runs without Gym installed."""

from __future__ import annotations

import asyncio
import json
import time
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from responses_api_agents.langchain_deepagents_agent.runtime_messages import to_gym_input


_summary_scope: ContextVar[list[dict[str, Any]] | None] = ContextVar("deepagents_summary_scope", default=None)


def observe_native_summarization() -> None:
    """Observe the pinned SDK boundary without changing its prompts or retry behavior.

    The isolated runner owns this process. Context-local state also separates
    simultaneous summaries in different native child agents.
    """
    from deepagents.middleware.summarization import SummarizationMiddleware

    original = SummarizationMiddleware._acreate_summary
    if getattr(original, "_gym_observed", False):
        return

    @wraps(original)
    async def observed(self, messages_to_summarize):
        records: list[dict[str, Any]] = []
        token = _summary_scope.set(records)
        try:
            summary = await original(self, messages_to_summarize)
        except BaseException as error:
            for record in records:
                record["outcome"] = "aborted" if isinstance(error, asyncio.CancelledError) else "failed"
            raise
        else:
            for record in records:
                record.update(outcome="completed", summary=summary)
            return summary
        finally:
            _summary_scope.reset(token)

    observed._gym_observed = True
    SummarizationMiddleware._acreate_summary = observed


class DeepAgentsObserver(AsyncCallbackHandler):
    """Record actual model/tool boundaries and task-tool child invocations."""

    raise_error = True
    run_inline = True

    def __init__(
        self, *, output: Path, task_id: str, rollout_id: str, model_ref: dict[str, str], sandbox_id: str
    ) -> None:
        self.output = output
        self.task_id, self.rollout_id = task_id, rollout_id
        self.model_ref, self.sandbox_id = model_ref, sandbox_id
        self.parents: dict[str, str | None] = {}
        self.children: dict[str, str] = {}
        self.invocations: dict[str, dict[str, Any]] = {"root": self._invocation("root")}
        self.tools: dict[str, dict[str, Any]] = {}
        self.pending_tool_errors: set[tuple[str, str]] = set()
        self.turns: list[dict[str, Any]] = []
        self.usages: list[dict[str, Any] | None] = []
        self.compactions: list[dict[str, Any]] = []
        self.gaps: list[dict[str, str]] = []
        self.initial_input_counts: dict[str, int] = {}
        self.status = "incomplete"
        self.error: str | None = None
        self.error_status: int | None = None
        self.stop_reason: str | None = None
        self.shell_results: dict[str, int] = {}
        self.runtime: dict[str, Any] | None = None

    @staticmethod
    def _invocation(name: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "kind": "agent_invocation",
            "invocation_id": name,
            "status": "incomplete",
            "model_calls": [],
            "conversation": [],
            **kwargs,
        }

    def owner(self, run_id: UUID | str | None) -> str:
        current = str(run_id) if run_id else None
        while current:
            if current in self.children:
                return self.children[current]
            current = self.parents.get(current)
        return "root"

    async def on_chain_start(
        self, serialized: dict | None, inputs: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        self.parents[str(run_id)] = str(parent_run_id) if parent_run_id else None

    async def on_chat_model_start(
        self, serialized: dict, messages: list, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        self.parents[str(run_id)] = str(parent_run_id) if parent_run_id else None

    async def on_tool_start(
        self, serialized: dict, input_str: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        key = str(run_id)
        self.parents[key] = str(parent_run_id) if parent_run_id else None
        owner = self.owner(parent_run_id)
        call_id = kwargs.get("tool_call_id")
        if not call_id:
            raise ValueError("DeepAgents tool callback omitted its native tool_call_id")
        name = serialized["name"]
        self.tools[key] = {
            "kind": "tool_call",
            "invocation_id": owner,
            "tool_call_id": call_id,
            "tool_name": name,
            "sandbox_id": self.sandbox_id,
            "started_at": time.time(),
            "status": "incomplete",
            "timing_source": "harness",
        }
        if name == "task":
            child = f"{owner}/{call_id}"
            self.children[key] = child
            self.invocations[child] = self._invocation(
                child, parent_invocation_id=owner, spawned_by_tool_call_id=call_id
            )
        self.save()

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        record = self.tools[key]
        # State-updating native tools (write_todos and task) return Command, not
        # ToolMessage. Retain only their actual message for this native call.
        messages = [output] if isinstance(output, ToolMessage) else []
        if isinstance(output, Command) and isinstance(output.update, dict):
            updates = output.update.get("messages", [])
            messages = updates if isinstance(updates, list) else [updates]
        messages = [
            message
            for message in messages
            if isinstance(message, ToolMessage) and message.tool_call_id == record["tool_call_id"]
        ]
        now = time.time()
        record.update(
            completed_at=now,
            duration_ms=max(0, (now - record["started_at"]) * 1000),
            status="failed" if any(message.status == "error" for message in messages) else "completed",
        )
        if key in self.shell_results:
            exit_code = self.shell_results.pop(key)
            record["status"] = "timeout" if exit_code == 124 else "failed" if exit_code else "completed"
            if exit_code:
                record["error_type"] = "CommandTimeout" if exit_code == 124 else "CommandFailed"
        if messages:
            self.invocations[record["invocation_id"]]["conversation"].extend(to_gym_input(messages))
        else:
            self.gaps.append(
                {
                    "code": "tool_output_unavailable",
                    "invocation_id": record["invocation_id"],
                    "detail": record["tool_call_id"],
                }
            )
        if key in self.children:
            self.invocations[self.children[key]]["status"] = record["status"]
        self.save()

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        key = str(run_id)
        record = self.tools[key]
        now = time.time()
        record.update(
            completed_at=now,
            duration_ms=max(0, (now - record["started_at"]) * 1000),
            status="failed",
            error_type=type(error).__name__,
        )
        if key in self.children:
            self.invocations[self.children[key]]["status"] = "failed"
        self.pending_tool_errors.add((record["invocation_id"], record["tool_call_id"]))
        self.gaps.append(
            {
                "code": "tool_output_unavailable",
                "invocation_id": record["invocation_id"],
                "detail": record["tool_call_id"],
            }
        )
        self.save()

    def _retain_tool_errors(self, *, owner: str, items: list[dict[str, Any]]) -> None:
        # The SDK constructs recoverable validation-error ToolMessages after
        # on_tool_error. Observe its actual next input; never synthesize the text
        # from the exception or borrow a result from another invocation.
        if not self.pending_tool_errors:
            return
        for item in items:
            key = (owner, item.get("call_id"))
            if item.get("type") != "function_call_output" or key not in self.pending_tool_errors:
                continue
            self.invocations[owner]["conversation"].append(dict(item))
            self.pending_tool_errors.remove(key)
            self.gaps = [
                gap
                for gap in self.gaps
                if not (
                    gap["code"] == "tool_output_unavailable"
                    and gap.get("invocation_id") == owner
                    and gap.get("detail") == item["call_id"]
                )
            ]

    def begin_model(self, *, run_id: UUID, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        owner = self.owner(run_id)
        invocation = self.invocations[owner]
        self._retain_tool_errors(owner=owner, items=request["input"])
        summary_scope = _summary_scope.get()
        if summary_scope is not None:
            if summary_scope:
                compaction = summary_scope[0]
                compaction["source_message_ids"].append(str(run_id))
                return owner, compaction
            compaction = {
                "kind": "context_compaction",
                "invocation_id": owner,
                "source_message_ids": [str(run_id)],
                "source_model_ref": self.model_ref,
                "observed_at": time.time(),
                "trigger": "context_pressure",
                "outcome": "unknown",
                "model_calls": [],
            }
            self.compactions.append(compaction)
            summary_scope.append(compaction)
            self.save()
            return owner, compaction
        if not invocation["conversation"]:
            invocation["conversation"] = list(request["input"])
            self.initial_input_counts[owner] = len(request["input"])
        turn_no = 1 + sum(t["invocation_id"] == owner for t in self.turns)
        turn = {
            "invocation_id": owner,
            "source_message_id": str(run_id),
            "source_model_ref": self.model_ref,
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "turn_no": turn_no,
            "timestamp": time.time(),
            "question": request["input"],
            "step_count": turn_no,
            "model_calls": [],
        }
        self.turns.append(turn)
        self.save()
        return owner, turn

    def end_model(self, *, owner: str, turn: dict[str, Any], response: dict[str, Any]) -> None:
        ref = {"model_ref": self.model_ref, "response_id": response["id"]}
        self.invocations[owner]["model_calls"].append(ref)
        if turn.get("kind") == "context_compaction":
            turn["summary"] = "".join(
                p.get("text", "")
                for item in response["output"]
                if item.get("type") == "message"
                for p in item.get("content", [])
            )
        else:
            self.invocations[owner]["conversation"].extend(response["output"])
            turn["answer"] = response["output"]
        # The collector joins every HTTP attempt through the native run-ID header.
        # A second response-ID reference here would count the successful attempt twice.
        self.usages.append(response.get("usage"))
        self.save()

    def save(self) -> None:
        """Atomically retain partial evidence after each native event."""
        payload = {
            "runtime": self.runtime,
            "output_items": self.invocations["root"]["conversation"][self.initial_input_counts.get("root", 0) :],
            "invocations": list(self.invocations.values()),
            "tools": list(self.tools.values()),
            "turns": self.turns,
            "usages": self.usages,
            "compactions": self.compactions,
            "gaps": self.gaps,
            "status": self.status,
            "error": self.error,
            "error_status": self.error_status,
            "stop_reason": self.stop_reason,
        }
        temporary = self.output.with_suffix(".partial")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.output)
