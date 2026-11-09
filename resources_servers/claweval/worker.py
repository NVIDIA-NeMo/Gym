# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own ClawEval task state in its Python environment, without running an assistant loop.

The parent exchanges JSON lines over private pipes. Native diagnostics go to stderr.
This module uses ClawEval's installed APIs; it contains no vendored ClawEval/OpenCode code.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TextIO
from uuid import uuid4


def validate_source(root: Path, revision: str) -> None:
    """Require the configured, clean native source before importing it."""
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if actual != revision or dirty.strip():
        raise ValueError(
            "ClawEval source must match claweval_revision and have no tracked, staged or untracked changes"
        )


def merge_config(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value


def match_dispatch(name: str, arguments: dict, dispatches: list) -> tuple[int, tuple, bool]:
    """Join a call to an execution, allowing repeated calls only when their results are indistinguishable."""
    matches = [
        (index, record)
        for index, record in enumerate(dispatches)
        if name.casefold()
        in {
            record[0].name.casefold(),
            f"claweval_{record[0].name}".casefold(),
            f"mcp__claweval__{record[0].name}".casefold(),
        }
        and record[0].input == arguments
    ]
    if not matches:
        raise ValueError("Harness tool transcript has no matching native dispatch")
    signatures = {
        json.dumps(
            {
                "result": record[1].model_dump(exclude={"tool_use_id"}),
                "status": record[2].response_status,
                "body": record[2].response_body,
                "media": [block.model_dump() for block in record[3]],
            },
            sort_keys=True,
        )
        for _, record in matches
    }
    if len(signatures) != 1:
        raise ValueError("Repeated native calls have distinguishable results and cannot be safely joined")
    index, record = matches[0]
    return index, record, len(matches) > 1


class TaskRuntime:
    """A single native task's services, sandbox, transcript and verifier."""

    def __init__(self) -> None:
        self.stack = contextlib.ExitStack()
        self.dispatches: list = []
        self.messages: list = []
        self.rounds = 0
        self.done = False
        self.usage: list[dict | None] = []
        self.finalized = False
        self.failure: str | None = None
        self.started = time.monotonic()

    def seed(self, payload: dict) -> dict:
        from claw_eval.config import Config, load_config
        from claw_eval.models.message import Message
        from claw_eval.models.task import TaskDefinition
        from claw_eval.models.trace import TraceStart
        from claw_eval.runner.dispatcher import ToolDispatcher
        from claw_eval.runner.loop import _build_initial_user_content
        from claw_eval.runner.sandbox_dispatcher import SandboxToolDispatcher
        from claw_eval.runner.sandbox_tools import SANDBOX_TOOLS
        from claw_eval.runner.system_prompt import build_system_prompt
        from claw_eval.trace.writer import TraceWriter
        from evaluation import run_multimodal as native

        self.payload = payload
        self.root = Path(payload["claweval_root"]).resolve()
        data = payload["task_data"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", data["task_id"]):
            raise ValueError("Invalid ClawEval task directory")
        self.path = (self.root / "tasks" / data["task_id"] / "task.yaml").resolve()
        if not self.path.is_relative_to(self.root / "tasks"):
            raise ValueError("Task path escaped the task root")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != data["task_sha256"]:
            raise ValueError("Native task digest changed; prepare the taskset again")
        self.task = TaskDefinition.from_yaml(self.path)
        if self.task.task_id != data["task_id"] or self.task.prompt.text != data["prompt"]:
            raise ValueError("Prepared task identity/prompt does not match the native task")
        if self.task.environment.enable_todo or self.task.environment.enable_compact:
            raise ValueError("Task requires native assistant tools, which an interchangeable harness cannot supply")
        path = Path(payload["claweval_config"])
        self.config_path = path if path.is_absolute() else self.root / path
        raw = load_config(self.config_path).model_dump()
        for section in ("model", "judge", "user_agent_model"):
            merge_config(raw[section], payload.get(f"{section}_overrides", {}))
        self.cfg = Config.model_validate(raw)
        self.output = Path(payload["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.fixture_root = Path(payload["fixture_root"] or self.root / "tasks").resolve()
        missing = native.validate_assets([self.path], self.fixture_root)
        if missing:
            raise FileNotFoundError("Missing native task assets: " + "; ".join(missing))
        self.workspace = self.output / "workspace"
        self.workspace.mkdir()
        native.stage_task_workspace(self.task, self.path.parent, self.fixture_root, self.workspace)
        native.isolate_service_ports(self.task, uuid4().hex)
        self.stack.enter_context(
            native.ServiceManager(
                self.task.services, cwd=self.root, mock_today=self.task.environment.mock_today, allow_reuse=False
            )
        )
        self.sandbox = self.stack.enter_context(
            native.RemoteSandbox(
                workspace=self.workspace,
                image=payload["sandbox_image"],
                dependencies=Path(payload["sandbox_dependencies"]),
                server=self.root / "src/claw_eval/sandbox/server.py",
                port=native.available_sandbox_port(payload["sandbox_port"]),
                log_path=self.output / "sandbox.log",
                ready_timeout=payload["sandbox_ready_timeout"],
            )
        )
        self.trace_id = uuid4().hex
        self.trace_path = self.output / "trace.jsonl"
        self.writer = self.stack.enter_context(TraceWriter(self.trace_path))
        self.writer.write_event(
            TraceStart(trace_id=self.trace_id, task_id=self.task.task_id, model=payload["agent_name"])
        )
        initial = _build_initial_user_content(
            self.task, trace_id=self.trace_id, writer=self.writer, model_cfg=self.cfg.model, media_cfg=self.cfg.media
        )
        if any(block.type not in {"text", "image"} for block in initial):
            raise ValueError("This pairing cannot preserve native audio/video input; select a compatible harness")
        self.messages = [Message(role="user", content=initial)]
        http_dispatcher = ToolDispatcher(self.task.get_endpoint_map())
        self.dispatcher = SandboxToolDispatcher(
            http_dispatcher,
            sandbox_url=self.sandbox.url,
            sandbox_instance_id=self.sandbox.instance_id,
            max_images_per_turn=self.cfg.media.max_images_per_turn,
            video_frame_budget=self.cfg.media.video_frame_budget,
            tool_image_max_dimension=self.cfg.media.tool_image_max_dimension,
            tool_image_quality=self.cfg.media.tool_image_quality,
            input_modalities=self.cfg.model.input_modalities,
        )
        self.stack.callback(self.dispatcher.close)
        existing = {tool.name for tool in self.task.tools}
        extra_tools = [tool for tool in SANDBOX_TOOLS if tool.name not in existing]
        self.tools = list(self.task.tools) + extra_tools
        self.system = build_system_prompt(self.task, self.cfg.prompt, extra_tools=extra_tools)
        self.user = native._make_user_agent(self.cfg, self.task)
        if self.user is not None:
            self.stack.callback(self.user.client.close)
        return {
            "instructions": self.system,
            "input": [
                {"type": "input_text", "text": block.text}
                if block.type == "text"
                else {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": f"data:{block.mime_type};base64,{block.data}",
                }
                for block in initial
            ],
            "tools": [tool.model_dump() for tool in self.tools],
            "max_user_rounds": self.task.user_agent.max_rounds if self.user is not None else 0,
            "max_turns": self.task.environment.max_turns,
            "timeout_seconds": self.task.environment.timeout_seconds,
        }

    def tool(self, payload: dict) -> dict:
        from claw_eval.models.content import ToolUseBlock

        if self.finalized:
            raise ValueError("The task is already finalized")
        if payload["name"] not in {tool.name for tool in self.tools}:
            raise ValueError("Tool is not available for this task")
        call = ToolUseBlock(id=uuid4().hex, name=payload["name"], input=payload["arguments"])
        result, event, media = self.dispatcher.dispatch(call, self.trace_id)
        parts = list(result.content)
        self.dispatches.append((call, result, event, media or []))
        with (self.output / "dispatches.jsonl").open("a") as journal:
            journal.write(
                json.dumps({"call": call.model_dump(), "result": result.model_dump(), "event": event.model_dump()})
                + "\n"
            )
        if any(block.type != "image" for block in media or []):
            raise ValueError("This pairing cannot preserve native audio/video tool results")
        return {
            "content": [{"type": "text", "text": block.text} for block in parts]
            + [{"type": "image", "data": block.data, "mimeType": block.mime_type} for block in media or []],
            "isError": result.is_error,
        }

    def simulate(self, payload: dict) -> dict:
        from claw_eval.models.content import TextBlock
        from claw_eval.models.message import Message

        self.messages = []
        for item in payload["conversation"]:
            if item.get("role") in {"user", "assistant"}:
                content = item.get("content", "")
                text = (
                    content
                    if isinstance(content, str)
                    else "\n".join(
                        part["text"] for part in content if part.get("type") in {"text", "input_text", "output_text"}
                    )
                )
                self.messages.append(Message(role=item["role"], content=[TextBlock(text=text)]))
        self.usage.append(payload.get("usage"))
        if self.user is None or self.rounds >= self.task.user_agent.max_rounds:
            return {"done": True}
        text = self.user.generate_response(self.task.user_agent.persona, self.messages)
        self.done = text is None
        if text is None:
            return {"done": True}
        self.rounds += 1
        value = f"[user_agent]\n{text}"
        return {"done": False, "input": value}

    def verify(self, payload: dict) -> dict:
        if self.finalized:
            raise ValueError("Verification already ran")
        if self.failure is not None:
            raise RuntimeError(f"Native task infrastructure/content failed: {self.failure}; grading is prohibited")
        from claw_eval.models.content import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
        from claw_eval.models.message import Message
        from claw_eval.models.trace import AuditSnapshot, TraceEnd, TraceMessage
        from evaluation import run_multimodal as native

        self.finalized = True
        # A harness transcript is evidence, not a replacement for the task's actual side effects.
        self.writer.write_event(
            TraceMessage(trace_id=self.trace_id, message=Message(role="system", content=[TextBlock(text=self.system)]))
        )
        dispatches = list(self.dispatches)
        observed_calls = 0
        indistinguishable_joins = 0
        recorded_media: set[str] = set()
        (self.output / "conversation.json").write_text(json.dumps(payload["conversation"]))
        for item in payload["conversation"]:
            if item.get("type") == "function_call":
                name, arguments = item["name"], json.loads(item["arguments"])
                index, (call, result, event, media), repeated = match_dispatch(name, arguments, dispatches)
                indistinguishable_joins += int(repeated)
                dispatches.pop(index)
                observed_calls += 1
                call = ToolUseBlock(id=item["call_id"], name=call.name, input=call.input)
                self.writer.write_event(
                    TraceMessage(trace_id=self.trace_id, message=Message(role="assistant", content=[call]))
                )
                self.writer.write_event(event.model_copy(update={"tool_use_id": call.id}))
                result = ToolResultBlock(tool_use_id=call.id, content=result.content, is_error=result.is_error)
                self.writer.write_event(
                    TraceMessage(trace_id=self.trace_id, message=Message(role="user", content=[result]))
                )
                if media:
                    self.writer.write_event(
                        TraceMessage(trace_id=self.trace_id, message=Message(role="user", content=media))
                    )
                    recorded_media.update(block.data for block in media)
            elif item.get("role") in {"user", "assistant", "system", "developer"}:
                role = "system" if item["role"] == "developer" else item["role"]
                content = item.get("content", "")
                blocks = [TextBlock(text=content)] if isinstance(content, str) else []
                for part in content if isinstance(content, list) else []:
                    if part.get("type") in {"input_text", "output_text", "text"}:
                        blocks.append(TextBlock(text=part["text"]))
                    elif part.get("type") == "input_image":
                        header, data = part["image_url"].split(",", 1)
                        if data not in recorded_media:
                            blocks.append(ImageBlock(data=data, mime_type=header[5:].split(";", 1)[0]))
                    else:
                        raise ValueError(
                            "Harness conversation contains unsupported media; do not score a partial trace"
                        )
                self.writer.write_event(
                    TraceMessage(trace_id=self.trace_id, message=Message(role=role, content=blocks))
                )
        if dispatches:
            raise ValueError("Native tool executions are missing from the harness transcript")
        for service in self.task.services:
            if service.reset_endpoint:
                import httpx

                url = service.reset_endpoint.rsplit("/reset", 1)[0] + "/audit"
                response = httpx.get(url, timeout=5)
                response.raise_for_status()
                self.writer.write_event(
                    AuditSnapshot(
                        trace_id=self.trace_id, service_name=service.name, audit_url=url, audit_data=response.json()
                    )
                )
        usages = [usage for usage in self.usage if usage is not None]
        input_tokens = sum(usage["input_tokens"] for usage in usages)
        output_tokens = sum(usage["output_tokens"] for usage in usages)
        self.writer.write_event(
            TraceEnd(
                trace_id=self.trace_id,
                total_turns=payload["model_turns"],
                input_tokens=input_tokens,
                model_input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                wall_time_s=time.monotonic() - self.started,
                user_agent_rounds=self.rounds,
                user_agent_max_rounds=self.task.user_agent.max_rounds if self.user else 0,
                user_agent_done=self.done,
            )
        )
        count = native.stage_grader_files(self.task, self.path.parent, self.fixture_root, self.workspace)
        if count != len(self.task.sandbox_grader_files):
            raise RuntimeError("Native grader fixtures were not staged")
        snapshot = native._collect_env_snapshot(
            self.sandbox.url, self.task, sandbox_headers={"X-Claw-Sandbox-Instance": self.sandbox.instance_id}
        )
        native._save_env_snapshot(snapshot, self.trace_path, self.task.task_id)
        snapshot = native.add_local_grader_files(snapshot, self.task, self.path.parent, self.fixture_root)
        self.writer.close()
        start, messages, events, media, end, audit = native.load_trace(self.trace_path)
        grader = native.get_grader(self.task.task_id, tasks_dir=self.path.parent.parent, task_dir=self.path.parent)
        judge = native.build_judge(self.cfg, no_judge=self.payload["no_judge"])
        if judge is not None:
            self.stack.callback(judge.client.close)
        scores, judge_calls = native._grade_with_optional_params(
            grader,
            messages,
            events,
            self.task,
            audit_data=audit,
            judge=judge,
            media_events=media,
            env_snapshot=snapshot,
        )
        score = native.compute_task_score(scores)
        native._append_grading_to_trace(
            self.trace_path,
            trace_id=start.trace_id,
            task_id=self.task.task_id,
            scores=scores,
            task_score=score,
            passed=native.is_pass(score),
            judge_calls=judge_calls,
            user_agent_meta={"rounds_used": self.rounds, "done_reached": self.done},
        )
        result = {
            "reward": score,
            "task_score": score,
            "passed": native.is_pass(score),
            "reward_components": scores.model_dump(),
            "trace": str(self.trace_path),
            "native_tool_executions": observed_calls,
            "indistinguishable_dispatch_joins": indistinguishable_joins,
            "usage_complete": len(usages) == len(self.usage),
            "provenance": {
                "revision": self.payload["claweval_revision"],
                "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
                "task_sha256": self.payload["task_data"]["task_sha256"],
                "task_contract": native.task_contract_fingerprint(self.path, self.fixture_root),
            },
        }
        (self.output / "verification.json").write_text(json.dumps(result))
        return result

    def close(self) -> None:
        self.stack.close()


def isolate_protocol_stdio() -> tuple[TextIO, TextIO]:
    """Reserve private, non-inheritable RPC descriptors before native subprocesses start."""
    incoming = os.fdopen(os.dup(sys.stdin.fileno()), "r", encoding="utf-8")
    outgoing = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    with open(os.devnull) as empty:
        os.dup2(empty.fileno(), sys.stdin.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return incoming, outgoing


def main() -> None:
    """Serve one native task until the parent closes its pipe or requests teardown."""
    incoming, protocol = isolate_protocol_stdio()

    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt("ClawEval task worker terminated")

    signal.signal(signal.SIGTERM, terminate)
    runtime = TaskRuntime()
    try:
        for line in incoming:
            payload = json.loads(line)
            operation = payload.pop("operation")
            try:
                if operation == "seed":
                    root = Path(payload["claweval_root"]).resolve()
                    validate_source(root, payload["claweval_revision"])
                    sys.path[:0] = [str(root / "src"), str(root)]
                    os.chdir(root)
                result = getattr(runtime, operation)(payload) if operation != "close" else runtime.close()
                protocol.write(json.dumps({"result": result}) + "\n")
                protocol.flush()
                if operation == "close":
                    return
            except Exception as error:
                if operation in {"tool", "simulate"}:
                    runtime.failure = f"{type(error).__name__}: {error}"
                traceback.print_exc()
                protocol.write(
                    json.dumps({"error_type": type(error).__name__, "error": f"{type(error).__name__}: {error}"})
                    + "\n"
                )
                protocol.flush()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
