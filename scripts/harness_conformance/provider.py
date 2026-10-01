# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled Chat/Responses endpoints and private episode witnesses."""

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nemo_gym.base_responses_api_model import ModelCallCaptureConfig, install_model_call_capture
from nemo_gym.chat_streaming import synthesize_chat_completion_sse
from nemo_gym.responses_streaming import (
    flatten_namespace_tools,
    restore_namespace_tool_calls,
    synthesize_responses_sse,
)

from .scenarios import Scenario


@dataclass
class Probe:
    scenario: Scenario
    directory: Path
    attempts: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    seeded: int = 0
    finished: bool = False

    def save(self) -> None:
        """Persist independent observations even if collection fails."""
        witness = {
            "attempts": self.attempts,
            "tool_calls": self.tool_calls,
            "violations": self.violations,
            "verifications": self.verifications,
            "seeded": self.seeded,
            "finished": self.finished,
        }
        (self.directory / "witness.json").write_text(json.dumps(witness, indent=2) + "\n")

    def _tool(self, body: dict) -> tuple[str, dict, dict]:
        tools, namespaces = flatten_namespace_tools(body.get("tools"))
        for tool in tools:
            function = tool.get("function", tool)
            name = function.get("name", "")
            bare = name.rsplit("__", 1)[-1]
            properties = (function.get("parameters") or {}).get("properties", {})
            if bare in ("bash", "terminal", "shell", "shell_command", "exec_command"):
                index = len(self.tool_calls)
                token = f"probe_{uuid4().hex}"
                marker = self.directory / f"tool-{index}.txt"
                code = self.scenario.tool_exit_code if index == 0 else 0
                command = f"printf %s {shlex.quote(token)} > {shlex.quote(str(marker))}; printf %s {shlex.quote(token)}; exit {code}"
                if "cmd" in properties:
                    args = {"cmd": command}
                elif "command" in properties:
                    command_type = properties["command"].get("type")
                    args = {"command": ["sh", "-c", command] if command_type == "array" else command}
                else:
                    continue
                if "description" in (function.get("parameters") or {}).get("required", []):
                    args["description"] = "Run conformance shell check"
                self.tool_calls.append(
                    {
                        "id": f"call_{uuid4().hex}",
                        "name": name,
                        "arguments": args,
                        "token": token,
                        "marker": str(marker),
                        "exit_code": code,
                        "result_seen": False,
                        "outputs": [],
                    }
                )
                return name, args, namespaces
        raise ValueError("harness did not advertise a supported shell tool")

    def _observe_tool_results(self, body: dict) -> None:
        history = body.get("messages", body.get("input", []))
        for call in self.tool_calls:
            results = (
                [
                    item
                    for item in history
                    if isinstance(item, dict)
                    and (
                        item.get("role") == "tool"
                        and item.get("tool_call_id") == call["id"]
                        or item.get("type") == "function_call_output"
                        and item.get("call_id") == call["id"]
                    )
                ]
                if isinstance(history, list)
                else []
            )
            call["result_seen"] = call["result_seen"] or any(
                call["token"] in json.dumps(item.get("content", item.get("output"))) for item in results
            )
            for item in results:
                output = item.get("content", item.get("output"))
                if output is not None and output not in call["outputs"]:
                    call["outputs"].append(output)
            marker = Path(call["marker"])
            call["executed"] = marker.is_file() and marker.read_text() == call["token"]

    async def model(self, request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        self._observe_tool_results(body)
        index = len(self.attempts)
        status = 200
        if index < len(self.scenario.http_errors):
            status = self.scenario.http_errors[index]
        elif self.scenario.terminal_error:
            status = self.scenario.http_errors[-1]
        response_id = f"resp_{uuid4().hex}"
        namespaces = {}
        if status != 200:
            payload = {
                "error": {"message": "controlled model failure", "type": "conformance_error", "code": str(status)}
            }
        else:
            try:
                if self.finished:
                    raise ValueError("unexpected model request after the scripted final answer")
                if any(not call["result_seen"] or not call.get("executed") for call in self.tool_calls):
                    raise ValueError("tool execution or model-visible result was not witnessed")
                if len(self.tool_calls) < self.scenario.tool_steps:
                    name, arguments, namespaces = self._tool(body)
                    call = self.tool_calls[-1]
                    output = [
                        {
                            "type": "function_call",
                            "id": call["id"],
                            "call_id": call["id"],
                            "name": name,
                            "arguments": json.dumps(arguments),
                            "status": "completed",
                        }
                    ]
                else:
                    output = [
                        {
                            "type": "message",
                            "id": f"msg_{uuid4().hex}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "CONFORMANCE_DONE", "annotations": []}],
                        }
                    ]
                    self.finished = True
                if request.url.path.endswith("/chat/completions"):
                    message = {
                        "role": "assistant",
                        "content": "CONFORMANCE_DONE" if self.finished else None,
                        "reasoning_content": "Perform the prescribed check.",
                    }
                    if not self.finished:
                        message["tool_calls"] = [
                            {
                                "id": output[0]["call_id"],
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ]
                    payload = {
                        "id": response_id,
                        "object": "chat.completion",
                        "created": 1,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": "stop" if self.finished else "tool_calls",
                            }
                        ],
                    }
                    if self.scenario.usage:
                        payload["usage"] = {
                            "prompt_tokens": 20 + index,
                            "completion_tokens": 8,
                            "total_tokens": 28 + index,
                            "prompt_tokens_details": {"cached_tokens": 3},
                            "completion_tokens_details": {"reasoning_tokens": 2},
                        }
                else:
                    payload = {
                        "id": response_id,
                        "object": "response",
                        "created_at": 1,
                        "status": "completed",
                        "error": None,
                        "incomplete_details": None,
                        "model": body["model"],
                        "output": restore_namespace_tool_calls(output, namespaces),
                    }
                    if self.scenario.usage:
                        payload["usage"] = {
                            "input_tokens": 20 + index,
                            "output_tokens": 8,
                            "total_tokens": 28 + index,
                            "input_tokens_details": {"cached_tokens": 3},
                            "output_tokens_details": {"reasoning_tokens": 2},
                        }
            except ValueError as exc:
                status = 409
                self.violations.append(str(exc))
                payload = {"error": {"message": str(exc), "type": "probe_protocol_error"}}
        self.attempts.append(
            {"attempt_id": f"attempt-{index}", "request": body, "response": payload, "status_code": status}
        )
        self.save()
        if status != 200 or not body.get("stream"):
            return JSONResponse(payload, status_code=status, headers={"Retry-After": "0"})
        events = (
            synthesize_chat_completion_sse(payload, include_usage=True)
            if request.url.path.endswith("/chat/completions")
            else synthesize_responses_sse(payload)
        )
        return StreamingResponse(events, media_type="text/event-stream")

    def model_app(self) -> FastAPI:
        app = FastAPI()
        install_model_call_capture(
            app,
            ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=self.directory / "capture"),
            model_server_name="policy_model",
        )
        app.post("/v1/chat/completions", response_model=None)(self.model)
        app.post("/v1/responses", response_model=None)(self.model)
        return app

    def resources_app(self) -> FastAPI:
        app = FastAPI()

        async def seed(request: Request) -> dict:
            self.seeded += 1
            self.save()
            return {}

        async def verify(request: Request) -> dict:
            body = await request.json()
            output = body.get("response", {}).get("output", [])
            complete = any(
                item.get("type") == "message"
                and item.get("role") == "assistant"
                and any(
                    part.get("text") == "CONFORMANCE_DONE"
                    for part in item.get("content", [])
                    if isinstance(part, dict)
                )
                for item in output
            )
            reward = float(complete and self.scenario.expected_reward == 1.0)
            self.verifications.append({"reward": reward, "answer_seen": complete})
            self.save()
            return {**body, "reward": reward}

        for prefix in ("", "/ng-rollout/{rollout_id}"):
            app.post(prefix + "/seed_session")(seed)
            app.post(prefix + "/verify")(verify)
        return app
