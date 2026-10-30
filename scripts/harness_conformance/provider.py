# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled Chat/Responses endpoints and private episode witnesses."""

import asyncio
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.base_responses_api_model import ModelCallCaptureConfig, install_model_call_capture
from nemo_gym.chat_streaming import synthesize_chat_completion_sse
from nemo_gym.core import NeMoGymChatCompletion
from nemo_gym.judge import call_judge
from nemo_gym.observability.verifiers import observe_verifier
from nemo_gym.responses_streaming import (
    flatten_namespace_tools,
    restore_namespace_tool_calls,
    synthesize_responses_sse,
)
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from nemo_gym.server_utils import ServerClient

from .messages import is_title_request, messages_response, messages_sse
from .scenarios import Scenario


SHELL_TOOLS = ("bash", "Bash", "terminal", "shell", "shell_command", "exec_command", "code_exec")
# Harnesses that end an episode on a tool call rather than on a text answer.
FINISH_TOOLS = ("finish", "submit")
ANSWER = "CONFORMANCE_DONE"


@dataclass
class Probe:
    scenario: Scenario
    directory: Path
    harness: str = ""
    attempts: list[dict] = field(default_factory=list)
    auxiliary_calls: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    agent_responses: list[dict] = field(default_factory=list)
    judge_calls: list[dict] = field(default_factory=list)
    seeded: int = 0
    closed: int = 0
    finished: bool = False
    completion_proposed: bool = False
    finish_call: dict | None = None

    def save(self) -> None:
        """Persist independent observations even if collection fails."""
        witness = {
            "attempts": self.attempts,
            "auxiliary_calls": self.auxiliary_calls,
            "tool_calls": self.tool_calls,
            "violations": self.violations,
            "verifications": self.verifications,
            "agent_responses": self.agent_responses,
            "judge_calls": self.judge_calls,
            "seeded": self.seeded,
            "closed": self.closed,
            "finished": self.finished,
            "finish_call": self.finish_call,
        }
        (self.directory / "witness.json").write_text(json.dumps(witness, indent=2) + "\n")

    def _tool(self, body: dict) -> tuple[str, dict, dict]:
        tools, namespaces = flatten_namespace_tools(body.get("tools"))
        if self.harness == "terminus_2_sandboxed":
            # Terminus sends terminal commands inside JSON text, without advertising function tools.
            tools = [{"name": "terminal", "parameters": {"properties": {"command": {"type": "string"}}}}]
        for tool in tools:
            function = tool.get("function", tool)
            name = function.get("name", "")
            bare = name.rsplit("__", 1)[-1]
            properties = (function.get("parameters") or {}).get("properties", {})
            if bare in SHELL_TOOLS:
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

    def _finish(self, body: dict) -> tuple[str, dict, dict] | None:
        """Return a call to an advertised finish tool carrying the answer, or None to answer in text."""
        tools, namespaces = flatten_namespace_tools(body.get("tools"))
        if self.harness == "miniswe_sandboxed":
            for tool in tools:
                function = tool.get("function", tool)
                if function.get("name") == "bash":
                    return (
                        "bash",
                        {"command": f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo {ANSWER}"},
                        namespaces,
                    )
            raise ValueError("mini-SWE did not advertise its bash submission tool")
        for tool in tools:
            function = tool.get("function", tool)
            name = function.get("name", "")
            if name.rsplit("__", 1)[-1] not in FINISH_TOOLS:
                continue
            parameters = function.get("parameters") or {}
            properties = parameters.get("properties", {})
            args = {}
            for field_name in parameters.get("required", []):
                field_type = properties.get(field_name, {}).get("type")
                if field_type == "string":
                    args[field_name] = ANSWER
                elif field_type == "array":
                    args[field_name] = []
                else:
                    break
            else:
                return name, args, namespaces
        return None

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
            if self.harness == "terminus_2_sandboxed" and isinstance(history, list):
                results = [
                    item
                    for item in history
                    if isinstance(item, dict)
                    and item.get("role") == "user"
                    and call["token"] in json.dumps(item.get("content"))
                ]
                # A later terminal pane can still contain previous output.
                # Keep the first model-visible observation of this command.
                results = [] if call["result_seen"] else results[:1]
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
        messages_api = request.url.path.endswith("/messages")
        if messages_api and is_title_request(body):
            # Titles do not consume task replies or the prescribed task failures,
            # but remain in the witness and every canonical exchange comparison.
            payload = messages_response(
                body["model"],
                f"resp_{uuid4().hex}",
                {"type": "text", "text": '{"title":"Conformance check"}'},
                usage=self.scenario.usage,
                index=len(self.auxiliary_calls),
            )
            self.auxiliary_calls.append({"request": body, "response": payload, "status_code": 200})
            self.save()
            if body.get("stream"):
                return StreamingResponse(messages_sse(payload), media_type="text/event-stream")
            return JSONResponse(payload)
        protocol_body = (
            AnthropicConverter().anthropic_request_to_responses(body).model_dump(mode="json", exclude_none=True)
            if messages_api
            else body
        )
        self._observe_tool_results(protocol_body)
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
                    name, arguments, namespaces = self._tool(protocol_body)
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
                elif (finish := self._finish(protocol_body)) is not None:
                    name, arguments, namespaces = finish
                    self.finish_call = {"id": f"call_{uuid4().hex}", "name": name, "arguments": arguments}
                    output = [
                        {
                            "type": "function_call",
                            "id": self.finish_call["id"],
                            "call_id": self.finish_call["id"],
                            "name": name,
                            "arguments": json.dumps(arguments),
                            "status": "completed",
                        }
                    ]
                    self.finished = True
                else:
                    output = [
                        {
                            "type": "message",
                            "id": f"msg_{uuid4().hex}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": ANSWER, "annotations": []}],
                        }
                    ]
                    self.finished = True
                calls_tool = output[0]["type"] == "function_call"
                if self.harness == "terminus_2_sandboxed":
                    commands = (
                        [{"keystrokes": f"sh -c {shlex.quote(arguments['command'])}\n", "duration": 0.2}]
                        if calls_tool
                        else []
                    )
                    if calls_tool:
                        call["wire_id"] = False
                        call["arguments"] = commands[0]
                    text = json.dumps(
                        {
                            "analysis": "Perform the prescribed check." if calls_tool else ANSWER,
                            "plan": "Run the command." if calls_tool else ANSWER,
                            "commands": commands,
                            "task_complete": not calls_tool,
                        }
                    )
                    output = [
                        {
                            "type": "message",
                            "id": f"msg_{uuid4().hex}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}],
                        }
                    ]
                    if not calls_tool and not self.completion_proposed:
                        # Harbor asks the model to confirm completion once before returning.
                        self.completion_proposed = True
                        self.finished = False
                    calls_tool = False
                if messages_api:
                    payload = messages_response(
                        body["model"],
                        response_id,
                        {"type": "tool_use", "id": output[0]["call_id"], "name": name, "input": arguments}
                        if calls_tool
                        else {"type": "text", "text": ANSWER},
                        usage=self.scenario.usage,
                        index=index,
                    )
                elif request.url.path.endswith("/chat/completions"):
                    message = {
                        "role": "assistant",
                        "content": None if calls_tool else ANSWER,
                        "reasoning_content": "Perform the prescribed check.",
                    }
                    if calls_tool:
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
                                "finish_reason": "tool_calls" if calls_tool else "stop",
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
                        "parallel_tool_calls": body.get("parallel_tool_calls", False),
                        "tool_choice": body.get("tool_choice", "auto"),
                        "tools": body.get("tools", []),
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
        if messages_api:
            events = messages_sse(payload)
        elif request.url.path.endswith("/chat/completions"):
            events = synthesize_chat_completion_sse(payload, include_usage=True)
        else:
            events = synthesize_responses_sse(payload)
        return StreamingResponse(events, media_type="text/event-stream")

    def model_app(self, *, assistant_message_header: bytes | None = None) -> FastAPI:
        app = FastAPI()
        install_model_call_capture(
            app,
            ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=self.directory / "capture"),
            model_server_name="policy_model",
            assistant_message_header=assistant_message_header,
        )
        app.post("/v1/chat/completions", response_model=None)(self.model)
        app.post("/v1/responses", response_model=None)(self.model)
        app.post("/v1/messages", response_model=None)(self.model)
        return app

    def judge_app(self) -> FastAPI:
        """A separate captured Model Server used only by the verifier."""
        app = FastAPI()
        install_model_call_capture(
            app,
            ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=self.directory / "capture"),
            model_server_name="judge_model",
        )

        @app.post("/v1/chat/completions")
        async def judge(request: Request) -> dict:
            body = await request.json()
            payload = {
                "id": f"judge_{uuid4().hex}",
                "object": "chat.completion",
                "created": int(time()),
                "model": "conformance-judge",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "1.0"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 13, "completion_tokens": 2, "total_tokens": 15},
            }
            self.judge_calls.append({"request": body, "status_code": 200, "response": payload})
            self.save()
            return payload

        return app

    def resources_app(
        self, *, server_client: ServerClient | None = None, sandbox_access: dict | None = None
    ) -> FastAPI:
        app = FastAPI()
        app.add_middleware(RolloutContextMiddleware)
        app.add_middleware(SessionMiddleware, secret_key=uuid4().hex, session_cookie="probe_resources_session")

        async def seed(request: Request) -> dict:
            body = await request.json()
            if "resources_session_id" in body:
                session = ResourcesSeedSessionRequest.model_validate(body)
                request.session["resources_session_id"] = session.resources_session_id
            self.seeded += 1
            self.save()
            result = {"resources_session_id": session.resources_session_id} if "resources_session_id" in body else {}
            if sandbox_access is not None:
                result["sandbox_access"] = sandbox_access
            return result

        async def close(body: ResourcesCloseSessionRequest) -> dict:
            self.closed += 1
            self.save()
            return {"resources_session_id": body.resources_session_id}

        async def bash(request: Request) -> JSONResponse:
            body = await request.json()
            process = await asyncio.create_subprocess_exec(
                "sh",
                "-c",
                body["command"],
                cwd=self.directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return JSONResponse(
                {
                    "stdout": stdout.decode(errors="replace"),
                    "stderr": stderr.decode(errors="replace"),
                    "exit_code": process.returncode,
                },
                status_code=200 if process.returncode == 0 else 400,
            )

        async def verify(request: Request) -> dict:
            body = await request.json()
            output = body.get("response", {}).get("output", [])
            answered = any(
                item.get("type") == "message"
                and item.get("role") == "assistant"
                and any(part.get("text") == ANSWER for part in item.get("content", []) if isinstance(part, dict))
                for item in output
            )
            if self.harness == "terminus_2_sandboxed":
                for item in output:
                    if item.get("type") != "message" or item.get("role") != "assistant":
                        continue
                    for part in item.get("content", []):
                        try:
                            content = json.loads(part.get("text", ""))
                        except (ValueError, TypeError, AttributeError):
                            continue
                        if isinstance(content, dict):
                            answered |= content.get("task_complete") is True and content.get("analysis") == ANSWER
            # A harness that ends on a finish tool may never emit a final assistant message.
            finish_name = (self.finish_call or {}).get("name", "").rsplit("__", 1)[-1]
            finished_by_tool = bool(finish_name) and any(
                item.get("type") == "function_call"
                and item.get("name", "").rsplit("__", 1)[-1] == finish_name
                and ANSWER in str(item.get("arguments"))
                for item in output
            )
            complete = answered or finished_by_tool
            reward = float(complete and self.scenario.expected_reward == 1.0)
            grade = {}
            if self.scenario.judge_verifier:
                if server_client is None:
                    raise RuntimeError("judge_verifier requires a ServerClient")
                judgement = await call_judge(
                    server_client,
                    server_name="judge_model",
                    url_path="/v1/chat/completions",
                    json={
                        "model": "conformance-judge",
                        "messages": [{"role": "user", "content": json.dumps(output)}],
                        "max_tokens": 16,
                    },
                    response_model=NeMoGymChatCompletion,
                )
                reward = float(judgement.choices[0].message.content)
                grade = {"judgement_id": judgement.id}
            self.verifications.append({"reward": reward, "answer_seen": complete})
            self.save()
            return {**body, **grade, "reward": reward, "evaluation_completed": True, "mask_sample": False}

        for prefix in ("", "/ng-rollout/{rollout_id}"):
            app.post(prefix + "/seed_session")(seed)
            app.post(prefix + "/close_session")(close)
            app.post(prefix + "/verify")(observe_verifier(verify, enabled=True))
            app.post(prefix + "/bash")(bash)
        return app
