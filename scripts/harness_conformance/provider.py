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
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymChatCompletion
from nemo_gym.models import (
    ModelCallCaptureConfig,
    flatten_namespace_tools,
    install_model_call_capture,
    message_content_to_text,
    restore_namespace_tool_calls,
    synthesize_chat_completion_sse,
    synthesize_responses_sse,
)
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    RolloutContextMiddleware,
    observe_verifier,
)
from nemo_gym.verification import call_judge

from .scenarios import Scenario


SHELL_TOOLS = ("bash", "terminal", "shell", "shell_command", "exec_command", "code_exec", "exec", "python_cell")
# Harnesses that end an episode on a tool call rather than on a text answer.
FINISH_TOOLS = ("finish", "submit")
ANSWER = "CONFORMANCE_DONE"

# The reply vllm_model gives a context-length refusal when propagate_context_overflow_errors is on: the engine's
# HTTP 400 in OpenAI's error envelope with the code that tells a harness to compact.
CONTEXT_OVERFLOW_ERROR = {
    "error": {
        "message": "This model's maximum context length is 32768 tokens. However, you requested 32818 tokens in the "
        "messages, Please reduce the length of the messages.",
        "type": "invalid_request_error",
        "param": None,
        "code": "context_length_exceeded",
    }
}
# vllm_model's default for the same refusal: an empty completion with zero usage and a length stop.
LENGTH_STOP_USAGE = {
    "chat": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
    "responses": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def harness_kill_command(token: str, marker: Path) -> str:
    """A shell command that records ``token`` and then SIGKILLs the supervised harness it runs under.

    Gym's process supervisor starts a harness with ``NEMO_GYM_SUPERVISOR=<identity>`` and gives itself the value
    ``detached``. The command walks up from its own shell to the outermost ancestor that carries the identity (the
    harness) and kills those ancestors, outermost first. It then kills every other process that carries the identity,
    such as a tool the harness started in another session, and last itself. The supervisor never carries the identity,
    so it survives and reports the harness's exit. Without a supervised ancestor the command kills nothing and exits
    97. Linux only (``/proc``, GNU ``grep -z``), like the probes that use it.
    """
    record = f"printf %s {shlex.quote(token)} > {shlex.quote(str(marker))}"
    return (
        f"{record}; p=$$; id=; tree=; "
        "while [ -r /proc/$p/environ ]; do "
        "v=$(tr '\\0' '\\n' 2>/dev/null < /proc/$p/environ | sed -n 's/^NEMO_GYM_SUPERVISOR=//p'); "
        'if [ -n "$v" ] && [ "$v" != detached ] && { [ -z "$id" ] || [ "$v" = "$id" ]; }; then '
        'id=$v; tree="$p $tree"; '
        'elif [ -n "$id" ]; then break; fi; '
        "p=$(sed 's/.*) //' /proc/$p/stat | cut -d ' ' -f 2); "
        '[ "$p" -gt 1 ] 2>/dev/null || break; '
        "done; "
        '[ -n "$id" ] || exit 97; '
        'for q in $tree; do [ "$q" = $$ ] || kill -KILL "$q" 2>/dev/null; done; '
        'for e in $(grep -lzxF "NEMO_GYM_SUPERVISOR=$id" /proc/[0-9]*/environ 2>/dev/null); do '
        'q=${e#/proc/}; q=${q%/environ}; [ "$q" = $$ ] || kill -KILL "$q" 2>/dev/null; '
        "done; "
        "kill -KILL $$"
    )


# Identify the actual summary instruction, never the availability of tools or a
# token threshold. These prefixes are exercised against the pinned CLI requests.
SUMMARY_PROMPTS = (
    "Create a new anchored summary from the conversation history.",
    "Update the existing anchored summary using the new conversation history.",
    "You are performing a CONTEXT CHECKPOINT COMPACTION.",
    "The context window is approaching its limit. Please create a concise summary",
    "Create a structured handoff summary for a later assistant",
)


def is_compaction_request(body: dict) -> bool:
    messages = body.get("messages", body.get("input", []))
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return False
    last = messages[-1]
    return last.get("role") == "user" and message_content_to_text(last.get("content")).startswith(SUMMARY_PROMPTS)


@dataclass
class Probe:
    scenario: Scenario
    directory: Path
    attempts: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    agent_responses: list[dict] = field(default_factory=list)
    agent_replies: list[dict] = field(default_factory=list)
    judge_calls: list[dict] = field(default_factory=list)
    seeded: int = 0
    closed: int = 0
    finished: bool = False
    finish_call: dict | None = None
    compaction_summary: str = field(
        default_factory=lambda: f"COMPACTION_SUMMARY_{uuid4().hex}: continue the shell checks."
    )
    compaction_response_ids: list[str] = field(default_factory=list)

    def save(self) -> None:
        """Persist independent observations even if collection fails."""
        # A call that ends the harness has no later request to observe its execution.
        for call in self.tool_calls:
            if call.get("ends_harness"):
                self._observe_execution(call)
        witness = {
            "attempts": self.attempts,
            "tool_calls": self.tool_calls,
            "violations": self.violations,
            "verifications": self.verifications,
            "agent_responses": self.agent_responses,
            "agent_replies": self.agent_replies,
            "judge_calls": self.judge_calls,
            "seeded": self.seeded,
            "closed": self.closed,
            "finished": self.finished,
            "finish_call": self.finish_call,
        }
        if self.scenario.compaction is not None:
            witness["compaction"] = {
                "summary": self.compaction_summary,
                "response_ids": self.compaction_response_ids,
            }
        (self.directory / "witness.json").write_text(json.dumps(witness, indent=2) + "\n")

    async def seed_sandbox(self) -> dict:
        """Optional Resources-owned sandbox for harnesses that borrow their task environment."""
        return {}

    async def close_sandbox(self) -> None:
        """Release a task sandbox, when the probe owns one."""

    def _tool(self, body: dict) -> tuple[str, dict, dict]:
        tools, namespaces = flatten_namespace_tools(body.get("tools"))
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
                ends_harness = self.scenario.kill_harness and index == self.scenario.tool_steps - 1
                command = (
                    harness_kill_command(token, marker)
                    if ends_harness
                    else f"printf %s {shlex.quote(token)} > {shlex.quote(str(marker))}; printf %s {shlex.quote(token)}; exit {code}"
                )
                if bare == "python_cell" and "code" in properties:
                    args = {
                        "code": (
                            f"result = await self.shell.run({('sh -c ' + shlex.quote(command))!r})\n"
                            "print(result.stdout, end='')\n"
                            "if result.returncode:\n"
                            "    raise RuntimeError(f'shell exited with code {result.returncode}')"
                        )
                    }
                elif "cmd" in properties:
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
                        **({"ends_harness": True} if ends_harness else {}),
                    }
                )
                return name, args, namespaces
        raise ValueError("harness did not advertise a supported shell tool")

    def _finish(self, body: dict) -> tuple[str, dict, dict] | None:
        """Return a call to an advertised finish tool carrying the answer, or None to answer in text."""
        tools, namespaces = flatten_namespace_tools(body.get("tools"))
        for tool in tools:
            function = tool.get("function", tool)
            name = function.get("name", "")
            if name.rsplit("__", 1)[-1] == "python_cell":
                return (
                    name,
                    {
                        "code": (
                            "from nooa_bench.bench_agent import TaskResult\n"
                            f"return_result(TaskResult(solution_description={ANSWER!r}, "
                            f"evidence={ANSWER!r}, how_to_verify={ANSWER!r}))"
                        )
                    },
                    namespaces,
                )
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

    def _accepted_result_ids(self, call: dict, history: list) -> set[str]:
        """Require the original ID unless a provider-specific replay rule applies."""
        return {call["id"]}

    def _observe_tool_results(self, body: dict) -> None:
        history = body.get("messages", body.get("input", []))
        if not isinstance(history, list):
            history = []
        for call in self.tool_calls:
            result_ids = self._accepted_result_ids(call, history)
            results = [
                item
                for item in history
                if isinstance(item, dict)
                and (
                    item.get("role") == "tool"
                    and item.get("tool_call_id") in result_ids
                    or item.get("type") == "function_call_output"
                    and item.get("call_id") in result_ids
                )
            ]
            call["result_seen"] = call["result_seen"] or any(
                call["token"] in json.dumps(item.get("content", item.get("output"))) for item in results
            )
            for item in results:
                output = item.get("content", item.get("output"))
                if output is not None and output not in call["outputs"]:
                    call["outputs"].append(output)
            self._observe_execution(call)

    @staticmethod
    def _observe_execution(call: dict) -> None:
        marker = Path(call["marker"])
        call["executed"] = marker.is_file() and marker.read_text() == call["token"]

    async def _hold(self, request: Request, body: dict, attempt: dict) -> None:
        """Never answer: hold the request until the harness disconnects, as a model that never replies.

        The harness's own deadline ends it. The exchange then ends as a cancelled call, which capture records without
        a response, and the witness keeps the attempt without a status.
        """
        self.attempts.append({**attempt, "request": body, "response": None, "status_code": None})
        self.save()
        while not await request.is_disconnected():
            await asyncio.sleep(0.1)
        raise asyncio.CancelledError("the harness disconnected from a request the provider never answers")

    async def model(self, request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        self._observe_tool_results(body)
        index = len(self.attempts)
        attempt = {"attempt_id": f"attempt-{index}", "received_at": time()}
        headers = {"Retry-After": "0"}
        status = 200
        if index < len(self.scenario.http_errors):
            status = self.scenario.http_errors[index]
        elif self.scenario.terminal_error:
            status = self.scenario.http_errors[-1]
        response_id = f"resp_{uuid4().hex}"
        namespaces = {}
        compacting = self.scenario.compaction is not None and is_compaction_request(body)
        if status != 200:
            payload = {
                "error": {"message": "controlled model failure", "type": "conformance_error", "code": str(status)}
            }
        else:
            try:
                if self.finished:
                    raise ValueError("unexpected model request after the scripted final answer")
                scripted = len(self.tool_calls) >= self.scenario.tool_steps
                if self.scenario.kill_harness and scripted:
                    raise ValueError("unexpected model request after the scripted tool call ended the harness")
                # A summary can cover only older history. Policy requests must still
                # show every executed result before receiving the next scripted action.
                if not compacting and any(
                    not call["result_seen"] or not call.get("executed") for call in self.tool_calls
                ):
                    raise ValueError("tool execution or model-visible result was not witnessed")
                length_stop = scripted and not compacting and self.scenario.ending == "length_stop"
                output = None
                if scripted and not compacting and self.scenario.ending == "no_reply":
                    await self._hold(request, body, attempt)
                elif scripted and not compacting and self.scenario.ending == "context_overflow_error":
                    status, payload, headers = 400, CONTEXT_OVERFLOW_ERROR, {}
                elif length_stop:
                    output = [
                        {
                            "type": "message",
                            "id": f"msg_{uuid4().hex}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "", "annotations": []}],
                        }
                    ]
                elif compacting:
                    if self.scenario.compaction is not True or not self.tool_calls or self.compaction_response_ids:
                        raise ValueError("unexpected compaction request outside the single summary stage")
                    output = [
                        {
                            "type": "message",
                            "id": f"msg_{uuid4().hex}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": self.compaction_summary, "annotations": []}],
                        }
                    ]
                elif len(self.tool_calls) < self.scenario.tool_steps:
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
                elif (finish := self._finish(body)) is not None:
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
                if output is not None:
                    payload = self._reply(
                        request,
                        body,
                        output,
                        namespaces,
                        response_id,
                        index,
                        compacting=compacting,
                        length=length_stop,
                    )
                if compacting:
                    self.compaction_response_ids.append(response_id)
            except ValueError as exc:
                status = 409
                self.violations.append(str(exc))
                payload = {"error": {"message": str(exc), "type": "probe_protocol_error"}}
        self.attempts.append({**attempt, "request": body, "response": payload, "status_code": status})
        self.save()
        if status != 200 or not body.get("stream"):
            return JSONResponse(payload, status_code=status, headers=headers)
        events = (
            synthesize_chat_completion_sse(payload, include_usage=True)
            if request.url.path.endswith("/chat/completions")
            else synthesize_responses_sse(payload)
        )
        return StreamingResponse(events, media_type="text/event-stream")

    def _reply(
        self,
        request: Request,
        body: dict,
        output: list[dict],
        namespaces: dict,
        response_id: str,
        index: int,
        *,
        compacting: bool,
        length: bool,
    ) -> dict:
        """The Chat or Responses body for one scripted model reply.

        A ``length`` reply is vllm_model's default answer to a context overflow: an empty assistant message with zero
        usage, finished by ``length`` (Chat) or incomplete for ``max_output_tokens`` (Responses).
        """
        calls_tool = output[0]["type"] == "function_call"
        prompt_tokens = 20 + index
        completion_tokens = 11 if compacting else 8
        if self.scenario.compaction is not None:
            policy_index = index - len(self.compaction_response_ids)
            prompt_tokens = (
                500 if compacting else 27000 + policy_index if not self.compaction_response_ids else 100 + policy_index
            )
        if request.url.path.endswith("/chat/completions"):
            message = {
                "role": "assistant",
                "content": None if calls_tool or length else output[0]["content"][0]["text"],
            }
            if not compacting and not length:
                message["reasoning_content"] = "Perform the prescribed check."
            if calls_tool:
                message["tool_calls"] = [
                    {
                        "id": output[0]["call_id"],
                        "type": "function",
                        "function": {"name": output[0]["name"], "arguments": output[0]["arguments"]},
                    }
                ]
            payload = {
                "id": response_id,
                "object": "chat.completion",
                "created": 1,
                "model": body.get("model") or "conformance-model",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "length" if length else "tool_calls" if calls_tool else "stop",
                    }
                ],
            }
            if length:
                payload["usage"] = LENGTH_STOP_USAGE["chat"]
            elif self.scenario.usage:
                payload["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 0 if compacting else 2},
                }
            return payload
        payload = {
            "id": response_id,
            "object": "response",
            "created_at": 1,
            "status": "incomplete" if length else "completed",
            "error": None,
            "incomplete_details": {"reason": "max_output_tokens"} if length else None,
            "model": body.get("model") or "conformance-model",
            "output": restore_namespace_tool_calls(output, namespaces),
            "parallel_tool_calls": body.get("parallel_tool_calls", False),
            "tool_choice": body.get("tool_choice", "auto"),
            "tools": body.get("tools", []),
        }
        if length:
            payload["metadata"] = {"nemo_gym_context_window_exceeded": "true"}
            payload["usage"] = LENGTH_STOP_USAGE["responses"]
        elif self.scenario.usage:
            payload["usage"] = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 0 if compacting else 2},
            }
        return payload

    def observe_agent(self, app: ASGIApp) -> ASGIApp:
        """Witness the HTTP status of each agent reply to the Environment Server's ``/v1/responses`` call."""

        async def observed(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http" or not scope["path"].endswith("/v1/responses"):
                await app(scope, receive, send)
                return

            async def watch(message: Message) -> None:
                if message["type"] == "http.response.start":
                    self.agent_replies.append({"http_status": message["status"], "replied_at": time()})
                    self.save()
                await send(message)

            await app(scope, receive, watch)

        return observed

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

    def resources_app(self, *, server_client: ServerClient | None = None) -> FastAPI:
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
            if "resources_session_id" in body:
                result.update(await self.seed_sandbox())
            return result

        async def close(body: ResourcesCloseSessionRequest) -> dict:
            await self.close_sandbox()
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


class OpenClawProbe(Probe):
    """Match OpenClaw replay IDs while retaining the original captured identities."""

    def _accepted_result_ids(self, call: dict, history: list) -> set[str]:
        result_ids = super()._accepted_result_ids(call, history)
        # The pinned custom-provider replay strips the underscore from our short
        # UUID IDs. Require the matching assistant call as well as the tool result.
        replay_id = call["id"].replace("_", "")
        for item in history:
            if not isinstance(item, dict) or item.get("role") != "assistant":
                continue
            for replay in item.get("tool_calls") or []:
                function = replay.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments", ""))
                except (TypeError, ValueError):
                    continue
                if (
                    replay.get("id") == replay_id
                    and function.get("name") == call["name"]
                    and arguments == call["arguments"]
                ):
                    result_ids.add(replay_id)
        return result_ids
