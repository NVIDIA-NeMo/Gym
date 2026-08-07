# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
from asyncio import Semaphore
from contextlib import suppress
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.kiro_agent.setup_kiro import ensure_kiro_cli


LOG = logging.getLogger(__name__)


class KiroACPError(RuntimeError):
    pass


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    items = list(body_input)
    system_message: Optional[str] = None

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")) for part in content
                )
            system_message = content or ""
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")) for part in content
                )
            user_message = content or ""
            break

    return user_message, system_message


def _tool_name(state: dict[str, Any]) -> str:
    metadata = state.get("_meta") if isinstance(state.get("_meta"), dict) else {}
    candidate = metadata.get("toolName") or metadata.get("name") or state.get("kind") or state.get("title")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(candidate or "kiro_tool")).strip("_").lower()
    return name[:64] or "kiro_tool"


def _tool_arguments(state: dict[str, Any]) -> str:
    raw_input = state.get("rawInput")
    if isinstance(raw_input, (dict, list)):
        return json.dumps(raw_input, separators=(",", ":"))
    if raw_input is None:
        return "{}"
    return str(raw_input)


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := _content_text(item)))
    if not isinstance(value, dict):
        return str(value)

    value_type = value.get("type")
    if value_type == "content":
        return _content_text(value.get("content"))
    if value_type == "text":
        return str(value.get("text") or "")
    if value_type == "diff":
        return json.dumps(
            {key: value.get(key) for key in ("path", "oldText", "newText") if key in value},
            separators=(",", ":"),
        )
    return json.dumps(value, separators=(",", ":"), default=str)


def _tool_output(state: dict[str, Any]) -> str:
    raw_output = state.get("rawOutput")
    if isinstance(raw_output, (dict, list)):
        return json.dumps(raw_output, separators=(",", ":"), default=str)
    if raw_output is not None:
        return str(raw_output)
    return _content_text(state.get("content"))


def _usage_from_events(events: list[dict[str, Any]]) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    context_used = 0

    for event in events:
        update = (event.get("params") or {}).get("update") or {}
        if update.get("sessionUpdate") == "usage_update":
            context_used = int(update.get("used") or context_used)

        candidates = [update.get("usage"), (event.get("result") or {}).get("usage")]
        for usage in candidates:
            if not isinstance(usage, dict):
                continue
            input_tokens = int(usage.get("inputTokens") or usage.get("input_tokens") or input_tokens)
            output_tokens = int(usage.get("outputTokens") or usage.get("output_tokens") or output_tokens)

    if not input_tokens and not output_tokens:
        input_tokens = context_used
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def parse_kiro_events(events: list[dict[str, Any]]) -> tuple[list[Any], dict[str, int]]:
    timeline: list[tuple[str, str]] = []
    messages: dict[str, str] = {}
    tools: dict[str, dict[str, Any]] = {}
    completed_tools: set[str] = set()
    message_index = 0

    for event in events:
        if event.get("method") != "session/update":
            continue
        update = (event.get("params") or {}).get("update")
        if not isinstance(update, dict):
            continue
        update_type = update.get("sessionUpdate")

        if update_type == "agent_message_chunk":
            text = _content_text(update.get("content"))
            if not text:
                continue
            message_id = str(update.get("messageId") or "")
            if timeline and timeline[-1][0] == "message" and messages.get(timeline[-1][1]) is not None:
                current_key = timeline[-1][1]
                current_id = current_key.split(":", 1)[0]
                if message_id and current_id != message_id:
                    current_key = ""
            else:
                current_key = ""
            if not current_key:
                message_index += 1
                current_key = f"{message_id or 'message'}:{message_index}"
                messages[current_key] = ""
                timeline.append(("message", current_key))
            messages[current_key] += text
            continue

        if update_type not in {"tool_call", "tool_call_update"}:
            continue
        call_id = str(update.get("toolCallId") or "")
        if not call_id:
            continue
        if call_id not in tools:
            tools[call_id] = {"toolCallId": call_id}
            timeline.append(("tool", call_id))
        tools[call_id].update({key: value for key, value in update.items() if value is not None})
        if update.get("status") in {"completed", "failed"} and call_id not in completed_tools:
            timeline.append(("tool_output", call_id))
            completed_tools.add(call_id)

    output_items: list[Any] = []
    for item_type, item_id in timeline:
        if item_type == "message":
            text = messages[item_id]
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        elif item_type == "tool":
            state = tools[item_id]
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=_tool_arguments(state),
                    call_id=item_id,
                    name=_tool_name(state),
                    type="function_call",
                    id=item_id,
                    status="completed",
                )
            )
        else:
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=item_id,
                    output=_tool_output(tools[item_id]),
                    status="completed",
                )
            )

    return output_items, _usage_from_events(events)


class KiroAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    concurrency: int = 8
    command: str = "kiro-cli"
    api_key: str = ""
    model: Optional[str] = None
    effort: Optional[str] = None
    agent_engine: str = "v2"
    trust_all_tools: bool = True
    trust_tools: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/kiro_agent/workspaces"
    system_prompt: Optional[str] = None
    timeout: int = 900
    extra_args: list[str] = Field(default_factory=list)

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class KiroAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class KiroAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class KiroAgent(SimpleResponsesAPIAgent):
    config: KiroAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if command == "kiro-cli" and shutil.which(command) is None:
            ensure_kiro_cli()
        if not command or shutil.which(command) is None:
            LOG.warning("Kiro CLI command %r is not on PATH", self.config.command)

    def _workspace(self, system_prompt: Optional[str]) -> tuple[Path, Path, Optional[str]]:
        work_dir = Path(self.config.workspace_root).expanduser() / f"kiro_{uuid4().hex[:8]}"
        if not work_dir.is_absolute():
            work_dir = Path.cwd() / work_dir
        kiro_home = work_dir / ".kiro-home"
        kiro_home.mkdir(parents=True)

        agent_name = None
        if system_prompt:
            agent_name = "nemo-gym"
            agents_dir = work_dir / ".kiro" / "agents"
            agents_dir.mkdir(parents=True)
            agent_config = {
                "name": agent_name,
                "description": "NeMo Gym rollout agent",
                "prompt": system_prompt,
                "tools": ["*"],
                "allowedTools": ["*"],
                "includeMcpJson": False,
            }
            (agents_dir / f"{agent_name}.json").write_text(json.dumps(agent_config, indent=2))

        return work_dir, kiro_home, agent_name

    def _env(self, kiro_home: Path) -> dict[str, str]:
        env = {
            **os.environ,
            "KIRO_HOME": str(kiro_home),
            "KIRO_LOG_NO_COLOR": "1",
            "NO_COLOR": "1",
        }
        if self.config.api_key:
            env["KIRO_API_KEY"] = self.config.api_key
        env.update({key: value for key, value in self.config.env.items() if value})
        return env

    def _command(self, agent_name: Optional[str]) -> list[str]:
        cmd = [*self.config.command_parts, "acp", "--agent-engine", self.config.agent_engine]
        if agent_name:
            cmd += ["--agent", agent_name]
        if self.config.model:
            cmd += ["--model", self.config.model]
        if self.config.effort:
            cmd += ["--effort", self.config.effort]
        if self.config.trust_all_tools:
            cmd.append("--trust-all-tools")
        elif self.config.trust_tools:
            cmd += ["--trust-tools", self.config.trust_tools]
        cmd += self.config.extra_args
        return cmd

    @staticmethod
    async def _send(proc: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise KiroACPError("Kiro ACP stdin is unavailable")
        proc.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
        await proc.stdin.drain()

    async def _handle_agent_request(self, proc: asyncio.subprocess.Process, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        if request_id is None:
            return
        if message.get("method") == "session/request_permission":
            options = (message.get("params") or {}).get("options") or []
            allowed = next(
                (
                    option
                    for option in options
                    if isinstance(option, dict) and str(option.get("kind") or "").startswith("allow_")
                ),
                None,
            )
            outcome = (
                {"outcome": "selected", "optionId": allowed.get("optionId")} if allowed else {"outcome": "cancelled"}
            )
            await self._send(proc, {"jsonrpc": "2.0", "id": request_id, "result": {"outcome": outcome}})
            return

        await self._send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not supported by NeMo Gym"},
            },
        )

    async def _request(
        self,
        proc: asyncio.subprocess.Process,
        request_id: int,
        method: str,
        params: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> Any:
        await self._send(
            proc,
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        )
        if proc.stdout is None:
            raise KiroACPError("Kiro ACP stdout is unavailable")

        while True:
            line = await proc.stdout.readline()
            if not line:
                raise KiroACPError(f"Kiro ACP closed before responding to {method}")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                LOG.debug("Ignoring non-JSON Kiro ACP output: %s", line.decode(errors="replace")[:200])
                continue
            if not isinstance(message, dict):
                continue
            events.append(message)
            if message.get("method"):
                await self._handle_agent_request(proc, message)
                continue
            if message.get("id") != request_id:
                continue
            if message.get("error"):
                raise KiroACPError(f"Kiro ACP {method} failed: {message['error']}")
            return message.get("result")

    async def _acp_turn(
        self,
        instruction: str,
        system_prompt: Optional[str],
    ) -> tuple[list[Any], dict[str, int], str, str]:
        work_dir, kiro_home, agent_name = self._workspace(system_prompt)
        proc: Optional[asyncio.subprocess.Process] = None
        stderr_task: Optional[asyncio.Task[bytes]] = None
        events: list[dict[str, Any]] = []
        error: Optional[BaseException] = None
        result: Optional[tuple[list[Any], dict[str, int], str, str]] = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command(agent_name),
                cwd=str(work_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(kiro_home),
            )
            if proc.stderr is not None:
                stderr_task = asyncio.create_task(proc.stderr.read())

            initialize = await self._request(
                proc,
                0,
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {},
                    "clientInfo": {"name": "nemo-gym", "version": "0.1.0"},
                },
                events,
            )
            if not isinstance(initialize, dict) or initialize.get("protocolVersion") != 1:
                raise KiroACPError(f"Kiro returned an unsupported ACP version: {initialize!r}")

            new_session = await self._request(
                proc,
                1,
                "session/new",
                {"cwd": str(work_dir), "mcpServers": []},
                events,
            )
            session_id = new_session.get("sessionId") if isinstance(new_session, dict) else None
            if not session_id:
                raise KiroACPError(f"Kiro did not return an ACP session ID: {new_session!r}")

            prompt_result = await self._request(
                proc,
                2,
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "text", "text": instruction}]},
                events,
            )
            stop_reason = prompt_result.get("stopReason", "") if isinstance(prompt_result, dict) else ""
            output_items, usage = parse_kiro_events(events)
            if not any(getattr(item, "type", None) == "message" for item in output_items):
                raise KiroACPError("Kiro completed without an assistant message")
            result = output_items, usage, self.config.model or "kiro", stop_reason
        except BaseException as exc:
            error = exc
        finally:
            if proc is not None:
                if proc.stdin is not None:
                    proc.stdin.close()
                if proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        with suppress(ProcessLookupError):
                            proc.kill()
                        await proc.wait()
            stderr = await stderr_task if stderr_task is not None else b""
            shutil.rmtree(work_dir, ignore_errors=True)

        if error is not None:
            stderr_text = stderr.decode(errors="replace").strip()
            detail = f": {stderr_text[:500]}" if stderr_text else ""
            if isinstance(error, asyncio.CancelledError):
                raise error
            raise KiroACPError(f"{error}{detail}") from error
        if result is None:
            raise KiroACPError("Kiro ACP turn produced no result")
        return result

    async def _run_kiro(
        self,
        instruction: str,
        system_prompt: Optional[str],
    ) -> tuple[list[Any], dict[str, int], str, str]:
        try:
            return await asyncio.wait_for(self._acp_turn(instruction, system_prompt), timeout=self.config.timeout)
        except asyncio.TimeoutError as exc:
            raise KiroACPError(f"Kiro timed out after {self.config.timeout}s") from exc

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [part for part in (self.config.system_prompt, input_system) if part]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        output_items, usage, model_name, _ = await self._run_kiro(user_message, system_prompt)

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def run(self, request: Request, body: KiroAgentRunRequest) -> KiroAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies
            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"
            return KiroAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    KiroAgent.run_webserver()
