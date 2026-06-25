# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import shlex
import shutil
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
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
from responses_api_agents.codex_agent.setup_codex import ensure_codex


LOG = logging.getLogger(__name__)


def parse_codex_events(stdout: str) -> tuple[list[Any], dict[str, int]]:
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")

        if etype == "turn.completed":
            usage = event.get("usage") or {}
            input_tokens += int(usage.get("input_tokens") or 0) + int(usage.get("cached_input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            continue
        if etype != "item.completed":
            continue

        item = event.get("item") or {}
        itype = item.get("type")
        if itype == "agent_message" and (item.get("text") or "").strip():
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=item["text"], annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        elif itype == "command_execution":
            call_id = item.get("id") or f"call-{uuid4().hex[:8]}"
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=json.dumps({"command": item.get("command", "")}),
                    call_id=call_id,
                    name="shell",
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=str(item.get("aggregated_output") or ""),
                    status="completed",
                )
            )

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


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
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
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
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            user_message = content or ""
            break

    return user_message, system_message

# fields codex sends that the gym model server rejects
_CODEX_REQUEST_KEYS_TO_DROP = (
    "stream",
    "store",
    "include",
    "reasoning",
    "prompt_cache_key",
    "client_metadata",
    "previous_response_id",
    "instructions",  # folded into `input` as a system message instead below
    "metadata",
    "text",
)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for p in value:
            if isinstance(p, dict):
                text = p.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(value)


def _sanitize_codex_input_items(input_items: list) -> list[dict]:
    out: list[dict] = []
    for item in input_items:
        if not isinstance(item, dict):
            # if already a validated model object pass through
            out.append(item)
            continue
        itype = item.get("type")

        if itype == "reasoning":
            summary = item.get("summary")
            if not isinstance(summary, list):
                summary = [] if summary is None else [summary]
            reasoning = {"type": "reasoning", "id": item.get("id") or f"rs_{uuid4().hex[:24]}", "summary": summary}
            if item.get("encrypted_content") is not None:
                reasoning["encrypted_content"] = item["encrypted_content"]
            out.append(reasoning)

        elif itype == "function_call":
            arguments = item.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            fc = {
                "type": "function_call",
                "call_id": str(item.get("call_id") or item.get("id") or ""),
                "name": str(item.get("name") or ""),
                "arguments": arguments,
            }
            if item.get("id") is not None:
                fc["id"] = str(item["id"])
            if item.get("status") is not None:
                fc["status"] = item["status"]
            out.append(fc)

        elif itype == "function_call_output":
            fco = {
                "type": "function_call_output",
                "call_id": str(item.get("call_id") or ""),
                "output": _coerce_text(item.get("output")),
            }
            if item.get("id") is not None:
                fco["id"] = str(item["id"])
            if item.get("status") is not None:
                fco["status"] = item["status"]
            out.append(fco)

        elif itype == "message" or "role" in item:
            # normalize any message (user/assistant/system/developer) to EasyInputMessage(str content)
            out.append(
                {"type": "message", "role": item.get("role") or "user", "content": _coerce_text(item.get("content"))}
            )

        else:
            # unknown item type the union won't accept; drop it rather than 422 the whole turn.
            LOG.warning("dropping unsupported codex input item type %r", itype)

    return out


def _sanitize_codex_responses_body(body: dict) -> dict:
    out = {k: v for k, v in body.items() if k not in _CODEX_REQUEST_KEYS_TO_DROP}

    input_items = _sanitize_codex_input_items(list(out.get("input") or []))
    instructions = body.get("instructions")
    if instructions:
        input_items = [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": instructions}]}
        ] + input_items
    out["input"] = input_items

    tools = out.get("tools")
    if tools is not None:
        function_tools = [t for t in tools if isinstance(t, dict) and t.get("type") == "function"]
        if function_tools:
            out["tools"] = function_tools
        else:
            out.pop("tools", None)
            out.pop("tool_choice", None)
            out.pop("parallel_tool_calls", None)

    return out


def _response_to_codex_sse(response: dict) -> str:
    response = dict(response)
    response.setdefault("object", "response")
    response["status"] = "completed"
    output_items = list(response.get("output") or [])

    in_progress = {k: response[k] for k in ("id", "object", "model") if k in response}
    in_progress.update({"status": "in_progress", "output": []})

    events: list[tuple[str, dict]] = [("response.created", {"response": in_progress})]
    for idx, item in enumerate(output_items):
        events.append(("response.output_item.done", {"output_index": idx, "item": item}))
    events.append(("response.completed", {"response": response}))

    chunks = []
    for event_type, payload in events:
        chunks.append(f"data: {json.dumps({'type': event_type, **payload})}\n\n")
    return "".join(chunks)


class CodexAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "codex"
    model: str = "nvidia/qwen/qwen3-next-80b-a3b-instruct"
    model_provider: str = "nvinf"
    # used only when model_server is unset (direct, non-buffered eval against an external endpoint
    # whose /v1/responses streams and accepts function tools).
    base_url: Optional[str] = None
    api_key_env: str = "NVIDIA_API_KEY"
    wire_api: str = "responses"  # "chat" removed in modern codex
    # nemo-rl asserts request temperature/top_p matches generation config; shim pins them
    temperature: float = 1.0
    top_p: float = 1.0
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/codex_agent/workspaces"
    system_prompt: Optional[str] = None
    timeout: int = 900
    extra_args: list[str] = []
    codex_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class CodexAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class CodexAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class CodexAgent(SimpleResponsesAPIAgent):
    config: CodexAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/runs/{run_token}/v1/responses")(self._responses_sse_shim)
        return app

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_codex(self.config.codex_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("codex command %r is not on PATH yet", self.config.command)

    def _own_base_url(self) -> str:
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.name)
        return self.server_client._build_server_base_url(cfg)

    def _resolve_base_url(self, request: Request) -> Optional[str]:
        if self.config.model_server is not None:
            run_token = self.run_token_from_request(request)
            if run_token:
                return f"{self._own_base_url().rstrip('/')}/runs/{run_token}/v1"
        return self.config.base_url

    def _resolve_model(self) -> str:
        if self.config.model_server is not None:
            return str(self.config.model_server.name)
        return self.config.model

    def _write_config_toml(self, codex_home: Path, request: Request) -> None:
        provider = self.config.model_provider
        base_url = self._resolve_base_url(request)
        lines = [
            f'model = "{self._resolve_model()}"',
            f'model_provider = "{provider}"',
            "",
            # the multi_agent namespace tool is rejected by non-OpenAI /v1/responses endpoints
            "[features]",
            "multi_agent = false",
            "",
            f"[model_providers.{provider}]",
            f'name = "{provider}"',
        ]
        if base_url:
            lines.append(f'base_url = "{base_url}"')
        lines.append(f'env_key = "{self.config.api_key_env}"')
        lines.append(f'wire_api = "{self.config.wire_api}"')
        (codex_home / "config.toml").write_text("\n".join(lines) + "\n")

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"codex_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    async def _responses_sse_shim(self, run_token: str, request: Request) -> StreamingResponse:
        body = await request.json()
        sanitized = _sanitize_codex_responses_body(body)
        sanitized["temperature"] = self.config.temperature
        sanitized["top_p"] = self.config.top_p

        model_resp = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=f"/runs/{run_token}/v1/responses",
            json=sanitized,
            cookies=request.cookies,
        )
        await raise_for_status(model_resp)
        response_json = await get_response_json(model_resp)

        sse = _response_to_codex_sse(response_json)

        async def event_stream():
            yield sse

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async def _run_codex(
        self, request: Request, instruction: str, system_prompt: Optional[str]
    ) -> tuple[list[Any], dict[str, int], str]:
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        codex_home = work_dir / ".codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        self._write_config_toml(codex_home, request)
        env = {**os.environ, "CODEX_HOME": str(codex_home)}
        env.update({k: v for k, v in self.config.env.items() if v})

        cmd = [
            *self.config.command_parts,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C",
            str(work_dir),
            *self.config.extra_args,
            prompt,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                LOG.warning("codex timed out after %ds", self.config.timeout)
                return [], {"input_tokens": 0, "output_tokens": 0}, self.config.model

            if proc.returncode not in (0, None):
                LOG.warning("codex exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])
            output_items, usage = parse_codex_events(stdout.decode(errors="replace"))
            return output_items, usage, self.config.model
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        output_items, usage, model_name = await self._run_codex(request, user_message, system_prompt)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("codex produced no assistant message; padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

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


if __name__ == "__main__":
    CodexAgent.run_webserver()
