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
import shutil
import subprocess
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict

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
from responses_api_agents.claude_code_agent.setup_claude_code import ensure_claude_code


LOG = logging.getLogger(__name__)


def _extract_text(content: list[Any]) -> str:
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _extract_thinking(content: list[Any]) -> str:
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("thinking", "reasoning"):
            parts.append(b.get("thinking") or b.get("text") or "")
    return "\n".join(p for p in parts if p)


def parse_stream_json(stdout: str) -> tuple[list[Any], dict]:
    """Convert claude -p --output-format=stream-json stdout into (output_items, usage)."""
    raw_events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    output_items: list[Any] = []
    pending_calls: dict[str, dict] = {}
    buffered_think: str | None = None
    total_input = 0
    total_output = 0

    for event in raw_events:
        etype = event.get("type")

        if etype == "result":
            usage = event.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)

        elif etype == "assistant":
            message = event.get("message", {})
            content = message.get("content") or []
            usage = message.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)

            if not isinstance(content, list):
                content = []

            think = _extract_thinking(content)
            if think:
                buffered_think = (buffered_think + "\n" + think) if buffered_think else think

            text = _extract_text(content)
            if text:
                if buffered_think:
                    text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                    buffered_think = None
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                input_data = block.get("input") or {}
                arguments = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
                pending_calls[call_id] = {"name": block.get("name", ""), "call_id": call_id, "arguments": arguments}

        elif etype == "user":
            message = event.get("message", {})
            content = message.get("content") or []
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = block.get("tool_use_id", "")
                call_info = pending_calls.pop(tool_id, None)
                if call_info:
                    output_items.append(
                        NeMoGymResponseFunctionToolCall(
                            arguments=call_info["arguments"],
                            call_id=tool_id,
                            name=call_info["name"],
                            type="function_call",
                            id=tool_id,
                            status="completed",
                        )
                    )
                result_content = block.get("content") or ""
                if isinstance(result_content, list):
                    result_text = _extract_text(result_content)
                else:
                    result_text = str(result_content)
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_id,
                        output=result_text,
                        status="completed",
                    )
                )

    return output_items, {"input_tokens": total_input, "output_tokens": total_output}


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    """Return (user_message, system_message) from a responses body input list."""
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


class ClaudeCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    # When model_server is set, ANTHROPIC_BASE_URL is resolved from the Gym model
    # server's URL (requires the server to expose POST /v1/messages. None is pushed yet).
    # When None, anthropic_base_url is used directly.
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""  # pragma: allowlist secret
    anthropic_base_url: Optional[str] = None
    max_turns: int = 30
    timeout: int = 300
    system_prompt: Optional[str] = None
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None
    claude_code_version: Optional[str] = None
    thinking: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    # Runtime capability knobs. The default (bare=True, no mcp_config/settings)
    # reproduces the original isolated behavior: Claude Code skips auto-discovery
    # of skills, hooks, plugins, MCP servers, auto memory, and CLAUDE.md.
    bare: bool = True
    mcp_config: Optional[str] = None
    settings: Optional[str] = None


class ClaudeCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class ClaudeCodeAgent(SimpleResponsesAPIAgent):
    config: ClaudeCodeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_claude_code(self.config.claude_code_version)
        try:
            ver = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
            LOG.warning("claude-code version: %s", ver or "(unknown)")
        except Exception as exc:
            LOG.warning("could not determine claude-code version: %s", exc)

    def _resolve_base_url(self) -> str:
        if self.config.model_server:
            cfg = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            return self.server_client._build_server_base_url(cfg)
        return self.config.anthropic_base_url or ""

    def _build_settings(self) -> dict[str, Any]:
        """Settings written into the run's CLAUDE_CONFIG_DIR.

        The base settings disable telemetry/attribution. When ``config.settings`` points at a
        JSON file, its contents are layered on top: top-level keys override, and the ``env`` block
        is shallow-merged so the telemetry defaults are preserved unless explicitly overridden.
        """
        settings: dict[str, Any] = {
            "env": {
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        }
        if self.config.settings:
            user_settings = json.loads(Path(self.config.settings).expanduser().read_text())
            user_env = user_settings.get("env") or {}
            settings = {**settings, **user_settings, "env": {**settings["env"], **user_env}}
        return settings

    def _setup_config_dir(self) -> Path:
        """Create a per-run CLAUDE_CONFIG_DIR and stage settings into it.

        The directory lives for the duration of a single ``_run_claude_code`` call and is the
        staging seam for capabilities discovered from CLAUDE_CONFIG_DIR (e.g. skills under
        ``<dir>/skills/``). The caller is responsible for removing it.
        """
        claude_config_dir = Path.home() / ".claude_code_agent" / uuid4().hex
        claude_config_dir.mkdir(parents=True)
        (claude_config_dir / "settings.json").write_text(json.dumps(self._build_settings()))
        return claude_config_dir

    def _build_command(self, model: str, instruction: str, system_prompt: Optional[str] = None) -> list[str]:
        """Construct the ``claude`` CLI argv from config.

        ``--bare`` is only passed when ``config.bare`` is True; it disables auto-discovery of
        skills, hooks, plugins, MCP servers, auto memory, and CLAUDE.md. Explicit capabilities
        like ``--mcp-config`` are passed regardless of ``--bare`` since they are not auto-discovered.
        """
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if self.config.bare:
            cmd.append("--bare")
        cmd += ["--max-turns", str(self.config.max_turns), "--model", model]
        if self.config.mcp_config:
            cmd += ["--mcp-config", self.config.mcp_config]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        if self.config.allowed_tools:
            cmd += ["--allowedTools", self.config.allowed_tools]
        if self.config.disallowed_tools:
            cmd += ["--disallowedTools", self.config.disallowed_tools]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if self.config.max_thinking_tokens is not None:
            cmd += ["--max-thinking-tokens", str(self.config.max_thinking_tokens)]
        cmd += ["--", instruction]
        return cmd

    async def _run_claude_code(self, instruction: str, system_prompt: Optional[str] = None) -> tuple[str, str]:
        """Run claude -p --output-format=stream-json and return (stdout, model_name)."""
        base_url = self._resolve_base_url()
        # Keep full model name for local/custom endpoints; strip provider prefix for real Anthropic API.
        model = self.config.model if base_url else self.config.model.split("/")[-1]
        api_key = self.config.anthropic_api_key

        claude_config_dir = self._setup_config_dir()
        try:
            env = {
                **os.environ,
                "ANTHROPIC_API_KEY": api_key,  # pragma: allowlist secret
                "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                "CLAUDE_CODE_SUBAGENT_MODEL": model,
                "IS_SANDBOX": "1",
                "CLAUDE_CONFIG_DIR": str(claude_config_dir),
            }
            if base_url:
                env["ANTHROPIC_BASE_URL"] = base_url
                env["ANTHROPIC_AUTH_TOKEN"] = api_key or "local"

            cmd = self._build_command(model, instruction, system_prompt=system_prompt)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                LOG.warning("claude-code timed out after %ds", self.config.timeout)
                return "", model

            if proc.returncode not in (0, None):
                LOG.warning("claude-code exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            LOG.debug("claude-code stdout (%d chars): %s", len(stdout), stdout[:2000].decode(errors="replace"))
            return stdout.decode(errors="replace"), model
        finally:
            shutil.rmtree(claude_config_dir, ignore_errors=True)

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

        stdout, model_name = await self._run_claude_code(user_message, system_prompt=system_prompt)
        output_items, usage = parse_stream_json(stdout)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("claude-code produced no assistant message; padding empty output")
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

    async def run(self, request: Request, body: ClaudeCodeAgentRunRequest) -> ClaudeCodeAgentVerifyResponse:
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

            return ClaudeCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    ClaudeCodeAgent.run_webserver()
