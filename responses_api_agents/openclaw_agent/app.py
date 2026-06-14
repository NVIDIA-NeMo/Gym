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
import copy
import json
import logging
import os
import shlex
import shutil
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, ClassVar, Optional
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
from responses_api_agents.openclaw_agent.setup_openclaw import ensure_openclaw


LOG = logging.getLogger(__name__)


def _decode_last_json_dict_suffix(raw: str) -> Optional[dict[str, Any]]:
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and not text[start + consumed :].strip():
            return obj
    return None


def _text_from_openclaw_payloads(envelope: dict[str, Any]) -> str:
    payloads = envelope.get("payloads")
    if not isinstance(payloads, list):
        payloads = []
    parts = [p["text"].strip() for p in payloads if isinstance(p, dict) and (p.get("text") or "").strip()]
    if parts:
        return "\n\n".join(parts)
    final = (envelope.get("meta") or {}).get("finalAssistantVisibleText")
    return final.strip() if isinstance(final, str) else ""


def parse_openclaw_output(stdout: str) -> tuple[list[Any], dict[str, int]]:
    envelope = _decode_last_json_dict_suffix(stdout)
    if not envelope:
        return [], {"input_tokens": 0, "output_tokens": 0}

    text = _text_from_openclaw_payloads(envelope)
    output_items: list[Any] = []
    if text:
        output_items.append(
            NeMoGymResponseOutputMessage(
                id="msg-0",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
        )

    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    usage = agent_meta.get("usage") if isinstance(agent_meta.get("usage"), dict) else {}
    cache_read = int(usage.get("cacheRead") or 0)
    input_tokens = int(usage.get("input") or 0) + cache_read
    output_tokens = int(usage.get("output") or 0)
    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


def parse_openclaw_session(session_text: str) -> list[Any]:
    """Convert an OpenClaw session .jsonl into Gym output items, including tool calls"""
    output_items: list[Any] = []
    for line in session_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            continue

        if role == "assistant":
            texts = [b["text"] for b in content if isinstance(b, dict) and (b.get("text") or "").strip()]
            if texts:
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text="\n".join(texts), annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                args = block.get("arguments")
                arguments = json.dumps(args) if isinstance(args, (dict, list)) else str(args or "")
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=arguments,
                        call_id=call_id,
                        name=block.get("name", ""),
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )

        elif role == "toolResult":
            call_id = message.get("toolCallId", "")
            result_text = "".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=result_text,
                    status="completed",
                )
            )

    return output_items


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


class OpenClawAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    concurrency: int = 32
    command: str = "openclaw"
    model: str = "nvinf/nvidia/meta/llama-3.3-70b-instruct"
    node_bin_dir: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/openclaw_agent/workspaces"
    openclaw_agent_id: str = "main"
    thinking: str = "off"
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    openclaw_config: dict[str, Any] = Field(default_factory=dict)
    openclaw_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenClawAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenClawAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class OpenClawAgent(SimpleResponsesAPIAgent):
    """Runs the OpenClaw CLI (openclaw agent --local --json)"""

    config: OpenClawAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # deny the interactive "message" channel so the headless agent finishes
    _HEADLESS_TOOL_DENY: ClassVar[tuple[str, ...]] = ("message",)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_openclaw(self.config.openclaw_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("openclaw command %r is not on PATH yet", self.config.command)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenClawAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _merge_headless_tool_denies(self, cfg: dict[str, Any]) -> None:
        tools = cfg.setdefault("tools", {})
        deny = tools.get("deny")
        if not isinstance(deny, list):
            deny = []
        merged = list(dict.fromkeys([item for item in deny if isinstance(item, str)] + list(self._HEADLESS_TOOL_DENY)))
        tools["deny"] = merged

    def _build_openclaw_config(self, base: dict[str, Any]) -> dict[str, Any]:
        cfg = copy.deepcopy(base)
        self._merge_headless_tool_denies(cfg)
        self._deep_merge(cfg, copy.deepcopy(self.config.openclaw_config))
        return cfg

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"openclaw_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _env(self, home: Path) -> dict[str, str]:
        env = {**os.environ, "HOME": str(home)}
        if self.config.node_bin_dir:
            env["PATH"] = f"{self.config.node_bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    async def _run_exec(
        self, args: list[str], *, cwd: Optional[str], env: dict[str, str], timeout: int
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"Timed out after {timeout}s: {shlex.join(args)}") from None
        return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    @staticmethod
    def _session_file(envelope: Optional[dict[str, Any]]) -> Optional[Path]:
        meta = (envelope or {}).get("meta") if isinstance(envelope, dict) else None
        agent_meta = meta.get("agentMeta") if isinstance(meta, dict) else None
        session_file = agent_meta.get("sessionFile") if isinstance(agent_meta, dict) else None
        return Path(session_file) if isinstance(session_file, str) and session_file else None

    async def _run_openclaw(
        self, instruction: str, system_prompt: Optional[str]
    ) -> tuple[list[Any], dict[str, int], str]:
        """setup and run agent. returns (output_items, usage, model_name)."""
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        home = work_dir / ".openclaw-home"
        home.mkdir(parents=True, exist_ok=True)
        env = self._env(home)

        try:
            await self._run_exec(
                [*self.config.command_parts, "setup", "--non-interactive", "--accept-risk", "--mode", "local"],
                cwd=str(work_dir),
                env=env,
                timeout=self.config.setup_timeout,
            )

            config_path = home / ".openclaw" / "openclaw.json"
            if not config_path.is_file():
                raise RuntimeError(f"openclaw setup did not produce a config at {config_path}")
            base_cfg = json.loads(config_path.read_text())
            config_path.write_text(json.dumps(self._build_openclaw_config(base_cfg), indent=2) + "\n")

            cmd = [
                *self.config.command_parts,
                "agent",
                "--local",
                "--json",
                "--agent",
                self.config.openclaw_agent_id,
                "--thinking",
                self.config.thinking,
                "--model",
                self.config.model,
                "--message",
                prompt,
                *self.config.extra_args,
            ]
            code, stdout, stderr = await self._run_exec(cmd, cwd=str(work_dir), env=env, timeout=self.config.timeout)
            if code:
                LOG.warning("openclaw exited %d: %s", code, stderr[:500])
            LOG.debug("openclaw stdout (%d chars): %s", len(stdout), stdout[:2000])

            envelope = _decode_last_json_dict_suffix(stdout)
            usage = parse_openclaw_output(stdout)[1]

            # prefer the session file, try final text if not (shouldnt happen)
            output_items: list[Any] = []
            session_path = self._session_file(envelope)
            if session_path and session_path.is_file():
                output_items = parse_openclaw_session(session_path.read_text(errors="replace"))
            if not output_items and envelope is not None:
                output_items = parse_openclaw_output(stdout)[0]
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

        output_items, usage, model_name = await self._run_openclaw(user_message, system_prompt)

        if not output_items:
            LOG.warning("OpenClaw produced no assistant message. Padding empty output")
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

    async def run(self, request: Request, body: OpenClawAgentRunRequest) -> OpenClawAgentVerifyResponse:
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

            return OpenClawAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    OpenClawAgent.run_webserver()
