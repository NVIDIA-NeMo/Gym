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
import sqlite3
from asyncio import Semaphore
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
from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode


LOG = logging.getLogger(__name__)


def parse_opencode_session(db_path: Path) -> tuple[list[Any], dict[str, int]]:
    """convert the sqlite session db into Gym format including tool calls"""
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    if not db_path.is_file():
        return output_items, {"input_tokens": 0, "output_tokens": 0}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        roles = {r["id"]: json.loads(r["data"]).get("role") for r in con.execute("select id, data from message")}
        rows = con.execute("select message_id, data from part order by time_created").fetchall()
    finally:
        con.close()

    for r in rows:
        try:
            part = json.loads(r["data"])
        except (json.JSONDecodeError, TypeError):
            continue
        ptype = part.get("type")

        if ptype == "step-finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            output_tokens += int(tokens.get("output") or 0)
            continue

        if roles.get(r["message_id"]) != "assistant":
            continue

        if ptype == "text" and (part.get("text") or "").strip():
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=part["text"], annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        elif ptype == "tool":
            state = part.get("state") or {}
            call_id = part.get("callID") or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=part.get("tool", ""),
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )
            if state.get("output") is not None:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=str(state.get("output")),
                        status="completed",
                    )
                )

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


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


class OpenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    concurrency: int = 8
    command: str = "opencode"
    model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/opencode_agent/workspaces"
    thinking: bool = True
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    opencode_config: dict[str, Any] = Field(default_factory=dict)
    opencode_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class OpenCodeAgent(SimpleResponsesAPIAgent):
    """Runs the CLI (opencode run --format=json)"""

    config: OpenCodeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_opencode(self.config.opencode_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("opencode command %r is not on PATH yet", self.config.command)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenCodeAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"opencode_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _write_opencode_config(self, work_dir: Path) -> None:
        if not self.config.opencode_config:
            return
        config = self._deep_merge({}, copy.deepcopy(self.config.opencode_config))
        (work_dir / "opencode.json").write_text(json.dumps(config, indent=2))

    def _env(self, data_home: str) -> dict[str, str]:
        env = {**os.environ, "XDG_DATA_HOME": data_home}
        if self.config.openai_base_url:
            env["OPENAI_BASE_URL"] = self.config.openai_base_url
        if self.config.openai_api_key:
            env["OPENAI_API_KEY"] = self.config.openai_api_key
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    async def _run_opencode(
        self, instruction: str, system_prompt: Optional[str]
    ) -> tuple[list[Any], dict[str, int], str]:
        """Run one headless opencode run. Returns (output_items, usage, model_name)."""
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        data_home = work_dir / ".opencode-data"
        data_home.mkdir(parents=True, exist_ok=True)
        self._write_opencode_config(work_dir)
        env = self._env(str(data_home))

        cmd = [*self.config.command_parts, "run", "-m", self.config.model, "--dir", str(work_dir)]
        if self.config.thinking:
            cmd.append("--thinking")
        cmd.extend(self.config.extra_args)
        cmd.append(prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                LOG.warning("opencode timed out after %ds", self.config.timeout)
                return [], {"input_tokens": 0, "output_tokens": 0}, self.config.model

            if proc.returncode not in (0, None):
                LOG.warning("opencode exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            output_items, usage = parse_opencode_session(data_home / "opencode" / "opencode.db")
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

        output_items, usage, model_name = await self._run_opencode(user_message, system_prompt)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("OpenCode produced no assistant message. Padding empty output")
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

    async def run(self, request: Request, body: OpenCodeAgentRunRequest) -> OpenCodeAgentVerifyResponse:
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

            return OpenCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    OpenCodeAgent.run_webserver()
