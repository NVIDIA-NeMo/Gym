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
import signal
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
from responses_api_agents.kilocode_agent.setup_kilo import ensure_kilo


LOG = logging.getLogger(__name__)


def _think_message(index: int, text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg-{index}",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def parse_kilo_events(stdout: str) -> tuple[list[Any], dict[str, int]]:
    """Convert ``kilo run --format json`` stdout into (output_items, usage).

    Kilo (an OpenCode fork) emits one JSON object per line for each root-session
    ``message.part.updated`` event: ``{"type", "timestamp", "sessionID", "part"|"error"}``. The
    ``part`` objects reuse OpenCode's part shapes, so the mapping mirrors the OpenCode agent's
    sqlite parser: ``text`` -> assistant message, ``tool_use`` -> function_call (+ output),
    ``step_finish`` -> token usage, ``reasoning`` -> buffered <think> block prepended to the next
    assistant message (a trailing reasoning run is surfaced on its own so it is not dropped).
    """
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    buffered_think: Optional[str] = None
    # kilo's `--format json` writer emits every message.part.updated event twice with the same part
    # id (persists with KILO_NO_DAEMON=1, so it is the writer, not a daemon double-subscription). The
    # two copies are identical for every event type we parse, so first-wins dedup by part id counts
    # each tool call, message, and token total once. Verified against @kilocode/cli 7.4.15.
    seen_part_ids: set[str] = set()

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        etype = event.get("type")
        part = event.get("part") or {}
        if not isinstance(part, dict):
            part = {}

        part_id = part.get("id")
        if part_id is not None:
            if part_id in seen_part_ids:
                continue
            seen_part_ids.add(part_id)

        if etype == "step_finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            output_tokens += int(tokens.get("output") or 0)

        elif etype == "reasoning":
            text = (part.get("text") or "").strip()
            if text:
                buffered_think = (buffered_think + "\n" + text) if buffered_think else text

        elif etype == "text":
            text = part.get("text") or ""
            if not text.strip():
                continue
            if buffered_think:
                text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                buffered_think = None
            output_items.append(_think_message(len(output_items), text))

        elif etype == "tool_use":
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
            output = state.get("output")
            if output is None and state.get("status") == "error":
                output = state.get("error") or ""
            if output is not None:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=str(output),
                        status="completed",
                    )
                )

        elif etype == "error":
            LOG.warning("kilo run reported error event: %s", str(event.get("error"))[:500])

    # A run that ends on reasoning (some vLLM reasoning parsers route the final answer through the
    # reasoning channel) is surfaced as a think-tagged message rather than dropped.
    if buffered_think:
        output_items.append(_think_message(len(output_items), f"<think>\n{buffered_think}\n</think>"))

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


class KiloCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    concurrency: int = 8
    command: str = "kilo"
    model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/kilocode_agent/workspaces"
    repo_dir: Optional[str] = None
    thinking: bool = False
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    # written to kilo.json in the run dir (OpenCode-compatible config schema)
    kilo_config: dict[str, Any] = Field(default_factory=dict)
    kilo_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class KiloCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class KiloCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class KiloCodeAgent(SimpleResponsesAPIAgent):
    """Runs the Kilo Code CLI (``kilo run --auto --format json``).

    Kilo runs its own tools internally; we parse its JSON event stream into Gym format and use the
    resources server to verify. Eval-only: token IDs and logprobs are not wired up and it does not
    use a Gym model server yet.
    """

    config: KiloCodeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_kilo(self.config.kilo_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("kilo command %r is not on PATH yet", self.config.command)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                KiloCodeAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"kilo_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _repo_dir(self, fallback: Path) -> Path:
        """Return the configured persistent repository or the temporary fallback."""
        if not self.config.repo_dir:
            return fallback
        root = Path(self.config.repo_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _write_kilo_config(self, work_dir: Path) -> None:
        if not self.config.kilo_config:
            return
        config = self._deep_merge({}, copy.deepcopy(self.config.kilo_config))
        (work_dir / "kilo.json").write_text(json.dumps(config, indent=2))

    def _env(self, data_home: str, config_home: str) -> dict[str, str]:
        # Per-run isolation. KILO_NO_DAEMON=1 forces a fresh embedded server (kilo's daemon client is
        # enabled iff this is unset), so concurrent rollouts share no daemon db/state. KILO_DB=:memory:
        # keeps sessions ephemeral (we read stdout, not the db). XDG_DATA_HOME/XDG_CONFIG_HOME point
        # inside the run dir so global ~/.local/share/kilo and ~/.config/kilo never leak in. Indexing
        # is handled by --pure (see _build_command), not here: KILO_DISABLE_CODEBASE_INDEXING only
        # triggers on the literal value "vscode-no-workspace", not a truthy one, so it is unusable.
        env = {
            **os.environ,
            "KILO_NO_DAEMON": "1",
            "KILO_DB": ":memory:",
            "XDG_DATA_HOME": data_home,
            "XDG_CONFIG_HOME": config_home,
        }
        if self.config.openai_base_url:
            env["OPENAI_BASE_URL"] = self.config.openai_base_url
        if self.config.openai_api_key:
            env["OPENAI_API_KEY"] = self.config.openai_api_key
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    def _build_command(self, project_dir: Path, prompt: str) -> list[str]:
        cmd = [
            *self.config.command_parts,
            "run",
            "--auto",
            # --pure disables external plugins, so codebase indexing never starts; built-in
            # bash/edit/webfetch tools and the custom provider are unaffected.
            "--pure",
            "--format",
            "json",
            "-m",
            self.config.model,
            "--dir",
            str(project_dir),
        ]
        if self.config.thinking:
            cmd.append("--thinking")
        cmd.extend(self.config.extra_args)
        # `--` terminates option parsing so a prompt beginning with '-' is passed as the message.
        cmd += ["--", prompt]
        return cmd

    @staticmethod
    def _kill_process_group(proc: "asyncio.subprocess.Process") -> None:
        # `kilo` is an npm shim whose child is the real binary; killing only the shim orphans the
        # child holding the stdout pipe, so kill the whole process group (start_new_session=True).
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()

    async def _run_kilo(self, instruction: str, system_prompt: Optional[str]) -> tuple[list[Any], dict[str, int], str]:
        """Run one headless kilo run. Returns (output_items, usage, model_name)."""
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_home = work_dir / ".kilo-data"
        config_home = work_dir / ".kilo-config"
        data_home.mkdir(parents=True, exist_ok=True)
        config_home.mkdir(parents=True, exist_ok=True)
        self._write_kilo_config(project_dir)
        env = self._env(str(data_home), str(config_home))
        cmd = self._build_command(project_dir, prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                self._kill_process_group(proc)
                await proc.communicate()
                LOG.warning("kilo timed out after %ds", self.config.timeout)
                return [], {"input_tokens": 0, "output_tokens": 0}, self.config.model

            if proc.returncode not in (0, None):
                LOG.warning("kilo exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            output_items, usage = parse_kilo_events(stdout.decode(errors="replace"))
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

        output_items, usage, model_name = await self._run_kilo(user_message, system_prompt)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("Kilo produced no assistant message. Padding empty output")
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

    async def run(self, request: Request, body: KiloCodeAgentRunRequest) -> KiloCodeAgentVerifyResponse:
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

            return KiloCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    KiloCodeAgent.run_webserver()
