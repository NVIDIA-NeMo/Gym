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

"""Run Harbor's Terminus-2 loop as a NeMo Gym agent server."""

import asyncio
import json
import logging
import os
import shlex
import shutil
import signal
import sys
import tempfile
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import Request
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


LOG = logging.getLogger(__name__)


def _message_text(item: Any) -> str:
    content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in content)


def _extract_instruction(body_input: Any) -> tuple[str, Optional[str]]:
    """Return the last user instruction and all leading system/developer text."""
    if isinstance(body_input, str):
        return body_input, None

    system_parts: list[str] = []
    user_message = ""
    for item in body_input or []:
        role = getattr(item, "role", None) if not isinstance(item, dict) else item.get("role")
        text = _message_text(item)
        if role in {"system", "developer"} and text:
            system_parts.append(text)
        elif role == "user" and text:
            user_message = text
    return user_message, "\n\n".join(system_parts) or None


class LocalEnvironment:
    """Execute Harbor environment commands in one local workspace."""

    def __init__(self, cwd: Path, command_timeout_sec: float) -> None:
        self.cwd = cwd.resolve()
        self.command_timeout_sec = command_timeout_sec

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | float | None = None,
    ) -> ExecResult:
        run_cwd = Path(cwd).resolve() if cwd else self.cwd
        if not run_cwd.is_dir():
            raise FileNotFoundError(f"Terminus-2 working directory does not exist: {run_cwd}")

        process_env = os.environ.copy()
        if env:
            process_env.update({str(key): str(value) for key, value in env.items()})

        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            cwd=str(run_cwd),
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timeout = self.command_timeout_sec if timeout_sec is None else float(timeout_sec)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.CancelledError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.communicate()
            raise
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = await proc.communicate()
            message = stderr.decode(errors="replace")
            message += f"\nCommand timed out after {timeout:g} seconds"
            return ExecResult(stdout=stdout.decode(errors="replace"), stderr=message.strip(), return_code=124)

        return ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=int(proc.returncode or 0),
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path).resolve()
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target:
            await asyncio.to_thread(shutil.copy2, source, target)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.upload_file(source_path, str(target_path))


class LocalTmuxSession(TmuxSession):
    """Use Harbor's Linux launcher and a BSD-compatible macOS launcher."""

    @property
    def _tmux_start_session(self) -> str:
        if sys.platform != "darwin":
            return super()._tmux_start_session

        session_name = shlex.quote(self._session_name)
        login_shell = shlex.quote("bash --login")
        pipe_command = shlex.quote(f"cat > {shlex.quote(str(self._logging_path))}")
        return (
            "export TERM=xterm-256color && "
            "export SHELL=/bin/bash && "
            f"tmux new-session -x {self._pane_width} -y {self._pane_height} -d "
            f"-s {session_name} {login_shell} \\; pipe-pane -t {session_name} {pipe_command}"
        )


class StandaloneTerminus2(Terminus2NemoGym):
    """Run Terminus-2 in a unique tmux session."""

    async def _handle_llm_interaction(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Track Terminus-2's consecutive two-response completion handshake."""
        result = await super()._handle_llm_interaction(*args, **kwargs)
        if result[1]:
            self._consecutive_completion_claims = getattr(self, "_consecutive_completion_claims", 0) + 1
        else:
            self._consecutive_completion_claims = 0
        return result

    @property
    def finished_naturally(self) -> bool:
        return getattr(self, "_consecutive_completion_claims", 0) >= 2

    async def setup(self, environment: LocalEnvironment) -> None:
        self._consecutive_completion_claims = 0
        self._standalone_session_name = f"terminus-2-{uuid4().hex[:12]}"
        recording_path = self.logs_dir / "recording.cast" if self._record_terminal_session else None
        self._session = LocalTmuxSession(
            session_name=self._standalone_session_name,
            environment=environment,
            logging_path=self.logs_dir / "terminus_2.pane",
            local_asciinema_recording_path=recording_path,
            remote_asciinema_recording_path=recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
        )
        await self._session.start()

    async def close(self, environment: LocalEnvironment) -> None:
        if self._session is not None:
            try:
                await self._session.stop()
            except Exception as exc:
                self.logger.warning("Could not stop Terminus-2 recording cleanly: %s", exc)
        session_name = getattr(self, "_standalone_session_name", None)
        if session_name:
            await environment.exec(
                f"tmux kill-session -t {shlex.quote(session_name)}",
                timeout_sec=30,
            )


class Terminus2AgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 8
    model: Optional[str] = None
    workspace_root: Optional[str] = None
    system_prompt: Optional[str] = None
    max_turns: Optional[int] = None
    parser_name: Literal["json", "xml"] = "json"
    temperature: float = 0.7
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "default"]] = None
    collect_rollout_details: bool = False
    enable_summarize: bool = True
    proactive_summarization_threshold: int = 8000
    max_thinking_tokens: Optional[int] = None
    model_info: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_input_tokens": 262144,
            "max_output_tokens": 81920,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }
    )
    trajectory_config: dict[str, Any] = Field(default_factory=lambda: {"raw_content": False})
    tmux_pane_width: int = 160
    tmux_pane_height: int = 40
    store_all_messages: bool = False
    record_terminal_session: bool = False
    interleaved_thinking: bool = False
    model_timeout_sec: float = 2400
    command_timeout_sec: float = 1800
    timeout: float = 10800
    keep_logs: bool = False
    logs_root: str = "outputs/terminus_2_agent/runs"


class Terminus2AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    agent_timeout_error: int = 0
    context_length_exceeded_error: int = 0


class Terminus2Agent(SimpleResponsesAPIAgent):
    """NeMo Gym Responses API wrapper for the Terminus-2 terminal loop."""

    config: Terminus2AgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _workspace(self) -> Path:
        workspace = Path(self.config.workspace_root).expanduser() if self.config.workspace_root else Path.cwd()
        workspace = workspace.resolve()
        if not workspace.is_dir():
            raise FileNotFoundError(f"Terminus-2 workspace does not exist: {workspace}")
        return workspace

    def _logs_dir(self) -> Path:
        if not self.config.keep_logs:
            return Path(tempfile.mkdtemp(prefix="nemo-gym-terminus-2-"))
        root = Path(self.config.logs_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        path = root / f"run-{int(time())}-{uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _model_name(self, body: NeMoGymResponseCreateParamsNonStreaming) -> str:
        global_model = getattr(self.server_client, "global_config_dict", {}).get("policy_model_name")
        return self.config.model or body.model or global_model or "model"

    def _rollout_id(self, request: Optional[Request]) -> Optional[str]:
        path_params = getattr(request, "path_params", None)
        return path_params.get("rollout_id") if isinstance(path_params, Mapping) else None

    def _build_agent(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        logs_dir: Path,
        api_base: str,
    ) -> StandaloneTerminus2:
        temperature = body.temperature if body.temperature is not None else self.config.temperature
        return StandaloneTerminus2(
            logs_dir=logs_dir,
            model_name=self._model_name(body),
            max_turns=self.config.max_turns,
            parser_name=self.config.parser_name,
            api_base=api_base,
            temperature=temperature,
            reasoning_effort=self.config.reasoning_effort,
            collect_rollout_details=self.config.collect_rollout_details,
            enable_summarize=self.config.enable_summarize,
            proactive_summarization_threshold=self.config.proactive_summarization_threshold,
            max_thinking_tokens=self.config.max_thinking_tokens,
            model_info=self.config.model_info,
            trajectory_config=self.config.trajectory_config,
            tmux_pane_width=self.config.tmux_pane_width,
            tmux_pane_height=self.config.tmux_pane_height,
            store_all_messages=self.config.store_all_messages,
            record_terminal_session=self.config.record_terminal_session,
            interleaved_thinking=self.config.interleaved_thinking,
            responses_create_params=body.model_dump(exclude_none=True),
            nemo_model_server_timeout_sec=self.config.model_timeout_sec,
        )

    async def _run_terminus(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        instruction: str,
        api_base: str,
    ) -> tuple[dict[str, Any], AgentContext, dict[str, bool], bool, bool]:
        logs_dir = self._logs_dir()
        environment = LocalEnvironment(self._workspace(), self.config.command_timeout_sec)
        context = AgentContext()
        agent = self._build_agent(body, logs_dir, api_base)
        timed_out = False
        try:
            try:
                await agent.setup(environment)
                try:
                    await asyncio.wait_for(agent.run(instruction, environment, context), timeout=self.config.timeout)
                except asyncio.TimeoutError:
                    timed_out = True
                    LOG.warning("Terminus-2 timed out after %gs", self.config.timeout)
            finally:
                await agent.close(environment)

            trajectory_path = logs_dir / "trajectory.json"
            flags_path = logs_dir / "agent_error_flags.json"
            trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else {"steps": []}
            flags = json.loads(flags_path.read_text()) if flags_path.exists() else {}
            return trajectory, context, flags, timed_out, agent.finished_naturally
        finally:
            if not self.config.keep_logs:
                shutil.rmtree(logs_dir, ignore_errors=True)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_instruction, input_system = _extract_instruction(body.input)
        instruction_parts = [part for part in (self.config.system_prompt, input_system, user_instruction) if part]
        instruction = "\n\n".join(instruction_parts)

        api_base = self.resolve_model_base_url(
            self.config.model_server.name,
            self._rollout_id(request),
        )
        trajectory, context, flags, timed_out, finished_naturally = await self._run_terminus(
            body, instruction, api_base
        )
        output_items = HarborAgentUtils.trajectory_to_responses(trajectory)
        if not output_items:
            output_items = [
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                ).model_dump()
            ]

        usage = HarborAgentUtils.extract_usage(
            {
                "agent_result": {
                    "n_input_tokens": context.n_input_tokens or 0,
                    "n_output_tokens": context.n_output_tokens or 0,
                    "n_cache_tokens": context.n_cache_tokens or 0,
                }
            },
            trajectory,
        )
        response = HarborAgentUtils.get_default_response_object()
        response.update(
            {
                "created_at": int(time()),
                "model": self._model_name(body),
                "output": output_items,
                "parallel_tool_calls": False,
                "temperature": body.temperature,
                "top_p": body.top_p,
                "tool_choice": body.tool_choice,
                "tools": body.tools,
                "usage": usage,
                "metadata": {
                    "terminus_2": json.dumps(
                        {
                            "context": context.model_dump(mode="json"),
                            "agent_error_flags": flags,
                            "timed_out": timed_out,
                            "finished_naturally": finished_naturally,
                        }
                    )
                },
            }
        )
        return NeMoGymResponse.model_validate(response)

    async def run(self, request: Request, body: Terminus2AgentRunRequest) -> Terminus2AgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_response)
            cookies = seed_response.cookies

            agent_response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_response)
            cookies = agent_response.cookies
            response_json = await get_response_json(agent_response)

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": response_json},
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            verify_json = await get_response_json(verify_response)

            gym_response = NeMoGymResponse.model_validate(response_json)
            turns = sum(
                1
                for item in gym_response.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            terminus_metadata_json = (gym_response.metadata or {}).get("terminus_2", "{}")
            try:
                terminus_metadata = json.loads(terminus_metadata_json)
            except (json.JSONDecodeError, TypeError):
                terminus_metadata = {}
            flags = terminus_metadata.get("agent_error_flags", {})
            timed_out = bool(terminus_metadata.get("timed_out"))
            naturally = bool(terminus_metadata.get("finished_naturally"))

            return Terminus2AgentVerifyResponse.model_validate(
                verify_json
                | {
                    "turns_used": turns,
                    "finished_naturally": naturally and not timed_out,
                    "agent_timeout_error": int(timed_out),
                    "context_length_exceeded_error": int(flags.get("context_length_exceeded", False)),
                }
            )


if __name__ == "__main__":
    Terminus2Agent.run_webserver()
