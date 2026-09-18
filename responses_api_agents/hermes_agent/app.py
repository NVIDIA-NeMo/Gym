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
import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
from asyncio import Semaphore
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from time import time
from typing import Any, Callable, Optional
from uuid import uuid4

import model_tools  # noqa: F401  # fail-fast if hermes-agent isn't installed  # pyright: ignore[reportMissingImports]
from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxPtyError, SandboxPtySession
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.hermes_agent.observability import HermesAgentObserver, normalize_hermes_messages


def _trajectory_to_output_items(messages, n_input):
    output_items = []
    for item in messages[n_input:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "") or ""
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "") for c in content)
        if role == "assistant":
            reasoning_text = item.get("reasoning") or ""
            if reasoning_text:
                content = ResponsesConverter._parse_think_tags(content)[1]
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rsn-{len(output_items)}",
                        summary=[NeMoGymSummary(type="summary_text", text=reasoning_text)],
                        type="reasoning",
                    )
                )
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=content, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=item.get("prompt_token_ids") or [],
                    generation_token_ids=item.get("generation_token_ids") or [],
                    generation_log_probs=item.get("generation_log_probs") or [],
                    routed_experts=item.get("routed_experts"),
                )
            )
            for tc in item.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not fn:
                    continue
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=fn.get("arguments", ""),
                        call_id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        type="function_call",
                        id=tc.get("id"),
                        status="completed",
                    )
                )
        elif role == "tool":
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=item.get("tool_call_id", ""),
                    output=content,
                    status="completed",
                )
            )
    return output_items


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"
_SANDBOX_RUNTIME_DIR = "/tmp/nemo-gym-hermes-runtime-26bb847a"
_SANDBOX_UV = f"{_SANDBOX_RUNTIME_DIR}/uv"
_SANDBOX_PYTHON = f"{_SANDBOX_RUNTIME_DIR}/venv/bin/python"
_SANDBOX_RUNNER = f"{_SANDBOX_RUNTIME_DIR}/sandbox_runner.py"
_SANDBOX_OBSERVER = f"{_SANDBOX_RUNTIME_DIR}/sandbox_observer.py"
_HERMES_REQUIREMENT = (
    "hermes-agent @ https://github.com/cmunley1/hermes-agent/archive/26bb847a88493342ca1b194e0455b479073ae21d.tar.gz"
)
_AGENT_SESSION_ID_KEY = "agent_session_id"


@dataclass
class HermesAgentSessionState:
    request: AgentSeedSessionRequest
    sandbox: AsyncSandbox
    workdir: str
    session_dir: str
    runner_session: SandboxPtySession | None = None
    runner_exit_task: asyncio.Task[int] | None = None
    observations: AgentObservationBundle | None = None


# if ray close sys.stderr mid-request, write to the original fd
class _SafeStderrHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = sys.__stderr__
            if stream is None:
                return
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            pass


if not LOG.handlers:
    LOG.addHandler(_SafeStderrHandler(level=logging.WARNING))


def _split_input_to_user_and_history(input_items) -> tuple[str, list[dict], Optional[str]]:
    items = list(input_items)
    system_message: Optional[str] = None
    if items:
        first = items[0]
        first_role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        first_content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
        if first_role == "system":
            if isinstance(first_content, list):
                first_content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in first_content
                )
            system_message = first_content or ""
            items = items[1:]

    user_message = ""
    history: list[dict] = []
    for idx, item in enumerate(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if isinstance(content, list):
            content = "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        content = content or ""
        if idx == len(items) - 1 and role == "user":
            user_message = content
        else:
            history.append({"role": role, "content": content})
    return user_message, history, system_message


class HermesAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model: Optional[str] = None
    concurrency: int = 32
    max_turns: int = 90
    max_tokens: Optional[int] = None
    enabled_toolsets: Optional[list[str]] = None
    disabled_toolsets: Optional[list[str]] = None
    temperature: float | None = None
    terminal_backend: str = "local"
    terminal_timeout: int = 180
    sandbox_install_timeout_seconds: float = 900.0
    sandbox_runner_poll_seconds: float = 0.25
    session_close_timeout_seconds: float = 30.0
    system_prompt: Optional[str] = None
    compression_enabled: bool = True
    compression_threshold: float = 0.85
    chat_template_kwargs_enabled: bool = True
    api_key: Optional[str] = None
    delegation_max_iterations: int = 50
    checkpoints_enabled: bool = False


class HermesAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class HermesAgent(SimpleResponsesAPIAgent):
    config: HermesAgentConfig
    sem: Semaphore = None
    # Set of agents currently running run_conversation, plus a flag tracking whether the single
    # shared SIGTERM dispatcher has been installed on the event loop. See _ensure_sigterm_handler.
    active_agents: set = None
    interrupted_agents: set = None
    sigterm_installed: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def seed_agent_session(
        self,
        request: Request,
        body: AgentSeedSessionRequest,
    ) -> AgentSeedSessionResponse:
        agent_session_id = f"agent-session-{uuid4().hex}"
        state = await self._initialize_agent_session_state(agent_session_id, body)
        self._agent_sessions[agent_session_id] = state
        request.session[_AGENT_SESSION_ID_KEY] = agent_session_id
        return AgentSeedSessionResponse(agent_session_id=agent_session_id)

    async def close_agent_session(
        self,
        request: Request,
        body: AgentCloseSessionRequest,
    ) -> AgentCloseSessionResponse:
        agent_session_id = request.session.get(_AGENT_SESSION_ID_KEY)
        if body.agent_session_id != agent_session_id:
            raise ValueError("agent_session_id does not match the session cookie")
        state = self._require_agent_session(agent_session_id)
        if body.episode_id != state.request.episode_id:
            raise ValueError("episode_id does not match the seeded agent session")
        observations = await self._close_agent_session_state(state)
        del self._agent_sessions[agent_session_id]
        request.session.pop(_AGENT_SESSION_ID_KEY, None)
        return AgentCloseSessionResponse(
            agent_session_id=agent_session_id,
            agent_observations=observations,
        )

    def _require_agent_session(self, agent_session_id: str) -> HermesAgentSessionState:
        try:
            return self._agent_sessions[agent_session_id]
        except KeyError as error:
            raise ValueError(f"Unknown agent_session_id: {agent_session_id}") from error

    @staticmethod
    def _agent_session_id_from_request(request: Request | None) -> str | None:
        if request is None:
            return None
        try:
            agent_session_id = request.session.get(_AGENT_SESSION_ID_KEY)
        except (AssertionError, AttributeError):
            return None
        return agent_session_id if isinstance(agent_session_id, str) else None

    def _ensure_sigterm_handler(self) -> None:
        """Install exactly one SIGTERM handler on the event loop that interrupts *every* in-flight
        agent. Registering a fresh per-call handler is unsafe under concurrency: add_signal_handler
        replaces the previous handler, so concurrent responses() calls clobber each other and the
        first to finish removes the only remaining handler — leaving later SIGTERMs unhandled and
        their trajectories lost. A single dispatcher over `active_agents` avoids that race."""
        if self.sigterm_installed:
            return
        import signal

        def _dispatch():
            for ag in list(self.active_agents):
                self.interrupted_agents.add(id(ag))
                if hasattr(ag, "interrupt"):
                    ag.interrupt("timeout")

        try:
            asyncio.get_event_loop().add_signal_handler(signal.SIGTERM, _dispatch)
            self.sigterm_installed = True
        except (NotImplementedError, OSError):
            pass  # not supported on this platform (e.g. Windows, non-main thread)

    def _build_config(self) -> str:
        import yaml

        config: dict[str, Any] = {
            "model": self._model_name(),
            "provider": "auto",
            "toolsets": ["hermes-cli"],
            "agent": {"max_turns": self.config.max_turns},
            "memory": {
                "memory_enabled": False,
                "user_profile_enabled": False,
            },
            "compression": {
                "enabled": self.config.compression_enabled,
                "threshold": self.config.compression_threshold,
            },
            "terminal": {
                "backend": self.config.terminal_backend,
                "timeout": self.config.terminal_timeout,
            },
            "delegation": {
                "max_iterations": self.config.delegation_max_iterations,
            },
            "checkpoints": {
                "enabled": self.config.checkpoints_enabled,
            },
        }
        return yaml.dump(config, default_flow_style=False)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Process-local Hermes sessions require num_workers=1")
        self.sem = Semaphore(self.config.concurrency)
        self.active_agents = set()
        self.interrupted_agents = set()
        self._agent_sessions: dict[str, HermesAgentSessionState] = {}
        # hermes-agent reads these from env (cli.py / batch_runner.py); env vars are
        # process-global, so multiple HermesAgent instances in one process share them
        os.environ["TERMINAL_ENV"] = self.config.terminal_backend
        os.environ["TERMINAL_TIMEOUT"] = str(self.config.terminal_timeout)

        # Build config.yaml with config parameters
        hermes_home = tempfile.mkdtemp(prefix="hermes_agent_")
        atexit.register(shutil.rmtree, hermes_home, True)
        with open(os.path.join(hermes_home, "config.yaml"), "w") as _f:
            _f.write(self._build_config())
        os.environ["HERMES_HOME"] = hermes_home

    async def _initialize_agent_session_state(
        self,
        agent_session_id: str,
        body: AgentSeedSessionRequest,
    ) -> HermesAgentSessionState:
        if body.sandbox_access is None:
            raise ValueError("Hermes requires sandbox_access for an episode session")
        if self.config.enabled_toolsets != ["terminal"]:
            raise ValueError("Hermes sandbox access requires enabled_toolsets: [terminal]")
        connection = body.sandbox_access.connection
        if not isinstance(connection, DirectSandboxConnection):
            raise ValueError("Hermes currently supports only direct sandbox connections")
        provider_config = resolve_provider_config(
            connection.provider_config_ref,
            get_global_config_dict(),
        )
        provider = create_provider(provider_config)
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise

        session_dir = f"/tmp/nemo-gym-hermes-sessions/{agent_session_id}"
        try:
            uv_path = shutil.which("uv")
            if uv_path is None:
                raise RuntimeError("Hermes agent server requires uv to install the sandbox runtime")
            prepare = await sandbox.exec(
                f"mkdir -p {quote(_SANDBOX_RUNTIME_DIR)} {quote(session_dir)}",
                cwd=body.sandbox_access.workdir,
                timeout_s=30,
            )
            if prepare.return_code != 0:
                raise RuntimeError(prepare.stderr or prepare.stdout or "Failed to prepare Hermes sandbox paths")
            await sandbox.upload(uv_path, _SANDBOX_UV)
            install = await sandbox.exec(
                (
                    f"chmod 755 {quote(_SANDBOX_UV)}; "
                    f"if [ ! -x {quote(_SANDBOX_PYTHON)} ]; then "
                    f"{quote(_SANDBOX_UV)} venv {quote(_SANDBOX_RUNTIME_DIR + '/venv')} --python 3.13; "
                    f"{quote(_SANDBOX_UV)} pip install --python {quote(_SANDBOX_PYTHON)} "
                    f"{quote(_HERMES_REQUIREMENT)}; "
                    "fi"
                ),
                cwd=body.sandbox_access.workdir,
                timeout_s=self.config.sandbox_install_timeout_seconds,
            )
            if install.return_code != 0:
                raise RuntimeError(install.stderr or install.stdout or "Hermes sandbox installation failed")
            await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), _SANDBOX_RUNNER)
            await sandbox.upload(Path(__file__).with_name("sandbox_observer.py"), _SANDBOX_OBSERVER)
        except BaseException:
            await sandbox.disconnect()
            raise

        return HermesAgentSessionState(
            request=body,
            sandbox=sandbox,
            workdir=body.sandbox_access.workdir,
            session_dir=session_dir,
        )

    async def _terminate_sandbox_runner(self, state: HermesAgentSessionState) -> None:
        runner_session = state.runner_session
        runner_exit_task = state.runner_exit_task
        if runner_session is None:
            return
        try:
            if runner_exit_task is not None and not runner_exit_task.done():
                await runner_session.send_signal("SIGTERM")
                try:
                    await asyncio.wait_for(
                        asyncio.shield(runner_exit_task),
                        timeout=self.config.session_close_timeout_seconds,
                    )
                except TimeoutError:
                    await runner_session.send_signal("SIGKILL")
                    await asyncio.wait_for(
                        asyncio.shield(runner_exit_task),
                        timeout=self.config.session_close_timeout_seconds,
                    )
            elif runner_exit_task is not None:
                runner_exit_task.exception()
        except SandboxPtyError:
            if runner_exit_task is not None and not runner_exit_task.done():
                raise
        finally:
            await runner_session.close()
            state.runner_session = None
            state.runner_exit_task = None

    async def _close_agent_session_state(
        self,
        state: HermesAgentSessionState,
    ) -> AgentObservationBundle:
        await self._terminate_sandbox_runner(state)
        await state.sandbox.exec(
            f"rm -rf {quote(state.session_dir)}",
            cwd=state.workdir,
            timeout_s=self.config.session_close_timeout_seconds,
        )
        await state.sandbox.disconnect()
        observations = state.observations
        if observations is None:
            observations = AgentObservationBundle(
                source="hermes", gaps=[ObservationGap(code="observation_capture_failed")]
            )
        return observations

    def _model_name(self) -> str:
        return self.config.model or str(self.config.model_server.name)

    @staticmethod
    async def _upload_json(sandbox: AsyncSandbox, remote_path: str, payload: dict[str, Any]) -> None:
        with tempfile.TemporaryDirectory(prefix="hermes_sandbox_upload_") as directory:
            local_path = Path(directory) / "payload.json"
            local_path.write_text(json.dumps(payload))
            await sandbox.upload(local_path, remote_path)

    @staticmethod
    async def _download_json(sandbox: AsyncSandbox, remote_path: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="hermes_sandbox_download_") as directory:
            local_path = Path(directory) / "payload.json"
            await sandbox.download(remote_path, local_path)
            payload = json.loads(local_path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"Hermes sandbox payload at {remote_path} is not an object")
        return payload

    def _sandbox_observations(
        self,
        result: dict[str, Any],
        model_calls: list[ModelCallRef],
        raw_observations: Any,
    ) -> AgentObservationBundle:
        if isinstance(raw_observations, dict):
            try:
                model_calls_by_id = {call.response_id: call for call in model_calls if call.response_id}
                records: list[AgentInvocation | ToolCallObservation | ContextCompactionObservation] = []
                for raw_invocation in raw_observations.get("invocations") or []:
                    response_ids = raw_invocation.get("model_response_ids") or []
                    invocation_id = str(raw_invocation["invocation_id"])
                    records.append(
                        AgentInvocation(
                            invocation_id=invocation_id,
                            parent_invocation_id=raw_invocation.get("parent_invocation_id"),
                            status=raw_invocation.get("status", "unknown"),
                            model_calls=[
                                model_calls_by_id[response_id]
                                for response_id in response_ids
                                if response_id in model_calls_by_id
                            ],
                            conversation=normalize_hermes_messages(
                                raw_invocation.get("messages") or [],
                                id_prefix=invocation_id,
                            ),
                        )
                    )
                records.extend(
                    ToolCallObservation.model_validate(tool) for tool in raw_observations.get("tools") or []
                )
                records.extend(
                    ContextCompactionObservation.model_validate(compaction)
                    for compaction in raw_observations.get("compactions") or []
                )
                return AgentObservationBundle(
                    source="hermes",
                    records=records,
                    gaps=[ObservationGap.model_validate(gap) for gap in raw_observations.get("gaps") or []],
                )
            except Exception as error:
                LOG.exception("failed to validate sandbox Hermes observations")
                return AgentObservationBundle(
                    source="hermes",
                    gaps=[
                        ObservationGap(
                            code="observation_capture_failed",
                            detail=type(error).__name__,
                        )
                    ],
                )

        messages = result.get("messages") or []
        invocation_status = (
            "failed"
            if result.get("error") or result.get("failed")
            else ("completed" if result.get("completed", True) else "incomplete")
        )
        records: list[AgentInvocation | ToolCallObservation] = [
            AgentInvocation(
                invocation_id="root",
                status=invocation_status,
                model_calls=model_calls,
                conversation=normalize_hermes_messages(messages),
            )
        ]
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                if not isinstance(function, dict):
                    continue
                records.append(
                    ToolCallObservation(
                        invocation_id="root",
                        tool_call_id=str(tool_call.get("id") or ""),
                        tool_name=str(function.get("name") or ""),
                        timing_source="harness",
                        status="completed",
                    )
                )
        return AgentObservationBundle(source="hermes", records=records)

    async def _run_sandbox_episode(
        self,
        *,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming,
        agent_session_id: str,
        state: HermesAgentSessionState,
    ) -> AgentEpisode:
        user_message, history, input_system = _split_input_to_user_and_history(body.input)
        input_path = f"{state.session_dir}/input.json"
        output_path = f"{state.session_dir}/output.json"
        stdout_path = f"{state.session_dir}/stdout.log"
        stderr_path = f"{state.session_dir}/stderr.log"
        payload = {
            "agent_session_id": agent_session_id,
            "chat_template_kwargs_enabled": self.config.chat_template_kwargs_enabled,
            "config_yaml": self._build_config(),
            "disabled_toolsets": self.config.disabled_toolsets,
            "enabled_toolsets": self.config.enabled_toolsets,
            "history": history,
            "max_tokens": self.config.max_tokens,
            "max_turns": self.config.max_turns,
            "model": self._model_name(),
            "system_message": self.config.system_prompt or input_system,
            "temperature": self.config.temperature,
            "terminal_timeout": self.config.terminal_timeout,
            "user_message": user_message,
        }
        await self._upload_json(state.sandbox, input_path, payload)
        try:
            state.runner_session = await state.sandbox.pty.create(
                command=(
                    f"{quote(_SANDBOX_PYTHON)} {quote(_SANDBOX_RUNNER)} "
                    f"{quote(input_path)} {quote(output_path)} "
                    f">{quote(stdout_path)} 2>{quote(stderr_path)}"
                ),
                cwd=state.workdir,
                pty=False,
            )
        except NotImplementedError as error:
            raise ValueError("Hermes requires a sandbox provider with PTY process sessions") from error
        state.runner_exit_task = asyncio.create_task(state.runner_session.wait_exit())

        model_cookies: Any = None
        model_calls: list[ModelCallRef] = []
        request_index = 0
        try:
            while True:
                request_path = f"{state.session_dir}/model-request-{request_index}.json"
                status = await state.sandbox.exec(
                    (
                        f"if [ -f {quote(output_path)} ]; then echo output; "
                        f"elif [ -f {quote(request_path)} ]; then echo request; "
                        "else echo running; fi"
                    ),
                    cwd=state.workdir,
                    timeout_s=30,
                )
                state_name = (status.stdout or "").strip()
                if state_name == "running" and state.runner_exit_task is not None and state.runner_exit_task.done():
                    state_name = "exited"
                if state_name == "request":
                    model_request = await self._download_json(state.sandbox, request_path)
                    for client_option in ("extra_headers", "extra_query", "timeout"):
                        model_request.pop(client_option, None)
                    extra_body = model_request.pop("extra_body", None)
                    if isinstance(extra_body, dict):
                        model_request = extra_body | model_request
                    try:
                        model_response = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path=self.url_path_for_request("/v1/chat/completions", request),
                            json=model_request,
                            cookies=model_cookies,
                        )
                        await raise_for_status(model_response)
                        model_cookies = model_response.cookies
                        response_payload = await get_response_json(model_response)
                        response_id = response_payload.get("id") if isinstance(response_payload, dict) else None
                        if isinstance(response_id, str) and response_id:
                            model_calls.append(
                                ModelCallRef(model_ref=self.config.model_server, response_id=response_id)
                            )
                        relay_payload = {"response": response_payload}
                    except Exception as error:
                        relay_payload = {"error": str(error)}
                    await self._upload_json(
                        state.sandbox,
                        f"{state.session_dir}/model-response-{request_index}.json",
                        relay_payload,
                    )
                    await state.sandbox.exec(
                        f"rm -f {quote(request_path)}",
                        cwd=state.workdir,
                        timeout_s=30,
                    )
                    request_index += 1
                    continue
                if state_name == "output":
                    output = await self._download_json(state.sandbox, output_path)
                    if output.get("error") is not None:
                        raise RuntimeError(
                            f"Hermes sandbox runner failed: {output['error']}\n{output.get('traceback', '')}"
                        )
                    result = output.get("result")
                    runtime = output.get("runtime")
                    if not isinstance(result, dict) or not isinstance(runtime, dict):
                        raise RuntimeError("Hermes sandbox runner returned an invalid output")
                    response = self._response_from_result(
                        body=body,
                        result=result,
                        model_name=self._model_name(),
                        fail_on_error=True,
                        n_input=len(history) + 1,
                    )
                    response.metadata = {
                        **(response.metadata or {}),
                        "harness_execution": "sandbox",
                        "harness_hostname": str(runtime.get("hostname") or ""),
                        "harness_pid": str(runtime.get("pid") or ""),
                        "harness_python": str(runtime.get("python") or ""),
                    }
                    return AgentEpisode(
                        response=response,
                        observations=self._sandbox_observations(
                            result,
                            model_calls,
                            output.get("observations"),
                        ),
                    )
                if state_name == "exited":
                    logs = await state.sandbox.exec(
                        f"cat {quote(stderr_path)} 2>/dev/null || true",
                        cwd=state.workdir,
                        timeout_s=30,
                    )
                    raise RuntimeError(f"Hermes sandbox runner exited without output: {logs.stdout or ''}")
                await asyncio.sleep(self.config.sandbox_runner_poll_seconds)
        finally:
            await self._terminate_sandbox_runner(state)

    def _response_from_result(
        self,
        *,
        body: NeMoGymResponseCreateParamsNonStreaming,
        result: dict[str, Any],
        model_name: str,
        fail_on_error: bool,
        interrupted_by_dispatch: bool = False,
        n_input: int = 0,
    ) -> NeMoGymResponse:
        if fail_on_error and result.get("error"):
            raise RuntimeError(f"Hermes agent failed: {result['error']}")

        messages = result.get("messages") or []
        output_items = _trajectory_to_output_items(messages, n_input)
        has_assistant_message = any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        )
        if not has_assistant_message:
            LOG.warning(
                "Hermes agent ended without an assistant message. Padding empty assistant message: error=%r",
                result.get("error"),
            )
            last_valid = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, dict)
                    and message.get("role") == "assistant"
                    and message.get("generation_token_ids")
                ),
                None,
            )
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text=result.get("error") or "", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=last_valid["prompt_token_ids"] if last_valid else [],
                    generation_token_ids=last_valid["generation_token_ids"] if last_valid else [],
                    generation_log_probs=(last_valid.get("generation_log_probs") if last_valid else None) or [],
                    routed_experts=last_valid.get("routed_experts") if last_valid else None,
                )
            )

        agent_completed = bool(result.get("completed", True))
        was_interrupted = bool(result.get("interrupted")) or interrupted_by_dispatch
        harness_error = result.get("error")
        agent_failed = bool(harness_error) or bool(result.get("failed"))
        metadata: dict[str, str] = {
            "interrupted": "true" if was_interrupted else "false",
            "failed": "true" if result.get("failed") else "false",
            "partial": "true" if result.get("partial") else "false",
        }
        if isinstance(result.get("api_calls"), int):
            metadata["turns"] = str(result["api_calls"])
        if harness_error:
            metadata["hermes_error"] = str(harness_error)[:2000]

        response_error = None
        if harness_error:
            from openai.types.responses import ResponseError  # pyright: ignore[reportMissingImports]

            response_error = ResponseError(code="server_error", message=str(harness_error)[:2000])

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            status="failed" if agent_failed else ("completed" if agent_completed else "incomplete"),
            error=response_error,
            metadata=metadata,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
        observation_collector: Optional[Callable[[AgentObservationBundle], None]] = None,
    ) -> NeMoGymResponse:
        from run_agent import AIAgent  # from hermes-agent on path  # pyright: ignore[reportMissingImports]

        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, history, input_system = _split_input_to_user_and_history(body.input)
        system_message = self.config.system_prompt or input_system

        base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        model_name = self._model_name()

        agent = AIAgent(
            base_url=base_url,
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY", "gym"),  # pragma: allowlist secret
            model=model_name,
            use_streaming=False,
            temperature=self.config.temperature,
            insert_reasoning=True,
            max_iterations=self.config.max_turns,
            max_tokens=self.config.max_tokens,
            enabled_toolsets=self.config.enabled_toolsets,
            disabled_toolsets=self.config.disabled_toolsets,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
            save_trajectories=False,
        )
        _original_build_api_kwargs = agent._build_api_kwargs

        def _patched_build_api_kwargs(api_messages):
            kw = _original_build_api_kwargs(api_messages)
            if not self.config.chat_template_kwargs_enabled:
                return kw
            ctk = kw.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})
            ctk.setdefault("enable_thinking", True)
            ctk["truncate_history_thinking"] = False
            return kw

        agent._build_api_kwargs = _patched_build_api_kwargs
        observer = None
        if observation_collector is not None:
            try:
                observer = HermesAgentObserver(model_ref=self.config.model_server).instrument(agent)
            except Exception:
                LOG.exception("failed to initialize Hermes observability")

        # Interrupt the agent cleanly on SIGTERM so run_conversation returns with partial messages
        # instead of being killed mid-turn (which would leave response.json unwritten). A single
        # shared dispatcher interrupts every in-flight agent; we just register this one in the set.
        self._ensure_sigterm_handler()
        agent_id = id(agent)
        self.active_agents.add(agent)

        result = None
        agent_error: Optional[BaseException] = None
        interrupted_by_dispatch = False
        try:
            result = await asyncio.to_thread(
                agent.run_conversation,
                user_message,
                system_message,
                history,
                task_id=None,
            )
        except BaseException as exc:
            agent_error = exc
            raise
        finally:
            self.active_agents.discard(agent)
            interrupted_by_dispatch = agent_id in self.interrupted_agents
            self.interrupted_agents.discard(agent_id)
            if observation_collector is not None:
                try:
                    observations = (
                        observer.finish(result, error=agent_error)
                        if observer is not None
                        else AgentObservationBundle(
                            source="hermes",
                            gaps=[ObservationGap(code="observation_capture_failed")],
                        )
                    )
                except Exception:
                    LOG.exception("failed to finish Hermes observability")
                    observations = AgentObservationBundle(
                        source="hermes",
                        gaps=[ObservationGap(code="observation_capture_failed")],
                    )
                try:
                    observation_collector(observations)
                except Exception:
                    LOG.exception("failed to return Hermes observations")

        return self._response_from_result(
            body=body,
            result=result,
            model_name=model_name,
            fail_on_error=False,
            interrupted_by_dispatch=interrupted_by_dispatch,
            n_input=len(history) + 1,
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        agent_session_id = self._agent_session_id_from_request(request)
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        if isinstance(agent_session_id, str):
            if not isinstance(rollout_id, str):
                raise ValueError("Agent sessions require an attempt-qualified rollout path")
            state = self._require_agent_session(agent_session_id)
            if state.request.episode_id.capture_key != rollout_id:
                raise ValueError("Agent-session episode_id does not match the rollout route")
            episode = await self._run_sandbox_episode(
                request=request,
                body=body,
                agent_session_id=agent_session_id,
                state=state,
            )
            state.observations = episode.observations
            return episode.response
        if not isinstance(rollout_id, str):
            return await self._create_response(body)
        episode = await self._create_episode(body, rollout_id=rollout_id)
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: str,
    ) -> AgentEpisode:
        observations: Optional[AgentObservationBundle] = None

        def collect(bundle: AgentObservationBundle) -> None:
            nonlocal observations
            observations = bundle

        response = await self._create_response(
            body,
            rollout_id=rollout_id,
            observation_collector=collect,
        )
        if observations is None:
            observations = AgentObservationBundle(
                source="hermes",
                gaps=[ObservationGap(code="observation_capture_failed")],
            )
        observations.gaps.append(
            ObservationGap(
                code=(
                    "no_sandbox_runtime"
                    if self.config.terminal_backend == "local"
                    else "sandbox_observation_unavailable"
                ),
                detail=(
                    None
                    if self.config.terminal_backend == "local"
                    else f"terminal_backend={self.config.terminal_backend}"
                ),
            )
        )
        return AgentEpisode(response=response, observations=observations)

    async def run(self, request: Request, body: HermesAgentRunRequest) -> HermesAgentVerifyResponse:
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

            rollout_id = self.rollout_id_from_run(body)
            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            raw_observations = (
                agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None) if rollout_id is not None else None
            )
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )

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

            result = verify_json | {"turns_used": turns, "finished_naturally": naturally}
            if observations is not None:
                result["ng_agent_observations"] = observations.model_dump(mode="json")
            return HermesAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    HermesAgent.run_webserver()
