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
import signal
import sys
import tempfile
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, Callable, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentObservationBundle,
    ObservationGap,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.hermes_agent.observability import build_hermes_observations
from responses_api_agents.hermes_agent.setup_hermes import ensure_hermes
from responses_api_agents.hermes_agent.trajectory import project_hermes_response_messages


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"
_RESPONSES_CONVERTER = ResponsesConverter(return_token_id_information=False)


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


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in content or []
    )


def _split_chat_messages(messages) -> tuple[str, list[dict], Optional[str]]:
    """Adapt converted Chat Completions messages to Hermes's run_conversation arguments."""
    items = [dict(item) for item in messages]
    system_message: Optional[str] = None
    if items and items[0].get("role") == "system":
        system_message = _content_to_text(items.pop(0).get("content"))

    if items and items[-1].get("role") == "user":
        user_message = _content_to_text(items.pop().get("content"))
    else:
        user_message = ""

    history = items
    return user_message, history, system_message


def _result_to_response(
    body: NeMoGymResponseCreateParamsNonStreaming,
    result: dict[str, Any],
    *,
    model_name: str,
    n_input: int,
) -> NeMoGymResponse:
    messages = project_hermes_response_messages(result.get("messages") or [])
    output_items = _RESPONSES_CONVERTER.chat_completions_messages_to_responses_items(messages[n_input:])

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
            input_tokens=0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        ),
    )


class HermesAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Explicit opt-in: upstream Hermes does not return Gym token fields inline.
    token_id_capture: bool = False
    model: Optional[str] = None
    concurrency: int = 32
    max_turns: int = 90
    max_tokens: Optional[int] = None
    enabled_toolsets: Optional[list[str]] = None
    disabled_toolsets: Optional[list[str]] = None
    temperature: float | None = None
    terminal_backend: str = "local"
    terminal_timeout: int = 180
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
    # Set of managed Hermes child processes, plus a flag tracking whether the single
    # shared SIGTERM dispatcher has been installed on the event loop.
    active_processes: set = None
    sigterm_installed: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _hermes_python: Path = PrivateAttr()

    def _ensure_sigterm_handler(self) -> None:
        """Install one SIGTERM handler that asks every Hermes child to finish its partial trajectory."""
        if self.sigterm_installed:
            return

        def _dispatch():
            for process in list(self.active_processes):
                if process.returncode is None:
                    process.send_signal(signal.SIGTERM)

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
        self.sem = Semaphore(self.config.concurrency)
        self.active_processes = set()
        self._hermes_python = ensure_hermes()

    def _model_name(self) -> str:
        return self.config.model or str(self.config.model_server.name)

    def _request_overrides(self) -> dict[str, Any]:
        """Build request fields that Hermes no longer exposes as constructor arguments."""
        overrides: dict[str, Any] = {}
        if self.config.temperature is not None:
            overrides["temperature"] = self.config.temperature
        if self.config.chat_template_kwargs_enabled:
            overrides["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "truncate_history_thinking": False,
                }
            }
        return overrides

    async def _run_hermes_subprocess(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], AgentObservationBundle | None]:
        with tempfile.TemporaryDirectory(prefix="nemo_gym_hermes_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            hermes_home = temp_dir / "home"
            hermes_home.mkdir()
            (hermes_home / "config.yaml").write_text(self._build_config(), encoding="utf-8")
            request_path = temp_dir / "request.json"
            response_path = temp_dir / "response.json"
            request_path.write_text(json.dumps(payload), encoding="utf-8")

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.update(
                {
                    "HERMES_HOME": str(hermes_home),
                    "HERMES_YOLO_MODE": "1",
                    "HERMES_ACCEPT_HOOKS": "1",
                    "TERMINAL_ENV": self.config.terminal_backend,
                    "TERMINAL_TIMEOUT": str(self.config.terminal_timeout),
                }
            )

            process = await asyncio.create_subprocess_exec(
                str(self._hermes_python),
                str(Path(__file__).with_name("runner.py")),
                str(request_path),
                str(response_path),
                cwd=temp_dir,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            self._ensure_sigterm_handler()
            self.active_processes.add(process)
            try:
                _, stderr = await process.communicate()
            except asyncio.CancelledError:
                if process.returncode is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                raise
            finally:
                self.active_processes.discard(process)

            stderr_text = stderr.decode(errors="replace") if stderr else ""
            if process.returncode != 0:
                raise RuntimeError(
                    f"Hermes runtime exited with status {process.returncode}"
                    + (f": {stderr_text.strip()}" if stderr_text.strip() else "")
                )
            if not response_path.is_file():
                raise RuntimeError("Hermes runtime exited without writing a response")

            raw = json.loads(response_path.read_text(encoding="utf-8"))
            result = raw.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Hermes runtime returned an invalid result")
            raw_observations = raw.get("observations")
            observations = None
            if isinstance(raw_observations, dict):
                try:
                    observations = build_hermes_observations(
                        raw_observations,
                        model_ref=self.config.model_server,
                    )
                except Exception:
                    LOG.exception("failed to project Hermes observations")
                    observations = AgentObservationBundle(
                        source="hermes",
                        gaps=[ObservationGap(code="observation_capture_failed")],
                    )
            return result, observations

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
        observation_collector: Optional[Callable[[AgentObservationBundle], None]] = None,
    ) -> NeMoGymResponse:
        chat_params = _RESPONSES_CONVERTER.responses_to_chat_completion_create_params(body)
        user_message, history, input_system = _split_chat_messages(chat_params.messages)
        system_message = self.config.system_prompt or input_system

        base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        model_name = self._model_name()
        result, observations = await self._run_hermes_subprocess(
            {
                "user_message": user_message,
                "system_message": system_message,
                "history": history,
                "base_url": base_url,
                "api_key": self.config.api_key or os.environ.get("OPENAI_API_KEY", "gym"),  # pragma: allowlist secret
                "model": model_name,
                "max_iterations": self.config.max_turns,
                "max_tokens": self.config.max_tokens,
                "enabled_toolsets": self.config.enabled_toolsets,
                "disabled_toolsets": self.config.disabled_toolsets,
                "request_overrides": self._request_overrides(),
                "capture_observations": observation_collector is not None,
            }
        )
        if observation_collector is not None:
            try:
                observation_collector(
                    observations
                    or AgentObservationBundle(
                        source="hermes",
                        gaps=[ObservationGap(code="observation_capture_failed")],
                    )
                )
            except Exception:
                LOG.exception("failed to return Hermes observations")

        # AIAgent omits the system message from returned messages.
        return _result_to_response(body, result, model_name=model_name, n_input=len(history) + 1)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
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
