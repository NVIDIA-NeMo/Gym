# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import tempfile
from os import environ
from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from fastapi import Request
from harbor.agents.terminus_2 import Terminus2
from harbor.models.agent.context import AgentContext
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers.base import SandboxPtySession
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    get_server_url,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)


class Terminus2AgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_turns: int | None
    parser_name: str = "json"
    enable_summarize: bool
    proactive_summarization_threshold: int
    use_responses_api: bool
    tmux_pane_width: int
    tmux_pane_height: int
    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_timeout: float


class Terminus2AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class NeMoGymSandboxEnvironment:
    """The Harbor environment surface used by Terminus 2, backed by AsyncSandbox."""

    def __init__(self, sandbox: AsyncSandbox, logs_dir: Path, pty_session: SandboxPtySession | None = None):
        self._sandbox = sandbox
        self._pty_session = pty_session
        self.default_user = None
        self.trial_paths = SimpleNamespace(agent_dir=logs_dir)

    async def exec(
        self,
        command: str,
        timeout_sec: float | None = None,
        user: str | int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **_: Any,
    ) -> Any:
        if self._pty_session is not None and cwd is None and env is None and user is None:
            result = await self._sandbox.pty.exec(command, session=self._pty_session, timeout_s=timeout_sec)
        else:
            result = await self._sandbox.exec(command, cwd=cwd, env=env, timeout_s=timeout_sec, user=user)
        return SimpleNamespace(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            return_code=result.return_code,
        )

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self._sandbox.exec(f"test -d {json.dumps(path)}", user=user)
        return result.return_code == 0


def _instruction(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    messages: list[str] = []
    for item in input_value or []:
        value = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        if not isinstance(value, dict):
            messages.append(str(value))
            continue
        content = value.get("content", "")
        if isinstance(content, str):
            messages.append(content)
        elif isinstance(content, list):
            messages.extend(
                str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("text")
            )
    return "\n\n".join(messages)


class Terminus2Agent(SimpleResponsesAPIAgent):
    config: Terminus2AgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_sandboxes: dict[str, tuple[AsyncSandbox, SandboxPtySession]] = {}

    async def _connect_sandbox(self, sandbox_id: str, pty_session_id: str) -> tuple[AsyncSandbox, SandboxPtySession]:
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        sandbox = await AsyncSandbox.connect({"sandbox_id": sandbox_id}, provider=provider)
        pty_session = await sandbox.pty.attach(session_id=pty_session_id, takeover=True)
        return sandbox, pty_session

    async def _execute(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming,
        sandbox: AsyncSandbox,
        pty_session: SandboxPtySession | None,
    ) -> NeMoGymResponse:
        instruction = _instruction(body.input)

        model_base_url = (
            self.base_url_for_run(base_url=get_server_url(self.config.model_server.name), body=await request.json())
            + "/v1"
        )
        # Dummy api key for LiteLLM to use
        environ["OPENAI_API_KEY"] = "dummy"

        with tempfile.TemporaryDirectory(prefix="nemo-gym-terminus-2-") as log_dir:
            environment = NeMoGymSandboxEnvironment(sandbox, Path(log_dir), pty_session)
            context = AgentContext()
            agent = Terminus2(
                logs_dir=Path(log_dir),
                model_name=f"openai/{self.config.model_server.name}",
                api_base=model_base_url,
                max_turns=self.config.max_turns,
                parser_name=self.config.parser_name,
                temperature=None,
                reasoning_effort=None,
                enable_summarize=self.config.enable_summarize,
                proactive_summarization_threshold=self.config.proactive_summarization_threshold,
                use_responses_api=self.config.use_responses_api,
                tmux_pane_width=self.config.tmux_pane_width,
                tmux_pane_height=self.config.tmux_pane_height,
                record_terminal_session=False,
                store_all_messages=True,
            )

            await environment.exec("mkdir -p /logs/agent", user="root")
            await agent.setup(environment)

            async with asyncio.timeout(self.config.sandbox_timeout):
                await agent.run(instruction, environment, context)

            await agent._session.stop()

        messages = (context.metadata or {}).get("all_messages", [])
        final_content = ""
        if messages and isinstance(messages[-1], dict):
            final_content = str(messages[-1].get("content") or "")
        usage = NeMoGymResponseUsage(
            input_tokens=context.n_input_tokens or 0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=context.n_cache_tokens or 0),
            output_tokens=context.n_output_tokens or 0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=(context.n_input_tokens or 0) + (context.n_output_tokens or 0),
        )
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model_server.name,
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=final_content, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
        )

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        session_key = request.session[SESSION_ID_KEY]
        sandbox, pty_session = self._session_sandboxes[session_key]
        return await self._execute(request, body, sandbox, pty_session)

    async def run(self, request: Request, body: Terminus2AgentRunRequest) -> Terminus2AgentVerifyResponse:
        cookies = request.cookies
        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies
        seed_session_result = await seed_session_response.json()

        sandbox_id = seed_session_result["sandbox_handle"]
        pty_session_id = seed_session_result["pty_session_id"]

        sandbox, pty_session = await self._connect_sandbox(sandbox_id, pty_session_id)
        session_key = request.session[SESSION_ID_KEY]
        self._session_sandboxes[session_key] = (sandbox, pty_session)

        response = await self._execute(request, body.responses_create_params, sandbox, pty_session)

        verification = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response.model_dump()},
            cookies=cookies,
        )
        await raise_for_status(verification)

        self._session_sandboxes.pop(session_key)
        await pty_session.close()
        await sandbox.stop()

        return Terminus2AgentVerifyResponse.model_validate(await get_response_json(verification))


if __name__ == "__main__":
    Terminus2Agent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Terminus2Agent.run_webserver()  # noqa: F401
