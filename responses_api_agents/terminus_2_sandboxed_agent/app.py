# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path
from time import perf_counter, time
from traceback import format_exc
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Request
from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.agent.context import AgentContext
from harbor.models.metric.usage_info import UsageInfo
from harbor.utils.logger import logger as harbor_logger
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
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
    tmux_pane_width: int
    tmux_pane_height: int
    dump_trajectory: bool = False
    debug: bool = False
    model_context_limit: int = 1_000_000
    model_output_limit: int | None = None

    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    sandbox_timeout: float
    remote_tmux_binary_path: Optional[str]


class Terminus2AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    terminus2_completed: bool
    command_exec_times: List[float]
    model_call_times: List[float]
    average_command_exec_time: float
    average_model_call_time: float
    total_command_exec_time: float
    total_model_call_time: float
    command_exec_time_pct: float
    model_call_time_pct: float
    terminus2_time_taken: float
    model_calls_gt_10min: int
    num_compactions: int


class NeMoGymSandboxEnvironment:
    """The Harbor environment surface used by Terminus 2, backed by AsyncSandbox."""

    def __init__(self, sandbox: AsyncSandbox, logs_dir: Path, session_id: str):
        self._sandbox = sandbox
        self.default_user = None
        self.trial_paths = SimpleNamespace(agent_dir=logs_dir)
        self.session_id = session_id

    async def exec(
        self,
        command: str,
        timeout_sec: float | None = None,
        user: str | int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **_: Any,
    ) -> Any:
        result = await self._sandbox.exec(command, timeout_s=timeout_sec, cwd=cwd, user=user, env=env)

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


class NeMoGymLLM(BaseLLM):
    """Responses-only Harbor LLM adapter backed by NeMo Gym's aiohttp client."""

    def __init__(
        self,
        client: NeMoGymAsyncOpenAI,
        model_name: str,
        model_context_limit: int,
        model_output_limit: int | None,
    ):
        super().__init__()
        self._client = client
        self._model_name = model_name
        self._model_context_limit = model_context_limit
        self._model_output_limit = model_output_limit
        self.trajectory: list[NeMoGymResponseOutputItem] = []
        self._times_spent = []
        self._last_input_items = []
        self._model_calls_gt_10min = 0

    @staticmethod
    def _input_items(message_history: list[dict[str, Any]], prompt: str) -> list[NeMoGymEasyInputMessage]:
        messages = [*message_history, {"role": "user", "content": prompt}]
        return [
            NeMoGymEasyInputMessage(role=message.get("role", "user"), content=message.get("content", ""))
            for message in messages
        ]

    @staticmethod
    def _response_text(response: NeMoGymResponse) -> tuple[str, str | None]:
        content: list[str] = []
        reasoning: list[str] = []
        for item in response.output:
            if isinstance(item, NeMoGymResponseOutputMessage):
                content.extend(part.text for part in item.content if isinstance(part, NeMoGymResponseOutputText))
            elif isinstance(item, NeMoGymResponseReasoningItem):
                reasoning.extend(summary.text for summary in item.summary)
        return "".join(content), "\n".join(reasoning) or None

    async def call(self, prompt: str, **kwargs: Any) -> LLMResponse:
        message_history = kwargs.pop("message_history", [])
        kwargs.pop("previous_response_id", None)
        kwargs.pop("logging_path", None)
        if kwargs:
            raise NotImplementedError(f"NeMoGymLLM does not support call options: {sorted(kwargs)}")

        input_items = self._input_items(message_history, prompt)
        response = None
        start_time = perf_counter()
        max_attempts = 3  # Hardcode 3 attempts for now
        for attempt in range(max_attempts):
            try:
                async with asyncio.timeout(delay=60 * 10):  # Hardcoded to match litellm default timeout
                    response = NeMoGymResponse.model_validate(
                        await self._client.create_response(
                            model=self._model_name,
                            input=[item.model_dump(mode="json", exclude_none=True) for item in input_items],
                        )
                    )
                    break
            except TimeoutError:
                self._model_calls_gt_10min += 1
                print(
                    f"Hit LiteLLM default 10min timeout on model call, attempt {attempt + 1} / {max_attempts}",
                    file=sys.stderr,
                )
        self._times_spent.append(perf_counter() - start_time)
        if not response:
            raise TimeoutError(f"Failed to query model endpoint due to timeouts after {max_attempts} attempts!")

        if len(self._last_input_items) >= len(input_items):
            # Compacted
            self.trajectory.extend([*input_items, *response.output])
        else:
            self.trajectory.extend([input_items[-1], *response.output])
        self._last_input_items = input_items.copy()

        usage = response.usage
        usage_info = None
        if usage is not None:
            usage_info = UsageInfo(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                cache_tokens=usage.input_tokens_details.cached_tokens or 0,
                cost_usd=0.0,
            )
        content, reasoning_content = self._response_text(response)
        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            model_name=response.model,
            usage=usage_info,
            response_id=response.id,
        )

    def get_model_context_limit(self) -> int:
        return self._model_context_limit

    def get_model_output_limit(self) -> int | None:
        return self._model_output_limit


class NeMoGymTerminus2(Terminus2):
    """Terminus 2 with NeMo Gym model calls and optional Harbor file trajectories."""

    def __init__(self, *args: Any, llm: NeMoGymLLM, dump_trajectory: bool, **kwargs: Any):
        self._nemo_gym_llm = llm
        self._dump_trajectory_enabled = dump_trajectory
        self._times_spent = []
        self._num_compactions = 0
        super().__init__(*args, **kwargs)

    def _init_llm(self, *args: Any, **kwargs: Any) -> BaseLLM:
        return self._nemo_gym_llm

    def _count_total_tokens(self, chat: Any) -> int:
        return sum(len(str(message.get("content", ""))) // 4 for message in chat.messages)

    def _dump_trajectory_with_continuation_index(self, continuation_index: int) -> None:
        if self._dump_trajectory_enabled:
            super()._dump_trajectory_with_continuation_index(continuation_index)

    async def _execute_commands(self, *args, **kwargs):
        start_time = perf_counter()
        res = await super()._execute_commands(*args, **kwargs)
        self._times_spent.append(perf_counter() - start_time)

        return res

    async def _check_proactive_summarization(self, *args, **kwargs):
        res = await super()._check_proactive_summarization(*args, **kwargs)
        if res:
            self._num_compactions += 1
        return res


class Terminus2Agent(SimpleResponsesAPIAgent):
    config: Terminus2AgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_sandboxes: dict[str, AsyncSandbox] = {}

        if not self.config.debug:
            harbor_logger.setLevel(logging.WARNING)

    async def _connect_sandbox(self, sandbox_id: str) -> AsyncSandbox:
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        sandbox = await AsyncSandbox.connect({"sandbox_id": sandbox_id}, provider=provider)
        return sandbox

    async def _execute(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming,
        sandbox: AsyncSandbox,
    ) -> Tuple[NeMoGymResponse, Dict[str, Any]]:
        start_time = perf_counter()
        instruction = _instruction(body.input)

        model_base_url = (
            self.base_url_for_run(base_url=get_server_url(self.config.model_server.name), body=await request.json())
            + "/v1"
        )
        llm = NeMoGymLLM(
            client=NeMoGymAsyncOpenAI(base_url=model_base_url, api_key="dummy", internal=True),
            model_name=self.config.model_server.name,
            model_context_limit=self.config.model_context_limit,
            model_output_limit=self.config.model_output_limit,
        )

        with tempfile.TemporaryDirectory(prefix="nemo-gym-terminus-2-") as log_dir:
            environment = NeMoGymSandboxEnvironment(sandbox, Path(log_dir), request.session[SESSION_ID_KEY])
            context = AgentContext()
            agent = NeMoGymTerminus2(
                logs_dir=Path(log_dir),
                model_name=self.config.model_server.name,
                max_turns=self.config.max_turns,
                parser_name=self.config.parser_name,
                enable_summarize=self.config.enable_summarize,
                proactive_summarization_threshold=self.config.proactive_summarization_threshold,
                tmux_pane_width=self.config.tmux_pane_width,
                tmux_pane_height=self.config.tmux_pane_height,
                record_terminal_session=False,
                llm=llm,
                dump_trajectory=self.config.dump_trajectory,
            )

            await environment.exec("mkdir -p /logs/agent", user="root")
            if self.config.remote_tmux_binary_path:
                # We add the /usr/local/bin path at the end to not supersede and pre-existing orderings.
                tmux_install_result = await sandbox.exec(
                    f"""mkdir -p /usr/local/bin \
&& cp {self.config.remote_tmux_binary_path} /usr/local/bin/tmux \
&& chmod +x /usr/local/bin/tmux \
&& export PATH=$PATH:/usr/local/bin \
&& tmux -V""",
                )
                assert tmux_install_result.return_code == 0, tmux_install_result
            else:
                print(
                    "Downloading and installing tmux in the sandbox. Please consider mounting or uploading the appropriate tmux binary instead!",
                    file=sys.stderr,
                )
            await agent.setup(environment)

            try:
                async with asyncio.timeout(self.config.sandbox_timeout):
                    await agent.run(instruction, environment, context)
                terminus2_completed = True
            except TimeoutError:
                terminus2_completed = False
            except:
                terminus2_completed = False
                print(f"Hit exception while running Terminus2: {format_exc()}", file=sys.stderr)
            finally:
                pass

        usage = NeMoGymResponseUsage(
            input_tokens=context.n_input_tokens or 0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=context.n_cache_tokens or 0),
            output_tokens=context.n_output_tokens or 0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=(context.n_input_tokens or 0) + (context.n_output_tokens or 0),
        )
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model_server.name,
            object="response",
            output=llm.trajectory,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
        )

        total_time = perf_counter() - start_time
        total_command_exec_time = sum(agent._times_spent)
        total_model_call_time = sum(llm._times_spent)
        metrics = {
            "terminus2_completed": terminus2_completed,
            "command_exec_times": agent._times_spent,
            "model_call_times": llm._times_spent,
            "average_command_exec_time": total_command_exec_time / max(len(agent._times_spent), 1),
            "average_model_call_time": total_model_call_time / max(len(llm._times_spent), 1),
            "total_command_exec_time": total_command_exec_time,
            "total_model_call_time": total_model_call_time,
            "command_exec_time_pct": 100 * total_command_exec_time / total_time,
            "model_call_time_pct": 100 * total_model_call_time / total_time,
            "terminus2_time_taken": total_time,
            "model_calls_gt_10min": llm._model_calls_gt_10min,
            "num_compactions": agent._num_compactions,
        }
        return response, metrics

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        session_key = request.session[SESSION_ID_KEY]
        sandbox = self._session_sandboxes[session_key]
        response, _ = await self._execute(request, body, sandbox)
        return response

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

        sandbox = await self._connect_sandbox(sandbox_id)
        session_key = request.session[SESSION_ID_KEY]
        self._session_sandboxes[session_key] = sandbox

        response, metrics = await self._execute(request, body.responses_create_params, sandbox)

        verification = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response.model_dump()},
            cookies=cookies,
        )
        await raise_for_status(verification)

        self._session_sandboxes.pop(session_key)
        try:
            await sandbox.stop()
        except:
            print("Failed to stop sandbox", format_exc(), file=sys.stderr)

        result = await get_response_json(verification)
        result.update(metrics)
        return Terminus2AgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    Terminus2Agent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Terminus2Agent.run_webserver()  # noqa: F401
