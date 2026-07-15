# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""A NeMo Gym agent that runs a serialized Haystack pipeline as its rollout harness.

The Haystack ``Pipeline`` (a YAML that contains a ``haystack.components.agents.agent.Agent`` with
Haystack-side tools) is deserialized and warmed up **once** at startup and shared across all
requests, so expensive component/tool initialization is paid a single time rather than per rollout.
``HaystackAgent.responses()`` drives that shared pipeline with the request's input messages. The
Agent's ``chat_generator`` is a ``NeMoGymResponsesChatGenerator`` that calls a native NeMo Gym model
server, so Haystack's own Agent loop is what repeatedly calls the model. Per-request session state
(cookies, usage) is isolated across concurrent rollouts via ``contextvars`` in the generator. NeMo
Gym contributes the model (via the generator) and verification (via the resources server); tools
live entirely inside the Haystack pipeline.
"""

from pathlib import Path
from typing import Any

from fastapi import Request, Response
from haystack import Pipeline
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.haystack_agent import chat_generator
from responses_api_agents.haystack_agent.chat_generator import (
    NeMoGymResponsesChatGenerator,
    messages_to_responses_input,
    responses_input_to_messages,
)


# Re-exported for backwards compatibility (e.g. the example notebook imports it from ``app``).
__all__ = ["HaystackAgent", "NeMoGymResponsesChatGenerator"]


# Request-body fields the Haystack pipeline owns; never forwarded to the model call. Everything
# else the row set (temperature, max_output_tokens, ...) is forwarded as ``generation_kwargs``.
_PIPELINE_OWNED_FIELDS = {"input", "tools", "instructions", "stream"}


class HaystackAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Path (absolute, or relative to this agent directory) to the serialized Haystack pipeline
    # that defines the Agent and its Haystack-side tools.
    pipeline_yaml: str
    # Name of the Agent component inside the pipeline.
    agent_component_name: str = "agent"


class HaystackAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HaystackAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class HaystackAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class HaystackAgent(SimpleResponsesAPIAgent):
    config: HaystackAgentConfig

    # Deserialized once at startup and shared (safely) across all requests.
    _pipeline: Any = PrivateAttr(default=None)
    _agent: Any = PrivateAttr(default=None)
    _generator: Any = PrivateAttr(default=None)

    def model_post_init(self, context):
        pipeline_path = Path(self.config.pipeline_yaml)
        if not pipeline_path.is_absolute():
            pipeline_path = Path(__file__).parent / pipeline_path
        self._pipeline_text = pipeline_path.read_text()

        # Deserialize (parse YAML + import modules + instantiate every component/tool) and warm up
        # once, so this cost is paid at startup rather than on every rollout. The pipeline is then
        # shared: Haystack's Agent/Pipeline keep all per-run state in locals, and the generator's
        # per-run session state is isolated via contextvars (see chat_generator._current_run_state).
        self._pipeline = Pipeline.loads(self._pipeline_text, unsafe=True)
        self._agent = self._pipeline.get_component(self.config.agent_component_name)
        generator = getattr(self._agent, "chat_generator", None)
        if not isinstance(generator, NeMoGymResponsesChatGenerator):
            raise RuntimeError(
                f"Component '{self.config.agent_component_name}' in {self.config.pipeline_yaml} must be a Haystack "
                f"Agent whose chat_generator is a NeMoGymResponsesChatGenerator, got {type(generator).__name__}."
            )
        # server_name is constant across requests (from config), so set it once here.
        generator.server_name = self.config.model_server.name
        self._generator = generator
        self._pipeline.warm_up()  # idempotent; loads any heavy components a single time.
        return super().model_post_init(context)

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        messages = responses_input_to_messages(body.input)

        # Forward the row's sampling params to every model call. Haystack threads generation_kwargs
        # to the generator, where _build_params applies it after the static kwargs (so request wins).
        generation_kwargs = body.model_dump(exclude_unset=True, exclude=_PIPELINE_OWNED_FIELDS)

        # Run the shared pipeline. Install a request-scoped generator state (seeded with this
        # request's cookies) so concurrent rollouts don't clobber each other's cookies/usage. The
        # generator reads/writes this state through contextvars; because Haystack awaits its
        # run_async directly in this request's event-loop task, the writes are visible here.
        run_state = chat_generator._GenRunState(cookies=request.cookies)
        token = chat_generator._current_run_state.set(run_state)
        try:
            result = await self._pipeline.run_async(
                {self.config.agent_component_name: {"messages": messages, "generation_kwargs": generation_kwargs}}
            )
        finally:
            chat_generator._current_run_state.reset(token)
        all_messages = result[self.config.agent_component_name]["messages"]

        # The Agent returns [<system prompt?>, <seeded input>, <generated ...>]. A configured
        # system_prompt renders to exactly one system message (Haystack validates this), so the
        # generated trajectory is everything after that prefix.
        system_offset = 1 if getattr(self._agent, "system_prompt", None) else 0
        generated = all_messages[system_offset + len(messages) :]
        output_items = messages_to_responses_input(generated, output=True)

        if run_state.last_response is None:
            raise RuntimeError("The Haystack Agent completed without any NeMo Gym model call.")

        model_response = run_state.last_response
        model_response.output = output_items
        model_response.usage = run_state.usage

        # Propagate the incoming (resources-server session) cookies and any model-server
        # cookies so downstream verification stays on the same session.
        for k, v in request.cookies.items():
            response.set_cookie(k, v)
        for k, v in (run_state.cookies or {}).items():
            response.set_cookie(k, v)

        return model_response

    async def run(self, request: Request, body: HaystackAgentRunRequest) -> HaystackAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies

        verify_request = HaystackAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return HaystackAgentVerifyResponse.model_validate(await get_response_json(verify_response))

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    HaystackAgent.run_webserver()
