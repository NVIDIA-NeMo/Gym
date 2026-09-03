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
"""A NeMo Gym agent that runs a shared, serialized Haystack pipeline as its rollout harness.

The pipeline is deserialized once and shared across requests. Haystack's ``Agent`` drives the
model/tool loop through ``NeMoGymResponsesChatGenerator``. Configured local and MCP tools remain
on that shared pipeline; a request may additionally supply function schemas which become ephemeral
HTTP tools for that rollout only. Per-rollout state, including separate model-server and
resources-server cookie jars, is stored in context variables so concurrent rollouts cannot leak
session state into each other.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import Request, Response
from haystack import Pipeline, logging
from haystack.tools import flatten_tools_or_toolsets, warm_up_tools
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    NEMO_GYM_MCP_METADATA_KEY,
    NEMO_GYM_MCP_SESSION_TOKEN_HEADER,
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
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.haystack_agent import chat_generator
from responses_api_agents.haystack_agent.chat_generator import (
    NeMoGymResponsesChatGenerator,
    chat_messages_to_responses,
    chat_messages_usage,
    responses_input_to_messages,
)
from responses_api_agents.haystack_agent.http_tool import HTTPTool
from responses_api_agents.haystack_agent.mcp_toolset import (
    close_rollout_mcp_sessions,
    configure_mcp_url,
    context_aware_mcp_tool_names,
    has_context_aware_mcp_toolset,
)


# Re-exported for backwards compatibility (e.g. the example notebook imports it from ``app``).
__all__ = ["HaystackAgent", "NeMoGymResponsesChatGenerator"]


# Request-body fields the Haystack pipeline owns; never forwarded to the model call. Everything
# else the row set (temperature, max_output_tokens, ...) is forwarded as ``generation_kwargs``.
_PIPELINE_OWNED_FIELDS = {"input", "tools", "instructions", "stream"}
logger = logging.getLogger(__name__)


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

    def _get_agent_and_generator(self, pipeline: Any) -> tuple[Any, NeMoGymResponsesChatGenerator]:
        agent = pipeline.get_component(self.config.agent_component_name)
        generator = getattr(agent, "chat_generator", None)
        if not isinstance(generator, NeMoGymResponsesChatGenerator):
            raise RuntimeError(
                f"Component '{self.config.agent_component_name}' in {self.config.pipeline_yaml} must be a Haystack "
                f"Agent whose chat_generator is a NeMoGymResponsesChatGenerator, got {type(generator).__name__}."
            )
        generator.server_name = self.config.model_server.name
        return agent, generator

    def _resources_mcp_url(self) -> str:
        resource_config = get_first_server_config_dict(
            self.server_client.global_config_dict, self.config.resources_server.name
        )
        resources_base_url = self.server_client._build_server_base_url(resource_config)
        return f"{resources_base_url.rstrip('/')}/mcp"

    def _runtime_http_tools(self, schemas: list[Any]) -> list[HTTPTool]:
        http_tools = []
        for schema in schemas:
            if not isinstance(schema, Mapping):
                raise ValueError("HTTP environment tools must be function-tool objects.")
            http_tools.append(HTTPTool(schema, self.server_client, self.config.resources_server.name))

        http_names = [tool.name for tool in http_tools]
        if len(http_names) != len(set(http_names)):
            raise ValueError("HTTP environment tool names must be unique within a request.")
        return http_tools

    def _tools_for_http_request(self, schemas: list[Any]) -> list[Any]:
        """Combine configured tools with request-scoped HTTP tools, preferring MCP on collisions."""
        http_tools = self._runtime_http_tools(schemas)
        http_names = {tool.name for tool in http_tools}

        configured_tools = getattr(self._agent, "tools", None)
        # Haystack can flatten a toolset only after it has discovered its concrete tools. Gym's
        # MCP toolset performs a schema-only ``tools/list`` here; authenticated MCP clients are
        # still created lazily, per rollout, when an MCP tool is actually invoked.
        warm_up_tools(configured_tools)
        mcp_names = context_aware_mcp_tool_names(configured_tools)
        mcp_overrides = http_names & mcp_names
        if mcp_overrides:
            http_tools = [tool for tool in http_tools if tool.name not in mcp_overrides]
        http_names = {tool.name for tool in http_tools}

        tools = []
        for tool in flatten_tools_or_toolsets(configured_tools):
            if tool.name in http_names:
                logger.warning(
                    "HTTP environment tool '{tool_name}' overrides a configured local tool with the same name.",
                    tool_name=tool.name,
                )
                continue
            tools.append(tool)
        return [*tools, *http_tools]

    def _validate_mcp_configuration(self, mcp_enabled: bool) -> None:
        """Require the pipeline's MCP configuration to match this rollout's Resources Server."""
        has_mcp_toolset = has_context_aware_mcp_toolset(getattr(self._agent, "tools", None))
        if has_mcp_toolset and not mcp_enabled:
            raise RuntimeError(
                "The Haystack pipeline configures ContextAwareMCPToolset, but the Resources Server did not "
                "enable MCP for this rollout. Enable expose_tools_over_mcp on the Resources Server or remove "
                "the MCP toolset from the pipeline."
            )
        if mcp_enabled and not has_mcp_toolset:
            logger.warning(
                "The Resources Server enabled MCP for this rollout, but the Haystack pipeline has no "
                "ContextAwareMCPToolset; MCP tools will not be available to the model."
            )

    def model_post_init(self, context):
        pipeline_path = Path(self.config.pipeline_yaml)
        if not pipeline_path.is_absolute():
            pipeline_path = Path(__file__).parent / pipeline_path
        self._pipeline_text = pipeline_path.read_text()

        # Deserialize once. Haystack warms components on the first run; MCP schemas are then
        # shared, while authenticated clients are created from each request's context.
        self._pipeline = Pipeline.loads(self._pipeline_text, unsafe=True)
        self._agent, self._generator = self._get_agent_and_generator(self._pipeline)
        if getattr(self._agent, "user_prompt", None) is not None:
            raise RuntimeError(
                "HaystackAgent pipelines must not set Agent.user_prompt. The Responses request's input is the "
                "complete user context; Haystack appends user_prompt after that input, which would be "
                "mistaken for generated output when the trajectory is reconstructed."
            )
        tools = getattr(self._agent, "tools", None)
        if has_context_aware_mcp_toolset(tools):
            configure_mcp_url(tools, self._resources_mcp_url())
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

        if getattr(self._agent, "system_prompt", None) and any(
            getattr(item, "role", None) == "system" for item in body.input
        ):
            raise ValueError(
                "The Responses request includes a system message, but the Haystack Agent also has a configured "
                "system_prompt. Configure only one system instruction source."
            )

        messages = responses_input_to_messages(body.input)

        # Forward the row's sampling params to every model call. Haystack threads generation_kwargs
        # to the generator, where _build_params applies it after the static kwargs (so request wins).
        generation_kwargs = body.model_dump(exclude_unset=True, exclude=_PIPELINE_OWNED_FIELDS)

        # Run the shared pipeline. Keep server cookie jars separate: ``/run`` seeds the Resources
        # Server session and its cookie must survive every model turn so direct HTTP tools operate
        # on that same environment. Model responses may set unrelated cookies, which are only sent
        # back to the model server on later turns.
        mcp_headers = {}
        session_token = request.headers.get(NEMO_GYM_MCP_SESSION_TOKEN_HEADER)
        if session_token:
            mcp_headers[NEMO_GYM_MCP_SESSION_TOKEN_HEADER] = session_token
        self._validate_mcp_configuration(mcp_enabled=bool(session_token))
        run_state = chat_generator._GenRunState(resources_server_cookies=request.cookies, mcp_headers=mcp_headers)
        token = chat_generator._current_run_state.set(run_state)
        try:
            agent_inputs = {"messages": messages, "generation_kwargs": generation_kwargs}
            if body.tools:
                agent_inputs["tools"] = self._tools_for_http_request(body.tools)
            result = await self._pipeline.run_async({self.config.agent_component_name: agent_inputs})
        finally:
            try:
                close_rollout_mcp_sessions(run_state)
            finally:
                chat_generator._current_run_state.reset(token)
        all_messages = result[self.config.agent_component_name]["messages"]

        # The Agent returns [<system prompt?>, <seeded input>, <generated ...>]. A configured
        # system_prompt renders to exactly one system message (Haystack validates this), so the
        # generated trajectory is everything after that prefix.
        system_offset = 1 if getattr(self._agent, "system_prompt", None) else 0
        generated = all_messages[system_offset + len(messages) :]
        output_items = chat_messages_to_responses(generated, output=True)

        if run_state.last_response is None:
            raise RuntimeError("The Haystack Agent completed without any NeMo Gym model call.")

        model_response = run_state.last_response
        model_response.output = output_items
        model_response.usage = chat_messages_usage(generated)

        # Forward the Resources Server session so ``run()`` can verify against the same state after
        # this internal ``/v1/responses`` call. Preserve model-server cookies separately as well,
        # allowing a model server that uses sessions to continue them on a later request.
        for k, v in request.cookies.items():
            response.set_cookie(k, v)
        for k, v in (run_state.resources_server_cookies or {}).items():
            response.set_cookie(k, v)
        for k, v in (run_state.model_server_cookies or {}).items():
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
        seed_session_json = await get_response_json(seed_session_response)
        cookies = seed_session_response.cookies

        # MCP auto-exposure returns a signed, rollout-scoped token. Forward it only to this
        # request; responses() installs it in the ContextVar used by the shared toolset.
        response_headers: dict[str, str] = {}
        mcp_metadata = seed_session_json.get(NEMO_GYM_MCP_METADATA_KEY)
        if isinstance(mcp_metadata, dict):
            mcp_headers = mcp_metadata.get("headers")
            if isinstance(mcp_headers, dict):
                response_headers = {str(key): str(value) for key, value in mcp_headers.items()}

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
            headers=response_headers,
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
