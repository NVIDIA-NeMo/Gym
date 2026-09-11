# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simple-agent loop with opt-in shared semantic context management."""

import json
from typing import Any

from fastapi import Request, Response
from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.context_management import ContextHistoryConfig, ContextManagedResponsesClient, LogicalCCResult
from nemo_gym.context_management.result import capture_rollout_id
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


class SimpleAgentWithCompactionConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = Field(default=256, ge=1)
    token_id_capture: bool = True
    context_history: ContextHistoryConfig = Field(default_factory=ContextHistoryConfig)


class SimpleAgentWithCompactionRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentWithCompactionVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentWithCompactionVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    context_compaction_result: LogicalCCResult


class ContextManagedAgentResponse(NeMoGymResponse):
    context_compaction_result: LogicalCCResult


def _without_raw_images(items: list[dict] | str) -> list[dict] | str:
    """Keep the semantic skeleton; raw assets have one separate transport owner."""
    if isinstance(items, str):
        return items
    projected_items = []
    for item in items:
        projected = dict(item)
        if isinstance(projected.get("content"), list):
            projected["content"] = [
                part
                for part in projected["content"]
                if not (isinstance(part, dict) and part.get("type") in {"input_image", "image", "image_url"})
            ]
        projected_items.append(projected)
    return projected_items


class SimpleAgentWithCompaction(SimpleResponsesAPIAgent):
    config: SimpleAgentWithCompactionConfig

    async def _tool_response_items(self, output: str, call_id: str) -> list:
        """Agent-specific observation decoding; manufactured/image agents may override."""
        return [NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=output)]

    async def _seed_session_response_messages(self, seed_session_response) -> list[NeMoGymEasyInputMessage]:
        return []

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        logical_rollout_id: str,
        resources_server_cookies: Any = None,
        seed_observations: list | None = None,
    ) -> tuple[NeMoGymResponse, LogicalCCResult, dict]:
        client = ContextManagedResponsesClient(
            server_client=self.server_client,
            model_server=self.config.model_server,
            logical_rollout_id=logical_rollout_id,
            config=self.config.context_history,
            initial_request=body,
            seed_observations=seed_observations or (),
        )
        cookies = dict(resources_server_cookies or {})
        usage = None
        outcome = "max_steps"
        for _ in range(self.config.max_steps):
            model_response = await client.create()
            usage = accumulate_response_usage(usage, model_response.usage)
            if model_response.error is not None or model_response.status == "failed":
                outcome = "execution_failure"
                break
            if model_response.incomplete_details:
                outcome = (
                    "max_output_tokens"
                    if model_response.incomplete_details.reason == "max_output_tokens"
                    else "execution_failure"
                )
                break
            calls = [item for item in model_response.output if item.type == "function_call"]
            if not calls and any(
                item.type == "message" and item.role == "assistant" for item in model_response.output
            ):
                outcome = "completed"
                break
            for call in calls:
                try:
                    arguments = json.loads(call.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {exc!r}"})
                else:
                    # Definite HTTP errors are model-visible observations, as in simple_agent.
                    # Transport/read errors propagate: replaying a mutating tool is unsafe.
                    tool_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{call.name}",
                        json=arguments,
                        cookies=cookies,
                        _retry=False,
                    )
                    tool_output = (await tool_response.content.read()).decode()
                    cookies.update(tool_response.cookies)
                try:
                    observations = await self._tool_response_items(tool_output, call.call_id)
                except ValidationError as exc:
                    observations = [
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=call.call_id,
                            output=json.dumps({"error": f"Invalid tool envelope: {exc!r}"}),
                        )
                    ]
                client.append_observation(observations)
        # ID still identifies the last selected action while output accumulates the
        # ordinary semantic conversation for verification.
        result = client.finish(model_response, outcome=outcome)
        response = NeMoGymResponse.model_validate(
            model_response.model_dump() | {"output": client.output_items, "usage": usage}
        )
        cookies.update(client.cookies)
        return response, result, cookies

    def _require_capture(self, logical_rollout_id: str | None) -> str:
        if not self._token_id_capture_enabled() or logical_rollout_id is None:
            raise ValueError("This agent requires training token capture and a framework logical rollout ID")
        capture_rollout_id(logical_rollout_id, 0)
        return logical_rollout_id

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> ContextManagedAgentResponse:
        owner = self._require_capture(request.path_params.get("rollout_id"))
        model_response, result, cookies = await self._create_episode(
            body, logical_rollout_id=owner, resources_server_cookies=request.cookies
        )
        for key, value in cookies.items():
            response.set_cookie(key, value)
        return ContextManagedAgentResponse.model_validate(
            model_response.model_dump() | {"context_compaction_result": result}
        )

    async def run(
        self, request: Request, body: SimpleAgentWithCompactionRunRequest
    ) -> SimpleAgentWithCompactionVerifyResponse:
        owner = self._require_capture(self.rollout_id_from_run(body))
        cookies = dict(request.cookies)
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
            _retry=False,
        )
        await raise_for_status(seed_response)
        cookies.update(seed_response.cookies)
        seed_observations = await self._seed_session_response_messages(seed_response)
        model_response, result, cookies = await self._create_episode(
            body.responses_create_params,
            logical_rollout_id=owner,
            resources_server_cookies=cookies,
            seed_observations=seed_observations,
        )
        payload = body.model_dump() | {"response": model_response.model_dump()}
        if self.config.skip_verification:
            verified = payload | {"reward": float(self.config.skip_verification_reward), "verification_skipped": True}
        else:
            payload["response"]["output"] = [
                *(item.model_dump() for item in seed_observations),
                *payload["response"]["output"],
            ]
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=SimpleAgentWithCompactionVerifyRequest.model_validate(payload).model_dump(),
                cookies=cookies,
                _retry=False,
            )
            await raise_for_status(verify_response)
            verified = await get_response_json(verify_response)
        # Reward belongs to the ordinary verifier; this metadata never grants validity.
        # Project only after verification: the media arena exports each raw image
        # once, while the verifier above still sees every observation occurrence.
        transport_response = model_response.model_dump()
        transport_response["output"] = _without_raw_images(transport_response["output"])
        input_echo = dict(verified["responses_create_params"])
        input_echo["input"] = _without_raw_images(input_echo["input"])
        return SimpleAgentWithCompactionVerifyResponse.model_validate(
            verified
            | {
                "response": transport_response,
                "responses_create_params": input_echo,
                "context_compaction_result": result,
            }
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/aggregate_metrics", json=body, _retry=False
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SimpleAgentWithCompaction.run_webserver()
