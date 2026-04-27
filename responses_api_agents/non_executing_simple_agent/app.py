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
import json
import os
from typing import Any

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

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


def _debug_http_errors_enabled() -> bool:
    return os.environ.get("NEMO_GYM_DEBUG_HTTP_ERRORS", "").lower() in {"1", "true", "yes", "on"}


def _truncate_debug_value(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... [truncated {len(value) - limit} chars]"


def _debug_body_summary(body: Any) -> dict[str, Any]:
    if body is None:
        return {}
    if hasattr(body, "model_dump"):
        body_dict = body.model_dump(exclude_unset=True)
    elif isinstance(body, dict):
        body_dict = body
    else:
        return {"body_type": type(body).__name__}

    responses_create_params = body_dict.get("responses_create_params") or body_dict
    tools = responses_create_params.get("tools") or []
    return {
        "_ng_task_index": body_dict.get("_ng_task_index"),
        "_ng_rollout_index": body_dict.get("_ng_rollout_index"),
        "source_record_id": body_dict.get("source_record_id"),
        "distractor_style": body_dict.get("distractor_style"),
        "tool_schema_mode": body_dict.get("tool_schema_mode"),
        "tool_union_mode": body_dict.get("tool_union_mode"),
        "num_distractors": body_dict.get("num_distractors"),
        "tool_name": body_dict.get("tool_name"),
        "tool_payload_key": body_dict.get("tool_payload_key"),
        "max_output_tokens": responses_create_params.get("max_output_tokens"),
        "num_tools": len(tools),
    }


class NonExecutingSimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class NonExecutingSimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NonExecutingSimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class NonExecutingSimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class NonExecutingSimpleAgent(SimpleResponsesAPIAgent):
    config: NonExecutingSimpleAgentConfig

    async def _raise_for_status_with_debug(self, response: Any, stage: str, body: Any = None) -> None:
        try:
            await raise_for_status(response)
        except Exception:
            if _debug_http_errors_enabled():
                try:
                    response_body = await response.text()
                except Exception as text_error:
                    response_body = f"<failed to read response body: {text_error!r}>"
                print(
                    "[non_executing_simple_agent] downstream HTTP error "
                    f"stage={stage!r} status={getattr(response, 'status', None)} "
                    f"url={getattr(response, 'url', None)} "
                    f"request_summary={json.dumps(_debug_body_summary(body), sort_keys=True)} "
                    f"response_body={_truncate_debug_value(response_body)!r}"
                )
            raise

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
        )
        await self._raise_for_status_with_debug(model_response, "responses:model_server:/v1/responses", body)
        model_response_json = await get_response_json(model_response)

        try:
            parsed_response = NeMoGymResponse.model_validate(model_response_json)
        except ValidationError as e:
            raise RuntimeError(
                f"Received an invalid response from model server: {json.dumps(model_response_json)}"
            ) from e

        # Preserve session cookies for /run verification, but do not inspect or execute tool calls.
        for k, v in (*request.cookies.items(), *model_response.cookies.items()):
            response.set_cookie(k, v)

        return parsed_response

    async def run(
        self,
        request: Request,
        body: NonExecutingSimpleAgentRunRequest,
    ) -> NonExecutingSimpleAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await self._raise_for_status_with_debug(seed_session_response, "run:resources_server:/seed_session", body)
        cookies = seed_session_response.cookies

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await self._raise_for_status_with_debug(response, "run:agent:/v1/responses", body)
        cookies = response.cookies

        verify_request = NonExecutingSimpleAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await self._raise_for_status_with_debug(verify_response, "run:resources_server:/verify", body)
        return NonExecutingSimpleAgentVerifyResponse.model_validate(await get_response_json(verify_response))

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
    NonExecutingSimpleAgent.run_webserver()
