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
import uuid
from collections.abc import Mapping

from fastapi import Body, Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym._checkpoint import (
    RESOURCE_REQUEST_ID_HEADER,
    AgentBoundaryKind,
    AgentBoundaryRecord,
    PendingModelPayload,
)
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import (
    MODEL_CALL_ID_HEADER,
    current_attempt_index,
    current_logical_rollout_id,
    current_rollout_id,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


class ToolSimulationAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class ToolSimulationAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ToolSimulationAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ToolSimulationAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class ToolSimulationAgent(SimpleResponsesAPIAgent):
    config: ToolSimulationAgentConfig
    checkpoint_continuation_supported = True
    checkpoint_resource_dependencies_supported = True

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_response = await self.retry_checkpoint_refusal(
            lambda: self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_request("/v1/responses", request),
                json=body,
            ),
            request=request,
        )

        # Model calls are expected to always succeed.
        await raise_for_status(model_response)
        model_response_json = await get_response_json(model_response)

        headers = getattr(model_response, "headers", None)
        if isinstance(headers, Mapping) and headers.get(MODEL_CALL_ID_HEADER) is not None:
            response.headers[MODEL_CALL_ID_HEADER] = headers[MODEL_CALL_ID_HEADER]

        try:
            return NeMoGymResponse.model_validate(model_response_json)
        except ValidationError as e:
            raise RuntimeError(
                f"Received an invalid response from the model server: {json.dumps(model_response_json)}"
            ) from e

    async def run(
        self,
        request: Request,
        body: ToolSimulationAgentRunRequest = Body(),
    ) -> ToolSimulationAgentVerifyResponse:
        config = self.config
        continuation = self.checkpoint_continuation(body, request)
        execution = self.checkpoint_execution(request)

        async def commit_boundary(record: AgentBoundaryRecord) -> None:
            if execution is not None:
                await self.checkpoint_participant().commit_boundary(execution, record)

        logical_rollout_id = current_logical_rollout_id()
        attempt_index = current_attempt_index()
        if continuation is None and execution is not None:
            if logical_rollout_id is None or attempt_index is None:
                raise RuntimeError("checkpointed tool-simulation execution is missing its rollout identity")
            await commit_boundary(
                AgentBoundaryRecord(
                    rollout_id=logical_rollout_id,
                    attempt_index=attempt_index,
                    boundary_index=0,
                    turn_index=0,
                    boundary_kind=AgentBoundaryKind.TURN_COMPLETE,
                    output_items=[],
                )
            )

        pending_model = continuation.pending_model if continuation is not None else None
        if pending_model is not None:
            response_json = pending_model.response
            verify_request_id = pending_model.resource_request_id
        else:
            execution_headers = self.checkpoint_execution_headers()
            model_response = await self.retry_checkpoint_refusal(
                lambda: self.server_client.post(
                    server_name=config.name,
                    url_path=self.url_path_for_run("/v1/responses", body),
                    json=body.responses_create_params,
                    **({"headers": execution_headers} if execution_headers is not None else {}),
                ),
                request=request,
                checkpointable_model_wait=True,
            )
            await raise_for_status(model_response)
            response_json = await get_response_json(model_response)

            if execution is not None:
                capture_key = current_rollout_id()
                if logical_rollout_id is None or attempt_index is None or capture_key is None:
                    raise RuntimeError("checkpointed tool-simulation model call is missing its rollout identity")
                headers = getattr(model_response, "headers", None)
                model_call_id = headers.get(MODEL_CALL_ID_HEADER) if isinstance(headers, Mapping) else None
                model_call_id = model_call_id or response_json.get("id")
                if not isinstance(model_call_id, str) or not model_call_id:
                    raise RuntimeError("checkpointed tool-simulation model response is missing its model-call ID")
                verify_request_id = uuid.uuid4().hex
                usage = response_json.get("usage")
                await commit_boundary(
                    AgentBoundaryRecord(
                        rollout_id=logical_rollout_id,
                        attempt_index=attempt_index,
                        boundary_index=1,
                        turn_index=1,
                        boundary_kind=AgentBoundaryKind.PENDING_MODEL,
                        pending_model=PendingModelPayload(
                            model_call_id=model_call_id,
                            response=response_json,
                            usage=usage if isinstance(usage, dict) else None,
                            pending_action_cursor=0,
                            resource_request_id=verify_request_id,
                        ),
                        output_items=[],
                        last_committed_model_capture_key=capture_key,
                        last_committed_model_call_id=model_call_id,
                    )
                )
            else:
                verify_request_id = uuid.uuid4().hex

        if config.skip_verification:
            result = body.model_dump() | {
                "response": response_json,
                "reward": float(config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            verify_request = ToolSimulationAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": response_json}
            )

            async def verify() -> dict:
                verify_response = await self.retry_checkpoint_refusal(
                    lambda: self.server_client.post(
                        server_name=config.resources_server.name,
                        url_path="/verify",
                        json=verify_request.model_dump(),
                        **(
                            {"headers": {RESOURCE_REQUEST_ID_HEADER: verify_request_id}}
                            if execution is not None
                            else {}
                        ),
                    ),
                    request=request,
                )
                await raise_for_status(verify_response)
                return await get_response_json(verify_response)

            result = await self.checkpointable_external_wait(verify, request=request)

        return ToolSimulationAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    ToolSimulationAgent.run_webserver()
