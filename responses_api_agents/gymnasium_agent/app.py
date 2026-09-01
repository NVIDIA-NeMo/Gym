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

"""Agent for GymnasiumServer resources servers (resources_servers.gymnasium) which implements the Gymnasium API."""

import logging
import uuid
from collections.abc import Mapping
from typing import Optional

from fastapi import Body, Request, Response
from pydantic import ConfigDict, Field, TypeAdapter

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.checkpoint import AgentBoundaryRecord
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputItem,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.rollout_correlation import MODEL_CALL_ID_HEADER, current_attempt_index, current_logical_rollout_id
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.gymnasium import EnvResetResponse, EnvStepResponse


_LOGGER = logging.getLogger(__name__)
_INPUT_ITEMS_ADAPTER = TypeAdapter(list[NeMoGymResponseInputItem])


class GymnasiumAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = Field(10, ge=1)


class GymnasiumAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class GymnasiumRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    terminated: bool = False
    truncated: bool = False
    info: dict = {}


class GymnasiumAgent(SimpleResponsesAPIAgent):
    config: GymnasiumAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_resp = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_resp)
        result = NeMoGymResponse.model_validate(await get_response_json(model_resp))
        for k, v in model_resp.cookies.items():
            response.set_cookie(k, v)
        return result

    async def run(self, request: Request, body: GymnasiumAgentRunRequest) -> GymnasiumRunResponse:
        # Preserve auth/routing cookies and then merge in any session cookies
        # issued by the resource server during the rollout.
        env_cookies = dict(request.cookies)
        model_url_path = self.url_path_for_run("/v1/responses", body)
        continuation = self.checkpoint_continuation(body)

        supports_explicit_close = False
        try:
            if continuation is None:
                reset_resp = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/reset",
                    json=body.model_dump(),
                    cookies=env_cookies,
                )
                await raise_for_status(reset_resp)
                if reset_resp.cookies:
                    env_cookies.update(reset_resp.cookies)
                # A successful reset owns a stateful server slot even if response
                # decoding or schema validation fails.
                reset_payload = await get_response_json(reset_resp)
                if isinstance(reset_payload, dict) and isinstance(reset_payload.get("info"), dict):
                    supports_explicit_close = reset_payload["info"].get("supports_explicit_close") is True
                reset_data = EnvResetResponse.model_validate(reset_payload)
            else:
                reset_data = EnvResetResponse.model_validate(continuation.agent_state["reset_data"])
                env_cookies = dict(continuation.agent_state.get("env_cookies") or env_cookies)
            supports_explicit_close = supports_explicit_close or (
                (reset_data.info or {}).get("supports_explicit_close") is True
            )
            result = await self._run_open_episode(
                body,
                model_url_path,
                reset_data,
                env_cookies,
                continuation=continuation,
            )
        except BaseException:
            if supports_explicit_close:
                # Preserve the original model/transport/cancellation failure.
                try:
                    await self._close_environment(env_cookies)
                except Exception:
                    _LOGGER.exception("Failed to close Gymnasium environment after rollout error")
            raise

        if not supports_explicit_close:
            return result
        try:
            await self._close_environment(env_cookies)
        except Exception as exc:
            _LOGGER.exception("Completed Gymnasium rollout, but environment cleanup failed")
            result = result.model_copy(
                update={
                    "info": {
                        **(result.info or {}),
                        "cleanup_warning": {
                            "operation": "close",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                }
            )
        return result

    async def _close_environment(self, env_cookies) -> None:
        """Close an environment that advertised the optional endpoint."""

        close_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/close",
            json={},
            cookies=env_cookies,
        )
        await raise_for_status(close_resp)

    async def _run_open_episode(
        self,
        body: GymnasiumAgentRunRequest,
        model_url_path: str,
        reset_data: EnvResetResponse,
        env_cookies,
        *,
        continuation: Optional[AgentBoundaryRecord] = None,
    ) -> GymnasiumRunResponse:
        """Drive an already-reset episode; :meth:`run` owns its cleanup."""

        base_body = body.responses_create_params.model_copy(deep=True)
        if isinstance(base_body.input, str):
            base_body.input = [NeMoGymEasyInputMessage(role="user", content=base_body.input)]
        if reset_data.observation:
            base_body.input = list(base_body.input) + [
                NeMoGymEasyInputMessage(role="user", content=reset_data.observation)
            ]

        new_outputs = []
        total_reward = 0.0
        usage = None
        model_server_cookies = None
        step_data = EnvStepResponse(terminated=False, truncated=True, reward=0.0)
        last_model_response = None
        finished = False
        start_step = 0
        if continuation is not None:
            new_outputs.extend(_INPUT_ITEMS_ADAPTER.validate_python(continuation.output_items))
            total_reward = float(continuation.agent_state.get("total_reward", 0.0))
            usage = NeMoGymResponseUsage.model_validate(continuation.usage) if continuation.usage is not None else None
            model_server_cookies = continuation.agent_state.get("model_server_cookies") or None
            step_data = EnvStepResponse.model_validate(continuation.agent_state["step_data"])
            start_step = continuation.boundary_index

        for step in range(start_step + 1, self.config.max_steps + 1):
            new_body = base_body.model_copy(update={"input": base_body.input + new_outputs})

            model_resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            model_call_id = None
            headers = getattr(model_resp, "headers", None)
            if self._checkpoint_participant is not None and isinstance(headers, Mapping):
                model_call_id = headers.get(MODEL_CALL_ID_HEADER)
            await raise_for_status(model_resp)
            model_response = NeMoGymResponse.model_validate(await get_response_json(model_resp))
            model_server_cookies = model_resp.cookies
            last_model_response = model_response

            new_outputs.extend(model_response.output)

            usage = accumulate_response_usage(usage, model_response.usage)

            step_body = body.model_dump() | {"response": model_response.model_dump()}
            if (reset_data.info or {}).get("supports_step_idempotency") is True:
                step_body["_ng_step_request_id"] = uuid.uuid4().hex
            step_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/step",
                json=step_body,
                cookies=env_cookies,
            )
            await raise_for_status(step_resp)
            step_data = EnvStepResponse.model_validate(await get_response_json(step_resp))
            total_reward += step_data.reward
            if step_resp.cookies:
                env_cookies.update(step_resp.cookies)

            for tool_output in (step_data.info or {}).get("tool_outputs", []):
                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_output["call_id"],
                        output=tool_output["output"],
                    )
                )

            if step_data.observation:
                new_outputs.append(NeMoGymEasyInputMessage(role="user", content=step_data.observation))

            if self._checkpoint_participant is not None:
                logical_rollout_id = current_logical_rollout_id()
                attempt_index = current_attempt_index()
                if logical_rollout_id is not None and attempt_index is not None:
                    await self.checkpoint_participant().commit_boundary(
                        AgentBoundaryRecord(
                            rollout_id=logical_rollout_id,
                            attempt_index=attempt_index,
                            boundary_index=step,
                            output_items=[item.model_dump(mode="json") for item in new_outputs],
                            usage=usage.model_dump(mode="json") if usage is not None else None,
                            last_committed_model_call_id=model_call_id,
                            agent_state={
                                "reset_data": reset_data.model_dump(mode="json"),
                                "step_data": step_data.model_dump(mode="json"),
                                "total_reward": total_reward,
                                "model_server_cookies": dict(model_server_cookies or {}),
                                "env_cookies": dict(env_cookies),
                            },
                        )
                    )

            if step_data.terminated or step_data.truncated:
                finished = True
                break

        if not finished:
            step_data = step_data.model_copy(update={"truncated": True})

        last_model_response.output = new_outputs
        last_model_response.usage = usage

        return GymnasiumRunResponse(
            responses_create_params=base_body,
            response=last_model_response,
            reward=total_reward,
            terminated=step_data.terminated,
            truncated=step_data.truncated,
            info=step_data.info,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    GymnasiumAgent.run_webserver()
