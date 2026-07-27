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

import copy
import logging

from fastapi import Body, Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.gymnasium import EnvResetResponse, EnvStepResponse


_LOGGER = logging.getLogger(__name__)


def _validate_tightening_patch(
    *,
    tool_name: str,
    property_name: str,
    original: dict,
    patch: dict,
) -> None:
    """Reject environment patches that add or loosen a task-row schema."""

    supported = {"minimum", "maximum", "enum", "const"}
    unsupported = set(patch) - supported
    if unsupported:
        raise ValueError(
            f"tool_contract override {tool_name}.{property_name} uses unsupported keys: {sorted(unsupported)}"
        )
    if "minimum" in patch:
        value = patch["minimum"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"tool_contract override {tool_name}.{property_name}.minimum must be numeric")
        if "minimum" in original and value < original["minimum"]:
            raise ValueError(f"tool_contract may not loosen {tool_name}.{property_name}.minimum")
    if "maximum" in patch:
        value = patch["maximum"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"tool_contract override {tool_name}.{property_name}.maximum must be numeric")
        if "maximum" in original and value > original["maximum"]:
            raise ValueError(f"tool_contract may not loosen {tool_name}.{property_name}.maximum")
    if "enum" in patch:
        values = patch["enum"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"tool_contract override {tool_name}.{property_name}.enum must be a non-empty list")
        if "enum" in original and not set(values).issubset(set(original["enum"])):
            raise ValueError(f"tool_contract may not loosen {tool_name}.{property_name}.enum")
    if "const" in patch:
        value = patch["const"]
        if "const" in original and value != original["const"]:
            raise ValueError(f"tool_contract may not change {tool_name}.{property_name}.const")
        if "enum" in original and value not in original["enum"]:
            raise ValueError(f"tool_contract may not loosen {tool_name}.{property_name}.const")


def _apply_tool_contract(
    body: NeMoGymResponseCreateParamsNonStreaming,
    contract: object,
) -> NeMoGymResponseCreateParamsNonStreaming:
    """Apply an optional resource-server tool contract to one rollout body.

    Gymnasium task rows normally own their Responses API tool schemas. A
    stateful environment can narrow that surface after reset when the
    episode's tier/topology is known. The contract is intentionally limited
    to filtering function names and tightening existing property schemas; it
    cannot add tools or loosen a task-row bound.
    """

    if contract is None:
        return body
    if not isinstance(contract, dict):
        raise ValueError("tool_contract must be an object")
    allowed_names = contract.get("allowed_names")
    overrides = contract.get("parameter_overrides", {})
    if (
        not isinstance(allowed_names, list)
        or not allowed_names
        or any(not isinstance(name, str) or not name for name in allowed_names)
        or len(set(allowed_names)) != len(allowed_names)
    ):
        raise ValueError("tool_contract.allowed_names must be a non-empty unique string list")
    if not isinstance(overrides, dict):
        raise ValueError("tool_contract.parameter_overrides must be an object")

    allowed = set(allowed_names)
    filtered: list[dict] = []
    for original in body.tools:
        if not isinstance(original, dict):
            raise ValueError("tool_contract can only constrain dictionary function tools")
        name = original.get("name")
        if original.get("type") != "function" or name not in allowed:
            continue
        tool = copy.deepcopy(original)
        per_tool = overrides.get(name, {})
        if not isinstance(per_tool, dict):
            raise ValueError(f"tool_contract override for {name!r} must be an object")
        properties = tool.get("parameters", {}).get("properties", {})
        for property_name, patch in per_tool.items():
            if property_name not in properties or not isinstance(patch, dict):
                raise ValueError(f"tool_contract override {name}.{property_name} must target an existing property")
            _validate_tightening_patch(
                tool_name=name,
                property_name=property_name,
                original=properties[property_name],
                patch=patch,
            )
            # Overrides are produced by the environment and may only tighten
            # the checked-in schema. The resource server remains the final
            # authority and still validates every call.
            properties[property_name].update(copy.deepcopy(patch))
        filtered.append(tool)

    missing = allowed - {tool["name"] for tool in filtered}
    if missing:
        raise ValueError(f"task row is missing tool(s) required by tool_contract: {sorted(missing)}")
    return body.model_copy(update={"tools": filtered})


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
        # A rollout starts a fresh resource-server session. Do not forward
        # caller or reverse-proxy cookies across the internal service boundary;
        # retain only cookies issued by the resource server itself.
        env_cookies = {}
        model_url_path = self.url_path_for_run("/v1/responses", body)

        reset_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/reset",
            json=body.model_dump(),
            cookies=env_cookies,
        )
        await raise_for_status(reset_resp)
        if reset_resp.cookies:
            env_cookies.update(reset_resp.cookies)

        try:
            # A successful reset owns a stateful server slot even if response
            # decoding or schema validation fails, so validation belongs
            # inside the same cleanup boundary as the rollout itself.
            reset_data = EnvResetResponse.model_validate(await get_response_json(reset_resp))
            result = await self._run_open_episode(body, model_url_path, reset_data, env_cookies)
        except BaseException:
            # Preserve the original model/transport/cancellation failure.  A
            # failed best-effort close is logged here; the
            # resource server's own budget and orphan reaper remain the final
            # safety net.
            try:
                close_resp = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/close",
                    json={},
                    cookies=env_cookies,
                )
                await raise_for_status(close_resp)
            except Exception:
                _LOGGER.exception("Failed to close Gymnasium environment after rollout error")
            raise

        try:
            close_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/close",
                json={},
                cookies=env_cookies,
            )
            await raise_for_status(close_resp)
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

    async def _run_open_episode(
        self,
        body: GymnasiumAgentRunRequest,
        model_url_path: str,
        reset_data: EnvResetResponse,
        env_cookies,
    ) -> GymnasiumRunResponse:
        """Drive an already-reset episode; :meth:`run` owns its cleanup."""

        base_body = body.responses_create_params.model_copy(deep=True)
        base_body = _apply_tool_contract(
            base_body,
            (reset_data.info or {}).get("tool_contract"),
        )
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

        for _ in range(self.config.max_steps):
            new_body = base_body.model_copy(update={"input": base_body.input + new_outputs})

            model_resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_resp)
            model_response = NeMoGymResponse.model_validate(await get_response_json(model_resp))
            model_server_cookies = model_resp.cookies
            last_model_response = model_response

            new_outputs.extend(model_response.output)

            usage = accumulate_response_usage(usage, model_response.usage)

            step_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/step",
                json=body.model_dump() | {"response": model_response.model_dump()},
                cookies=env_cookies,
            )
            await raise_for_status(step_resp)
            step_data = EnvStepResponse.model_validate(await get_response_json(step_resp))
            total_reward += step_data.reward
            if step_resp.cookies:
                env_cookies.update(step_resp.cookies)

            if step_data.terminated or step_data.truncated:
                finished = True
                break

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
