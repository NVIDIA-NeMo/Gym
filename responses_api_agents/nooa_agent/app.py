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

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import aiohttp
from fastapi import Body, HTTPException, Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_TERMINAL_KEY
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.nooa_agent.config import NOOAAgentConfig
from responses_api_agents.nooa_agent.observability import adapt_response_for_verify, finalize_observations
from responses_api_agents.nooa_agent.runner import EmbeddedNOOARunner, NOOARunRequest, NOOARunResult


NOOA_TERMINATION_REASON_KEY = "nooa_termination_reason"
NOOA_TERMINATION_ERROR_KEY = "nooa_termination_error"


class _TransientInfrastructureError(RuntimeError):
    """Marks a retryable failure from a downstream service."""


class _EpisodeTimeoutExceeded(TimeoutError):
    """Marks expiration of the configured NOOA episode budget."""


def _is_transient_infrastructure_error(error: BaseException) -> bool:
    """Apply Stirrup's retry policy to downstream HTTP and connection failures."""

    if isinstance(error, aiohttp.ClientResponseError):
        return 500 <= error.status < 600
    if isinstance(error, aiohttp.ClientConnectionError):
        return True
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return any(
        _is_transient_infrastructure_error(nested)
        for nested in (error.__cause__, error.__context__)
        if nested is not None
    )


class NOOAAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NOOAAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


def _merge_cookies(current: dict[str, str], response: Any) -> dict[str, str]:
    current.update({name: morsel.value for name, morsel in response.cookies.items()})
    return current


class NOOAAgent(SimpleResponsesAPIAgent):
    """Embedded NOOA adapter that keeps Gym authoritative for every external interaction."""

    config: NOOAAgentConfig
    sem: Any = None
    runner: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)
        self.runner = EmbeddedNOOARunner(
            invocation=self.config.nooa,
            server_client=self.server_client,
            model_server_name=self.config.model_server.name,
            resources_server_name=self.config.resources_server.name,
            max_steps=self.config.max_steps,
        )
        super().model_post_init(context)

    async def _run_episode(
        self,
        body: NOOAAgentRunRequest,
        *,
        model_url_path: str,
        model_cookies: dict[str, str],
        resource_cookies: dict[str, str],
        rollout_id: str | None = None,
    ) -> NOOARunResult:
        rollout_id = rollout_id or maybe_rollout_id_from_run_body(body) or uuid4().hex
        return await self.runner.run(
            NOOARunRequest(
                row=body,
                rollout_id=rollout_id,
                model_url_path=model_url_path,
                model_cookies=model_cookies,
                resource_cookies=resource_cookies,
            )
        )

    def _finalize_run_result(self, run_result: NOOARunResult) -> tuple[NeMoGymResponse, AgentObservationBundle]:
        verify_response, verify_gaps = adapt_response_for_verify(run_result.episode.response, run_result.return_value)
        observations = finalize_observations(
            run_result.episode.observations,
            extra_gaps=verify_gaps,
            termination_reason=run_result.termination_reason,
            termination_error=run_result.termination_error,
        )
        return verify_response, observations

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        run_body = NOOAAgentRunRequest(responses_create_params=body)
        cookies = dict(request.cookies)
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        try:
            async with self.sem, asyncio.timeout(self.config.run_timeout_secs):
                run_result = await self._run_episode(
                    run_body,
                    model_url_path=self.url_path_for_request("/v1/responses", request),
                    model_cookies=dict(cookies),
                    resource_cookies=dict(cookies),
                    rollout_id=rollout_id if isinstance(rollout_id, str) else None,
                )
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail=f"NOOA argument mapping failed for /v1/responses: {error}",
            ) from error
        for name, value in (run_result.model_cookies | run_result.resource_cookies).items():
            response.set_cookie(name, value)
        return run_result.episode.response

    async def run(
        self,
        request: Request,
        response: Response,
        body: NOOAAgentRunRequest,
    ) -> NOOAAgentVerifyResponse:
        record = body.model_dump()
        try:
            async with self.sem:
                result = await self._run_once(request, body, record)
        except _TransientInfrastructureError as error:
            cause = error.__cause__ or error
            result = self._failure_response(
                record,
                f"{type(cause).__name__}: {cause}",
                failure_class="transient",
            )
        except _EpisodeTimeoutExceeded:
            result = self._failure_response(
                record,
                f"NOOA episode exceeded run_timeout_secs={self.config.run_timeout_secs}s",
                failure_class="timeout_exceeded",
                terminal=True,
            )
        except Exception as error:  # noqa: BLE001 -- isolate one rollout from the batch
            result = self._failure_response(
                record,
                f"{type(error).__name__}: {error}",
                failure_class="legitimate",
            )

        for name, value in (result.model_extra or {}).pop("_response_cookies", {}).items():
            response.set_cookie(name, value)
        return result

    async def _run_once(
        self,
        request: Request,
        body: NOOAAgentRunRequest,
        record: dict[str, Any],
    ) -> NOOAAgentVerifyResponse:
        try:
            return await self._run_once_unclassified(request, body, record)
        except _EpisodeTimeoutExceeded:
            raise
        except Exception as error:
            if _is_transient_infrastructure_error(error):
                raise _TransientInfrastructureError(str(error)) from error
            raise

    async def _run_once_unclassified(
        self,
        request: Request,
        body: NOOAAgentRunRequest,
        record: dict[str, Any],
    ) -> NOOAAgentVerifyResponse:
        resource_cookies = dict(request.cookies)
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=record,
            cookies=resource_cookies,
        )
        await raise_for_status(seed)
        _merge_cookies(resource_cookies, seed)

        try:
            async with asyncio.timeout(self.config.run_timeout_secs) as episode_timeout:
                run_result = await self._run_episode(
                    body,
                    model_url_path=self.url_path_for_run("/v1/responses", body),
                    model_cookies=dict(request.cookies),
                    resource_cookies=resource_cookies,
                )
        except TimeoutError as error:
            if not episode_timeout.expired():
                raise
            raise _EpisodeTimeoutExceeded from error

        projected, observations = self._finalize_run_result(run_result)
        response_json = projected.model_dump(mode="json")
        if self.config.skip_verification:
            result: dict[str, Any] = record | {
                "response": response_json,
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            verify = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=record | {"response": response_json},
                cookies=resource_cookies,
            )
            await raise_for_status(verify)
            _merge_cookies(resource_cookies, verify)
            result = await get_response_json(verify)
        if run_result.termination_reason is not None:
            result[NOOA_TERMINATION_REASON_KEY] = run_result.termination_reason
            result[NOOA_TERMINATION_ERROR_KEY] = run_result.termination_error
        result["ng_agent_observations"] = observations.model_dump(mode="json")
        result["_response_cookies"] = run_result.model_cookies | run_result.resource_cookies
        return NOOAAgentVerifyResponse.model_validate(result)

    def _failure_response(
        self,
        record: dict[str, Any],
        error: str,
        *,
        failure_class: str,
        terminal: bool = False,
    ) -> NOOAAgentVerifyResponse:
        response = NeMoGymResponse(
            id="nooa_agent_failure",
            created_at=0.0,
            model="nooa",
            object="response",
            output=[
                {
                    "id": "nooa_failure_message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "", "annotations": []}],
                }
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        routing: dict[str, Any] = {
            NG_FAILURE_CLASS_KEY: failure_class,
            "error": error,
        }
        if terminal:
            routing[NG_TERMINAL_KEY] = True
        return NOOAAgentVerifyResponse.model_validate(
            record | {"response": response.model_dump(mode="json"), "reward": 0.0} | routing
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)
        async with asyncio.timeout(self.config.run_timeout_secs):
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/aggregate_metrics",
                json=body,
            )
            await raise_for_status(response)
            return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    NOOAAgent.run_webserver()
