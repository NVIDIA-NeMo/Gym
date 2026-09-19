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
from responses_api_agents.nooa_agent.observability import ensure_verifier_final_message, finalize_observation_gaps
from responses_api_agents.nooa_agent.runner import (
    ArgumentMappingError,
    EmbeddedNOOARunner,
    NOOARunFailure,
    NOOARunRequest,
    NOOARunResult,
)


NOOA_TERMINATION_REASON_KEY = "nooa_termination_reason"
NOOA_TERMINATION_ERROR_KEY = "nooa_termination_error"


class _EpisodeTimeoutExceeded(TimeoutError):
    """Marks expiration of the configured NOOA episode budget."""

    def __init__(self, result: NOOARunResult | None = None) -> None:
        super().__init__("NOOA episode timed out")
        self.result = result


class NOOACookieConflictError(ValueError):
    """Raised when model and resources services return different values for one cookie."""


def _identity(body: NOOAAgentRunRequest, rollout_id: str | None = None) -> dict[str, str]:
    row = body.model_dump()
    task_id = next(
        (
            str(row[key])
            for key in ("task_id", "problem_id", "instance_id", "_ng_task_index")
            if row.get(key) is not None
        ),
        "unknown",
    )
    rollout_id = rollout_id or maybe_rollout_id_from_run_body(body)
    if rollout_id is None and row.get("_ng_task_index") is not None and row.get("_ng_rollout_index") is not None:
        rollout_id = f"{row['_ng_task_index']}-{row['_ng_rollout_index']}"
    return {"task_id": task_id, "rollout_id": rollout_id or uuid4().hex}


def _evidence(result: NOOARunResult, observations: AgentObservationBundle) -> dict[str, Any]:
    fields: dict[str, Any] = {"ng_agent_observations": observations.model_dump(mode="json")}
    if result.trajectory is not None:
        fields["ng_trajectory"] = result.trajectory.model_copy(update={"gaps": observations.gaps}).model_dump(
            mode="json"
        )
    return fields


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
    """Run request that preserves benchmark-specific fields for the NOOA agent."""

    model_config = ConfigDict(extra="allow")


class NOOAAgentVerifyResponse(BaseVerifyResponse):
    """Verification response containing NOOA agent observations and benchmark fields."""

    model_config = ConfigDict(extra="allow")

    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


def _merge_cookies(current: dict[str, str], response: Any) -> None:
    current.update({name: morsel.value for name, morsel in response.cookies.items()})


def _merge_downstream_cookies(model_cookies: dict[str, str], resource_cookies: dict[str, str]) -> dict[str, str]:
    conflicts = sorted(
        name
        for name in model_cookies.keys() & resource_cookies.keys()
        if model_cookies[name] != resource_cookies[name]
    )
    if conflicts:
        names = ", ".join(repr(name) for name in conflicts)
        raise NOOACookieConflictError(
            f"NOOA model and resources services returned conflicting values for cookie(s): {names}"
        )
    return model_cookies | resource_cookies


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

    def _finalize_run_result(self, run_result: NOOARunResult) -> tuple[NeMoGymResponse, AgentObservationBundle]:
        verify_response, verify_gaps = ensure_verifier_final_message(
            run_result.episode.response, run_result.return_value
        )
        observations = finalize_observation_gaps(
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
        try:
            async with self.sem, asyncio.timeout(self.config.run_timeout_secs):
                run_result = await self.runner.run(
                    NOOARunRequest(
                        row=run_body,
                        model_url_path=self.url_path_for_request("/v1/responses", request),
                        model_cookies=dict(cookies),
                        resource_cookies=dict(cookies),
                        **_identity(run_body, request.path_params.get("rollout_id")),
                    )
                )
        except ArgumentMappingError as error:
            raise HTTPException(
                status_code=422,
                detail=f"NOOA argument mapping failed for /v1/responses: {error}",
            ) from error
        for name, value in _merge_downstream_cookies(run_result.model_cookies, run_result.resource_cookies).items():
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
                result = await self._execute_rollout(request, body, record)
        except NOOACookieConflictError:
            raise
        # Preserve the terminal episode timeout: the generic classifier treats its
        # TimeoutError base class as transient.
        except _EpisodeTimeoutExceeded as error:
            result = self._failure_response(
                record,
                f"NOOA episode exceeded run_timeout_secs={self.config.run_timeout_secs}s",
                failure_class="timeout_exceeded",
                terminal=True,
                partial=error.result,
            )
        except Exception as error:  # noqa: BLE001 -- isolate one rollout from the batch
            failure_class = "transient" if _is_transient_infrastructure_error(error) else "legitimate"
            result = self._failure_response(
                record,
                f"{type(error).__name__}: {error}",
                failure_class=failure_class,
                partial=error.result if isinstance(error, NOOARunFailure) else None,
            )

        for name, value in (result.model_extra or {}).pop("_response_cookies", {}).items():
            response.set_cookie(name, value)
        return result

    async def _execute_rollout(
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
                run_result = await self.runner.run(
                    NOOARunRequest(
                        row=body,
                        model_url_path=self.url_path_for_run("/v1/responses", body),
                        model_cookies=dict(request.cookies),
                        resource_cookies=resource_cookies,
                        **_identity(body),
                    )
                )
        except TimeoutError as error:
            if not episode_timeout.expired():
                raise
            raise _EpisodeTimeoutExceeded(getattr(error.__cause__, "nooa_result", None)) from error

        try:
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
                result = record | await get_response_json(verify)
            if run_result.termination_reason is not None:
                result[NOOA_TERMINATION_REASON_KEY] = run_result.termination_reason
                result[NOOA_TERMINATION_ERROR_KEY] = run_result.termination_error
            result.update(_evidence(run_result, observations))
            result["_response_cookies"] = _merge_downstream_cookies(
                run_result.model_cookies, run_result.resource_cookies
            )
            return NOOAAgentVerifyResponse.model_validate(result)
        except NOOACookieConflictError:
            raise
        except Exception as error:
            raise NOOARunFailure(error, run_result) from error

    def _failure_response(
        self,
        record: dict[str, Any],
        error: str,
        *,
        failure_class: str,
        terminal: bool = False,
        partial: NOOARunResult | None = None,
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
        if partial is not None:
            response = partial.episode.response
            observations = finalize_observation_gaps(
                partial.episode.observations, termination_reason=failure_class, termination_error=error
            )
            routing.update(_evidence(partial, observations))
            routing["_response_cookies"] = _merge_downstream_cookies(partial.model_cookies, partial.resource_cookies)
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
