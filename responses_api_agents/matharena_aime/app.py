# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MathArena AIME: one policy response and at most one gold-blind format repair."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


# Exact _build_last_chance_prompt(None), including leading/trailing newlines.
# MathArena b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b, runner.py. See NOTICE.
FORMAT_REPAIR_PROMPT = (
    "\nYour last message does not provide a final answer in a way that follows the formatting instructions.\n"
    "Please based on the conversation history, report the final answer again within \\boxed{}.\n"
    "If you did not find the answer, please use \\boxed{None}.\n"
    "Do not reason about the problem again or use tools, simply try to extract the final answer from the previous reasoning.\n"
    "Boxed answers in thinking/reasoning stages will be ignored; only the final response message is considered.\n"
)
_TURNS_KEY = "_ng_matharena_turn_responses"
_CHECK_KEY = "_ng_matharena_format_check"


class FormatRetryDecision(BaseModel):
    needs_format_retry: StrictBool
    parser_warning: int
    valid: StrictBool
    verifier_error: str | None = None


class MathArenaAIMEAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    seed_offset: int = Field(default=0, ge=0)


class MathArenaAIMERunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MathArenaAIMEVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    turn_responses: list[NeMoGymResponse]
    format_retry_check: FormatRetryDecision
    format_retry_count: Literal[0, 1]


def _policy_params(
    body: NeMoGymResponseCreateParamsNonStreaming, *, seed_offset: int
) -> NeMoGymResponseCreateParamsNonStreaming:
    """Offset Gym's per-rollout seed once; preserve standalone unseeded requests."""
    params = body.model_copy(deep=True)
    metadata = params.metadata or {}
    extra_body = json.loads(metadata.get("extra_body", "{}"))
    if not isinstance(extra_body, dict):
        raise ValueError("metadata.extra_body must encode an object")
    if "seed" in extra_body:
        seed = extra_body["seed"]
        if type(seed) is not int:
            raise ValueError("metadata.extra_body.seed must be an integer")
        extra_body["seed"] = seed + seed_offset
        params.metadata = metadata | {"extra_body": json.dumps(extra_body)}
    return params


class MathArenaAIMEAgent(SimpleResponsesAPIAgent):
    """Use the resource's strict parser only to decide whether formatting needs repair."""

    config: MathArenaAIMEAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        params = _policy_params(body, seed_offset=self.config.seed_offset)
        running_input = (
            [NeMoGymEasyInputMessage(role="user", content=params.input)]
            if isinstance(params.input, str)
            else list(params.input)
        )
        chronological_output: list[Any] = []
        turns: list[NeMoGymResponse] = []
        cookies = dict(request.cookies)
        usage = None
        for turn_index in range(2):
            turn_body = params.model_copy(update={"input": list(running_input)})
            http_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_request("/v1/responses", request),
                json=turn_body,
                cookies=dict(cookies),
            )
            await raise_for_status(http_response)
            cookies.update(dict(http_response.cookies or {}))
            try:
                turn = NeMoGymResponse.model_validate(await get_response_json(http_response))
            except ValidationError as exc:
                raise RuntimeError(f"Invalid policy response on MathArena AIME turn {turn_index + 1}") from exc
            turns.append(turn)
            chronological_output.extend(turn.output)
            running_input.extend(turn.output)
            usage = accumulate_response_usage(usage, turn.usage)
            if turn_index:
                break
            check_http = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/needs_format_retry",
                json={"response": turn.model_dump(mode="json")},
                cookies=dict(cookies),
            )
            await raise_for_status(check_http)
            cookies.update(dict(check_http.cookies or {}))
            decision = FormatRetryDecision.model_validate(await get_response_json(check_http))
            if not decision.valid or not decision.needs_format_retry:
                break
            repair = NeMoGymEasyInputMessage(role="user", content=FORMAT_REPAIR_PROMPT)
            running_input.append(repair)
            chronological_output.append(repair)
        for key, value in cookies.items():
            response.set_cookie(key, value)
        combined = turns[-1].model_copy(deep=True)
        combined.output = chronological_output
        combined.usage = usage
        return combined.model_copy(update={_TURNS_KEY: turns, _CHECK_KEY: decision})

    async def run(self, request: Request, body: MathArenaAIMERunRequest) -> MathArenaAIMEVerifyResponse:
        cookies = dict(request.cookies)
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(mode="json"),
            cookies=dict(cookies),
        )
        await raise_for_status(seed)
        cookies.update(dict(seed.cookies or {}))
        http_response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params.model_dump(mode="json"),
            cookies=dict(cookies),
        )
        await raise_for_status(http_response)
        cookies.update(dict(http_response.cookies or {}))
        combined = await get_response_json(http_response)
        turns = [NeMoGymResponse.model_validate(item) for item in combined.pop(_TURNS_KEY, [])]
        decision = FormatRetryDecision.model_validate(combined.pop(_CHECK_KEY, {}))
        retry_count = int(decision.valid and decision.needs_format_retry)
        if len(turns) != retry_count + 1:
            raise RuntimeError("MathArena AIME self-call did not preserve the expected per-turn response audit")
        verification = body.model_dump(mode="json") | {
            "response": combined,
            "turn_responses": [turn.model_dump(mode="json") for turn in turns],
            "format_retry_check": decision.model_dump(mode="json"),
            "format_retry_count": retry_count,
        }
        if self.config.skip_verification:
            result = verification | {
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            verified = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verification,
                cookies=dict(cookies),
            )
            await raise_for_status(verified)
            result = verification | await get_response_json(verified)
        if not decision.valid:
            result.update(
                reward=0.0,
                valid=False,
                mask_sample=True,
                failure_kind="matharena_aime:format_check_failed",
                failure_reason=decision.verifier_error or "The format-retry decision could not be measured",
            )
        return MathArenaAIMEVerifyResponse.model_validate(result)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        if self.config.skip_verification:
            return AggregateMetrics()
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":  # pragma: no cover
    MathArenaAIMEAgent.run_webserver()
