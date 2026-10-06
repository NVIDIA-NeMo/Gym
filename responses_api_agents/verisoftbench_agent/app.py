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
"""VeriSoftBench candidate-plus-repair agent.

One rollout is one of upstream's k proof candidates and its repair chain:

1. Generate a candidate from the task's prompt (system + user turn).
2. Verify it with the ``verisoftbench`` resources server.
3. If it failed and rounds remain, send upstream's repair prompt as a fresh
   single-turn request (the same system prompt, upstream's ``fix.txt`` user
   message) and verify the repaired proof, forwarding the previous round's
   proof, lemmas and error so the server can apply upstream's fallbacks.
4. Stop at the first verified round or after ``max_repair_rounds`` repairs.

The rollout's ``reward`` is 1.0 if any round verified. ``solved_without_repair``
records whether round 0 alone verified. The server holds no per-rollout state;
everything a later round needs travels in the request.
"""

import logging
from typing import Any, Optional

from fastapi import Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)

# Upstream: 3 repair rounds per failed candidate.
DEFAULT_MAX_REPAIR_ROUNDS = 3


class VeriSoftBenchAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_repair_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS


class VeriSoftBenchAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class VeriSoftBenchAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class VeriSoftBenchAgentVerifyResponse(BaseVerifyResponse):
    """The final round's verify response plus the rollout-level summary."""

    model_config = ConfigDict(extra="allow")
    solved_without_repair: float = 0.0
    repair_rounds_used: int = 0
    total_rounds: int = 0
    attempts: Optional[list[dict[str, Any]]] = None


def system_messages(params: NeMoGymResponseCreateParamsNonStreaming) -> list[Any]:
    """The system turn(s) of the task prompt; upstream resends the same system prompt on repair."""
    if isinstance(params.input, str):
        return []
    return [
        m
        for m in params.input
        if getattr(m, "role", None) == "system" or (isinstance(m, dict) and m.get("role") == "system")
    ]


def repair_params(
    params: NeMoGymResponseCreateParamsNonStreaming, repair_prompt: str
) -> NeMoGymResponseCreateParamsNonStreaming:
    """A fresh single-turn request with the task's system turn and upstream's repair user turn.

    Every sampling parameter of the original request is kept; only ``input`` changes.
    """
    new_input = [*system_messages(params), NeMoGymEasyInputMessage(role="user", content=repair_prompt)]
    return params.model_copy(update={"input": new_input}, deep=True)


def output_text(response_json: dict[str, Any]) -> str:
    for item in reversed(response_json.get("output") or []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            return "".join(
                part.get("text", "") for part in item.get("content") or [] if part.get("type") == "output_text"
            )
    return ""


class VeriSoftBenchAgent(SimpleResponsesAPIAgent):
    config: VeriSoftBenchAgentConfig

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        for k, v in model_response.cookies.items():
            response.set_cookie(k, v)
        return NeMoGymResponse.model_validate(await model_response.json())

    async def run(self, request: Request, body: VeriSoftBenchAgentRunRequest) -> VeriSoftBenchAgentVerifyResponse:
        cookies = request.cookies
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        attempts: list[dict[str, Any]] = []
        params = body.responses_create_params
        repair_state: Optional[dict[str, Any]] = None
        verify_result: dict[str, Any] = {}
        round_index = 0
        while True:
            gen_response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=params,
                cookies=cookies,
            )
            await raise_for_status(gen_response)
            cookies = gen_response.cookies
            model_response_json = await gen_response.json()

            verify_request = body.model_dump()
            verify_request["response"] = model_response_json
            verify_request["repair_round"] = round_index
            verify_request["repair_state"] = repair_state
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request,
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            cookies = verify_response.cookies
            verify_result = await verify_response.json()

            attempts.append(
                {
                    "repair_round": round_index,
                    "input": params.model_dump(),
                    "response": model_response_json,
                    "generation": output_text(model_response_json),
                    "status": verify_result.get("status"),
                    "reward": verify_result.get("reward", 0.0),
                    "proof": verify_result.get("proof"),
                    "aux_lemmas": verify_result.get("aux_lemmas"),
                    "err_msg": verify_result.get("err_msg"),
                    "lean_time_s": verify_result.get("lean_time_s"),
                }
            )
            LOG.info("task %s round %d: %s", verify_result.get("task_id"), round_index, verify_result.get("status"))

            if verify_result.get("harness_failure", 0.0) >= 1.0 or not verify_result.get("needs_repair", False):
                break
            if round_index >= self.config.max_repair_rounds:
                break
            repair_prompt = verify_result.get("repair_prompt")
            if not repair_prompt:
                LOG.warning("round %d: needs_repair without repair_prompt", round_index)
                break
            params = repair_params(params, repair_prompt)
            repair_state = verify_result.get("repair_state")
            round_index += 1

        final = VeriSoftBenchAgentVerifyResponse.model_validate(verify_result)
        final.solved_without_repair = 1.0 if attempts and attempts[0]["reward"] >= 1.0 else 0.0
        final.repair_rounds_used = round_index
        final.total_rounds = round_index + 1
        final.attempts = attempts
        return final

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server.

        Gym aggregates through the agent endpoint; the generic implementation would never
        reach the server's ``compute_metrics``/``get_key_metrics``, so the benchmark's
        ``any_of_k/*`` estimand and headline selection live or die on this forward.
        """
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    VeriSoftBenchAgent.run_webserver()
