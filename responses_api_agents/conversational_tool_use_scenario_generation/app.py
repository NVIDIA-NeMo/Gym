# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate customer scenarios as one Gym rollout per domain."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from random import random
from time import time
from typing import Any, ClassVar, Literal, Optional

from fastapi import Body, Request
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.conversational_tool_use_scenario_generation.assets import (
    ScenarioAssets,
    load_assets,
)


class CustomerScenario(BaseModel):
    """
    Customer scenario. A definition of a customer who is contacting a customer service representative. This includes a general description of the customer, and information about the specific situation the customer is in and the tasks they are trying to complete.
    """

    customer_persona: str = Field(
        description=(
            "Customer's persona. This information defines the customer in general, "
            "not the specific situation they are in."
        ),
        validation_alias=AliasChoices("customer_persona", "persona"),
    )
    reason_for_contact: str = Field(
        description=("The reason for the customer to contact the customer service representative."),
        validation_alias=AliasChoices("reason_for_contact", "reason_for_call"),
    )
    customer_details: str = Field(
        description=(
            "Specific details about the customer that are relevant to the "
            "customer's tasks. This should be information that the customer "
            "can provide to the representative when the representative asks "
            "for details about the customer's tasks."
        )
    )
    unknown_info: Optional[str] = Field(default_factory=lambda: None)
    task_instructions: str = Field(
        description=(
            "Instructions for the customer about the specific situation the "
            "customer is in, the tasks they are trying to complete, and how to "
            "interact with the customer service representative."
        )
    )
    representative_domain: SkipJsonSchema[Optional[str]] = Field(
        description="The domain of the customer service representative.",
        default=None,
    )
    outside_policy_scope: SkipJsonSchema[Optional[bool]] = Field(
        description=(
            "A value that is true if the action that the customer service "
            "representative should take in response to the customer's request is "
            "not covered in the policy (and the representative should transfer the "
            "customer to a human agent), or false if the action that should be "
            "taken is covered in the policy."
        ),
        default=None,
    )

    def create_tuple(self) -> tuple[str, str, str, Optional[str], str]:
        return (
            self.customer_persona,
            self.reason_for_contact,
            self.customer_details,
            self.unknown_info,
            self.task_instructions,
        )


class CustomerScenarioCollection(BaseModel):
    """
    Customer scenario collection. A collection of customer scenarios.
    """

    scenarios: list[CustomerScenario] = Field(
        description="An array that contains the customer scenarios in the collection."
    )


def parse_scenarios(text: str) -> list[CustomerScenario]:
    canonical_text = text.strip().removeprefix("```json").removesuffix("```")
    return CustomerScenarioCollection.model_validate_json(canonical_text).scenarios


class ScenarioGenerationAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef


class ScenarioGenerationRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    domain_name: str
    policy: str
    tools: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioCallTrace(BaseModel):
    request_index: int
    completion_index: int
    outside_policy_scope: bool
    messages: list[dict[str, str]]
    status: Literal["success", "failed"]
    scenarios: list[CustomerScenario] = Field(default_factory=list)
    parsed_scenario_count: int = 0
    omitted_unknown_info_count: int = 0
    accepted_scenario_count: int = 0
    duplicate_scenario_count: int = 0
    raw_chat_completion: Optional[dict[str, Any]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class ScenarioGenerationResult(BaseModel):
    domain_name: str
    scenarios: list[CustomerScenario]


class ScenarioGenerationTrace(BaseModel):
    request_count: int
    scenarios_per_request: int
    outside_policy_scope_fraction: float
    successful_call_count: int
    failed_call_count: int
    calls: list[ScenarioCallTrace]


class ScenarioGenerationVerifyResponse(ScenarioGenerationRunRequest, BaseVerifyResponse):
    result: ScenarioGenerationResult
    generation_trace: ScenarioGenerationTrace


@dataclass
class _CallOutcome:
    trace: ScenarioCallTrace
    chat_completion: Optional[NeMoGymChatCompletion]


class ConversationalToolUseScenarioGenerationAgent(SimpleResponsesAPIAgent):
    REQUEST_COUNT: ClassVar[int] = 20
    SCENARIOS_PER_REQUEST: ClassVar[int] = 80
    OUTSIDE_POLICY_SCOPE_FRACTION: ClassVar[float] = 0.1

    config: ScenarioGenerationAgentConfig

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=body,
        )
        await raise_for_status(model_response)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    async def run(self, body: ScenarioGenerationRunRequest) -> ScenarioGenerationVerifyResponse:
        outside_scope_schedule = [random() < self.OUTSIDE_POLICY_SCOPE_FRACTION for _ in range(self.REQUEST_COUNT)]
        assets = load_assets()
        tasks = [
            asyncio.create_task(
                self._generate_one(
                    body=body,
                    request_index=request_index,
                    outside_policy_scope=outside_policy_scope,
                    assets=assets,
                )
            )
            for request_index, outside_policy_scope in enumerate(outside_scope_schedule)
        ]

        calls: list[ScenarioCallTrace] = []
        accepted: list[CustomerScenario] = []
        accepted_keys: set[tuple[str, str, str, Optional[str], str]] = set()
        last_artifact_completion: Optional[NeMoGymChatCompletion] = None

        for completion_index, task in enumerate(asyncio.as_completed(tasks)):
            outcome = await task
            trace = outcome.trace
            trace.completion_index = completion_index

            accepted_from_call = 0
            duplicate_from_call = 0
            for scenario in trace.scenarios:
                scenario_key = scenario.create_tuple()
                if scenario_key in accepted_keys:
                    duplicate_from_call += 1
                    continue
                accepted_keys.add(scenario_key)
                accepted.append(scenario)
                accepted_from_call += 1
            trace.accepted_scenario_count = accepted_from_call
            trace.duplicate_scenario_count = duplicate_from_call
            calls.append(trace)

            if trace.scenarios and outcome.chat_completion is not None:
                last_artifact_completion = outcome.chat_completion

        result = ScenarioGenerationResult(domain_name=body.domain_name, scenarios=accepted)
        generation_trace = ScenarioGenerationTrace(
            request_count=self.REQUEST_COUNT,
            scenarios_per_request=self.SCENARIOS_PER_REQUEST,
            outside_policy_scope_fraction=self.OUTSIDE_POLICY_SCOPE_FRACTION,
            successful_call_count=sum(call.status == "success" for call in calls),
            failed_call_count=sum(call.status == "failed" for call in calls),
            calls=calls,
        )
        response = (
            self._chat_completion_to_response(body, last_artifact_completion)
            if last_artifact_completion is not None
            else self._empty_response()
        )
        return ScenarioGenerationVerifyResponse.model_validate(
            body.model_dump(mode="json")
            | {
                "response": response.model_dump(mode="json"),
                "reward": 1.0,
                "result": result.model_dump(mode="json"),
                "generation_trace": generation_trace.model_dump(mode="json"),
            }
        )

    async def _generate_one(
        self,
        *,
        body: ScenarioGenerationRunRequest,
        request_index: int,
        outside_policy_scope: bool,
        assets: ScenarioAssets,
    ) -> _CallOutcome:
        system_message = assets.system_prompt.format(
            domain_policy=body.policy,
            policy_scope_instruction=("does not cover" if outside_policy_scope else "covers"),
        )
        user_message = assets.user_prompt.format(
            scenario_count=self.SCENARIOS_PER_REQUEST,
            scenarios_schema=assets.schema,
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        chat_completion: Optional[NeMoGymChatCompletion] = None
        try:
            http_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_run("/v1/chat/completions", body),
                json={"messages": messages},
            )
            await raise_for_status(http_response)
            chat_completion = NeMoGymChatCompletion.model_validate(await get_response_json(http_response))
            if not chat_completion.choices:
                raise ValueError("model returned no choices")
            response_text = chat_completion.choices[0].message.content
            if response_text is None:
                raise ValueError("model returned no text content")
            parsed_scenarios = parse_scenarios(response_text)
            scenarios = []
            omitted_unknown_info_count = 0
            for scenario in parsed_scenarios:
                if "unknown_info" not in scenario.model_fields_set:
                    omitted_unknown_info_count += 1
                    continue
                scenario.representative_domain = body.domain_name
                scenario.outside_policy_scope = outside_policy_scope
                scenarios.append(scenario)
            return _CallOutcome(
                trace=ScenarioCallTrace(
                    request_index=request_index,
                    completion_index=-1,
                    outside_policy_scope=outside_policy_scope,
                    messages=messages,
                    status="success",
                    scenarios=scenarios,
                    parsed_scenario_count=len(parsed_scenarios),
                    omitted_unknown_info_count=omitted_unknown_info_count,
                    raw_chat_completion=chat_completion.model_dump(mode="json"),
                ),
                chat_completion=chat_completion,
            )
        except Exception as exc:
            return _CallOutcome(
                trace=ScenarioCallTrace(
                    request_index=request_index,
                    completion_index=-1,
                    outside_policy_scope=outside_policy_scope,
                    messages=messages,
                    status="failed",
                    raw_chat_completion=(
                        chat_completion.model_dump(mode="json") if chat_completion is not None else None
                    ),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ),
                chat_completion=None,
            )

    def _chat_completion_to_response(
        self,
        body: ScenarioGenerationRunRequest,
        completion: NeMoGymChatCompletion,
    ) -> NeMoGymResponse:
        response_params = body.responses_create_params.model_copy(update={"model": completion.model})
        return ResponsesConverter(return_token_id_information=False).chat_completion_to_response(
            response_params,
            completion,
        )

    def _empty_response(self) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="conversational_tool_use_scenario_generation_empty",
            created_at=time(),
            model=self.config.model_server.name,
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )


if __name__ == "__main__":
    ConversationalToolUseScenarioGenerationAgent.run_webserver()
