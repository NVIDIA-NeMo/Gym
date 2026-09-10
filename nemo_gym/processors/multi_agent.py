# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-robin processing for independently hosted Responses API agents."""

from typing import Any, Literal, Optional

from fastapi import Body, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.agents.responses_api_agent import INTERNAL_TRAJECTORY_KEY
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.config_types import AgentServerRef, AggregateMetrics, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.processors.base import BaseProcessor, BaseProcessorConfig
from nemo_gym.server_utils import get_response_json, raise_for_status


EpisodeEventKind = Literal["response_item", "state", "termination"]


class ParticipantTurn(BaseModel):
    """One attributed agent invocation, including its exact model-visible input."""

    turn_index: int
    participant: str
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    agent_trajectory: Optional[dict[str, Any]] = None


class EpisodeEvent(BaseModel):
    """One ordered participant output, state observation, or termination event."""

    sequence: int
    turn_index: int
    kind: EpisodeEventKind
    participant: Optional[str] = None
    data: dict[str, Any]


class EpisodeStatus(BaseModel):
    """Resources-server response used to stop an episode and expose shared state."""

    model_config = ConfigDict(extra="allow")

    terminated: bool = False
    reason: Optional[str] = None
    state: dict[str, Any] = Field(default_factory=dict)


class MultiAgentRunRequest(BaseRunRequest):
    """Inputs for the focal participant and every independently configured peer."""

    model_config = ConfigDict(extra="allow")

    participant_responses_create_params: dict[str, NeMoGymResponseCreateParamsNonStreaming] = Field(
        default_factory=dict
    )


class MultiAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    focal_participant: str
    participant_trajectories: dict[str, list[ParticipantTurn]]
    episode_trajectory: list[EpisodeEvent]
    termination_reason: str
    turns_completed: int


class MultiAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    focal_participant: str
    participant_trajectories: dict[str, list[ParticipantTurn]]
    episode_trajectory: list[EpisodeEvent]
    termination_reason: str
    turns_completed: int


class MultiAgentProcessorConfig(BaseProcessorConfig):
    participants: dict[str, AgentServerRef] = Field(min_length=2)
    turn_order: list[str] = Field(min_length=2)
    focal_participant: str
    resources_server: ResourcesServerRef
    max_turns: int = Field(8, ge=1)
    status_url_path: str = "/episode_status"

    @model_validator(mode="after")
    def validate_participants(self) -> "MultiAgentProcessorConfig":
        participant_ids = set(self.participants)
        turn_ids = set(self.turn_order)
        unknown = turn_ids - participant_ids
        if unknown:
            raise ValueError(f"turn_order references unknown participants: {sorted(unknown)}")
        missing = participant_ids - turn_ids
        if missing:
            raise ValueError(f"turn_order omits configured participants: {sorted(missing)}")
        if len(turn_ids) != len(self.turn_order):
            raise ValueError("turn_order must contain each participant exactly once")
        if self.focal_participant not in participant_ids:
            raise ValueError(f"focal_participant {self.focal_participant!r} must be present in participants")
        return self


def _input_items(params: NeMoGymResponseCreateParamsNonStreaming) -> list[Any]:
    if isinstance(params.input, str):
        return [NeMoGymEasyInputMessage(role="user", content=params.input)]
    return list(params.input)


def _visible_text(response: NeMoGymResponse) -> str:
    return response.output_text.strip()


class MultiAgentProcessor(BaseProcessor):
    """Run independently configured participants in a validated round-robin order."""

    config: MultiAgentProcessorConfig

    def _params_by_participant(
        self,
        body: MultiAgentRunRequest,
    ) -> dict[str, NeMoGymResponseCreateParamsNonStreaming]:
        if self.config.focal_participant in body.participant_responses_create_params:
            raise ValueError(
                "participant_responses_create_params must not contain focal_participant "
                f"{self.config.focal_participant!r}; use responses_create_params for it"
            )
        unknown = set(body.participant_responses_create_params) - set(self.config.participants)
        if unknown:
            raise ValueError(f"request contains unknown participants: {sorted(unknown)}")
        params = {
            self.config.focal_participant: body.responses_create_params,
            **body.participant_responses_create_params,
        }
        missing = set(self.config.participants) - set(params)
        if missing:
            raise ValueError(f"request is missing response parameters for participants: {sorted(missing)}")
        return params

    async def _call_participant(
        self,
        *,
        participant: str,
        params: NeMoGymResponseCreateParamsNonStreaming,
        body: MultiAgentRunRequest,
        cookies: Any,
    ) -> tuple[NeMoGymResponse, Optional[dict[str, Any]], dict[str, Any]]:
        response = await self.server_client.post(
            server_name=self.config.participants[participant].name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=params,
            cookies=cookies,
        )
        await raise_for_status(response)
        response_json = await get_response_json(response)
        agent_trajectory = response_json.pop(INTERNAL_TRAJECTORY_KEY, None)
        return NeMoGymResponse.model_validate(response_json), agent_trajectory, dict(response.cookies)

    async def _episode_status(self, cookies: Any) -> tuple[EpisodeStatus, dict[str, Any]]:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=self.config.status_url_path,
            json={},
            cookies=dict(cookies),
        )
        await raise_for_status(response)
        return EpisodeStatus.model_validate(await get_response_json(response)), dict(response.cookies)

    async def run(self, request: Request, body: MultiAgentRunRequest) -> MultiAgentVerifyResponse:
        params_by_participant = self._params_by_participant(body)
        environment_cookies = dict(request.cookies)
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(mode="json"),
            cookies=environment_cookies,
        )
        await raise_for_status(seed_response)
        environment_cookies = dict(seed_response.cookies)
        environment_cookie_names = set(environment_cookies)
        participant_cookies: dict[str, dict[str, Any]] = {participant: {} for participant in self.config.participants}
        participant_inputs = {
            participant: _input_items(params) for participant, params in params_by_participant.items()
        }
        trajectories: dict[str, list[ParticipantTurn]] = {participant: [] for participant in self.config.participants}
        events: list[EpisodeEvent] = []
        focal_outputs = []
        last_focal_response: Optional[NeMoGymResponse] = None
        termination_reason = "max_turns"

        for turn_index in range(self.config.max_turns):
            order_index = turn_index % len(self.config.turn_order)
            participant = self.config.turn_order[order_index]
            next_participant = self.config.turn_order[(order_index + 1) % len(self.config.turn_order)]
            participant_params = params_by_participant[participant].model_copy(
                deep=True,
                update={"input": list(participant_inputs[participant])},
            )
            participant_response, agent_trajectory, response_cookies = await self._call_participant(
                participant=participant,
                params=participant_params,
                body=body,
                cookies=environment_cookies | participant_cookies[participant],
            )
            for key, value in response_cookies.items():
                if key in environment_cookie_names:
                    environment_cookies[key] = value
                else:
                    participant_cookies[participant][key] = value
            trajectories[participant].append(
                ParticipantTurn(
                    turn_index=turn_index,
                    participant=participant,
                    request=participant_params,
                    response=participant_response,
                    agent_trajectory=agent_trajectory,
                )
            )
            participant_inputs[participant].extend(participant_response.output)
            if participant == self.config.focal_participant:
                focal_outputs.extend(participant_response.output)
                last_focal_response = participant_response

            for output_item in participant_response.output:
                events.append(
                    EpisodeEvent(
                        sequence=len(events),
                        turn_index=turn_index,
                        kind="response_item",
                        participant=participant,
                        data=output_item.model_dump(mode="json"),
                    )
                )

            visible_text = _visible_text(participant_response)
            if visible_text:
                participant_inputs[next_participant].append(NeMoGymEasyInputMessage(role="user", content=visible_text))
            else:
                incomplete_reason = (
                    participant_response.incomplete_details.reason
                    if participant_response.incomplete_details is not None
                    else None
                )
                termination_reason = (
                    f"{participant}_{incomplete_reason}" if incomplete_reason else f"{participant}_produced_no_text"
                )
                events.append(
                    EpisodeEvent(
                        sequence=len(events),
                        turn_index=turn_index,
                        kind="termination",
                        participant=participant,
                        data={"reason": termination_reason},
                    )
                )
                break

            status, status_cookies = await self._episode_status(environment_cookies)
            environment_cookies.update(status_cookies)
            events.append(
                EpisodeEvent(
                    sequence=len(events),
                    turn_index=turn_index,
                    kind="state",
                    data=status.model_dump(mode="json"),
                )
            )
            if status.terminated:
                termination_reason = status.reason or "environment_terminated"
                events.append(
                    EpisodeEvent(
                        sequence=len(events),
                        turn_index=turn_index,
                        kind="termination",
                        data={"reason": termination_reason},
                    )
                )
                break

        if not events or events[-1].kind != "termination":
            events.append(
                EpisodeEvent(
                    sequence=len(events),
                    turn_index=max(0, sum(len(turns) for turns in trajectories.values()) - 1),
                    kind="termination",
                    data={"reason": termination_reason},
                )
            )

        if last_focal_response is None:
            raise RuntimeError(f"Focal participant {self.config.focal_participant!r} did not produce a response.")

        focal_response = last_focal_response.model_copy(update={"output": focal_outputs})
        verify_request = MultiAgentVerifyRequest.model_validate(
            body.model_dump(mode="json")
            | {
                "response": focal_response.model_dump(mode="json"),
                "focal_participant": self.config.focal_participant,
                "participant_trajectories": trajectories,
                "episode_trajectory": events,
                "termination_reason": termination_reason,
                "turns_completed": sum(len(turns) for turns in trajectories.values()),
            }
        )

        if self.config.skip_verification:
            result = verify_request.model_dump(mode="json") | {
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(mode="json"),
                cookies=dict(environment_cookies),
            )
            await raise_for_status(verify_response)
            result = await get_response_json(verify_response)
        return MultiAgentVerifyResponse.model_validate(result)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        if self.config.skip_verification:
            return await super().aggregate_metrics(body)
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))
