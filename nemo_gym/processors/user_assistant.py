# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn processing for independently hosted user and assistant agents."""

from typing import Any, Literal, Optional

from fastapi import Body, Request
from pydantic import BaseModel, ConfigDict, Field

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


Participant = Literal["assistant", "user"]
EpisodeEventKind = Literal["response_item", "state", "termination"]


class ParticipantTurn(BaseModel):
    """One attributed agent invocation, including its exact model-visible input."""

    turn_index: int
    participant: Participant
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    agent_trajectory: Optional[dict[str, Any]] = None


class EpisodeEvent(BaseModel):
    """One ordered participant output, state observation, or termination event."""

    sequence: int
    turn_index: int
    kind: EpisodeEventKind
    participant: Optional[Participant] = None
    data: dict[str, Any]


class EpisodeStatus(BaseModel):
    """Resources-server response used to stop an episode and expose shared state."""

    model_config = ConfigDict(extra="allow")

    terminated: bool = False
    reason: Optional[str] = None
    state: dict[str, Any] = Field(default_factory=dict)


class UserAssistantRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    user_responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class UserAssistantVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    assistant_trajectory: list[ParticipantTurn]
    user_trajectory: list[ParticipantTurn]
    episode_trajectory: list[EpisodeEvent]
    termination_reason: str
    turns_completed: int


class UserAssistantVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    assistant_trajectory: list[ParticipantTurn]
    user_trajectory: list[ParticipantTurn]
    episode_trajectory: list[EpisodeEvent]
    termination_reason: str
    turns_completed: int


class UserAssistantProcessorConfig(BaseProcessorConfig):
    assistant_agent: AgentServerRef
    user_agent: AgentServerRef
    resources_server: ResourcesServerRef
    max_turns: int = Field(8, ge=1)
    status_url_path: str = "/episode_status"


def _input_items(params: NeMoGymResponseCreateParamsNonStreaming) -> list[Any]:
    if isinstance(params.input, str):
        return [NeMoGymEasyInputMessage(role="user", content=params.input)]
    return list(params.input)


def _visible_text(response: NeMoGymResponse) -> str:
    return response.output_text.strip()


class UserAssistantProcessor(BaseProcessor):
    """Alternate user and assistant policies over one shared resources-server session."""

    config: UserAssistantProcessorConfig

    async def _call_participant(
        self,
        *,
        participant: Participant,
        params: NeMoGymResponseCreateParamsNonStreaming,
        body: UserAssistantRunRequest,
        cookies: Any,
    ) -> tuple[NeMoGymResponse, Optional[dict[str, Any]], dict[str, Any]]:
        agent_ref = self.config.assistant_agent if participant == "assistant" else self.config.user_agent
        response = await self.server_client.post(
            server_name=agent_ref.name,
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

    async def run(self, request: Request, body: UserAssistantRunRequest) -> UserAssistantVerifyResponse:
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
        participant_cookies: dict[Participant, dict[str, Any]] = {"assistant": {}, "user": {}}

        participant_inputs = {
            "assistant": _input_items(body.responses_create_params),
            "user": _input_items(body.user_responses_create_params),
        }
        params_by_participant = {
            "assistant": body.responses_create_params,
            "user": body.user_responses_create_params,
        }
        trajectories: dict[Participant, list[ParticipantTurn]] = {"assistant": [], "user": []}
        events: list[EpisodeEvent] = []
        assistant_outputs = []
        last_assistant_response: Optional[NeMoGymResponse] = None
        termination_reason = "max_turns"

        for turn_index in range(self.config.max_turns):
            participant: Participant = "assistant" if turn_index % 2 == 0 else "user"
            counterpart: Participant = "user" if participant == "assistant" else "assistant"
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
            if participant == "assistant":
                assistant_outputs.extend(participant_response.output)
                last_assistant_response = participant_response

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
                participant_inputs[counterpart].append(NeMoGymEasyInputMessage(role="user", content=visible_text))
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
                    turn_index=max(0, len(trajectories["assistant"]) + len(trajectories["user"]) - 1),
                    kind="termination",
                    data={"reason": termination_reason},
                )
            )

        if last_assistant_response is None:
            raise RuntimeError("The assistant did not produce a response.")

        assistant_response = last_assistant_response.model_copy(update={"output": assistant_outputs})
        verify_request = UserAssistantVerifyRequest.model_validate(
            body.model_dump(mode="json")
            | {
                "response": assistant_response.model_dump(mode="json"),
                "assistant_trajectory": trajectories["assistant"],
                "user_trajectory": trajectories["user"],
                "episode_trajectory": events,
                "termination_reason": termination_reason,
                "turns_completed": len(trajectories["assistant"]) + len(trajectories["user"]),
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
        return UserAssistantVerifyResponse.model_validate(result)

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
