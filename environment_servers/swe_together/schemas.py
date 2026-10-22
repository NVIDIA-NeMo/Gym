# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE-Together episode inputs and retained evidence."""

from time import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from environment_servers.swe_together.task import TaskData
from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseUsage
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnResult


class CandidateTurn(BaseModel):
    """One activation's reply, or explicitly identified evidence from deadline close."""

    activation_id: int = Field(ge=1)
    response: NeMoGymResponse
    duration_seconds: float = Field(default=0, ge=0)
    response_source: Literal["agent", "session_close"] = "agent"

    @classmethod
    def from_deadline_close(
        cls, close: AgentCloseSessionResponse, activation_id: int, duration: float
    ) -> "CandidateTurn":
        """Retain an interrupted activation without inventing output or token usage.

        Closing fences delayed requests and collects the adapter's native evidence.
        An activation that never established a native session remains unavailable.
        The complete cumulative trajectory and observations stay in the close receipt.
        """
        observations = close.agent_observations.model_dump(mode="json") if close.agent_observations else {}
        roots = [
            record
            for record in observations.get("records", [])
            if record.get("kind") == "agent_invocation"
            and not record.get("parent_invocation_id")
            and record.get("activation_id") == activation_id
        ]
        native_ids = {record.get("harness_session_id") for record in roots}
        if close.cleanup_confirmed is not True or len(native_ids) != 1 or None in native_ids:
            raise RuntimeError("Deadline close has no confirmed native activation evidence")
        if any(
            root.get("status") not in {"completed", "incomplete"}
            or root.get("error_type") not in {None, "cancelled", "timeout", "permission_denied"}
            for root in roots
        ):
            raise RuntimeError("Deadline close retained a failed native activation")
        response = NeMoGymResponse(
            id=f"swet-deadline-{close.agent_session_id}-{activation_id}",
            created_at=int(time()),
            object="response",
            model="",
            output=[],
            usage=None,
            status="incomplete",
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            metadata={
                "harness_session_id": next(iter(native_ids)),
                "environment_stop_reason": "session_budget_exhausted",
            },
        ).model_copy(
            update={
                "_ng_agent_observations": {
                    "source": observations.get("source"),
                    "records": roots,
                    "gaps": [{"code": "response_output_unavailable", "detail": "Interaction deadline close"}],
                }
            }
        )
        return cls(
            activation_id=activation_id, response=response, duration_seconds=duration, response_source="session_close"
        )

    @property
    def turn_complete(self) -> bool:
        return self.response.status == "completed"

    @property
    def stop_reason(self) -> str | None:
        records = (self.response.model_dump().get("_ng_agent_observations") or {}).get("records", [])
        root = next(
            (r for r in records if r.get("kind") == "agent_invocation" and not r.get("parent_invocation_id")), {}
        )
        return root.get("stop_reason") or root.get("error_type")


class SWETTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_data: TaskData
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None


class UserTurn(BaseModel):
    """The accepted simulator decision and the environment's stopping decision."""

    activation_id: int
    kind: Literal["continue", "stop"]
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None
    synthetic: bool = False
    stop_reason: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SWETResult(SingleAgentTurnResult):
    judge_score: float | None = None
    task_id: str
    ng_activations: list[CandidateTurn] = Field(default_factory=list)
    ng_steps: list[UserTurn] = Field(default_factory=list)
    ng_agent_close: AgentCloseSessionResponse | None = None


class SWETFailure(SingleAgentTurnFailure):
    """Execution failures retain their diagnostics without becoming candidate zeroes."""

    activations: list[CandidateTurn] = Field(default_factory=list)
    agent_close: AgentCloseSessionResponse | None = None
    cleanup_receipts: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_directory: str | None = None
    dependency_error: dict[str, JsonValue] | None = None


class SWETRequest(BaseEpisodeRequest[SWETTaskInput]):
    pass


class SWETResponse(BaseEpisodeResponse[SWETResult]):
    failure: SWETFailure | None = None


def episode_response(activations: list[CandidateTurn]) -> NeMoGymResponse:
    """Retain the final output and report usage over the whole collected episode.

    Each activation retains its own usage. A missing counter makes the episode
    total unknown, rather than silently reporting a partial sum as complete.
    """
    usages = [activation.response.usage for activation in activations]
    usage = NeMoGymResponseUsage.sum_from_list(usages) if all(item is not None for item in usages) else None
    return activations[-1].response.model_copy(update={"usage": usage}, deep=True)
