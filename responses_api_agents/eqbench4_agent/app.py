# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EQ-Bench 4 agent: a roleplay conversation between the policy and a simulated persona, then verification.

The conversation follows EQ-Bench 4 (https://github.com/EQ-bench/eqbench4, revision
93dcb7f5d430433ad49d501e672a98bcff2c9678), implemented independently:

* The policy opens. Each policy turn sees that turn's system prompt and the dialogue so far; the first turn also sees
  the opening message that states the presenting issue.
* The persona simulator answers each policy turn. It sees the conversation with the roles swapped, its own most recent
  reply in full (private ``<internal>`` monologue included) and earlier replies as dialogue only, and a
  ``[Turn k of n]`` marker before the policy's latest message. A reply needs a non-empty ``<internal>`` and
  ``<dialogue>`` without leaked control tags; an unusable reply is shown back with a correction and asked again.
* Each side speaks at most ``num_turns`` times. The persona ends the conversation early by answering
  ``<status>ended</status>``.

The resources server then judges the transcript against published transcripts of the same scenario.
"""

import logging
import re
from time import time
from typing import Any, Literal

from fastapi import Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.agents import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory import AgentInvocation, ModelCallRef, ObservationGap, TrajectoryRecord, TrajectoryTurn
from resources_servers.eqbench4.judging import strip_thinking


LOG = logging.getLogger(__name__)

_INTERNAL = re.compile(r"<internal>\s*(.*?)\s*</internal>", re.DOTALL | re.IGNORECASE)
_DIALOGUE = re.compile(r"<dialogue>\s*(.*?)\s*</dialogue>", re.DOTALL | re.IGNORECASE)
_STATUS = re.compile(r"<status>\s*(.*?)\s*</status>", re.DOTALL | re.IGNORECASE)
_CONTROL_TAG = re.compile(
    r"</?\s*(internal|state|dialogue|status)\s*/?>|\[/?\s*(INTERNAL|STATE|DIALOGUE|STATUS)\s*\]?>?", re.IGNORECASE
)
_ENDED = {"ended", "end", "done", "finished", "stop", "stopped"}

# The rollout's two speakers, each an invocation of the trajectory record: the policy, and the persona simulator,
# which Gym's evidence profile counts as a second agent of the episode (as nemo_user_sim's simulated user is).
POLICY_INVOCATION = "root"
PERSONA_INVOCATION = "persona"
_TASK_ID_KEYS = ("task_id", "problem_id", "instance_id", "_ng_task_index")


class EQBench4AgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    persona_model_server: ModelServerRef
    persona_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    # Persona replies per turn before the conversation is abandoned as a simulator failure.
    persona_max_attempts: int = Field(default=3, ge=1)
    # Shown to the persona after an unusable reply; ``{problems}`` lists what was wrong.
    persona_correction_message: str
    # Policy replies per turn, as in EQ-Bench 4; a turn whose replies all have no visible text ends the conversation,
    # which is then left out of the score with the reason recorded. A context-window overflow is not asked again.
    policy_max_attempts: int = Field(default=7, ge=1)


class EQBench4RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    num_turns: int = Field(ge=1)
    evaluated_system_prompts: list[str] = Field(min_length=1)
    persona_system_prompt: str


class EQBench4AgentVerifyResponse(BaseVerifyResponse):
    """The resources server's verify response, kept whole."""

    model_config = ConfigDict(extra="allow")


class PersonaReply:
    """A persona reply and what is wrong with it, if anything."""

    def __init__(self, text: str) -> None:
        self.text = text
        internal = _INTERNAL.search(text)
        dialogue = _DIALOGUE.search(text)
        status = _STATUS.search(text)
        self.dialogue = dialogue.group(1).strip() if dialogue else None
        self.ended = bool(status) and status.group(1).strip().lower() in _ENDED
        self.problems: list[str] = []
        if not (internal and internal.group(1).strip()):
            self.problems.append("no <internal> section")
        if self.dialogue is None:
            self.problems.append("no <dialogue> section")
        elif not self.dialogue:
            self.problems.append("an empty <dialogue> section")
        elif _CONTROL_TAG.search(self.dialogue):
            self.problems.append("control tags inside <dialogue>")


def persona_messages(persona_prompt: str, transcript: list[dict[str, str]], turn: int, num_turns: int) -> list[dict]:
    """The persona's view of the conversation: roles swapped, its latest reply in full, and the turn marker."""
    last_persona = max((index for index, message in enumerate(transcript) if message["role"] == "user"), default=-1)
    messages = [{"role": "system", "content": persona_prompt}]
    for index, message in enumerate(transcript):
        if message["role"] == "user":
            content = message["raw_response"] if index == last_persona else message["content"]
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": "user", "content": message["content"]})
    messages[-1]["content"] = f"[Turn {turn} of {num_turns}]\n\n{messages[-1]['content']}"
    return messages


def policy_stop_reason(response: NeMoGymResponse) -> str:
    """Why a policy reply has no visible text, from the model server's markings."""
    if (response.metadata or {}).get("nemo_gym_context_window_exceeded") == "true":
        return "context_limit"
    incomplete = response.incomplete_details.reason if response.incomplete_details else None
    refused = any(
        content.type == "refusal" for item in response.output if item.type == "message" for content in item.content
    )
    if refused or incomplete == "content_filter":
        return "content_filter"
    if incomplete == "max_output_tokens":
        return "output_budget"
    return "empty_reply"


def visible_text(response: NeMoGymResponse) -> str:
    parts = [
        content.text
        for item in response.output
        if item.type == "message"
        for content in item.content
        if content.type == "output_text"
    ]
    return strip_thinking("".join(parts))


class ConversationTrajectory:
    """The trajectory record of one conversation, kept while model-call capture is on.

    Every model call of the conversation is a turn: each policy reply (retries of an empty reply too) under the policy
    invocation, and each persona reply (retries of an unusable one too) under the persona invocation. A turn names its
    call by model server and response id, so Gym joins it to the captured call. Judge calls belong to the resources
    server's verifier invocation. A turn has no tool steps, and the conversation's grade is a fraction, not a
    resolution.
    """

    def __init__(self, task_id: str, rollout_id: str) -> None:
        self.task_id = task_id
        self.rollout_id = rollout_id
        self.turns: list[TrajectoryTurn] = []
        self.calls: dict[str, list[ModelCallRef]] = {POLICY_INVOCATION: [], PERSONA_INVOCATION: []}
        self.gaps: list[ObservationGap] = [
            ObservationGap(code="resolution_unavailable", invocation_id=POLICY_INVOCATION)
        ]

    def turn(
        self,
        invocation_id: str,
        model_server: ModelServerRef,
        started_at: float,
        question: Any,
        response: NeMoGymResponse,
    ) -> None:
        refs = [ModelCallRef(model_ref=model_server, response_id=response.id)] if response.id else []
        number = sum(turn.invocation_id == invocation_id for turn in self.turns) + 1
        if not refs:
            self.gaps.append(
                ObservationGap(
                    code="model_call_reference_unavailable", invocation_id=invocation_id, detail=f"turn:{number}"
                )
            )
        self.calls[invocation_id].extend(refs)
        reasoning = [item.model_dump(mode="json") for item in response.output if item.type == "reasoning"]
        self.turns.append(
            TrajectoryTurn(
                invocation_id=invocation_id,
                task_id=self.task_id,
                rollout_id=self.rollout_id,
                turn_no=number,
                timestamp=started_at,
                question=question,
                answer=[item.model_dump(mode="json") for item in response.output if item.type != "reasoning"],
                reasoning_content=reasoning or None,
                step_count=0,
                model_calls=refs,
            )
        )

    def record(
        self,
        *,
        policy_status: Literal["completed", "failed", "incomplete"],
        persona_status: Literal["completed", "failed", "incomplete"],
        stop_reason: str | None = None,
        error_type: str | None = None,
        failed: str | None = None,
    ) -> TrajectoryRecord:
        """The record so far. ``failed`` names the invocation whose call raised ``error_type``, if one did."""
        invocations = [
            AgentInvocation(
                invocation_id=POLICY_INVOCATION,
                in_session=False,
                status=policy_status,
                stop_reason=stop_reason,
                error_type=error_type if failed == POLICY_INVOCATION else None,
                model_calls=list(self.calls[POLICY_INVOCATION]),
            )
        ]
        if self.calls[PERSONA_INVOCATION] or failed == PERSONA_INVOCATION:
            invocations.append(
                AgentInvocation(
                    invocation_id=PERSONA_INVOCATION,
                    in_session=False,
                    status=persona_status,
                    error_type=error_type if failed == PERSONA_INVOCATION else None,
                    model_calls=list(self.calls[PERSONA_INVOCATION]),
                )
            )
        return TrajectoryRecord(
            task_id=self.task_id,
            rollout_id=self.rollout_id,
            invocations=invocations,
            turns=list(self.turns),
            tool_calls=[],
            gaps=list(self.gaps),
        )


class EQBench4Agent(SimpleResponsesAPIAgent):
    config: EQBench4AgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """One policy reply."""
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        for key, value in model_response.cookies.items():
            response.set_cookie(key, value)
        return NeMoGymResponse.model_validate(await model_response.json())

    async def run(self, request: Request, body: EQBench4RunRequest) -> EQBench4AgentVerifyResponse:
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed)
        cookies = seed.cookies

        # With model-call capture on, the conversation's trajectory record goes with the result, and with the failure
        # report when the conversation breaks off.
        rollout_id = self.rollout_id_from_run(body)
        trajectory: ConversationTrajectory | None = None
        if self._model_call_capture_enabled() and rollout_id is not None:
            extra = body.model_extra or {}
            task_id = next((str(extra[key]) for key in _TASK_ID_KEYS if extra.get(key) is not None), "unknown")
            trajectory = ConversationTrajectory(task_id, rollout_id)

        transcript: list[dict[str, str]] = []
        usage: NeMoGymResponseUsage | None = None
        policy_response: dict[str, Any] | None = None
        status: Literal["completed", "persona_ended", "persona_failed", "policy_failed"] = "completed"
        error: str | None = None
        stop_reason: str | None = None
        policy_turns = persona_turns = 0
        speaking = POLICY_INVOCATION
        try:
            while policy_turns < body.num_turns:
                params = body.responses_create_params.model_copy(deep=True)
                if transcript:
                    params.input = [
                        NeMoGymEasyInputMessage(role="system", content=body.evaluated_system_prompts[policy_turns]),
                        *(NeMoGymEasyInputMessage(role=m["role"], content=m["content"]) for m in transcript),
                    ]
                text = ""
                speaking = POLICY_INVOCATION
                for attempt in range(1, self.config.policy_max_attempts + 1):
                    started_at = time()
                    reply = await self.server_client.post(
                        server_name=self.config.name,
                        url_path=self.url_path_for_run("/v1/responses", body),
                        json=params,
                        cookies=cookies,
                    )
                    await raise_for_status(reply)
                    cookies = reply.cookies
                    policy_response = await reply.json()
                    parsed = NeMoGymResponse.model_validate(policy_response)
                    if trajectory is not None:
                        question = params.model_dump(mode="json")["input"]
                        trajectory.turn(POLICY_INVOCATION, self.config.model_server, started_at, question, parsed)
                    usage = accumulate_response_usage(usage, parsed.usage)
                    text = visible_text(parsed)
                    if text:
                        break
                    stop_reason = policy_stop_reason(parsed)
                    if stop_reason == "context_limit":
                        break  # The same conversation overflows again.
                if not text:
                    status = "policy_failed"
                    error = (
                        f"no visible policy reply in turn {policy_turns + 1} after {attempt} attempt(s): {stop_reason}"
                    )
                    LOG.warning(
                        "EQ-Bench 4 scenario %s: %s; the conversation is left out of the score",
                        body.model_extra.get("scenario_id"),
                        error,
                    )
                    break
                transcript.append({"role": "assistant", "content": text, "raw_response": text})
                policy_turns += 1

                # Every policy turn gets a persona reply, so the conversation ends on the persona's last turn.
                speaking = PERSONA_INVOCATION
                persona, persona_usage = await self._persona_reply(body, transcript, persona_turns + 1, trajectory)
                usage = accumulate_response_usage(usage, persona_usage)
                if persona.problems:
                    status = "persona_failed"
                    error = f"persona turn {persona_turns + 1}: " + "; ".join(persona.problems)
                    break
                transcript.append({"role": "user", "content": persona.dialogue, "raw_response": persona.text})
                persona_turns += 1
                if persona.ended:
                    status = "persona_ended"
                    break
        except BaseException as failure:
            if trajectory is not None:
                # The speaker whose call raised failed (or was cancelled); the other one stopped with it.
                outcome = "failed" if isinstance(failure, Exception) else "incomplete"
                request.state.nemo_gym_agent_trajectory = trajectory.record(
                    policy_status=outcome if speaking == POLICY_INVOCATION else "incomplete",
                    persona_status=outcome if speaking == PERSONA_INVOCATION else "incomplete",
                    error_type=type(failure).__name__,
                    failed=speaking,
                )
            raise

        record: TrajectoryRecord | None = None
        if trajectory is not None:
            record = trajectory.record(
                policy_status=(
                    "failed"
                    if status == "policy_failed"
                    else "incomplete"
                    if status == "persona_failed"
                    else "completed"
                ),
                persona_status="failed" if status == "persona_failed" else "completed",
                stop_reason=stop_reason if status == "policy_failed" else None,
            )
            request.state.nemo_gym_agent_trajectory = record

        # The verifier echoes the last policy reply, with usage covering every model call of the conversation, the
        # policy's and the persona's: Gym's rollout usage covers the whole invocation tree, which includes the persona
        # simulator. It is set before verify, because the verifier binds its grading to the response it saw, so a
        # response changed afterwards no longer matches the grading.
        if policy_response is not None:
            policy_response = policy_response | {"usage": usage.model_dump(mode="json") if usage else None}
        verify_request = body.model_dump() | {
            "response": policy_response,
            "transcript": transcript,
            "conversation_status": status,
            "conversation_error": error,
            "policy_stop_reason": stop_reason if status == "policy_failed" else None,
        }
        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request,
            cookies=cookies,
        )
        await raise_for_status(verify)
        payload = await verify.json()
        if record is not None:
            payload["ng_trajectory"] = record.model_dump(mode="json")
        result = EQBench4AgentVerifyResponse.model_validate(payload)
        if policy_response is None:
            # No policy reply to grade: the verifier masks the row with a placeholder response, which carries the
            # usage of the calls that were made.
            result.response.usage = usage
        return result

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """The resources server aggregates: it holds the published comparisons the rating needs."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def _persona_reply(
        self,
        body: EQBench4RunRequest,
        transcript: list[dict[str, str]],
        turn: int,
        trajectory: ConversationTrajectory | None = None,
    ) -> tuple[PersonaReply, NeMoGymResponseUsage | None]:
        """The persona's reply to the latest policy turn, and the usage of every persona call it took."""
        messages = persona_messages(body.persona_system_prompt, transcript, turn, body.num_turns)
        reply = PersonaReply("")
        usage: NeMoGymResponseUsage | None = None
        for attempt in range(1, self.config.persona_max_attempts + 1):
            params = self.config.persona_responses_create_params.model_copy(deep=True)
            params.input = [NeMoGymEasyInputMessage(role=m["role"], content=m["content"]) for m in messages]
            started_at = time()
            response = await self.server_client.post(
                server_name=self.config.persona_model_server.name,
                url_path="/v1/responses",
                json=params,
            )
            await raise_for_status(response)
            parsed = NeMoGymResponse.model_validate(await response.json())
            if trajectory is not None:
                trajectory.turn(PERSONA_INVOCATION, self.config.persona_model_server, started_at, messages, parsed)
            usage = accumulate_response_usage(usage, parsed.usage)
            reply = PersonaReply(visible_text(parsed))
            if not reply.problems:
                return reply, usage
            LOG.warning("Persona reply %d of turn %d unusable: %s", attempt, turn, "; ".join(reply.problems))
            messages = [
                *messages,
                {"role": "assistant", "content": reply.text},
                {
                    "role": "user",
                    "content": self.config.persona_correction_message.format(problems="; ".join(reply.problems)),
                },
            ]
        return reply, usage


if __name__ == "__main__":
    EQBench4Agent.run_webserver()
