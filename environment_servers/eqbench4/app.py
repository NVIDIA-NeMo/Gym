# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EQ-Bench 4 environment server: a roleplay conversation between the policy and a simulated persona, then grading.

The conversation follows EQ-Bench 4 (https://github.com/EQ-bench/eqbench4, revision
93dcb7f5d430433ad49d501e672a98bcff2c9678, licensed under BSD-3-Clause with the Commons Clause condition, Licensor:
Samuel J. Paech), implemented independently:

* The policy opens. Each policy turn sees that turn's system prompt and the dialogue so far; the first turn also sees
  the opening message that states the presenting issue.
* The persona simulator answers each policy turn. It sees the conversation with the roles swapped, its own most recent
  reply in full (private ``<internal>`` monologue included) and earlier replies as dialogue only, and a
  ``[Turn k of n]`` marker before the policy's latest message. A reply needs a non-empty ``<internal>`` and
  ``<dialogue>`` without leaked control tags; an unusable reply is shown back with a correction and asked again.
* Each side speaks at most ``num_turns`` times. The persona ends the conversation early by answering
  ``<status>ended</status>``.

Each episode seeds a resources server session and an agent session for each participant: the policy, and the persona
simulator, which is part of the environment the policy acts in. Each turn activates the policy's session with that
turn's full input, since the system prompt changes every turn, and the persona's session with its own view. The episode
then closes both agent sessions, sends the conversation to the resources server's ``/verify``, whose response is the
episode's result, and closes the resources server session.
"""

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import Body
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentOutcome,
    AgentOutcomePolicyConfig,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    OutcomeAction,
    agent_outcome_policy,
)
from nemo_gym.client import format_exception_with_response_body, get_response_json, raise_boundary_failure
from nemo_gym.config import (
    TOKEN_CAPTURE_BLOCK,
    TOKEN_CAPTURE_PATH_SEGMENT,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ResourcesServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import (
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    NgFailureEvent,
    accumulate_response_usage,
    model_sampling_path_segment,
)
from nemo_gym.metrics import add_perf_metrics
from nemo_gym.server_runtime import (
    CHECKPOINT_CLOSE_FOLLOWS_HEADER,
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SingleAgentTurnFailure,
    SingleAgentTurnResult,
    SingleAgentTurnTaskInput,
    dependency_failure,
    handled_episode_failure,
    is_handled_episode_failure,
    is_retryable_dependency_error,
)
from nemo_gym.trajectory import NG_AGENT_OBSERVATIONS_KEY, AgentObservationBundle
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.eqbench4.judging import strip_thinking
from resources_servers.eqbench4.task_data import (
    CONVERSATION_METADATA_KEY,
    ConversationStatus,
    PolicyStopReason,
    check_system_prompts_cover_turns,
)


LOG = logging.getLogger(__name__)

# The persona simulator broke the conversation: an environment failure, retried, never scored against the policy.
PERSONA_FAILED = "eqbench4:persona_failed"

_INTERNAL = re.compile(r"<internal>\s*(.*?)\s*</internal>", re.DOTALL | re.IGNORECASE)
_DIALOGUE = re.compile(r"<dialogue>\s*(.*?)\s*</dialogue>", re.DOTALL | re.IGNORECASE)
_STATUS = re.compile(r"<status>\s*(.*?)\s*</status>", re.DOTALL | re.IGNORECASE)
_CONTROL_TAG = re.compile(
    r"</?\s*(internal|state|dialogue|status)\s*/?>|\[/?\s*(INTERNAL|STATE|DIALOGUE|STATUS)\s*\]?>?", re.IGNORECASE
)
_ENDED = {"ended", "end", "done", "finished", "stop", "stopped"}


class EQBench4EnvironmentServerConfig(BaseEnvironmentServerConfig):
    """Bind the resources server that grades the conversation and the agents of its two participants."""

    model_config = ConfigDict(extra="forbid")

    resources_server: ResourcesServerRef
    # The policy: an agent whose session answers each activation from the full input it is given, such as
    # simple_agent with max_steps 1 and execute_tools false.
    agent_server: AgentServerRef
    # The persona simulator: an agent of the same kind, on the persona's model.
    persona_agent: AgentServerRef
    persona_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    # Persona replies per turn before the simulator is taken to have failed.
    persona_max_attempts: int = Field(default=3, ge=1)
    # Shown to the persona after an unusable reply; ``{problems}`` lists what was wrong.
    persona_correction_message: str
    # Policy replies per turn, as in EQ-Bench 4; a turn whose replies all have no visible text ends the conversation,
    # which the resources server scores as a loss to every opponent, with the reason recorded. A context-window
    # overflow is not asked again.
    policy_max_attempts: int = Field(default=7, ge=1)
    # What a failed policy or persona reply does to the episode, by its cause (see agent_outcome_policy).
    agent_outcome_policy: AgentOutcomePolicyConfig = Field(default_factory=AgentOutcomePolicyConfig)


class EQBench4Conversation(BaseModel):
    """The task fields that drive the conversation; the rest of the row is the resources server's."""

    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    num_turns: int = Field(ge=1)
    evaluated_system_prompts: list[str] = Field(min_length=1)
    persona_system_prompt: str

    @model_validator(mode="after")
    def _a_system_prompt_per_policy_turn(self) -> "EQBench4Conversation":
        check_system_prompts_cover_turns(self.evaluated_system_prompts, self.num_turns)
        return self


class EQBench4EpisodeRequest(BaseEpisodeRequest[SingleAgentTurnTaskInput]):
    """One scenario: its prepared row, as the resources server's flat task data and the policy's first request."""


class EQBench4EpisodeResponse(BaseEpisodeResponse[SingleAgentTurnResult]):
    """The resources server's verify response with both sessions' observations, or the episode's failure with what
    its participants did before it."""

    failure: SingleAgentTurnFailure | None = None


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


def policy_stop_reason(response: NeMoGymResponse) -> PolicyStopReason:
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


@dataclass
class _AgentSession:
    """An agent session the episode seeded, and what its close returned."""

    agent_name: str
    session_id: str = field(default_factory=lambda: f"agent-session-{uuid4().hex}")
    cookies: dict[str, str] = field(default_factory=dict)
    observations: AgentObservationBundle | None = None


@dataclass
class _ConversationState:
    """The conversation so far; a failure partway keeps what it holds as evidence."""

    transcript: list[dict[str, str]] = field(default_factory=list)
    status: ConversationStatus = "completed"
    error: str | None = None
    stop_reason: PolicyStopReason | None = None
    last_policy_reply: NeMoGymResponse | None = None
    # Every model call of both sessions, the policy's and the persona's: the rollout's usage covers the invocation tree
    # its observations publish, which holds both.
    usage: NeMoGymResponseUsage | None = None
    # Whether every call reported its usage; a partial sum would read as the whole conversation's.
    usage_complete: bool = True

    def add_usage(self, response: NeMoGymResponse) -> None:
        self.usage = accumulate_response_usage(self.usage, response.usage)
        self.usage_complete = self.usage_complete and response.usage is not None

    def fields(self) -> dict[str, Any]:
        """The conversation as the resources server's verify request reads it."""
        return {
            "transcript": self.transcript,
            "conversation_status": self.status,
            "conversation_error": self.error,
            "policy_stop_reason": self.stop_reason if self.status == "policy_failed" else None,
        }


class EQBench4EnvironmentServer(BaseEnvironmentServer[EQBench4EpisodeRequest, EQBench4EpisodeResponse]):
    """Run one EQ-Bench 4 conversation as a Gym episode and have the resources server grade it."""

    config: EQBench4EnvironmentServerConfig
    request_model = EQBench4EpisodeRequest
    response_model = EQBench4EpisodeResponse
    # The policy's calls carry a training framework's sampling values; the persona's carry null and keep its own.
    applies_model_sampling = True

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """The resources server's rating and conversation statistics, with performance metrics for the whole batch.

        The resources server holds the published comparisons the rating needs. Rollout collection derives each row's
        ``ng_perf``, which is reduced here once.
        """
        benchmark_body = body.model_copy(
            update={
                "verify_responses": [
                    {key: value for key, value in row.items() if key != "ng_perf"} for row in body.verify_responses
                ]
            }
        )
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=benchmark_body,
        )
        await raise_boundary_failure(response)
        metrics = AggregateMetrics.model_validate(await get_response_json(response))
        # The reduction is CPU-bound over every row, so keep it off the server's event loop.
        return await asyncio.to_thread(add_perf_metrics, metrics, body.verify_responses)

    async def run(self, request: EQBench4EpisodeRequest, cleanup: CleanupContext) -> EQBench4EpisodeResponse:
        task_input = request.task.task_input
        try:
            conversation = EQBench4Conversation.model_validate(task_input.task_data)
        except ValidationError as error:
            raise handled_episode_failure(
                stage="seed",
                operation="episode",
                failure_reason=f"Not an EQ-Bench 4 scenario: {error}"[:2000],
                terminal=True,
            ) from error

        # Every session ID is assigned, and its close registered, before it is seeded, so a lost seed reply cannot hide
        # a session from cleanup.
        resources_session_id = f"resources-session-{uuid4().hex}"
        resources_cookies: dict[str, str] = {}
        resources_cleanup = cleanup.register_cleanup(
            "resources session", lambda: self._close_resources(request, resources_session_id, resources_cookies)
        )
        policy = _AgentSession(self.config.agent_server.name)
        persona = _AgentSession(self.config.persona_agent.name)
        policy_cleanup = cleanup.register_cleanup("policy agent session", lambda: self._close_agent(request, policy))
        persona_cleanup = cleanup.register_cleanup(
            "persona agent session", lambda: self._close_agent(request, persona)
        )

        await self._seed_resources(request, resources_session_id, resources_cookies)
        try:
            await self._seed_agent(request, policy)
        except Exception as error:
            raise dependency_failure(error, stage="seed") from error
        try:
            await self._seed_agent(request, persona)
        except Exception as error:
            raise self._persona_failure(
                f"The persona simulator's agent session could not be seeded: {format_exception_with_response_body(error)}",
                terminal=not is_retryable_dependency_error(error),
            ) from error

        state = _ConversationState()
        try:
            await self._converse(request, conversation, state, policy, persona)
        except NgFailureEvent as failure:
            if is_handled_episode_failure(failure):
                # Each session's evidence of the conversation so far arrives with its close.
                for handle in (policy_cleanup, persona_cleanup):
                    try:
                        await handle.close()
                    except Exception:
                        pass  # Final unwind retries the close and logs its failure.
                # Observations only: the conversation broke off, so it has no answer to grade, and a resumed run
                # runs it again. A kept partial response would tell recovery to grade it instead.
                failure.episode_evidence |= {"agent_observations": published_observations(policy, persona)}
            raise

        # The conversation is complete: from here on, a failure keeps the response to grade, so recovery grades the
        # conversation again rather than running it again.
        graded = graded_response(state)
        # Close the agent sessions before grading: their observations are part of the result, not of what is graded.
        for handle in (policy_cleanup, persona_cleanup):
            try:
                await handle.close()
            except Exception as error:
                raise dependency_failure(
                    error,
                    stage="cleanup",
                    partial_response=graded,
                    agent_observations=published_observations(policy, persona),
                ) from error

        result = await self._verify(request, state, graded, resources_cookies, policy, persona)
        # Keep one bounded retry in final unwind without erasing a completed verdict.
        cleanup.register_cleanup("post-verification resources session", resources_cleanup.close)
        return EQBench4EpisodeResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=result)

    async def _converse(
        self,
        request: EQBench4EpisodeRequest,
        conversation: EQBench4Conversation,
        state: _ConversationState,
        policy: _AgentSession,
        persona: _AgentSession,
    ) -> None:
        """Alternate the policy and the persona until a side has spoken ``num_turns`` times or the conversation
        ends; ``state`` holds the conversation as it goes."""
        transcript = state.transcript
        policy_turns = persona_turns = 0
        while policy_turns < conversation.num_turns:
            params = request.task.task_input.responses_create_params.model_copy(deep=True)
            if transcript:
                params.input = [
                    NeMoGymEasyInputMessage(
                        role="system", content=conversation.evaluated_system_prompts[policy_turns]
                    ),
                    *(NeMoGymEasyInputMessage(role=m["role"], content=m["content"]) for m in transcript),
                ]
            text = ""
            attempt = 0
            for attempt in range(1, self.config.policy_max_attempts + 1):
                reply = await self._policy_reply(request, policy, params)
                state.last_policy_reply = reply
                state.add_usage(reply)
                if reply.status == "failed":
                    # The agent session reported a failed policy call; the outcome policy decides what it is.
                    action = agent_outcome_policy(reply, config=self.config.agent_outcome_policy)
                    if action != OutcomeAction.VERIFY:
                        raise self._failed_reply_failure(reply, terminal=action == OutcomeAction.FAIL)
                    # The request was refused as it stands, so asking again would be refused the same way.
                    state.stop_reason = policy_stop_reason(reply)
                    break
                text = visible_text(reply)
                if text:
                    break
                state.stop_reason = policy_stop_reason(reply)
                if state.stop_reason == "context_limit":
                    break  # The same conversation overflows again.
            if not text:
                state.status = "policy_failed"
                state.error = f"no visible policy reply in turn {policy_turns + 1} after {attempt} attempt(s): {state.stop_reason}"
                LOG.warning(
                    "EQ-Bench 4 scenario %s: %s; the conversation is scored as a loss to every opponent",
                    conversation.scenario_id,
                    state.error,
                )
                return
            transcript.append({"role": "assistant", "content": text, "raw_response": text})
            policy_turns += 1

            # Every policy turn gets a persona reply, so the conversation ends on the persona's last turn.
            reply = await self._persona_reply(request, persona, conversation, state, persona_turns + 1)
            transcript.append({"role": "user", "content": reply.dialogue, "raw_response": reply.text})
            persona_turns += 1
            if reply.ended:
                state.status = "persona_ended"
                return

    async def _policy_reply(
        self, request: EQBench4EpisodeRequest, policy: _AgentSession, params: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        try:
            return await self._activate(request, policy, params, trained=True)
        except Exception as error:
            raise dependency_failure(error, stage="agent") from error

    async def _persona_reply(
        self,
        request: EQBench4EpisodeRequest,
        persona: _AgentSession,
        conversation: EQBench4Conversation,
        state: _ConversationState,
        turn: int,
    ) -> PersonaReply:
        """The persona's usable reply to the latest policy turn, or the persona's failure."""
        messages = persona_messages(conversation.persona_system_prompt, state.transcript, turn, conversation.num_turns)
        reply = PersonaReply("")
        for attempt in range(1, self.config.persona_max_attempts + 1):
            params = self.config.persona_responses_create_params.model_copy(deep=True)
            params.input = [NeMoGymEasyInputMessage(role=m["role"], content=m["content"]) for m in messages]
            try:
                response = await self._activate(request, persona, params, trained=False)
            except Exception as error:
                raise self._persona_failure(
                    f"The persona simulator failed in turn {turn}: {format_exception_with_response_body(error)}",
                    terminal=not is_retryable_dependency_error(error),
                ) from error
            state.add_usage(response)
            if response.status == "failed":
                action = agent_outcome_policy(response, config=self.config.agent_outcome_policy)
                message = response.error.message if response.error else "the persona's agent reported a failure"
                raise self._persona_failure(
                    f"The persona simulator failed in turn {turn}: {message}", terminal=action == OutcomeAction.FAIL
                )
            reply = PersonaReply(visible_text(response))
            if not reply.problems:
                return reply
            LOG.warning("Persona reply %d of turn %d unusable: %s", attempt, turn, "; ".join(reply.problems))
            messages = [
                *messages,
                {"role": "assistant", "content": reply.text},
                {
                    "role": "user",
                    "content": self.config.persona_correction_message.format(problems="; ".join(reply.problems)),
                },
            ]
        raise self._persona_failure(
            f"no usable persona reply in turn {turn} after {self.config.persona_max_attempts} attempt(s): "
            + "; ".join(reply.problems),
            terminal=False,
        )

    async def _verify(
        self,
        request: EQBench4EpisodeRequest,
        state: _ConversationState,
        response: dict[str, Any],
        resources_cookies: dict[str, str],
        policy: _AgentSession,
        persona: _AgentSession,
    ) -> SingleAgentTurnResult:
        """Have the resources server grade the conversation in ``response`` (see ``graded_response``); its verify
        response, with both sessions' observations, is the episode's result."""
        task_input = request.task.task_input
        conversation = state.fields()
        body = (
            task_input.task_data
            | {
                "responses_create_params": task_input.responses_create_params.model_dump(
                    mode="json", exclude_unset=True
                ),
                "response": response,
            }
            | conversation
        )
        try:
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body,
                cookies=resources_cookies,
                # Final cleanup closes the session, so the resources server keeps tracking it until then.
                headers={CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"},
            )
            await raise_boundary_failure(verify_response)
            payload = await get_response_json(verify_response)
        except Exception as error:
            raise dependency_failure(
                error,
                stage="verification",
                partial_response=response,
                agent_observations=published_observations(policy, persona),
            ) from error
        return SingleAgentTurnResult.model_validate(
            payload | {NG_AGENT_OBSERVATIONS_KEY: published_observations(policy, persona)}
        )

    async def _activate(
        self,
        request: EQBench4EpisodeRequest,
        session: _AgentSession,
        params: NeMoGymResponseCreateParamsNonStreaming,
        *,
        trained: bool,
    ) -> NeMoGymResponse:
        """One activation of an agent session: ``params`` is its whole input."""
        response = await self.server_client.post(
            server_name=session.agent_name,
            url_path=self._responses_path(request, session.agent_name, trained=trained),
            json=params,
            cookies=session.cookies,
        )
        await raise_boundary_failure(response)
        session.cookies |= _cookies(response)
        return NeMoGymResponse.model_validate(await get_response_json(response))

    def _responses_path(self, request: EQBench4EpisodeRequest, agent_name: str, *, trained: bool) -> str:
        """The rollout-scoped activation path. Only the policy is what a training framework trains: its calls carry
        the episode's sampling values and, when capture is on, the training-token tag; the persona's carry neither."""
        capture_segment = ""
        if trained and not request.skip_token_capture:
            block = self.server_client.global_config_dict.get(TOKEN_CAPTURE_BLOCK) or {}
            agent_config = get_first_server_config_dict(self.server_client.global_config_dict, agent_name)
            if bool(block.get("enabled", False)) and (
                bool(block.get("all_agents", False)) or bool(agent_config.get("token_id_capture", False))
            ):
                capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}"
        sampling_segment = model_sampling_path_segment(self.model_sampling_segment(request, trained=trained))
        return f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}{sampling_segment}/v1/responses"

    async def _seed_resources(
        self, request: EQBench4EpisodeRequest, session_id: str, cookies: dict[str, str]
    ) -> ResourcesSeedSessionResponse:
        task_input = request.task.task_input
        try:
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=ResourcesSeedSessionRequest(
                    resources_session_id=session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    task_data=task_input.task_data,
                ).model_dump(mode="json"),
            )
            await raise_boundary_failure(response)
            cookies.update(_cookies(response))
            if not cookies:
                raise ValueError("Resources seed did not establish a session cookie")
            seed = ResourcesSeedSessionResponse.model_validate(await get_response_json(response))
            if seed.resources_session_id != session_id:
                raise ValueError("Resources seed returned a different resources_session_id")
        except Exception as error:
            raise dependency_failure(error, stage="seed") from error
        return seed

    async def _seed_agent(self, request: EQBench4EpisodeRequest, session: _AgentSession) -> None:
        response = await self.server_client.post(
            server_name=session.agent_name,
            url_path="/v1/agent_sessions",
            json=AgentSeedSessionRequest(
                agent_session_id=session.session_id,
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                tool_accesses=[],
            ).model_dump(mode="json"),
        )
        await raise_boundary_failure(response)
        session.cookies |= _cookies(response)
        seeded = AgentSeedSessionResponse.model_validate(await get_response_json(response))
        if seeded.agent_session_id != session.session_id:
            raise ValueError(f"{session.agent_name} seed returned a different agent_session_id")
        if not session.cookies:
            raise ValueError(f"{session.agent_name} seed did not establish a session cookie")

    async def _close_agent(self, request: EQBench4EpisodeRequest, session: _AgentSession) -> None:
        response = await self.server_client.post(
            server_name=session.agent_name,
            url_path="/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(
                agent_session_id=session.session_id, episode_id=request.episode_id
            ).model_dump(mode="json"),
            cookies=session.cookies,
        )
        await raise_boundary_failure(response)
        closed = AgentCloseSessionResponse.model_validate(await get_response_json(response))
        if closed.agent_session_id != session.session_id:
            raise ValueError(f"{session.agent_name} close returned a different agent_session_id")
        session.observations = scoped_observations(closed.agent_observations, session.agent_name)

    async def _close_resources(
        self, request: EQBench4EpisodeRequest, session_id: str, cookies: dict[str, str]
    ) -> None:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/close_session",
            json=ResourcesCloseSessionRequest(
                resources_session_id=session_id, episode_id=request.episode_id
            ).model_dump(mode="json"),
            cookies=cookies,
        )
        await raise_boundary_failure(response)

    @staticmethod
    def _persona_failure(failure_reason: str, *, terminal: bool) -> NgFailureEvent:
        """The persona simulator, part of the environment, broke the conversation."""
        return handled_episode_failure(
            stage="agent",
            operation="episode",
            failure_reason=failure_reason[:2000],
            terminal=terminal,
            failure_kind=PERSONA_FAILED,
        )

    @staticmethod
    def _failed_reply_failure(reply: NeMoGymResponse, *, terminal: bool) -> NgFailureEvent:
        """A failed policy call that the agent session reported, under the cause it gave."""
        failure_kind = None
        try:
            failure_kind = AgentOutcome.model_validate_json(
                (reply.metadata or {})[AGENT_OUTCOME_METADATA_KEY]
            ).failure_kind
        except (KeyError, ValidationError):
            pass
        return handled_episode_failure(
            stage="agent",
            operation="episode",
            failure_reason=(reply.error.message if reply.error else "The policy's agent reported a failure")[:2000],
            terminal=terminal,
            failure_kind=failure_kind if failure_kind != "unknown" else None,
        )


def graded_response(state: "_ConversationState") -> dict[str, Any]:
    """The response the resources server grades: the policy's last reply, with usage covering every model call of
    both sessions, carrying the conversation.

    A re-judge of the saved answer reads the conversation from it (see ``CONVERSATION_METADATA_KEY``): Gym rebuilds a
    re-judge's verify request from the task and the saved response alone.
    """
    response = state.last_policy_reply.model_dump(mode="json")
    response["usage"] = state.usage.model_dump(mode="json") if state.usage and state.usage_complete else None
    response["metadata"] = (response.get("metadata") or {}) | {CONVERSATION_METADATA_KEY: json.dumps(state.fields())}
    return response


def published_observations(*sessions: _AgentSession) -> AgentObservationBundle | None:
    """Every session's observations as one bundle: the rollout's invocation tree, which owns each of its model calls.

    Each session's invocations are already scoped by its agent instance (``scoped_observations``), so they stay apart.
    """
    bundles = [session.observations for session in sessions if session.observations is not None]
    if not bundles:
        return None
    return AgentObservationBundle(
        source=bundles[0].source,
        records=[record for bundle in bundles for record in bundle.records],
        gaps=[gap for bundle in bundles for gap in bundle.gaps],
    )


def scoped_observations(bundle: AgentObservationBundle | None, agent_instance: str) -> AgentObservationBundle | None:
    """An agent session's observations, with each invocation named after the agent instance that ran it.

    Both participants' sessions run in one rollout and name their activations alike (``root``, ``activation-2``, ...).
    Rollout collection joins a captured model call to the published invocation its client session ID names, so with
    the sessions' own names each session's calls would also join the other's invocations. Scoped, each call binds only
    to the invocation of its own session, through the call's response reference.
    """
    if bundle is None:
        return None

    def scoped(invocation_id: str | None) -> str | None:
        return f"{agent_instance}:{invocation_id}" if invocation_id is not None else None

    dumped = bundle.model_dump(mode="json")
    for record in dumped["records"]:
        for key in ("invocation_id", "parent_invocation_id"):
            if key in record:
                record[key] = scoped(record[key])
    for gap in dumped["gaps"]:
        gap["invocation_id"] = scoped(gap.get("invocation_id"))
    return AgentObservationBundle.model_validate(dumped)


def _cookies(response: Any) -> dict[str, str]:
    cookies = getattr(response, "cookies", None) or {}
    return {str(name): str(morsel.value) for name, morsel in cookies.items()} if isinstance(cookies, Mapping) else {}


if __name__ == "__main__":
    EQBench4EnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = EQBench4EnvironmentServer.run_webserver()  # noqa: F401
