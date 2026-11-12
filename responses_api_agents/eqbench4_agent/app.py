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
from resources_servers.eqbench4.judging import strip_thinking


LOG = logging.getLogger(__name__)

_INTERNAL = re.compile(r"<internal>\s*(.*?)\s*</internal>", re.DOTALL | re.IGNORECASE)
_DIALOGUE = re.compile(r"<dialogue>\s*(.*?)\s*</dialogue>", re.DOTALL | re.IGNORECASE)
_STATUS = re.compile(r"<status>\s*(.*?)\s*</status>", re.DOTALL | re.IGNORECASE)
_CONTROL_TAG = re.compile(
    r"</?\s*(internal|state|dialogue|status)\s*/?>|\[/?\s*(INTERNAL|STATE|DIALOGUE|STATUS)\s*\]?>?", re.IGNORECASE
)
_ENDED = {"ended", "end", "done", "finished", "stop", "stopped"}


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

        transcript: list[dict[str, str]] = []
        usage: NeMoGymResponseUsage | None = None
        policy_response: dict[str, Any] | None = None
        status: Literal["completed", "persona_ended", "persona_failed", "policy_failed"] = "completed"
        error: str | None = None
        stop_reason: str | None = None
        policy_turns = persona_turns = 0
        while policy_turns < body.num_turns:
            params = body.responses_create_params.model_copy(deep=True)
            if transcript:
                params.input = [
                    NeMoGymEasyInputMessage(role="system", content=body.evaluated_system_prompts[policy_turns]),
                    *(NeMoGymEasyInputMessage(role=m["role"], content=m["content"]) for m in transcript),
                ]
            text = ""
            for attempt in range(1, self.config.policy_max_attempts + 1):
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
                usage = accumulate_response_usage(usage, parsed.usage)
                text = visible_text(parsed)
                if text:
                    break
                stop_reason = policy_stop_reason(parsed)
                if stop_reason == "context_limit":
                    break  # The same conversation overflows again.
            if not text:
                status = "policy_failed"
                error = f"no visible policy reply in turn {policy_turns + 1} after {attempt} attempt(s): {stop_reason}"
                LOG.warning(
                    "EQ-Bench 4 scenario %s: %s; the conversation is left out of the score",
                    body.model_extra.get("scenario_id"),
                    error,
                )
                break
            transcript.append({"role": "assistant", "content": text, "raw_response": text})
            policy_turns += 1

            # Every policy turn gets a persona reply, so the conversation ends on the persona's last turn.
            persona = await self._persona_reply(body, transcript, persona_turns + 1)
            if persona.problems:
                status = "persona_failed"
                error = f"persona turn {persona_turns + 1}: " + "; ".join(persona.problems)
                break
            transcript.append({"role": "user", "content": persona.dialogue, "raw_response": persona.text})
            persona_turns += 1
            if persona.ended:
                status = "persona_ended"
                break

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
        result = EQBench4AgentVerifyResponse.model_validate(await verify.json())
        # The verifier echoes the last policy reply; usage covers every policy call of the conversation.
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
        self, body: EQBench4RunRequest, transcript: list[dict[str, str]], turn: int
    ) -> PersonaReply:
        messages = persona_messages(body.persona_system_prompt, transcript, turn, body.num_turns)
        reply = PersonaReply("")
        for attempt in range(1, self.config.persona_max_attempts + 1):
            params = self.config.persona_responses_create_params.model_copy(deep=True)
            params.input = [NeMoGymEasyInputMessage(role=m["role"], content=m["content"]) for m in messages]
            response = await self.server_client.post(
                server_name=self.config.persona_model_server.name,
                url_path="/v1/responses",
                json=params,
            )
            await raise_for_status(response)
            reply = PersonaReply(visible_text(NeMoGymResponse.model_validate(await response.json())))
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
        return reply


if __name__ == "__main__":
    EQBench4Agent.run_webserver()
