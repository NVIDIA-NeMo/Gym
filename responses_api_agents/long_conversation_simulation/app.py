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

"""Gym agent that simulates one long multi-session user/assistant conversation per rollout."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.agents import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.models import ResponsesConverter, chat_completion_usage_to_response_usage
from responses_api_agents.long_conversation_simulation.models import (
    LongConversationRunRequest,
    LongConversationVerifyResponse,
    SimulationStats,
)
from responses_api_agents.long_conversation_simulation.simulation import (
    Batch,
    CallRole,
    Prompts,
    Settings,
    TokenCounter,
    simulate,
)


class EmptyOutputError(RuntimeError):
    """The model returned no text (vLLM also answers a context overflow this way by default)."""


class LongConversationSimulationAgentConfig(BaseResponsesAPIAgentConfig):
    model_config = ConfigDict(extra="forbid")

    assistant_model_server: ModelServerRef
    user_model_server: ModelServerRef
    helper_model_server: ModelServerRef

    assistant_system_prompt_fpath: str
    check_include_question_prompt_fpath: str
    check_need_followup_prompt_fpath: str
    user_answer_question_prompt_fpath: str
    user_followup_question_prompt_fpath: str

    tiktoken_encoding: str | None = "cl100k_base"
    call_timeout_seconds: float = Field(default=1800.0, gt=0)
    max_call_retries: int = Field(default=3, ge=0)

    assistant_context_tokens: int = 28000
    assistant_recent_messages: int = 10
    assistant_summary_tokens: int = 1000
    assistant_min_memory_tokens: int = 8000
    user_memory_tokens: int = 19000
    user_recent_messages: int = 10
    user_summary_tokens: int = 2000
    user_prompt_max_tokens: int = 27000
    check_max_tokens: int = 25000
    max_clarifications: int = 2
    max_followups: int = 2
    batch_shortfall_allowance: int = 10


def is_transient(error: BaseException) -> bool:
    """A timeout, connection error, empty output or retryable HTTP status, here or anywhere in the
    exception's __cause__/__context__ chain."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (asyncio.TimeoutError, ClientConnectionError, EmptyOutputError)):
            return True
        if isinstance(current, ClientResponseError) and (
            current.status in (408, 409, 425, 429) or current.status >= 500
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class LongConversationSimulationAgent(SimpleResponsesAPIAgent):
    config: LongConversationSimulationAgentConfig

    def prompts(self) -> Prompts:
        read = lambda path: Path(path).read_text(encoding="utf-8")  # noqa: E731
        return Prompts(
            assistant_system=read(self.config.assistant_system_prompt_fpath),
            check_include_question=read(self.config.check_include_question_prompt_fpath),
            check_need_followup=read(self.config.check_need_followup_prompt_fpath),
            user_answer_question=read(self.config.user_answer_question_prompt_fpath),
            user_followup_question=read(self.config.user_followup_question_prompt_fpath),
        )

    def settings(self) -> Settings:
        return Settings(**{name: getattr(self.config, name) for name in Settings.__dataclass_fields__})

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.assistant_model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        for key, value in model_response.cookies.items():
            response.set_cookie(key, value)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    async def run(self, request: Request, body: LongConversationRunRequest) -> LongConversationVerifyResponse:
        servers = {
            "assistant": self.config.assistant_model_server,
            "user": self.config.user_model_server,
            "helper": self.config.helper_model_server,
        }
        calls: Counter[str] = Counter()
        failed: Counter[str] = Counter()
        usage: NeMoGymResponseUsage | None = None
        last_assistant: NeMoGymChatCompletion | None = None

        async def complete(role: CallRole, messages: list[dict[str, str]]) -> NeMoGymChatCompletion:
            model_response = await self.server_client.post(
                server_name=servers[role].name,
                url_path=self.url_path_for_run("/v1/chat/completions", body),
                json={"messages": messages},
                cookies=request.cookies,
            )
            await raise_for_status(model_response)
            completion = NeMoGymChatCompletion.model_validate(await get_response_json(model_response))
            if not (completion.choices and completion.choices[0].message.content):
                raise EmptyOutputError(f"{role} model returned no text")
            return completion

        async def call(role: CallRole, messages: list[dict[str, str]]) -> str:
            nonlocal usage, last_assistant
            for attempt in range(self.config.max_call_retries + 1):
                calls[role] += 1
                try:
                    completion = await asyncio.wait_for(complete(role, messages), self.config.call_timeout_seconds)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    failed[role] += 1
                    if attempt == self.config.max_call_retries or not is_transient(error):
                        raise
                    await asyncio.sleep(0.5 * 2 ** min(attempt, 4))
                    continue
                if role == "assistant":
                    # Usage covers the assistant model only; user and helper calls are auxiliary.
                    usage = accumulate_response_usage(usage, chat_completion_usage_to_response_usage(completion.usage))
                    last_assistant = completion
                return completion.choices[0].message.content
            raise AssertionError("unreachable")

        result = await simulate(
            call,
            self.prompts(),
            [
                Batch(**batch.model_dump(include={"messages", "current_plan", "previous_plans", "time_anchor"}))
                for batch in body.batches
            ],
            body.topic,
            body.theme,
            previous_plans_summary=body.previous_plans_summary,
            settings=self.settings(),
            count=TokenCounter(self.config.tiktoken_encoding),
        )
        if not result.chat or last_assistant is None:
            raise RuntimeError(
                f"no batch completed (stopped at batch {result.stopped_at_batch}, "
                f"{len(result.skipped_questions)} questions skipped)"
            )
        response_params = body.responses_create_params.model_copy(update={"model": last_assistant.model})
        final_response = ResponsesConverter(return_token_id_information=False).chat_completion_to_response(
            response_params, last_assistant
        )
        final_response.usage = usage
        payload = body.model_dump()
        payload.update(
            response=final_response,
            # The share of requested batches the conversation completed.
            reward=len(result.chat) / len(body.batches),
            chat=result.chat,
            simulation=SimulationStats(
                batches_requested=result.batches_requested,
                batches_completed=len(result.chat),
                stopped_at_batch=result.stopped_at_batch,
                skipped_questions=result.skipped_questions,
                interrupted_turns=result.interrupted_turns,
                calls=dict(calls),
                failed_calls=dict(failed),
            ),
        )
        return LongConversationVerifyResponse.model_validate(payload)


if __name__ == "__main__":
    LongConversationSimulationAgent.run_webserver()
