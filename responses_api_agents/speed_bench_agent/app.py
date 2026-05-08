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
"""SPEED-Bench multi-turn replay agent.

Skills' `SpecdecGenerationTask.process_single_datapoint` plays multi-turn
prompts one user turn at a time: send turn 1 → assistant 1 → send
turn 1 + assistant 1 + turn 2 → assistant 2, and so on. The accumulated
assistant outputs are returned as a list. Each turn is a separate model call,
so the spec-decode counters reflect per-turn prompt distributions, which
matters for parity.

Gym's `simple_agent` does NOT do this. Given a list of user messages with no
interspersed assistant responses, it sends them all in a single model call.

This agent reproduces Skills' behaviour:

1. Read `body.input` as the *entire* dialogue. Split into "user turns" — each
   user message is a turn boundary.
2. For each turn, send the conversation up to and including that turn,
   accumulate the model's assistant reply into the running conversation.
3. Aggregate token usage across all turns and return a single response that
   summarizes the multi-turn run.

System / developer messages at the front of `body.input` are kept as a
preamble across all turns. Existing assistant messages in `body.input` are
preserved (so a partially-completed conversation can be resumed) — only
*trailing* user messages after the last assistant trigger fresh model calls.

For tasks that only have a single user turn, this collapses to a single
model call (same shape as simple_agent).
"""

import json
import logging
from typing import List

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


class SpeedBenchAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class SpeedBenchAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SpeedBenchAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SpeedBenchAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _content_text(content) -> str:
    """Best-effort flatten of a message's `content` field to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text") or item.get("content")
            if text:
                chunks.append(text)
        return "".join(chunks)
    return ""


class SpeedBenchAgent(SimpleResponsesAPIAgent):
    """Multi-turn fixed-replay agent for SPEED-Bench."""

    config: SpeedBenchAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        # Identify how many *trailing* user turns need fresh model calls.
        # Anything before the first trailing user turn is treated as preamble.
        msgs = list(body.input)
        # Walk from the end backwards. Collect contiguous trailing user msgs
        # and stop at the first non-user.
        trailing_user_indices: List[int] = []
        for idx in range(len(msgs) - 1, -1, -1):
            m = msgs[idx]
            role = getattr(m, "role", None)
            if role is None and isinstance(m, dict):
                role = m.get("role")
            if role == "user":
                trailing_user_indices.append(idx)
                continue
            break
        trailing_user_indices.reverse()

        if not trailing_user_indices:
            # Nothing for the model to respond to. Defer to a single model call
            # to preserve simple_agent's "let the server decide" behaviour.
            trailing_user_indices = [len(msgs) - 1] if msgs else []

        # If only one user turn (trailing), this collapses to a single call.
        # Multiple trailing user turns get replayed one at a time.
        first_user_idx = trailing_user_indices[0]
        running_input = list(msgs[: first_user_idx + 1])  # preamble + first turn
        remaining_user_indices = trailing_user_indices[1:]

        accumulated_outputs = []
        accumulated_text_parts: List[str] = []
        usage = None
        last_response: NeMoGymResponse = None
        model_server_cookies = None
        resources_server_cookies = request.cookies

        async def _call_model(turn_input):
            new_body = body.model_copy(update={"input": turn_input})
            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            payload = await get_response_json(model_response)
            try:
                parsed = NeMoGymResponse.model_validate(payload)
            except ValidationError as e:
                raise RuntimeError(f"Received an invalid response from model server: {json.dumps(payload)}") from e
            return parsed, model_response.cookies

        # First turn (always at least one).
        last_response, model_server_cookies = await _call_model(running_input)

        def _gather_assistant_text(resp: NeMoGymResponse) -> str:
            parts: List[str] = []
            for item in resp.output:
                if isinstance(item, NeMoGymResponseOutputMessage) and item.role == "assistant":
                    for c in item.content:
                        if isinstance(c, NeMoGymResponseOutputText):
                            parts.append(c.text)
                        elif isinstance(c, dict) and c.get("type") == "output_text":
                            parts.append(c.get("text", ""))
            return "".join(parts)

        accumulated_outputs.extend(last_response.output)
        accumulated_text_parts.append(_gather_assistant_text(last_response))
        usage = last_response.usage
        last_response.usage = None

        # Subsequent turns: append the assistant reply and the next user turn,
        # then call again. Accumulate tokens into `usage`.
        for next_user_idx in remaining_user_indices:
            assistant_text = accumulated_text_parts[-1]
            running_input.append(NeMoGymEasyInputMessage(role="assistant", content=assistant_text))
            running_input.append(msgs[next_user_idx])
            last_response, model_server_cookies = await _call_model(running_input)
            accumulated_outputs.extend(last_response.output)
            accumulated_text_parts.append(_gather_assistant_text(last_response))
            if usage and last_response.usage:
                usage.input_tokens += last_response.usage.input_tokens
                usage.output_tokens += last_response.usage.output_tokens
                usage.total_tokens += last_response.usage.total_tokens
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0
            elif last_response.usage and not usage:
                usage = last_response.usage
            last_response.usage = None

        # Propagate any cookies the resources server set so /verify sees them.
        for k, v in (*resources_server_cookies.items(), *(model_server_cookies or {}).items()):
            response.set_cookie(k, v)

        last_response.output = accumulated_outputs
        last_response.usage = usage
        return last_response

    async def run(self, request: Request, body: SpeedBenchAgentRunRequest) -> SpeedBenchAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        api_response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(api_response)
        cookies = api_response.cookies

        verify_request = SpeedBenchAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(api_response)}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return SpeedBenchAgentVerifyResponse.model_validate(await get_response_json(verify_response))

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        api_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(api_response)
        return AggregateMetrics.model_validate(await get_response_json(api_response))


if __name__ == "__main__":
    SpeedBenchAgent.run_webserver()
