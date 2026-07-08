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
"""SimpleAgent with per-turn telemetry.

Behaviorally identical agent loop to ``responses_api_agents/simple_agent`` (same cookie
threading, malformed-tool-call handling, and termination conditions), but it additionally:

- records one telemetry record per model API call (turn): ISO timestamp, wall duration,
  per-turn input/output/total tokens, CACHED input tokens (which simple_agent zeroes in
  its aggregate — the per-turn values here come straight from the model server), reasoning
  tokens, tool-call names, and the slice of the final output list the turn produced;
- aggregates usage without discarding cached/reasoning token details;
- attaches the turn list to the final verify response (``turns``, ``num_turns``,
  ``trial_started_at``, ``trial_finished_at``) so rollout JSONL rows carry full per-turn
  traces for downstream telemetry export.

Correlation between the ``/run`` request and its inner ``/v1/responses`` call uses the
Responses API ``metadata`` field (key ``_turn_log_id``), which is stripped before any
model call.
"""

import json
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import BaseVerifyResponse
from nemo_gym.base_responses_api_agent import Body
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentRunRequest


TURN_LOG_ID_METADATA_KEY = "_turn_log_id"


class TurnLoggingAgentConfig(SimpleAgentConfig):
    pass


class TurnLoggingVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns: List[Dict[str, Any]]
    num_turns: int
    trial_started_at: Optional[str]
    trial_finished_at: Optional[str]


class TurnLoggingAgent(SimpleAgent):
    config: TurnLoggingAgentConfig

    # In-memory turn logs keyed by correlation id; entries are popped by run().
    turn_logs: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)

    @staticmethod
    def _strip_metadata(
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> tuple[NeMoGymResponseCreateParamsNonStreaming, Optional[str]]:
        """Extract the correlation id and rebuild the params without metadata set at all
        (so it is never forwarded to the model server)."""
        turn_log_id = (body.metadata or {}).get(TURN_LOG_ID_METADATA_KEY)
        if body.metadata is not None:
            body_dict = body.model_dump(exclude_unset=True)
            body_dict.pop("metadata", None)
            body = NeMoGymResponseCreateParamsNonStreaming.model_validate(body_dict)
        return body, turn_log_id

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        # PARITY NOTE: this loop mirrors simple_agent.SimpleAgent.responses step for step
        # (cookie threading, malformed-argument handling, termination). Keep it in sync.
        body, turn_log_id = self._strip_metadata(body.model_copy(deep=True))

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        turns: List[Dict[str, Any]] = []
        new_outputs = []
        usage = None
        cached_tokens_total = 0
        reasoning_tokens_total = 0
        step = 0
        model_server_cookies = None
        resources_server_cookies = request.cookies

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            turn_started_at = datetime.now(timezone.utc).isoformat()
            turn_t0 = monotonic()
            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e
            turn_duration_ms = round((monotonic() - turn_t0) * 1000, 1)

            output = model_response.output
            output_start_index = len(new_outputs)
            new_outputs.extend(output)

            # Per-turn usage, straight from the model server (cached tokens intact).
            turn_usage = model_response.usage
            turn_input = turn_usage.input_tokens if turn_usage else 0
            turn_output = turn_usage.output_tokens if turn_usage else 0
            turn_cached = 0
            turn_reasoning = 0
            if turn_usage and turn_usage.input_tokens_details:
                turn_cached = turn_usage.input_tokens_details.cached_tokens or 0
            if turn_usage and turn_usage.output_tokens_details:
                turn_reasoning = turn_usage.output_tokens_details.reasoning_tokens or 0
            cached_tokens_total += turn_cached
            reasoning_tokens_total += turn_reasoning

            if not usage:
                usage = model_response.usage
                model_response.usage = None

            if usage and model_response.usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]

            turn_assistant_text = "".join(
                content.text
                for message in all_output_messages
                for content in message.content
                if getattr(content, "type", None) == "output_text"
            )

            turn_record = {
                "turn": step - 1,
                "timestamp": turn_started_at,
                "duration_ms": turn_duration_ms,
                "model": model_response.model,
                "response_id": model_response.id,
                "status": model_response.status,
                "incomplete_reason": model_response.incomplete_details.reason
                if model_response.incomplete_details
                else None,
                "input_tokens": turn_input,
                "output_tokens": turn_output,
                "cached_input_tokens": turn_cached,
                "reasoning_tokens": turn_reasoning,
                "tool_call_names": [fn_call.name for fn_call in all_fn_calls],
                "num_tool_calls": len(all_fn_calls),
                "assistant_text": turn_assistant_text,
                "output_start_index": output_start_index,
                "output_end_index": None,  # set below, after tool outputs are appended
            }
            turns.append(turn_record)

            if model_response.incomplete_details:
                turn_record["output_end_index"] = len(new_outputs)
                break

            if not all_fn_calls and all_output_messages:
                turn_record["output_end_index"] = len(new_outputs)
                break

            for output_function_call in all_fn_calls:
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    tool_response = NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        output=json.dumps({"error": f"Invalid tool call arguments: {e!r}"}),
                    )
                    new_outputs.append(tool_response)
                    continue

                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=parsed_arguments,
                    cookies=resources_server_cookies,
                )
                resources_server_cookies = api_response.cookies

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)

            turn_record["output_end_index"] = len(new_outputs)

            if self.config.max_steps and step >= self.config.max_steps:
                break

        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        # Restore token detail sums that simple_agent zeroes out.
        if usage:
            if usage.input_tokens_details:
                usage.input_tokens_details.cached_tokens = cached_tokens_total
            if usage.output_tokens_details:
                usage.output_tokens_details.reasoning_tokens = reasoning_tokens_total

        if turn_log_id:
            self.turn_logs[turn_log_id] = turns

        model_response.output = new_outputs
        model_response.usage = usage
        return model_response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> TurnLoggingVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        # Correlate this trial's inner /v1/responses call so we can attach its turn log.
        turn_log_id = uuid4().hex
        responses_create_params = body.responses_create_params.model_copy(
            update={"metadata": {TURN_LOG_ID_METADATA_KEY: turn_log_id}}
        )

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies

        verify_request_dict = body.model_dump() | {"response": await get_response_json(response)}

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request_dict,
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        verify_response_dict = await get_response_json(verify_response)

        turns = self.turn_logs.pop(turn_log_id, [])
        return TurnLoggingVerifyResponse.model_validate(
            verify_response_dict
            | {
                "turns": turns,
                "num_turns": len(turns),
                "trial_started_at": turns[0]["timestamp"] if turns else None,
                "trial_finished_at": turns[-1]["timestamp"] if turns else None,
            }
        )


if __name__ == "__main__":
    TurnLoggingAgent.run_webserver()
