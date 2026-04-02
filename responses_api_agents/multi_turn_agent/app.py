# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Multi-turn agent with LLM user model.

Orchestrates a multi-turn dialogue between a policy model and a user model
(LLM simulating the human user). The interaction has two nested loops:

Outer loop (run): alternates between policy turns and user model turns.
    Each iteration = one conversational exchange. Controlled by max_turns.

Inner loop (responses): within a single policy turn, the model may make
    multiple tool calls before producing a final text response. This is
    the same tool-call loop as SimpleAgent. Controlled by max_steps_per_turn.

The full conversation trajectory (all turns interleaved) is sent to the
resources server for verification and reward computation.
"""

import json
import logging
from typing import List, Optional

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
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


# Module-level logger. Log messages are prefixed with the module path
# (e.g. responses_api_agents.multi_turn_agent.app) so they can be
# filtered in production logging configs.
LOG = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config and request/response schemas
# ──────────────────────────────────────────────────────────────────────


class MultiTurnAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef  # Required - Policy model (the model being trained/evaluated)
    user_model_server: ModelServerRef  # Required — LLM that simulates the human user
    max_turns: int  # Required — no safe default; each environment must set this
    max_steps_per_turn: Optional[int] = None  # None = unbounded; inner loop self-terminates
    user_model_system_prompt: str  # Required — defines the user model's persona/behavior
    user_model_stop_token: Optional[str] = None  # If the user model emits this, conversation ends
    user_model_tool_choice: Optional[str] = None  # None = API default ("auto"); "required" forces tool use


# extra="allow" lets the JSONL data include arbitrary task-specific fields
# (e.g. user_model_system_prompt overrides, verifier_metadata) that pass
# through to seed_session and verify without needing to be declared here.
class MultiTurnAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MultiTurnAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MultiTurnAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


# ──────────────────────────────────────────────────────────────────────
# Agent implementation
# ──────────────────────────────────────────────────────────────────────


class MultiTurnAgent(SimpleResponsesAPIAgent):
    """Agent that orchestrates multi-turn dialogue between a policy model and a user model."""

    config: MultiTurnAgentConfig

    # ── Inner loop: single policy turn ────────────────────────────────

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Handle one policy turn: model call + tool-call loop.

        This is the INNER loop. The model generates a response, and if it
        includes tool calls, those are routed to the resources server and
        the results are fed back. The loop repeats until:
          - The model produces text with no tool calls (natural completion)
          - max_steps_per_turn is reached
          - max_output_tokens is hit (context full)

        Called via HTTP from run() for each policy turn.
        Same logic as SimpleAgent.responses().
        """
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []  # Accumulates all outputs within this turn
        usage = None
        step = 0
        model_server_cookies = None
        resources_server_cookies = request.cookies

        while True:
            step += 1

            # Send the full context (original input + outputs so far in this turn) to the model
            new_body = body.model_copy(update={"input": body.input + new_outputs})

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

            output = model_response.output
            new_outputs.extend(output)

            # Accumulate token usage across steps within this turn
            if not usage:
                usage = model_response.usage
                model_response.usage = None

            if usage and model_response.usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            # Stop: context length exceeded
            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                break

            # Stop: model produced text with no tool calls (natural turn completion)
            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            # Execute each tool call against the resources server.
            # Matches simple_agent: no try/except, no raise. Tool errors are
            # valid feedback (e.g. invalid move) and become part of the
            # training trajectory.
            for output_function_call in all_fn_calls:
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                )
                # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                resources_server_cookies = api_response.cookies

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)

            # Stop: max tool-call steps within this turn
            if self.config.max_steps_per_turn and step >= self.config.max_steps_per_turn:
                break

        # Propagate cookies from both model and resources servers so downstream
        # calls (verify, next turn) can access session state from both.
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        model_response.usage = usage
        return model_response

    # ── Outer loop: multi-turn conversation ───────────────────────────

    async def run(self, request: Request, body: MultiTurnAgentRunRequest) -> MultiTurnAgentVerifyResponse:
        """Execute the multi-turn dialogue loop.

        This is the OUTER loop. For each turn:
          1. Policy turn — call self /v1/responses (which runs the inner loop)
          2. User model turn — generate the next user message via the user LLM
        After all turns, verify the full conversation for a reward.

        The conversation is represented as two parts:
          - original_input: the initial messages from the JSONL data (system prompt, first user message)
          - all_turn_outputs: everything generated during the conversation (policy responses,
            tool calls, tool results, user model messages) — grows each turn
        """
        cookies = request.cookies

        # Phase 1: Seed the resources server session (e.g. initialize game board)
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        # Separate the static initial input from the growing conversation.
        # original_params includes tools, temperature, etc. — reused each turn.
        original_params = body.responses_create_params.model_dump(exclude_unset=True)
        original_input = original_params.get("input", [])
        if isinstance(original_input, str):
            original_input = [{"role": "user", "content": original_input, "type": "message"}]

        all_turn_outputs = []  # Grows with each turn: policy outputs + user messages
        last_model_response_json = None  # Used as the base for the final verify response

        # Phase 2: Multi-turn conversation loop
        for turn in range(self.config.max_turns):
            LOG.info("Turn %d: Policy turn", turn)

            # Build this turn's input: original messages + everything from previous turns.
            # The tools, temperature, etc. from original_params carry forward unchanged.
            turn_params = {**original_params, "input": original_input + all_turn_outputs}

            # Call this agent's own /v1/responses endpoint, which runs the inner
            # tool-call loop (responses method above) via HTTP.
            policy_response = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=turn_params,
                cookies=cookies,
            )
            await raise_for_status(policy_response)
            cookies = policy_response.cookies
            model_response_json = await get_response_json(policy_response)
            last_model_response_json = model_response_json

            # Append this turn's policy outputs to the growing conversation
            policy_outputs = model_response_json.get("output", [])
            all_turn_outputs.extend(policy_outputs)

            # Outer stop: context length exceeded
            incomplete = model_response_json.get("incomplete_details")
            if incomplete and incomplete.get("reason") == "max_output_tokens":
                LOG.info("Turn %d: Context length exceeded, stopping", turn)
                break

            # Don't generate a user message after the final turn
            if turn >= self.config.max_turns - 1:
                break

            # Generate the next user message via the user LLM
            user_text = await self._generate_user_response(body, original_input, all_turn_outputs, cookies)
            if user_text is None:
                LOG.info("Turn %d: No user message generated, stopping", turn)
                break

            # Outer stop: user model emitted the configured stop token
            if self.config.user_model_stop_token and self.config.user_model_stop_token in user_text:
                LOG.info("Turn %d: User model stop token detected, stopping", turn)
                break

            LOG.info("Turn %d: User message: %s", turn, user_text[:100])
            user_msg = {"role": "user", "content": user_text, "type": "message"}
            all_turn_outputs.append(user_msg)

        # Phase 3: Verify the full conversation.
        # Build a single NeMoGymResponse containing ALL outputs from ALL turns
        # (policy outputs + user messages interleaved) and send to the resources
        # server for reward computation.
        final_response_json = dict(last_model_response_json)
        final_response_json["output"] = all_turn_outputs

        verify_request = MultiTurnAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": final_response_json}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return MultiTurnAgentVerifyResponse.model_validate(await get_response_json(verify_response))

    # ── User model interaction ────────────────────────────────────────

    async def _generate_user_response(
        self,
        body: MultiTurnAgentRunRequest,
        original_input: list,
        all_turn_outputs: list,
        cookies,
    ) -> Optional[str]:
        """Call the user LLM to generate the next user message.

        Builds the user model's input as:
          1. The user model system prompt (defines persona/behavior)
          2. The conversation so far (original user messages + all turn outputs),
             but with the policy's system/developer prompt stripped so the user
             model only sees its own system prompt.

        If the original request includes tools, those are passed to the user
        model too. If the user model makes tool calls, they are executed against
        the resources server in a loop (same pattern as the policy's inner loop).

        Only the user model's final text message is returned — its tool calls
        are NOT included in the conversation trajectory since they are internal
        to the user's "thinking" and not visible to the policy. The policy
        observes the effects of user tool calls through environment state
        (e.g. an updated game board) on its next turn.

        When user_model_tool_choice is "required", the model is forced to call
        a tool each iteration. If it never produces text, the last tool call
        result is returned as the user message so the conversation continues.

        Returns None only if the user model produces neither text nor tool calls.
        """
        # Per-task override from JSONL data takes precedence over config default
        user_system_prompt = body.model_dump().get("user_model_system_prompt") or self.config.user_model_system_prompt

        # Build user model input: its own system prompt + conversation history
        # (excluding the policy model's system/developer prompt to avoid confusion)
        user_model_input = [{"role": "system", "content": user_system_prompt, "type": "message"}]
        for msg in original_input:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role not in ("system", "developer"):
                user_model_input.append(msg)
        user_model_input.extend(all_turn_outputs)

        original_params = body.responses_create_params.model_dump(exclude_unset=True)
        user_model_params = {"input": user_model_input}

        tools = original_params.get("tools")
        if tools:
            user_model_params["tools"] = tools
        if self.config.user_model_tool_choice:
            user_model_params["tool_choice"] = self.config.user_model_tool_choice

        user_outputs = []
        resources_server_cookies = cookies

        max_user_steps = self.config.max_steps_per_turn or 10
        for step in range(max_user_steps):
            user_response = await self.server_client.post(
                server_name=self.config.user_model_server.name,
                url_path="/v1/responses",
                json={**user_model_params, "input": user_model_params["input"] + user_outputs},
                cookies=cookies,
            )
            await raise_for_status(user_response)
            user_response_json = await get_response_json(user_response)

            outputs = user_response_json.get("output", [])
            user_outputs.extend(outputs)

            # Stop: user model hit context limit
            incomplete = user_response_json.get("incomplete_details")
            if incomplete and incomplete.get("reason") == "max_output_tokens":
                break

            fn_calls = [o for o in outputs if o.get("type") == "function_call"]
            text_msgs = [o for o in outputs if o.get("type") == "message" and o.get("role") == "assistant"]

            # Stop: user model produced text with no tool calls
            if not fn_calls and text_msgs:
                break

            # Execute user model's tool calls against the resources server.
            # Matches the policy tool-call pattern: no raise, no try/except.
            for fn_call in fn_calls:
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{fn_call['name']}",
                    json=json.loads(fn_call["arguments"]),
                    cookies=resources_server_cookies,
                )
                resources_server_cookies = api_response.cookies

                user_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": fn_call["call_id"],
                        "output": (await api_response.content.read()).decode(),
                    }
                )

            # Safety: if no tool calls and no text, avoid infinite loop
            if not fn_calls and not text_msgs:
                break

        # Extract text from the user model's response.
        for output_item in reversed(user_outputs):
            if output_item.get("type") == "message" and output_item.get("role") == "assistant":
                for content in output_item.get("content", []):
                    if content.get("type") == "output_text":
                        return content.get("text", "")

        # Fallback: user model made tool calls but produced no text.
        # Use the last tool result as the user message so the policy
        # sees the environment state change and the conversation continues.
        for output_item in reversed(user_outputs):
            if output_item.get("type") == "function_call_output":
                return output_item.get("output", "Your turn.")

        return None

    # ── Metrics proxy ─────────────────────────────────────────────────

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    MultiTurnAgent.run_webserver()
