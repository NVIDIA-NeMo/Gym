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

"""Multi-turn agent with a simulated user.

Orchestrates a multi-turn dialogue between a policy model and a user
simulator. The user can be either a plain model server (an LLM with a
persona system prompt) or another agent server (with its own tool-call
loop). The interaction has two nested loops:

Outer loop (run): alternates between policy turns and user turns. Each
    iteration = one conversational exchange. Controlled by max_turns.

Inner loop (responses): within a single policy turn, the policy model
    may make multiple tool calls before producing a final text response.
    Same as SimpleAgent's loop. Controlled by max_steps_per_turn.

Each side keeps its OWN trajectory, stored from its owner's perspective:
    policy_trajectory — policy's own outputs (assistant messages, tool
        calls, tool outputs) interleaved with observations of what the
        user replied (labeled "user" from the policy's view).
    user_trajectory   — the user's persona system prompt, observations
        of what the policy said (labeled "user"), and the user's own
        replies (labeled "assistant").

Only the policy trajectory is sent to /verify for scoring; the user
trajectory is scaffolding for the simulated dialogue.
"""

import json
import logging
from typing import Any, List, Optional, Tuple, Union

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
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef, AgentServerRef
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
    user_model_server: Union[ModelServerRef, AgentServerRef]  # Required — LLM that simulates the human user
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
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _extract_text_from_outputs(outputs: list) -> str:
    """Pull a text observation out of a turn's output items.

    Strategy: return the last assistant-message text if any; otherwise the
    last function_call_output's content (useful when a side only emitted a
    tool call in a turn, e.g. O's make_move in tic-tac-toe). Empty string
    if neither is present.
    """
    for item in reversed(outputs):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for piece in item.get("content", []) or []:
                if piece.get("type") == "output_text":
                    text = piece.get("text")
                    if text:
                        return text
    for item in reversed(outputs):
        if item.get("type") == "function_call_output":
            out = item.get("output")
            if out:
                return out
    return ""


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
          2. User turn — call the user server (model or agent) via /v1/responses
        After all turns, verify the policy's conversation for a reward.

        Each side keeps its OWN trajectory, labeled from its perspective:
          - policy_trajectory: policy's own outputs (assistant, tool calls,
            tool outputs) plus the user's replies as "user" observations
          - user_trajectory:   user's persona system prompt, the policy's
            text as "user" observations, and the user's own replies as
            "assistant" messages
        Only the policy trajectory is sent to /verify.
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

        # If the user simulator is an agent, seed its session too.
        if isinstance(self.config.user_model_server, AgentServerRef):
            user_seed = await self.server_client.post(
                server_name=self.config.user_model_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(user_seed)
            cookies = user_seed.cookies

        # original_params carries tools/temperature/etc. — reused for every policy turn.
        original_params = body.responses_create_params.model_dump(exclude_unset=True)
        original_input = original_params.get("input", [])
        if isinstance(original_input, str):
            original_input = [{"role": "user", "content": original_input, "type": "message"}]

        # Per-task override from JSONL data takes precedence over config default
        user_system_prompt = (
            body.model_dump().get("user_model_system_prompt") or self.config.user_model_system_prompt
        )

        # Two trajectories, each stored from its owner's perspective.
        policy_trajectory: list = list(original_input)
        user_trajectory: list = [
            {"role": "system", "content": user_system_prompt, "type": "message"}
        ]

        last_model_response_json = None  # Used as the base for the final verify response

        # Phase 2: Multi-turn conversation loop
        for turn in range(self.config.max_turns):
            LOG.info("Turn %d: Policy turn", turn)

            # Policy sees its canonical history; call own /v1/responses which
            # runs the inner tool-call loop (responses method above) via HTTP.
            turn_params = {**original_params, "input": policy_trajectory}
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

            # Append this turn's policy outputs to its own trajectory (verbatim,
            # including reasoning/function_call/function_call_output items).
            policy_outputs = model_response_json.get("output", []) or []
            policy_trajectory.extend(policy_outputs)

            # User's observation of this turn = last text (or tool output)
            # the policy emitted. Labeled "user" from the user LLM's view.
            policy_text = _extract_text_from_outputs(policy_outputs)
            user_trajectory.append(
                {"role": "user", "content": policy_text, "type": "message"}
            )

            # Outer stop: context length exceeded
            incomplete = model_response_json.get("incomplete_details")
            if incomplete and incomplete.get("reason") == "max_output_tokens":
                LOG.info("Turn %d: Context length exceeded, stopping", turn)
                break

            # Don't generate a user message after the final turn
            if turn >= self.config.max_turns - 1:
                break

            # User turn — one call if agent (or tool-less model); tool-call
            # loop here if user is a ModelServerRef with tool_choice set.
            user_text, cookies = await self._call_user_server(
                user_trajectory=user_trajectory,
                original_params=original_params,
                cookies=cookies,
            )
            if not user_text:
                LOG.info("Turn %d: No user message generated, stopping", turn)
                break

            # Outer stop: user model emitted the configured stop token
            if self.config.user_model_stop_token and self.config.user_model_stop_token in user_text:
                LOG.info("Turn %d: User model stop token detected, stopping", turn)
                break

            LOG.info("Turn %d: User message: %s", turn, user_text[:100])
            # Append the user's reply to BOTH trajectories: as "assistant" in
            # the user's own history (what it just said), and as "user" in the
            # policy's history (an observation of what the other side said).
            user_trajectory.append(
                {"role": "assistant", "content": user_text, "type": "message"}
            )
            policy_trajectory.append(
                {"role": "user", "content": user_text, "type": "message"}
            )

        # Phase 3: Verify on the policy trajectory only.
        # The user trajectory is scaffolding for the simulated dialogue; the
        # resources server doesn't need to see it for scoring.
        final_response_json = dict(last_model_response_json or {})
        final_response_json["output"] = policy_trajectory

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

    async def _call_user_server(
        self,
        user_trajectory: list,
        original_params: dict,
        cookies,
    ) -> Tuple[str, Any]:
        """Call the user simulator with its own trajectory.

        `run()` maintains `user_trajectory` from the user LLM's perspective
        (its own replies as "assistant", the policy's observations as "user"),
        so no role swapping or trajectory filtering is needed here.

        Behavior by user_model_server type:
          - AgentServerRef: one call to /v1/responses; the agent runs its own
            inner tool-call loop internally.
          - ModelServerRef with user_model_tool_choice set (e.g. tic-tac-toe):
            run a tool-call loop here against the resources server so the
            user model can use tools.
          - ModelServerRef without tool_choice (e.g. workplace_assistant):
            single call, take the final text.

        Returns (user_text, updated_cookies). user_text is "" if the user
        produced neither a text message nor a tool result.
        """
        is_agent = isinstance(self.config.user_model_server, AgentServerRef)
        user_sees_tools = (not is_agent) and self.config.user_model_tool_choice is not None

        user_model_params: dict = {"input": user_trajectory}
        if user_sees_tools:
            tools = original_params.get("tools")
            if tools:
                user_model_params["tools"] = tools
            user_model_params["tool_choice"] = self.config.user_model_tool_choice

        # Simple path: one call. Agent handles its own tool-call loop internally;
        # a tool-less model just produces text.
        if is_agent or not user_sees_tools:
            user_response = await self.server_client.post(
                server_name=self.config.user_model_server.name,
                url_path="/v1/responses",
                json=user_model_params,
                cookies=cookies,
            )
            await raise_for_status(user_response)
            cookies = user_response.cookies
            user_response_json = await get_response_json(user_response)
            outputs = user_response_json.get("output", []) or []
            return _extract_text_from_outputs(outputs), cookies

        # Tool-call loop for a user model with tools (mirrors the policy's
        # inner loop in responses()): call model, execute tool calls against
        # the resources server, feed back, repeat until text or max steps.
        user_outputs: list = []
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
            cookies = user_response.cookies
            user_response_json = await get_response_json(user_response)

            outputs = user_response_json.get("output", []) or []
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

        return _extract_text_from_outputs(user_outputs), cookies

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
