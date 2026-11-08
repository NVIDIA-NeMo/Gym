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

"""Proof refinement agent with multi-turn self-correction.

This agent implements a verify-correction loop:
1. Generate initial proof attempt
2. Check it with the resources server
3. If failed and turns remaining: inject correction_prompt, generate again
4. Repeat until success or max turns exhausted

The resources server always provides error feedback on failure. The agent controls the retry loop and turn counting.
The loop runs in one of two places:

- ``/run`` (behind a ``legacy_agent`` relay) seeds the resources server and checks every attempt with ``/verify``.
- An agent session, seeded by an environment server such as ``single_agent_turn_legacy``, runs the loop inside one
  ``/v1/responses`` activation. It checks each attempt but the last allowed one with the resources server's
  ``/check_proof`` tool, through the session's direct HTTP grant. The environment server then sends the returned
  response to ``/verify`` once, which grades the final attempt.

Both send the model the same inputs, and both reward the final attempt's ``/verify`` result.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, Response
from pydantic import ConfigDict

from nemo_gym.agents import (
    ActivationConflictError,
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    OrderedOperationLedger,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.client import request as http_request
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseUsage,
    accumulate_response_usage,
)
from nemo_gym.server_runtime import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    DirectHTTPToolAccess,
    MCPToolAccess,
)
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap


LOG = logging.getLogger(__name__)

# The response field in which a session activation reports its attempts; math_formal_lean's /verify reads it.
PROOF_REFINEMENT_RESPONSE_KEY = "proof_refinement"


@dataclass
class AttemptRecord:
    """Record of a single proof attempt."""

    turn_index: int
    generation: str
    proof_status: str
    error_feedback: Optional[str] = None


class ProofRefinementAgentConfig(BaseResponsesAPIAgentConfig):
    """Configuration for the proof refinement agent."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_correction_turns: int = 0  # 0 = single-turn (no corrections)
    include_all_attempts: bool = True  # Include all attempts in output for training


class ProofRefinementRunRequest(BaseRunRequest):
    """Run request that forwards fields to the resources server."""

    model_config = ConfigDict(extra="allow")


class ProofRefinementVerifyRequest(BaseVerifyRequest):
    """Verify request with turn tracking."""

    model_config = ConfigDict(extra="allow")
    turn_index: int = 0


class ProofRefinementVerifyResponse(BaseVerifyResponse):
    """Verify response with attempt history."""

    model_config = ConfigDict(extra="allow")
    total_turns: int = 0  # How many turns were used
    all_attempts: Optional[List[Dict[str, Any]]] = None  # All attempt records if include_all_attempts=True


@dataclass
class ProofRefinementSessionState(AgentSessionState):
    # The resources server's direct HTTP grant, through which the loop calls /check_proof; None without corrections.
    check_access: Optional[DirectHTTPToolAccess]
    operations: OrderedOperationLedger[NeMoGymResponse] = field(default_factory=OrderedOperationLedger)
    # The first activation's path; a retry through another path (another capture segment) is refused.
    activation_path: Optional[str] = None


def generation_text(model_response_json: Dict[str, Any]) -> str:
    """The text an attempt record keeps: the first output text of the response's last message that has one."""
    text = ""
    for output in model_response_json.get("output") or []:
        if output.get("type") == "message" and output.get("content"):
            for content in output["content"]:
                if content.get("type") == "output_text":
                    text = content.get("text", "")
                    break
    return text


def correction_input(params: NeMoGymResponseCreateParamsNonStreaming, correction_prompt: str) -> Dict[str, Any]:
    """The next turn's request: the correction prompt alone (Nemotron single-turn style), with the task's sampling."""
    current_input: Dict[str, Any] = {
        "input": [{"role": "user", "content": correction_prompt}],
        "model": getattr(params, "model", None),
    }
    # Preserve the caller's sampling params and output cap (Responses names it max_output_tokens)
    for key in ["temperature", "max_output_tokens", "top_p"]:
        value = getattr(params, key, None)
        if value is not None:
            current_input[key] = value
    return current_input


class ProofRefinementAgent(SimpleResponsesAPIAgent):
    """Agent that implements multi-turn proof refinement with error feedback."""

    config: ProofRefinementAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Generate a proof.

        Without an agent session this is one model call: ``run()`` calls it for each turn and drives the loop.
        In an agent session it runs the whole correction loop and returns the final attempt's response.
        """
        session_id = self._agent_session_id_from_request(request)
        if session_id is not None:
            return await self._activate(request, session_id, body)

        model_response_json, model_cookies = await self._generate(
            self.url_path_for_request("/v1/responses", request), body, request.cookies
        )
        # Propagate cookies
        for k, v in model_cookies.items():
            response.set_cookie(k, v)

        return NeMoGymResponse.model_validate(model_response_json)

    async def _generate(
        self, url_path: str, body: NeMoGymResponseCreateParamsNonStreaming | Dict[str, Any], cookies: Any
    ) -> tuple[Dict[str, Any], Any]:
        """Make one model call; return the response JSON and the model server's cookies."""
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(body).model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=url_path,
            json=body,
            cookies=cookies,
        )
        await raise_for_status(model_response)
        return await get_response_json(model_response), model_response.cookies

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> ProofRefinementSessionState:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("Proof refinement agent does not support sandbox access")
        accesses = self.effective_tool_accesses(body)
        unsupported = [access.name for access in accesses if isinstance(access, MCPToolAccess) and access.required]
        if unsupported:
            raise AgentSeedRejectedError(
                f"Proof refinement agent does not support required MCP tool access: {', '.join(unsupported)}"
            )
        direct_accesses = [access for access in accesses if isinstance(access, DirectHTTPToolAccess)]
        if len(direct_accesses) > 1:
            raise AgentSeedRejectedError("Proof refinement agent supports at most one direct HTTP tool access")
        if not direct_accesses and self.config.max_correction_turns > 0:
            raise AgentSeedRejectedError(
                "Proof refinement agent checks proofs between correction turns through the resources server's "
                "direct HTTP grant; set resources_tool_transports: [direct_http] on the environment server"
            )
        return ProofRefinementSessionState(request=body, check_access=direct_accesses[0] if direct_accesses else None)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, ProofRefinementSessionState):
            raise TypeError("Expected proof refinement session state")
        await state.operations.close(timeout=30.0)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(
                source=self.config.name,
                gaps=[
                    ObservationGap(
                        code="observation_capture_failed",
                        detail="This agent does not record its model calls or proof checks as observations.",
                    )
                ],
            ),
        )

    async def _activate(
        self, request: Request, session_id: str, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """Run the session's one activation, or join it when the caller sends the same activation again."""
        state = self._require_agent_session(session_id)
        if not isinstance(state, ProofRefinementSessionState):
            raise TypeError("Expected proof refinement session state")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Activation requires the seeded session's rollout path")
        if state.activation_path is None:
            state.activation_path = request.url.path
        elif state.activation_path != request.url.path:
            raise HTTPException(409, "Activation path does not match the session's first activation")
        model_url_path = self.url_path_for_request("/v1/responses", request)
        try:
            return await state.operations.execute(
                index=1, request=body, operation=lambda: self._refine(state, body, model_url_path=model_url_path)
            )
        except ActivationConflictError as error:
            raise HTTPException(409, str(error)) from error

    async def _refine(
        self,
        state: ProofRefinementSessionState,
        params: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
    ) -> NeMoGymResponse:
        """Generate, check and correct until a proof passes or the turns run out, as run() does with /verify.

        The last allowed attempt is not checked: the environment server's /verify grades it.
        """
        all_attempts: List[Dict[str, Any]] = []
        current_input: NeMoGymResponseCreateParamsNonStreaming | Dict[str, Any] = params
        model_cookies: Any = {}
        turn_index = 0
        usage: Optional[NeMoGymResponseUsage] = None

        while True:
            LOG.info("Turn %d: Generating proof attempt", turn_index)
            model_response_json, model_cookies = await self._generate(model_url_path, current_input, model_cookies)
            turn_usage = model_response_json.get("usage")
            usage = accumulate_response_usage(
                usage, NeMoGymResponseUsage.model_validate(turn_usage) if turn_usage is not None else None
            )
            model_response = NeMoGymResponse.model_validate(model_response_json)
            # As run() records an attempt: the response as this agent's /v1/responses returns it,
            # and the task's whole request on the first turn, the correction input after.
            response_json = model_response.model_dump(mode="json")
            attempt_record: Dict[str, Any] = {
                "turn_index": turn_index,
                "input": current_input.model_dump(mode="json")
                if isinstance(current_input, NeMoGymResponseCreateParamsNonStreaming)
                else current_input,
                "response": response_json,
                "generation": generation_text(response_json),
            }
            all_attempts.append(attempt_record)

            if turn_index >= self.config.max_correction_turns:
                LOG.info("Turn %d: Max correction turns exhausted", turn_index)
                break

            check = await self._check_proof(state, model_response.output_text)
            needs_correction = check.get("needs_correction", False)
            attempt_record |= {
                "proof_status": check.get("proof_status", "unknown"),
                "reward": 0.0 if needs_correction else 1.0,
                "error_feedback": check.get("error_feedback"),
                "correction_prompt": check.get("correction_prompt"),
            }
            LOG.info(
                "Turn %d: proof_status=%s, needs_correction=%s",
                turn_index,
                attempt_record["proof_status"],
                needs_correction,
            )
            if not needs_correction:
                LOG.info("Turn %d: Proof check passed", turn_index)
                break
            correction_prompt = check.get("correction_prompt")
            if not correction_prompt:
                LOG.warning("Turn %d: needs_correction=True but no correction_prompt provided", turn_index)
                break

            LOG.info("Turn %d: Preparing correction turn with error feedback", turn_index)
            current_input = correction_input(params, correction_prompt)
            turn_index += 1

        refinement: Dict[str, Any] = {"total_turns": turn_index + 1}
        if self.config.include_all_attempts:
            refinement["all_attempts"] = all_attempts
        # The response is the last attempt's; usage covers every turn's model call, as run()'s result does.
        return model_response.model_copy(update={"usage": usage, PROOF_REFINEMENT_RESPONSE_KEY: refinement})

    async def _check_proof(self, state: ProofRefinementSessionState, proof: str) -> Dict[str, Any]:
        """Compile an attempt with the resources server's /check_proof, through the session's direct HTTP grant."""
        access = state.check_access
        if access is None:
            raise RuntimeError("This agent session has no direct HTTP grant to check proofs with")
        check_response = await http_request(
            method="POST",
            url=f"{str(access.base_url).rstrip('/')}/check_proof",
            json={"proof": proof},
            cookies=access.cookies,
            headers=dict(access.headers),
            _internal=True,
        )
        await raise_for_status(check_response)
        return await get_response_json(check_response)

    async def run(self, request: Request, body: ProofRefinementRunRequest) -> ProofRefinementVerifyResponse:
        """Execute the proof refinement loop.

        Flow:
        1. Seed the session with the resources server
        2. Generate initial proof attempt
        3. Verify the proof
        4. If failed and turns remaining:
           - Use correction_prompt from verify response as new input
           - Generate correction attempt
           - Verify again
        5. Repeat until success or max turns exhausted
        6. Return final verify response with all attempts recorded
        """
        cookies = request.cookies
        all_attempts: List[Dict[str, Any]] = []

        # 1. Seed the session
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        # Start the verify-correction loop
        current_input = body.responses_create_params
        turn_index = 0
        usage: Optional[NeMoGymResponseUsage] = None

        while True:
            LOG.info("Turn %d: Generating proof attempt", turn_index)

            # 2. Generate proof attempt
            gen_response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=current_input,
                cookies=cookies,
            )
            await raise_for_status(gen_response)
            cookies = gen_response.cookies
            model_response_json = await gen_response.json()
            turn_usage = model_response_json.get("usage")
            usage = accumulate_response_usage(
                usage, NeMoGymResponseUsage.model_validate(turn_usage) if turn_usage is not None else None
            )

            # 3. Verify the proof
            verify_request_data = body.model_dump()
            verify_request_data["response"] = model_response_json
            verify_request_data["turn_index"] = turn_index

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request_data,
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            cookies = verify_response.cookies
            verify_result = await verify_response.json()

            # Record this attempt with full details
            # Convert current_input to dict if it's a Pydantic model
            if hasattr(current_input, "model_dump"):
                input_dict = current_input.model_dump()
            else:
                input_dict = current_input

            attempt_record = {
                "turn_index": turn_index,
                "input": input_dict,  # Full input/prompt sent to model
                "response": model_response_json,  # Full model response with reasoning
                "generation": generation_text(model_response_json),  # Extracted generation text for convenience
                "proof_status": verify_result.get("proof_status", "unknown"),
                "reward": verify_result.get("reward", 0.0),
                "error_feedback": verify_result.get("error_feedback"),
                "correction_prompt": verify_result.get("correction_prompt"),  # The prompt for next turn
            }
            all_attempts.append(attempt_record)

            LOG.info(
                "Turn %d: proof_status=%s, reward=%s, needs_correction=%s",
                turn_index,
                verify_result.get("proof_status"),
                verify_result.get("reward"),
                verify_result.get("needs_correction"),
            )

            # 4. Check if we should continue
            needs_correction = verify_result.get("needs_correction", False)
            turns_remaining = self.config.max_correction_turns - turn_index

            if not needs_correction:
                # Success! (or failure with no correction available)
                LOG.info("Turn %d: Proof verification complete (reward=%s)", turn_index, verify_result.get("reward"))
                break

            if turns_remaining <= 0:
                # No more turns allowed
                LOG.info("Turn %d: Max correction turns exhausted", turn_index)
                break

            # 5. Prepare for next turn using correction_prompt
            correction_prompt = verify_result.get("correction_prompt")
            if not correction_prompt:
                LOG.warning("Turn %d: needs_correction=True but no correction_prompt provided", turn_index)
                break

            LOG.info("Turn %d: Preparing correction turn with error feedback", turn_index)

            current_input = correction_input(body.responses_create_params, correction_prompt)

            turn_index += 1

        # Build final response
        final_response = ProofRefinementVerifyResponse.model_validate(verify_result)
        final_response.total_turns = turn_index + 1
        # The verifier echoes the last turn's response; usage covers every turn's model call.
        final_response.response.usage = usage

        if self.config.include_all_attempts:
            final_response.all_attempts = all_attempts

        return final_response


if __name__ == "__main__":
    ProofRefinementAgent.run_webserver()
