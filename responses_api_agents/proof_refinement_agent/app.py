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
2. Verify with resources server
3. If failed and turns remaining: inject correction_prompt, generate again
4. Repeat until success or max turns exhausted

The resources server is stateless - it always provides error feedback on failure.
The agent controls the retry loop and turn counting.
"""

import logging
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from aiohttp import ClientResponse
from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nemo_gym._checkpoint.agent import LegacyRun, RestoredAgentSession, require_rollout
from nemo_gym._checkpoint.steps import StepMode, seed_verify_mode
from nemo_gym.base_resources_server import (
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
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


def _cookie_values(cookies: Any) -> dict[str, str]:
    return {name: str(getattr(value, "value", value)) for name, value in (cookies or {}).items()}


class _ProofContinuation(BaseModel):
    """Proof-loop progress at the boundary before the next external call."""

    model_config = ConfigDict(extra="forbid")

    next: Literal["seed", "model", "verify", "return"] = "seed"
    current_input: dict[str, Any]
    cookies: dict[str, str] = Field(default_factory=dict)
    verify_mode: StepMode = "wait"
    all_attempts: list[dict[str, Any]] = Field(default_factory=list)
    pending_response: dict[str, Any] | None = None
    verify_result: dict[str, Any] | None = None


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


class ProofRefinementAgent(SimpleResponsesAPIAgent):
    """Agent that implements multi-turn proof refinement with error feedback."""

    ray_enabled = False
    checkpoint_sessions_supported = True

    config: ProofRefinementAgentConfig

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        # The legacy /run boundary contains all proof progress; there is no other session state.
        return {key: {} for key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        if any(session.session for session in sessions):
            raise ValueError("Proof refinement sessions keep their state in the /run boundary")

    async def retire_agent_session(self, session_key: str) -> None:
        # The participant cancels the run and discards its boundary.
        pass

    async def _request_model(
        self, body: NeMoGymResponseCreateParamsNonStreaming, *, url_path: str, cookies: dict[str, str]
    ) -> ClientResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        return await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=url_path,
            json=body,
            cookies=cookies,
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Generate one response; the legacy /run owns multi-turn proof progress."""
        model_response = await self._request_model(
            body, url_path=self.url_path_for_request("/v1/responses", request), cookies=request.cookies
        )
        await raise_for_status(model_response)
        model_response_json = await get_response_json(model_response)

        # Propagate cookies
        for k, v in _cookie_values(model_response.cookies).items():
            response.set_cookie(k, v)

        return NeMoGymResponse.model_validate(model_response_json)

    async def run(self, request: Request, body: ProofRefinementRunRequest) -> ProofRefinementVerifyResponse:
        """Run or continue proof attempts, preserving completed work across checkpoints."""
        participant = self.checkpoint_participant
        if participant is None:
            return await self._run(request, body, legacy_run=None)
        episode_id = EpisodeId.from_capture_key(require_rollout(self.rollout_id_from_run(body)))
        async with participant.legacy_run(f"run:{episode_id.rollout_id}", episode_id) as legacy_run:
            return await self._run(request, body, legacy_run=legacy_run)

    async def _run(
        self, request: Request, body: ProofRefinementRunRequest, *, legacy_run: LegacyRun | None
    ) -> ProofRefinementVerifyResponse:
        continuation = legacy_run.continuation if legacy_run is not None else None
        state = (
            _ProofContinuation.model_validate(continuation)
            if continuation is not None
            else _ProofContinuation(
                current_input=body.responses_create_params.model_dump(mode="json"),
                cookies=_cookie_values(request.cookies),
            )
        )

        def step(mode: StepMode) -> AbstractAsyncContextManager[None]:
            return legacy_run.step(mode) if legacy_run is not None else nullcontext()

        while True:
            if legacy_run is not None:
                await legacy_run.boundary(state.model_dump(mode="json"))
            if state.next == "return":
                break

            if state.next == "seed":
                async with step("replay"):
                    seed_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/seed_session",
                        json=body.model_dump(),
                        cookies=state.cookies,
                    )
                    await raise_for_status(seed_response)
                state.cookies.update(_cookie_values(seed_response.cookies))
                state.verify_mode = seed_verify_mode(seed_response.headers)
                state.next = "model"
                continue

            turn_index = len(state.all_attempts)
            if state.next == "model":
                LOG.info("Turn %d: Generating proof attempt", turn_index)
                # Call the model directly so a completed proof cannot be lost between an agent
                # self-call returning and /run recording its next boundary. The model participant
                # holds this reply while a checkpoint is open, just as for SimpleAgent activations.
                async with step("replay"):
                    gen_response = await self._request_model(
                        NeMoGymResponseCreateParamsNonStreaming.model_validate(state.current_input),
                        url_path=self.url_path_for_run("/v1/responses", body),
                        cookies=state.cookies,
                    )
                if legacy_run is not None and gen_response.status == 409:
                    try:
                        payload = await get_response_json(gen_response)
                    except ValueError:
                        payload = None
                    error = payload.get("error") if isinstance(payload, dict) else None
                    if isinstance(error, dict) and error.get("code") == "checkpoint_parked":
                        # Re-enter the same boundary and park until admission reopens.
                        continue
                await raise_for_status(gen_response)
                state.pending_response = NeMoGymResponse.model_validate(
                    await get_response_json(gen_response)
                ).model_dump(mode="json")
                state.cookies.update(_cookie_values(gen_response.cookies))
                state.next = "verify"
                continue

            model_response_json = state.pending_response
            async with step(state.verify_mode):
                verify_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=body.model_dump() | {"response": model_response_json, "turn_index": turn_index},
                    cookies=state.cookies,
                )
                await raise_for_status(verify_response)
                verify_result = await get_response_json(verify_response)
            state.cookies.update(_cookie_values(verify_response.cookies))
            state.verify_result = verify_result

            # Record this attempt with full details
            generation_text = ""
            if model_response_json.get("output"):
                for output in model_response_json["output"]:
                    if output.get("type") == "message" and output.get("content"):
                        for content in output["content"]:
                            if content.get("type") == "output_text":
                                generation_text = content.get("text", "")
                                break

            attempt_record = {
                "turn_index": turn_index,
                "input": state.current_input,  # Full input/prompt sent to model
                "response": model_response_json,  # Full model response with reasoning
                "generation": generation_text,  # Extracted generation text for convenience
                "proof_status": verify_result.get("proof_status", "unknown"),
                "reward": verify_result.get("reward", 0.0),
                "error_feedback": verify_result.get("error_feedback"),
                "correction_prompt": verify_result.get("correction_prompt"),  # The prompt for next turn
            }
            state.all_attempts.append(attempt_record)
            state.pending_response = None
            state.next = "return"

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
                continue

            if turns_remaining <= 0:
                # No more turns allowed
                LOG.info("Turn %d: Max correction turns exhausted", turn_index)
                continue

            # 5. Prepare for next turn using correction_prompt
            correction_prompt = verify_result.get("correction_prompt")
            if not correction_prompt:
                LOG.warning("Turn %d: needs_correction=True but no correction_prompt provided", turn_index)
                continue

            LOG.info("Turn %d: Preparing correction turn with error feedback", turn_index)

            # Create new input with the correction prompt (Nemotron single-turn style)
            # Access Pydantic model attributes properly
            params = body.responses_create_params
            state.current_input = {
                "input": [{"role": "user", "content": correction_prompt}],
                "model": getattr(params, "model", None),
            }
            # Preserve any other params like temperature, max_tokens
            for key in ["temperature", "max_tokens", "top_p"]:
                value = getattr(params, key, None)
                if value is not None:
                    state.current_input[key] = value

            state.next = "model"

        # Build final response
        final_response = ProofRefinementVerifyResponse.model_validate(state.verify_result)
        final_response.total_turns = len(state.all_attempts)

        if self.config.include_all_attempts:
            final_response.all_attempts = state.all_attempts

        return final_response


if __name__ == "__main__":
    ProofRefinementAgent.run_webserver()
