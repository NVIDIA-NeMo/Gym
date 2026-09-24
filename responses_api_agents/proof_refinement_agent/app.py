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
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym._checkpoint import (
    RESOURCE_REQUEST_ID_HEADER,
    RESOURCE_STATE_REVISION_HEADER,
    AgentBoundaryKind,
    AgentBoundaryRecord,
    PendingModelPayload,
)
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
from nemo_gym.failure_kinds import AGENT_NO_GENERATION
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.rollout_correlation import (
    MODEL_CALL_CAPTURE_OUTCOME_HEADER,
    MODEL_CALL_ID_HEADER,
    ModelCallCaptureOutcome,
    current_rollout_id,
    take_checkpoint_parent,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


def _cookie_values(cookies: Any) -> dict[str, str]:
    return {
        name: str(getattr(value, "value", value)) for name, value in (cookies.items() if cookies is not None else ())
    }


def _resource_revision(response: Any, previous: int) -> int:
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping) and RESOURCE_STATE_REVISION_HEADER in headers:
        return int(headers[RESOURCE_STATE_REVISION_HEADER])
    return previous


class _ProofContinuation(BaseModel):
    """Agent-owned progress; a pending generated proof is stored in PendingModelPayload."""

    model_config = ConfigDict(extra="forbid")

    current_input: dict[str, Any]
    all_attempts: list[dict[str, Any]] = Field(default_factory=list)
    cookies: dict[str, str] = Field(default_factory=dict)
    finished: bool = False
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

    config: ProofRefinementAgentConfig
    checkpoint_continuation_supported = True
    checkpoint_resource_dependencies_supported = True

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Generate a model response (single turn, no tool calls).

        This is called for each generation turn. The verify-correction loop
        is handled in run().
        """
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_response = await self.retry_checkpoint_refusal(
            lambda: self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_request("/v1/responses", request),
                json=body,
                cookies=request.cookies,
            ),
            request=request,
        )
        await raise_for_status(model_response)
        model_response_json = await get_response_json(model_response)

        # The outer /run owns continuation state and needs the model ledger coordinate.
        headers = getattr(model_response, "headers", None)
        if isinstance(headers, Mapping):
            for header in (MODEL_CALL_ID_HEADER, MODEL_CALL_CAPTURE_OUTCOME_HEADER):
                if headers.get(header) is not None:
                    response.headers[header] = headers[header]
        for k, v in _cookie_values(model_response.cookies).items():
            response.set_cookie(k, v)

        return NeMoGymResponse.model_validate(model_response_json)

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
        execution = self.checkpoint_execution(request)
        continuation = self.checkpoint_continuation(body, request)
        state = (
            _ProofContinuation.model_validate(continuation.agent_state)
            if continuation is not None
            else _ProofContinuation(
                current_input=body.responses_create_params.model_dump(mode="json"),
                cookies=_cookie_values(request.cookies),
            )
        )
        turn_index = continuation.turn_index if continuation is not None else 0
        boundary_index = continuation.boundary_index if continuation is not None else 0
        pending_model = continuation.pending_model if continuation is not None else None
        model_call_id = continuation.last_committed_model_call_id if continuation is not None else None
        model_capture_key = continuation.last_committed_model_capture_key if continuation is not None else None
        resource_revision = (
            continuation.resource_state_revisions.get(self.config.resources_server.name, 0)
            if continuation is not None
            else 0
        )

        async def commit_boundary() -> None:
            if execution is None:
                return
            await self.checkpoint_participant().commit_boundary(
                execution,
                AgentBoundaryRecord(
                    rollout_id=execution.rollout_id,
                    attempt_index=execution.attempt_index,
                    boundary_index=boundary_index,
                    turn_index=turn_index,
                    boundary_kind=(
                        AgentBoundaryKind.PENDING_MODEL
                        if pending_model is not None
                        else AgentBoundaryKind.TURN_COMPLETE
                    ),
                    pending_model=pending_model,
                    output_items=[],
                    last_committed_model_capture_key=model_capture_key,
                    last_committed_model_call_id=model_call_id,
                    resource_state_revisions={self.config.resources_server.name: resource_revision},
                    agent_state=state.model_dump(mode="json"),
                ),
            )

        if continuation is None:
            seed_headers = {RESOURCE_REQUEST_ID_HEADER: uuid.uuid4().hex} if execution is not None else None
            seed_response = await self.retry_checkpoint_refusal(
                lambda: self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/seed_session",
                    json=body.model_dump(),
                    cookies=state.cookies,
                    **({"headers": seed_headers} if seed_headers is not None else {}),
                ),
                request=request,
            )
            await raise_for_status(seed_response)
            state.cookies.update(_cookie_values(seed_response.cookies))
            resource_revision = _resource_revision(seed_response, resource_revision)
            await commit_boundary()
        # The shared participant already installs a restored boundary for this
        # attempt, including before its first model or verification wait.

        while not state.finished:
            LOG.info("Turn %d: Generating proof attempt", turn_index)

            if pending_model is not None:
                model_response_json = pending_model.response
                verify_request_id = pending_model.resource_request_id
            else:
                if turn_index > 0:
                    # Correction prompts replace the conversation with a new
                    # user message. Their capture starts a fresh root, even
                    # after restore; it does not extend the saved proof's tokens.
                    take_checkpoint_parent()
                execution_headers = self.checkpoint_execution_headers()
                gen_response = await self.retry_checkpoint_refusal(
                    lambda: self.server_client.post(
                        server_name=self.config.name,
                        url_path=self.url_path_for_run("/v1/responses", body),
                        json=state.current_input,
                        cookies=state.cookies,
                        **({"headers": execution_headers} if execution_headers is not None else {}),
                    ),
                    request=request,
                    checkpointable_model_wait=True,
                )
                await raise_for_status(gen_response)
                model_response_json = await get_response_json(gen_response)
                state.cookies.update(_cookie_values(gen_response.cookies))
                headers = getattr(gen_response, "headers", None)
                capture_result = self.model_call_capture_result(headers if isinstance(headers, Mapping) else None)
                if capture_result is not None and capture_result.outcome == ModelCallCaptureOutcome.CAPTURE_FAILED:
                    raise RuntimeError("model generation completed without durable token capture")
                if capture_result is not None and capture_result.outcome == ModelCallCaptureOutcome.NO_GENERATION:
                    model_response = NeMoGymResponse.model_validate(model_response_json)
                    if model_response.incomplete_details is None:
                        raise RuntimeError("no-generation model response must terminate as incomplete")
                    return ProofRefinementVerifyResponse.model_validate(
                        body.model_dump()
                        | {
                            "response": model_response_json,
                            "reward": 0.0,
                            "mask_sample": True,
                            "failure_kind": AGENT_NO_GENERATION,
                            "total_turns": turn_index + 1,
                            "all_attempts": state.all_attempts if self.config.include_all_attempts else None,
                        }
                    )

                verify_request_id = uuid.uuid4().hex
                if execution is not None:
                    model_call_id = (
                        capture_result.model_call_id if capture_result is not None else model_response_json.get("id")
                    )
                    model_capture_key = current_rollout_id()
                    if not isinstance(model_call_id, str) or not model_call_id or model_capture_key is None:
                        raise RuntimeError("checkpointed proof generation is missing its model-call coordinate")
                    pending_model = PendingModelPayload(
                        model_call_id=model_call_id,
                        response=model_response_json,
                        model_server_cookies=dict(state.cookies),
                        usage=model_response_json.get("usage"),
                        pending_action_cursor=0,
                        resource_request_id=verify_request_id,
                    )
                boundary_index += 1
                await commit_boundary()

            # 3. Verify the proof
            verify_request_data = body.model_dump()
            verify_request_data["response"] = model_response_json
            verify_request_data["turn_index"] = turn_index

            async def verify() -> tuple[dict[str, Any], dict[str, str], int]:
                verify_response = await self.retry_checkpoint_refusal(
                    lambda: self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/verify",
                        json=verify_request_data,
                        cookies=state.cookies,
                        **(
                            {"headers": {RESOURCE_REQUEST_ID_HEADER: verify_request_id}}
                            if execution is not None
                            else {}
                        ),
                    ),
                    request=request,
                )
                await raise_for_status(verify_response)
                return (
                    await get_response_json(verify_response),
                    _cookie_values(verify_response.cookies),
                    _resource_revision(verify_response, resource_revision),
                )

            verification = (
                await self.checkpointable_external_wait(verify, request=request)
                if self.config.checkpoint_replayable_verify
                else await verify()
            )
            verify_result, verify_cookies, resource_revision = verification
            state.cookies.update(verify_cookies)

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

            correction_prompt = verify_result.get("correction_prompt")
            if not needs_correction:
                # Success! (or failure with no correction available)
                LOG.info("Turn %d: Proof verification complete (reward=%s)", turn_index, verify_result.get("reward"))
                state.finished = True
            elif turns_remaining <= 0:
                # No more turns allowed
                LOG.info("Turn %d: Max correction turns exhausted", turn_index)
                state.finished = True
            elif not correction_prompt:
                LOG.warning("Turn %d: needs_correction=True but no correction_prompt provided", turn_index)
                state.finished = True
            else:
                LOG.info("Turn %d: Preparing correction turn with error feedback", turn_index)
                # Keep the existing correction prompt and sampling behavior.
                params = body.responses_create_params
                state.current_input = {
                    "input": [{"role": "user", "content": correction_prompt}],
                    "model": getattr(params, "model", None),
                }
                for key in ["temperature", "max_tokens", "top_p"]:
                    value = getattr(params, key, None)
                    if value is not None:
                        state.current_input[key] = value
                turn_index += 1

            state.verify_result = verify_result if state.finished else None
            pending_model = None
            boundary_index += 1
            await commit_boundary()

        # Build final response
        final_response = ProofRefinementVerifyResponse.model_validate(state.verify_result)
        final_response.total_turns = turn_index + 1

        if self.config.include_all_attempts:
            final_response.all_attempts = state.all_attempts

        return final_response


if __name__ == "__main__":
    ProofRefinementAgent.run_webserver()
