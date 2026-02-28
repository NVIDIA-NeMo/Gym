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
2. Verify with resource server
3. If failed and turns remaining: inject correction_prompt, generate again
4. Repeat until success or max turns exhausted

The resource server is stateless - it always provides error feedback on failure.
The agent controls the retry loop and turn counting.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict

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
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import raise_for_status


LOG = logging.getLogger(__name__)


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
    """Run request that forwards fields to the resource server."""

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

        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        model_response_json = await model_response.json()

        # Propagate cookies
        for k, v in model_response.cookies.items():
            response.set_cookie(k, v)

        return NeMoGymResponse.model_validate(model_response_json)

    async def run(self, request: Request, body: ProofRefinementRunRequest) -> ProofRefinementVerifyResponse:
        """Execute the proof refinement loop.

        Flow:
        1. Seed the session with the resource server
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

        while True:
            LOG.info("Turn %d: Generating proof attempt", turn_index)

            # 2. Generate proof attempt
            gen_response = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=current_input,
                cookies=cookies,
            )
            await raise_for_status(gen_response)
            cookies = gen_response.cookies
            model_response_json = await gen_response.json()

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
            generation_text = ""
            if model_response_json.get("output"):
                for output in model_response_json["output"]:
                    if output.get("type") == "message" and output.get("content"):
                        for content in output["content"]:
                            if content.get("type") == "output_text":
                                generation_text = content.get("text", "")
                                break

            # Convert current_input to dict if it's a Pydantic model
            if hasattr(current_input, "model_dump"):
                input_dict = current_input.model_dump()
            else:
                input_dict = current_input

            attempt_record = {
                "turn_index": turn_index,
                "input": input_dict,  # Full input/prompt sent to model
                "response": model_response_json,  # Full model response with reasoning
                "generation": generation_text,  # Extracted generation text for convenience
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

            # Create new input with the correction prompt (Nemotron single-turn style)
            # Access Pydantic model attributes properly
            params = body.responses_create_params
            current_input = {
                "input": [{"role": "user", "content": correction_prompt}],
                "model": getattr(params, "model", None),
            }
            # Preserve any other params like temperature, max_tokens
            for key in ["temperature", "max_tokens", "top_p"]:
                value = getattr(params, key, None)
                if value is not None:
                    current_input[key] = value

            turn_index += 1

        # Build final response
        final_response = ProofRefinementVerifyResponse.model_validate(verify_result)
        final_response.total_turns = turn_index + 1

        if self.config.include_all_attempts:
            final_response.all_attempts = all_attempts

        return final_response


if __name__ == "__main__":
    ProofRefinementAgent.run_webserver()
