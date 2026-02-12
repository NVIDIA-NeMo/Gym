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

"""COBOL evaluation agent with multi-turn error correction.

Implements a verify-correction loop:
1. Generate initial COBOL code attempt
2. Verify by compiling and running tests
3. If failed and turns remaining: build correction prompt with errors, generate again
4. Repeat until success or max turns exhausted

Follows the proof_refinement_agent pattern.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseRunRequest,
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


# ---------------------------------------------------------------------------
# Error correction prompt template (from DomainForge cobol_with_errors.txt)
# ---------------------------------------------------------------------------

ERROR_CORRECTION_TEMPLATE = """\
Write complete, compilable COBOL code for GnuCOBOL 3.2+. Fix the errors in your previous attempt based on the feedback below. Output ONLY the COBOL program with no explanations or commentary.

## Previous Attempt Information

### Original Problem:
{problem_description}

### Your Previous Code:
{previous_code}

### Errors Encountered:
{error_feedback}

---

## Coding Standards
- Use free format (-free), WS- prefix for variables, PIC clauses for all items
- Read input silently from stdin (no prompts), display only the final result
- Use NUMVAL/NUMVAL-C for numeric parsing, structured programming (PERFORM, no GO TO)

## Requirements (must all be true)
1. Compiles with GnuCOBOL (free format)
2. Reads input silently from stdin (no prompts)
3. Displays only the required final output (no labels/extra text)
4. Fixes all compilation errors mentioned in the feedback
5. Implements all specified test case logic correctly

IMPORTANT: Output ONLY the COBOL code without any markdown formatting, explanations, or commentary. Start with IDENTIFICATION DIVISION."""


class CobolEvalAgentConfig(BaseResponsesAPIAgentConfig):
    """Configuration for the COBOL evaluation agent."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_correction_turns: int = 3
    include_all_attempts: bool = True


class CobolEvalRunRequest(BaseRunRequest):
    """Run request forwarded to the resource server."""

    model_config = ConfigDict(extra="allow")


class CobolEvalVerifyResponse(BaseVerifyResponse):
    """Verify response with attempt history."""

    model_config = ConfigDict(extra="allow")
    total_turns: int = 0
    all_attempts: Optional[List[Dict[str, Any]]] = None


class CobolEvalAgent(SimpleResponsesAPIAgent):
    """Agent that implements multi-turn COBOL code generation with error feedback."""

    config: CobolEvalAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Generate a model response (single turn)."""
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

        for k, v in model_response.cookies.items():
            response.set_cookie(k, v)

        return NeMoGymResponse.model_validate(model_response_json)

    async def run(self, request: Request, body: CobolEvalRunRequest) -> CobolEvalVerifyResponse:
        """Execute the COBOL evaluation loop with error correction."""
        cookies = request.cookies
        all_attempts: List[Dict[str, Any]] = []

        # Seed the session
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        current_input = body.responses_create_params
        turn_index = 0

        while True:
            LOG.info("Turn %d: Generating COBOL code attempt", turn_index)

            # Generate code attempt
            gen_response = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=current_input,
                cookies=cookies,
            )
            await raise_for_status(gen_response)
            cookies = gen_response.cookies
            model_response_json = await gen_response.json()

            # Verify the code
            verify_request_data = body.model_dump()
            verify_request_data["response"] = model_response_json

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request_data,
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            cookies = verify_response.cookies
            verify_result = await verify_response.json()

            # Extract generation text for the record
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
                "input": input_dict,
                "response": model_response_json,
                "generation": generation_text,
                "reward": verify_result.get("reward", 0.0),
                "compilation_success": verify_result.get("compilation_success"),
                "compilation_errors": verify_result.get("compilation_errors"),
                "tests_passed": verify_result.get("tests_passed"),
                "tests_total": verify_result.get("tests_total"),
                "extracted_code": verify_result.get("extracted_code"),
            }
            all_attempts.append(attempt_record)

            LOG.info(
                "Turn %d: reward=%s, compilation=%s, tests=%s/%s",
                turn_index,
                verify_result.get("reward"),
                verify_result.get("compilation_success"),
                verify_result.get("tests_passed"),
                verify_result.get("tests_total"),
            )

            # Check if we should continue
            reward = verify_result.get("reward", 0.0)
            turns_remaining = self.config.max_correction_turns - turn_index

            if reward == 1.0:
                LOG.info("Turn %d: All tests passed", turn_index)
                break

            if turns_remaining <= 0:
                LOG.info("Turn %d: Max correction turns exhausted", turn_index)
                break

            # Build error feedback for correction prompt
            error_feedback = _build_error_feedback(verify_result)
            extracted_code = verify_result.get("extracted_code", "")

            if not error_feedback and not extracted_code:
                LOG.warning("Turn %d: No extractable code or error feedback, stopping", turn_index)
                break

            # Extract the original problem from the user message
            problem_description = _extract_problem_description(body.responses_create_params)

            # Add line numbers to previous code
            if extracted_code:
                lines = extracted_code.split("\n")
                numbered_lines = [f"{i:>5} | {line}" for i, line in enumerate(lines, 1)]
                previous_code = "\n".join(numbered_lines)
            else:
                previous_code = "(No code was extracted from your response)"

            correction_prompt = ERROR_CORRECTION_TEMPLATE.format(
                problem_description=problem_description,
                previous_code=previous_code,
                error_feedback=error_feedback,
            )

            LOG.info("Turn %d: Preparing correction turn", turn_index)

            # Build new input with the correction prompt
            params = body.responses_create_params
            current_input = {
                "input": [{"role": "user", "content": correction_prompt}],
                "model": getattr(params, "model", None),
            }
            for key in ["temperature", "max_tokens", "top_p"]:
                value = getattr(params, key, None)
                if value is not None:
                    current_input[key] = value

            turn_index += 1

        # Build final response
        final_response = CobolEvalVerifyResponse.model_validate(verify_result)
        final_response.total_turns = turn_index + 1

        if self.config.include_all_attempts:
            final_response.all_attempts = all_attempts

        return final_response


def _build_error_feedback(verify_result: Dict[str, Any]) -> str:
    """Build error feedback string from verify result."""
    parts: List[str] = []

    # Compilation errors
    if not verify_result.get("compilation_success", True):
        errors = verify_result.get("compilation_errors", [])
        if errors:
            parts.append("COMPILATION ERRORS:")
            for err in errors:
                parts.append(f"  {err}")

    # Test failures
    test_results = verify_result.get("test_results", [])
    failed_tests = [tr for tr in test_results if not tr.get("passed")]
    if failed_tests:
        parts.append("TEST FAILURES:")
        for tr in failed_tests:
            parts.append(f"  Test {tr.get('test_id', '?')}:")
            parts.append(f"    Expected: {tr.get('expected', '?')}")
            parts.append(f"    Got:      {tr.get('actual', '(empty)')}")
            if tr.get("error"):
                parts.append(f"    Error:    {tr['error']}")

    # No code extracted
    if verify_result.get("extracted_code") is None and not parts:
        parts.append("No COBOL code could be extracted from your response.")
        parts.append("Make sure to output complete COBOL code starting with IDENTIFICATION DIVISION.")

    return "\n".join(parts)


def _extract_problem_description(params: Any) -> str:
    """Extract the user's problem description from the responses_create_params."""
    input_messages = getattr(params, "input", None)
    if input_messages is None:
        if isinstance(params, dict):
            input_messages = params.get("input", [])
        else:
            return "(Problem description not available)"

    for msg in input_messages:
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role == "user":
            return content

    return "(Problem description not available)"


if __name__ == "__main__":
    CobolEvalAgent.run_webserver()
