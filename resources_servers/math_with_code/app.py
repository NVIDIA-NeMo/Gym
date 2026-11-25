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
"""Math problems the model solves with a Python tool, graded on the final ``\\boxed{}`` answer alone.

The Python tool is a Simple Agent Python tool that the task rows opt into: the model's code runs in the agent's
sandbox, never in this server. This server only grades, comparing answers without running anything.
"""

import json
from typing import Optional

from fastapi import Request

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class MathWithCodeResourcesServerConfig(BaseResourcesServerConfig):
    pass


class PythonMathRunRequest(BaseRunRequest):
    expected_result: str  # Add the expected result field
    expected_code_contains: str = ""  # Optional validation


class PythonMathVerifyRequest(PythonMathRunRequest, BaseVerifyRequest):
    pass


class PythonMathVerifyResponse(BaseVerifyResponse):
    extracted_answer: Optional[str] = None
    accuracy: bool = False


class MathWithCodeResourcesServer(SimpleResourcesServer):
    config: MathWithCodeResourcesServerConfig
    # verify normalizes the boxed answer and compares strings and floats; it executes nothing.
    verifier_mode = "none"

    # The server holds no state for a session, but implements the session hooks, so verify keeps masking a sample
    # whose session already closed instead of scoring it.
    def has_session_state(self, session_id: str) -> bool:
        return False

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """Nothing to create: the verify request carries the answer and the grading data."""

    async def release_session_state(self, session_id: str) -> None:
        """Nothing to release."""

    async def verify(self, request: Request, body: PythonMathVerifyRequest) -> PythonMathVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self.verifying(session_id, allow_missing=True) as verification:
            if verification.is_lost:
                # An Environment Server verifies before it closes, so this verify is out of order; mask it.
                return PythonMathVerifyResponse(
                    **(body.model_dump() | lost_session_verification(f"math_with_code session {session_id} is closed"))
                )

            # The retry request carries the answer and grading data; no Python worker state is required.
            verification.retain(None)
            expected = body.expected_result

            # Extract actual answer from final assistant message
            actual = None
            for output in reversed(body.response.output):
                if output.type == "message" and output.role == "assistant":
                    text_content = ""
                    for content in output.content:
                        if content.type == "output_text":
                            text_content += content.text

                    actual = _extract_boxed_answer(text_content)
                    if actual is not None:
                        break

            # Fallback: the model may print the boxed answer inside executed code,
            # so search tool output (stdout) as well.
            if actual is None:
                for output in reversed(body.response.output):
                    if output.type == "function_call_output":
                        try:
                            tool_resp = json.loads(output.output)
                            stdout_text = tool_resp.get("stdout", "")
                        except (json.JSONDecodeError, AttributeError):
                            stdout_text = output.output
                        actual = _extract_boxed_answer(stdout_text)
                        if actual is not None:
                            break

            accuracy = _answers_match(actual, expected)
            reward = 1.0 if accuracy else 0.0

            return PythonMathVerifyResponse(
                **body.model_dump(),
                reward=reward,
                extracted_answer=actual,
                accuracy=accuracy,
            )


def _normalize_answer(s: str) -> str:
    """Strip common math delimiters and whitespace for fair comparison.

    Many expected_result values are wrapped in \\(...\\) or $...$,
    but model answers inside \\boxed{} never include those wrappers.
    Without this normalization, ~67% of answers can never match.
    """
    s = s.strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    if s.startswith("\\text{") and s.endswith("}"):
        s = s[6:-1].strip()
    s = " ".join(s.split())
    return s


def _answers_match(actual: Optional[str], expected: str) -> bool:
    """Compare answers after normalization, with numeric fallback."""
    if actual is None:
        return False

    norm_actual = _normalize_answer(actual)
    norm_expected = _normalize_answer(expected)

    # Exact match after normalization
    if norm_actual == norm_expected:
        return True

    # Numeric fallback (handles "42" vs "42.0", "0.5" vs ".5", etc.)
    try:
        fa = float(norm_actual)
        fe = float(norm_expected)
        if abs(fa - fe) < 1e-6 * max(1.0, abs(fe)):
            return True
    except (ValueError, OverflowError):
        pass

    return False


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content inside the last \\boxed{...} in text.

    Uses brace-depth tracking so nested braces are handled correctly
    (e.g. \\boxed{\\frac{1}{2}} correctly returns '\\frac{1}{2}').
    """
    marker = "\\boxed{"
    # Use the last occurrence so chain-of-thought intermediate boxes are skipped.
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    result = text[start : i - 1].strip()
    return result if result else None


# -----------

if __name__ == "__main__":
    MathWithCodeResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = MathWithCodeResourcesServer.run_webserver()  # noqa: F401
