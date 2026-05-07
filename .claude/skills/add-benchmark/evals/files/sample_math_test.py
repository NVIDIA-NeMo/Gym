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
"""Tests for the math algebra resources server."""

import pytest


@pytest.fixture
def server():
    from sample_math_server import MathAlgebraServer

    return MathAlgebraServer()


@pytest.mark.asyncio
async def test_verify_pass(server):
    """Correct answer should get reward 1.0."""
    result = await server.verify(
        {
            "output_text": "Let me solve this.\nx + 5 = 12\nx = 7\nAnswer: 7",
            "verifier_metadata": {"expected_answer": "7"},
        }
    )
    assert result["reward"] == 1.0
    assert result["extracted_answer"] == "7"


@pytest.mark.asyncio
async def test_verify_fail_wrong_answer(server):
    """Wrong answer should get reward 0.0."""
    result = await server.verify(
        {
            "output_text": "I think the answer is:\nAnswer: 5",
            "verifier_metadata": {"expected_answer": "7"},
        }
    )
    assert result["reward"] == 0.0
    assert result["reason"] == "wrong_answer"


@pytest.mark.asyncio
async def test_verify_fail_no_answer(server):
    """Missing 'Answer:' marker should get reward 0.0."""
    result = await server.verify(
        {
            "output_text": "The solution is 7, which we can verify by substituting back.",
            "verifier_metadata": {"expected_answer": "7"},
        }
    )
    assert result["reward"] == 0.0
    assert result["reason"] == "no_answer_marker"


@pytest.mark.asyncio
async def test_verify_fail_think_block(server):
    """Answer only inside think block should get 0.0 after stripping."""
    result = await server.verify(
        {
            "output_text": "<think>\nThe answer is 7.\nAnswer: 7\n</think>",
            "verifier_metadata": {"expected_answer": "7"},
        }
    )
    assert result["reward"] == 0.0
    assert result["reason"] == "no_answer_marker"
