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
import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.swerl_gen.app import (
    SWEGenResourcesServer,
    SWEGenResourcesServerConfig,
    SWEGenVerifyRequest,
    SWEGenVerifyResponse,
)


class AwaitableResult:
    """Small helper to mock an awaitable Ray ObjectRef-like return value."""

    def __init__(self, value):
        self._value = value

    def __await__(self):
        async def _coro():
            return self._value

        return _coro().__await__()


def create_test_config() -> SWEGenResourcesServerConfig:
    return SWEGenResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        num_processes=1,
        sandbox_timeout=600,
        debug=False,
        relaxed_formatting=False,
    )


def create_nemogym_response(text: str) -> NeMoGymResponse:
    """Create a minimal NeMoGymResponse with a single assistant message."""
    return NeMoGymResponse(
        id="successful_execution",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_successful_execution",
                "content": [
                    {
                        "annotations": [],
                        "text": text,
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def create_verify_request(
    text: str,
    reward_mode: str = "eval",
    relevant_file_contents: Dict[str, Any] | None = None,
) -> SWEGenVerifyRequest:
    if relevant_file_contents is None:
        relevant_file_contents = {}

    response = create_nemogym_response(text)

    return SWEGenVerifyRequest(
        responses_create_params={
            "input": [
                {
                    "role": "user",
                    "content": "You are a helpful assistant. Please fix the bug in the code.",
                },
            ],
            "parallel_tool_calls": False,
            "temperature": 0,
        },
        response=response,
        instance={
            "instance_id": "test_instance_123",
            "repo": "test_repo",
            "setup_script": "test_setup_script",
            "test_script": "test_test_script",
            "regression_script": "test_regression_script",
            "PASS_TO_PASS": "test_PASS_TO_PASS",
            "FAIL_TO_PASS": "test_FAIL_TO_PASS",
            "patch": "test_patch",
        },
        metadata={
            "relevant_file_contents": json.dumps(relevant_file_contents),
            "remove_repo_name": False,
            "image": "test_image",
        },
        mode=reward_mode,
    )


class TestApp:
    def test_sanity(self) -> None:
        config = create_test_config()
        SWEGenResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @patch("resources_servers.swerl_gen.app.compute_score")
    @patch("resources_servers.swerl_gen.app.extract_pred_patch")
    async def test_verify_patch_gen_successful_execution(
        self,
        mock_extract_pred_patch,
        mock_compute_score,
    ) -> None:
        """Test successful verification flow when a patch is extracted and compute_score succeeds."""
        server = SWEGenResourcesServer(
            config=create_test_config(),
            server_client=MagicMock(spec=ServerClient),
        )

        # Mock patch extraction to return a valid patch string
        model_patch = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py"
        mock_extract_pred_patch.return_value = {"model_patch": model_patch}

        # Mock compute_score.remote to avoid initializing Ray
        mock_compute_score.remote.return_value = AwaitableResult(
            (
                1.0,
                {
                    "status": "done",
                    "resolution": "RESOLVED_FULL",
                    "return_codes_after_patch": None,
                    "return_codes_before_patch": None,
                },
            )
        )

        verify_request = create_verify_request(
            text="I've successfully fixed the bug in the code.",
            relevant_file_contents={},
        )

        result = await server.verify(verify_request)

        assert isinstance(result, SWEGenVerifyResponse)
        assert result.reward == 1.0
        assert result.verification_result == {
            "status": "done",
            "resolution": "RESOLVED_FULL",
            "return_codes_after_patch": None,
            "return_codes_before_patch": None,
        }
        assert result.model_patch == model_patch

    @patch("resources_servers.swerl_gen.app.compute_score")
    @patch("resources_servers.swerl_gen.app.extract_repro_test")
    async def test_verify_test_gen_successful_execution(
        self,
        mock_extract_repro_test,
        mock_compute_score,
    ) -> None:
        """Test successful verification flow when a patch is extracted and compute_score succeeds."""
        server = SWEGenResourcesServer(
            config=create_test_config(),
            server_client=MagicMock(spec=ServerClient),
        )

        # Mock repro test extraction to return a valid repro_test_info_base64
        repro_test_info_base64 = "def test_issue():\n    if foo == bar:\n        exit(0)\n    else:\n        exit(2)"
        mock_extract_repro_test.return_value = {"repro_test_info_base64": repro_test_info_base64}

        # Mock compute_score.remote to avoid initializing Ray
        mock_compute_score.remote.return_value = AwaitableResult(
            (
                1.0,
                {
                    "status": "done",
                    "resolution": None,
                    "return_codes_after_patch": [0],
                    "return_codes_before_patch": [2],
                },
            )
        )

        verify_request = create_verify_request(
            text="I've successfully fixed the bug in the code.",
            reward_mode="repro-gen",
            relevant_file_contents={},
        )

        result = await server.verify(verify_request)

        assert isinstance(result, SWEGenVerifyResponse)
        assert result.reward == 1.0
        assert result.verification_result == {
            "status": "done",
            "resolution": None,
            "return_codes_after_patch": [0],
            "return_codes_before_patch": [2],
        }
        assert result.repro_test_info_base64 == repro_test_info_base64

    @patch("resources_servers.swerl_gen.app.extract_pred_patch")
    async def test_verify_extraction_failure(
        self,
        mock_extract_pred_patch,
    ) -> None:
        """Test verification flow when no patch can be extracted from the model output."""
        server = SWEGenResourcesServer(
            config=create_test_config(),
            server_client=MagicMock(spec=ServerClient),
        )

        # Simulate failure to extract a patch
        mock_extract_pred_patch.return_value = None

        verify_request = create_verify_request(
            text="Model output without a usable patch.",
            relevant_file_contents={},
        )

        result = await server.verify(verify_request)

        assert isinstance(result, SWEGenVerifyResponse)
        assert result.reward == 0.0
        assert result.verification_result is None
        assert result.model_patch is None
