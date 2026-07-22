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
import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.instruction_following.app import (
    InstructionFollowingResourcesServer,
    InstructionFollowingResourcesServerConfig,
    InstructionFollowingVerifyRequest,
)


class TestApp:
    def _create_server(self):
        """Helper to create server instance."""
        config = InstructionFollowingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        return InstructionFollowingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_request_via_verifier_metadata(
        self, instruction_ids, prompt, kwargs, response_content, request_id=1, grading_mode="binary"
    ):
        """Helper to create request using the benchmark verifier_metadata format."""
        response = NeMoGymResponse(
            id=f"resp_test_{request_id}",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": f"msg_test_{request_id}",
                    "content": [{"annotations": [], "text": response_content, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        return InstructionFollowingVerifyRequest(
            id=request_id,
            responses_create_params={"input": [{"role": "user", "content": prompt}]},
            response=response,
            verifier_metadata={
                "prompt": prompt,
                "instruction_id_list": instruction_ids,
                "kwargs": kwargs,
                "grading_mode": grading_mode,
            },
        )

    def _create_real_request(self, instruction_ids, prompt, kwargs, response_content, request_id=1, grading_mode=None):
        """Helper to create real request with NeMoGymResponse."""
        # Create real NeMoGymResponse object
        response = NeMoGymResponse(
            id=f"resp_test_{request_id}",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": f"msg_test_{request_id}",
                    "content": [
                        {
                            "annotations": [],
                            "text": response_content,
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

        # Create real request object
        req_kwargs = dict(
            id=request_id,
            instruction_id_list=instruction_ids,
            prompt=prompt,
            kwargs=kwargs,
            responses_create_params={"input": []},
            response=response,
        )
        if grading_mode is not None:
            req_kwargs["grading_mode"] = grading_mode
        return InstructionFollowingVerifyRequest(**req_kwargs)

    def _run_verify_test(self, real_request, expected_follow_all, expected_reward, expected_follow_list):
        """Helper to run verify method and check results."""
        server = self._create_server()

        # Run the actual verify method with real objects
        result = asyncio.run(server.verify(real_request))

        # Check the actual results
        assert result.follow_all_instructions == expected_follow_all
        assert result.reward == expected_reward
        assert result.follow_instruction_list == expected_follow_list

    def test_sanity(self) -> None:
        config = InstructionFollowingResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        InstructionFollowingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_instruction_following_imports(self) -> None:
        """Test that we can import and use the verifiable-instructions library."""
        from verifiable_instructions import instructions_registry

        # Test that we can access the instruction dictionary
        assert "detectable_format:title" in instructions_registry.INSTRUCTION_DICT
        assert "length_constraints:number_words" in instructions_registry.INSTRUCTION_DICT

        # Test that we can create an instruction instance
        title_cls = instructions_registry.INSTRUCTION_DICT["detectable_format:title"]
        title_instruction = title_cls("detectable_format:title")

        # Should have the required methods
        assert hasattr(title_instruction, "build_description")
        assert hasattr(title_instruction, "check_following")

    def test_title_positive(self):
        """Test the verify method with title instruction."""
        real_request = self._create_real_request(
            instruction_ids=["detectable_format:title"],
            prompt="Write the entire response with a title.",
            kwargs=[{}],
            response_content="<<My Title>>\n\nThis is the content of my response.",
        )
        self._run_verify_test(real_request, True, 1.0, [True])

    def test_title_negative(self):
        """Test title instruction - should fail."""
        real_request = self._create_real_request(
            instruction_ids=["detectable_format:title"],
            prompt="Write the entire response with a title.",
            kwargs=[{}],
            response_content="This response has no title.",
        )
        self._run_verify_test(real_request, False, 0.0, [False])

    def test_no_comma_positive(self):
        """Test punctuation:no_comma instruction - should pass."""
        real_request = self._create_real_request(
            instruction_ids=["punctuation:no_comma"],
            prompt="The output should not contain any commas.",
            kwargs=[{}],
            response_content="Hello world without commas at all",
        )
        self._run_verify_test(real_request, True, 1.0, [True])

    def test_no_comma_negative(self):
        """Test punctuation:no_comma instruction - should fail."""
        real_request = self._create_real_request(
            instruction_ids=["punctuation:no_comma"],
            prompt="The output should not contain any commas.",
            kwargs=[{}],
            response_content="Hello, world with commas",
        )
        self._run_verify_test(real_request, False, 0.0, [False])

    def test_word_count_positive(self):
        """Test length_constraints:number_words instruction - should pass."""
        real_request = self._create_real_request(
            instruction_ids=["length_constraints:number_words"],
            prompt="The response must contain at least 6 words.",
            kwargs=[{"num_words": 6, "relation": "at least"}],
            response_content="This response has exactly six words.",
            request_id=100,
        )
        self._run_verify_test(real_request, True, 1.0, [True])

    def test_word_count_negative(self):
        """Test length_constraints:number_words instruction - should fail."""
        real_request = self._create_real_request(
            instruction_ids=["length_constraints:number_words"],
            prompt="The response must contain at least 10 words.",
            kwargs=[{"num_words": 10, "relation": "at least"}],
            response_content="Too short.",
        )
        self._run_verify_test(real_request, False, 0.0, [False])

    def test_multiple_constraints_positive(self):
        """Test multiple constraints together - should pass both."""
        real_request = self._create_real_request(
            instruction_ids=["detectable_format:title", "punctuation:no_comma"],
            prompt="Write a response with a title and no commas.",
            kwargs=[{}, {}],
            response_content="<<My Great Title>>\n\nThis is a response without any commas at all.",
            request_id=200,
        )
        self._run_verify_test(real_request, True, 1.0, [True, True])

    def test_multiple_constraints_negative(self):
        """Test multiple constraints together - should fail one."""
        real_request = self._create_real_request(
            instruction_ids=["detectable_format:title", "punctuation:no_comma"],
            prompt="Write a response with a title and no commas.",
            kwargs=[{}, {}],
            response_content="<<My Great Title>>\n\nThis response has commas, which should fail.",
            request_id=201,
        )
        self._run_verify_test(real_request, False, 0.0, [True, False])

    def test_fractional_reward_half(self):
        """Fractional grading: half of constraints satisfied -> reward 0.5."""
        real_request = self._create_real_request(
            instruction_ids=["detectable_format:title", "punctuation:no_comma"],
            prompt="Write a response with a title and no commas.",
            kwargs=[{}, {}],
            response_content="<<My Great Title>>\n\n with a comma here, okay.",
            request_id=300,
            grading_mode="fraction",
        )
        self._run_verify_test(real_request, False, 0.5, [True, False])

    def test_verifier_metadata_path(self):
        """Benchmark format: fields supplied via verifier_metadata are resolved correctly."""
        real_request = self._create_request_via_verifier_metadata(
            instruction_ids=["detectable_format:title"],
            prompt="Write the entire response with a title.",
            kwargs=[{}],
            response_content="<<My Title>>\n\nThis is the content of my response.",
        )
        self._run_verify_test(real_request, True, 1.0, [True])

    def test_verifier_metadata_preserved_in_response(self):
        """verifier_metadata from the request should appear in the verify response."""
        vm = {
            "prompt": "Write the entire response with a title.",
            "instruction_id_list": ["detectable_format:title"],
            "kwargs": [{}],
            "grading_mode": "binary",
        }
        real_request = self._create_request_via_verifier_metadata(
            instruction_ids=vm["instruction_id_list"],
            prompt=vm["prompt"],
            kwargs=vm["kwargs"],
            response_content="<<My Title>>\n\nContent here.",
        )
        server = self._create_server()
        result = asyncio.run(server.verify(real_request))
        assert result.verifier_metadata == vm

    def test_missing_instruction_id_list_raises(self):
        """Omitting instruction_id_list (neither top-level nor in verifier_metadata) should raise."""
        with pytest.raises(ValidationError):
            InstructionFollowingVerifyRequest(
                id=1,
                responses_create_params={"input": []},
                response=NeMoGymResponse(
                    id="r", created_at=0.0, model="m", object="response",
                    output=[], parallel_tool_calls=True, tool_choice="auto", tools=[],
                ),
                prompt="some prompt",
                kwargs=[{}],
                # instruction_id_list intentionally omitted
            )

    def test_missing_kwargs_raises(self):
        """Omitting kwargs (neither top-level nor in verifier_metadata) should raise."""
        with pytest.raises(ValidationError):
            InstructionFollowingVerifyRequest(
                id=1,
                responses_create_params={"input": []},
                response=NeMoGymResponse(
                    id="r", created_at=0.0, model="m", object="response",
                    output=[], parallel_tool_calls=True, tool_choice="auto", tools=[],
                ),
                prompt="some prompt",
                instruction_id_list=["detectable_format:title"],
                # kwargs intentionally omitted
            )

    def test_missing_prompt_raises(self):
        """Omitting prompt (neither top-level nor in verifier_metadata) should raise."""
        with pytest.raises(ValidationError):
            InstructionFollowingVerifyRequest(
                id=1,
                responses_create_params={"input": []},
                response=NeMoGymResponse(
                    id="r", created_at=0.0, model="m", object="response",
                    output=[], parallel_tool_calls=True, tool_choice="auto", tools=[],
                ),
                instruction_id_list=["detectable_format:title"],
                kwargs=[{}],
                # prompt intentionally omitted
            )
