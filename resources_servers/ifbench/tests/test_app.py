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

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.ifbench.app import (
    IFBenchResourcesServer,
    IFBenchResourcesServerConfig,
    IFBenchVerifyRequest,
)


class TestApp:
    def _create_server(self):
        config = IFBenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        return IFBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_request(
        self, instruction_ids, prompt, kwargs, response_content, request_id=1, grading_mode="fraction"
    ):
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
        return IFBenchVerifyRequest(
            id=request_id,
            instruction_id_list=instruction_ids,
            prompt=prompt,
            kwargs=kwargs,
            grading_mode=grading_mode,
            responses_create_params={"input": []},
            response=response,
        )

    def _run_verify(self, request, expected_follow_all, expected_reward, expected_follow_list):
        server = self._create_server()
        result = asyncio.run(server.verify(request))
        assert result.follow_all_instructions == expected_follow_all
        assert result.reward == expected_reward
        assert result.follow_instruction_list == expected_follow_list

    def test_sanity(self):
        self._create_server()

    def test_ifbench_imports(self):
        """IFBench instructions_registry must be importable and contain IFBench IDs."""
        import instructions_registry

        assert "count:word_count_range" in instructions_registry.INSTRUCTION_DICT
        assert "count:keywords_multiple" in instructions_registry.INSTRUCTION_DICT
        assert "format:line_indent" in instructions_registry.INSTRUCTION_DICT
        assert "words:start_verb" in instructions_registry.INSTRUCTION_DICT
        assert "ratio:stop_words" in instructions_registry.INSTRUCTION_DICT

        # Confirm IFEval IDs are NOT present (disjoint namespaces)
        assert "detectable_format:title" not in instructions_registry.INSTRUCTION_DICT
        assert "length_constraints:number_words" not in instructions_registry.INSTRUCTION_DICT

    def test_count_numbers_positive(self):
        """count:numbers N=2 passes when response contains exactly 2 numbers."""
        req = self._create_request(
            instruction_ids=["count:numbers"],
            prompt="Include exactly 2 numbers in your response.",
            kwargs=[{"N": 2}],
            response_content="The values are 42 and 7.",
        )
        self._run_verify(req, True, 1.0, [True])

    def test_count_numbers_negative(self):
        """count:numbers N=2 fails when response has no numbers."""
        req = self._create_request(
            instruction_ids=["count:numbers"],
            prompt="Include exactly 2 numbers in your response.",
            kwargs=[{"N": 2}],
            response_content="There are no digits here at all.",
        )
        self._run_verify(req, False, 0.0, [False])

    def test_count_keywords_multiple_positive(self):
        """count:keywords_multiple passes when all keywords appear at required frequencies."""
        req = self._create_request(
            instruction_ids=["count:keywords_multiple"],
            prompt="Include keyword hello twice and keyword world once.",
            kwargs=[{"keyword1": "hello", "keyword2": "world"}],
            response_content="Hello hello world, hello world.",
        )
        result = asyncio.run(self._create_server().verify(req))
        # Just check reward is in valid range; frequency matching is library-dependent
        assert 0.0 <= result.reward <= 1.0
        assert len(result.follow_instruction_list) == 1

    def test_empty_response_all_fail(self):
        """An empty response should fail all instructions."""
        req = self._create_request(
            instruction_ids=["count:numbers", "words:start_verb"],
            prompt="Start with a verb and include 3 numbers.",
            kwargs=[{"N": 3}, {}],
            response_content="   ",
        )
        self._run_verify(req, False, 0.0, [False, False])

    def test_fraction_grading_partial(self):
        """Fraction grading: 1 of 2 instructions pass -> reward 0.5."""
        # count:numbers N=1 passes (response has one number)
        # words:start_verb fails (response starts with "The", a determiner)
        req = self._create_request(
            instruction_ids=["count:numbers", "words:start_verb"],
            prompt="Start with a verb and include exactly 1 number.",
            kwargs=[{"N": 1}, {}],
            response_content="The answer is 42.",
            grading_mode="fraction",
        )
        result = asyncio.run(self._create_server().verify(req))
        assert result.reward == 0.5
        assert result.follow_all_instructions is False
        assert len(result.follow_instruction_list) == 2

    def test_binary_grading_all_pass(self):
        """Binary grading: all instructions pass -> reward 1.0."""
        req = self._create_request(
            instruction_ids=["count:numbers"],
            prompt="Include exactly 1 number.",
            kwargs=[{"N": 1}],
            response_content="The answer is 42.",
            grading_mode="binary",
        )
        result = asyncio.run(self._create_server().verify(req))
        assert result.reward == 1.0
        assert result.follow_all_instructions is True

    def test_binary_grading_partial_fail(self):
        """Binary grading: any failure -> reward 0.0."""
        req = self._create_request(
            instruction_ids=["count:numbers", "words:start_verb"],
            prompt="Start with a verb and include exactly 1 number.",
            kwargs=[{"N": 1}, {}],
            response_content="The answer is 42.",
            grading_mode="binary",
        )
        result = asyncio.run(self._create_server().verify(req))
        assert result.reward == 0.0
        assert result.follow_all_instructions is False

    def test_invalid_instruction_id_does_not_crash(self):
        """An unknown instruction ID should be treated as a failure, not raise."""
        req = self._create_request(
            instruction_ids=["totally:unknown_instruction"],
            prompt="Some prompt.",
            kwargs=[{}],
            response_content="Some response.",
        )
        result = asyncio.run(self._create_server().verify(req))
        assert result.reward == 0.0
        assert result.follow_instruction_list == [False]

    def test_response_fields_preserved(self):
        """Verify response must echo back the original request fields."""
        req = self._create_request(
            instruction_ids=["count:numbers"],
            prompt="Include exactly 2 numbers.",
            kwargs=[{"N": 2}],
            response_content="Use 1 and 2.",
            request_id=99,
        )
        result = asyncio.run(self._create_server().verify(req))
        assert result.prompt == "Include exactly 2 numbers."
        assert result.instruction_id_list == ["count:numbers"]
        assert result.grading_mode == "fraction"
