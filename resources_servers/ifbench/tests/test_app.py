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

    def _create_request(self, instruction_ids, prompt, kwargs, response_content, request_id=1, grading_mode=None):
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
        return IFBenchVerifyRequest(**req_kwargs)

    def _run_verify(self, request):
        server = self._create_server()
        return asyncio.run(server.verify(request))

    def test_sanity(self):
        self._create_server()

    def test_ifbench_registry_imported(self):
        """All 57 IFBench instruction IDs must be present after ensure_ifbench()."""
        import instructions_registry

        expected_ids = [
            "count:numbers",
            "format:emoji",
            "count:conjunctions",
            "ratio:stop_words",
            "words:start_verb",
            "format:line_indent",
            "count:person_names",
            "count:keywords_multiple",
        ]
        for id_ in expected_ids:
            assert id_ in instructions_registry.INSTRUCTION_DICT, f"Missing instruction ID: {id_}"

    def test_empty_response_fails_all(self):
        """An empty response must fail every instruction and return reward=0."""
        request = self._create_request(
            instruction_ids=["count:numbers"],
            prompt="Include exactly 3 numbers in your response.",
            kwargs=[{"N": 3}],
            response_content="",
        )
        result = self._run_verify(request)
        assert result.reward == 0.0
        assert result.follow_all_instructions is False
        assert result.follow_instruction_list == [False]

    def test_unknown_instruction_fails_gracefully(self):
        """An unknown instruction ID must fail gracefully (no exception, reward=0)."""
        request = self._create_request(
            instruction_ids=["nonexistent:fake_instruction"],
            prompt="Some prompt.",
            kwargs=[{}],
            response_content="Some response.",
        )
        result = self._run_verify(request)
        assert result.reward == 0.0
        assert result.follow_instruction_list == [False]

    def test_count_numbers_positive(self):
        """count:numbers: response contains exactly N=2 numbers -> passes."""
        request = self._create_request(
            instruction_ids=["count:numbers"],
            prompt="Include exactly 2 numbers in your response.",
            kwargs=[{"N": 2}],
            response_content="The first answer is 42 and the second is 7.",
        )
        result = self._run_verify(request)
        assert result.follow_instruction_list == [True]
        assert result.reward == 1.0

    def test_count_numbers_negative(self):
        """count:numbers: response contains wrong number of numbers -> fails."""
        request = self._create_request(
            instruction_ids=["count:numbers"],
            prompt="Include exactly 2 numbers in your response.",
            kwargs=[{"N": 2}],
            response_content="There are no digits here at all.",
        )
        result = self._run_verify(request)
        assert result.follow_instruction_list == [False]
        assert result.reward == 0.0

    def test_fraction_grading(self):
        """With two instructions and one passing, fraction reward should be 0.5."""
        # count:numbers N=1 should pass; count:numbers N=99 should fail.
        request = self._create_request(
            instruction_ids=["count:numbers", "count:numbers"],
            prompt="Include exactly 1 number. Include exactly 99 numbers.",
            kwargs=[{"N": 1}, {"N": 99}],
            response_content="The answer is 42.",
            grading_mode="fraction",
        )
        result = self._run_verify(request)
        assert len(result.follow_instruction_list) == 2
        assert result.reward == 0.5

    def test_binary_grading_partial_fail(self):
        """Binary grading: one failing instruction -> reward=0 even if one passes."""
        request = self._create_request(
            instruction_ids=["count:numbers", "count:numbers"],
            prompt="Include exactly 1 number. Include exactly 99 numbers.",
            kwargs=[{"N": 1}, {"N": 99}],
            response_content="The answer is 42.",
            grading_mode="binary",
        )
        result = self._run_verify(request)
        assert result.reward == 0.0
        assert result.follow_all_instructions is False

    def test_follow_instruction_list_length_matches(self):
        """follow_instruction_list must always have the same length as instruction_id_list."""
        request = self._create_request(
            instruction_ids=["count:numbers", "count:conjunctions", "ratio:stop_words"],
            prompt="Some multi-constraint prompt.",
            kwargs=[{"N": 1}, {"small_n": 2}, {}],
            response_content="The answer is 42 and it is correct.",
        )
        result = self._run_verify(request)
        assert len(result.follow_instruction_list) == 3
        assert all(isinstance(v, bool) for v in result.follow_instruction_list)
        assert 0.0 <= result.reward <= 1.0
