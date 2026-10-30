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
from nemo_gym.client import ServerClient
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

    def _create_real_request(
        self, instruction_ids, prompt, kwargs, response_content, request_id=1, grading_mode="binary"
    ):
        """Helper to create real request with NeMoGymResponse."""
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
        return InstructionFollowingVerifyRequest(
            id=request_id,
            responses_create_params={"input": [{"role": "user", "content": prompt}]},
            response=response,
            verifier_metadata={
                "instruction_id_list": instruction_ids,
                "prompt": prompt,
                "kwargs": kwargs,
                "grading_mode": grading_mode,
            },
        )

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

    def _create_counting_composition_request(self, n_sent, n_words, response_content):
        """Helper to create a count:counting_composition request with the instruction text it is graded against."""
        return self._create_real_request(
            instruction_ids=["count:counting_composition"],
            prompt=(
                "Write 3 paragraphs, delimited by the markdown divider: * * *, "
                f"with exactly {n_sent} sentences each, with exactly {n_words} words in each sentence."
            ),
            kwargs=[{"n_sent": n_sent, "n_words": n_words}],
            response_content=response_content,
        )

    @pytest.mark.parametrize(
        "n_sent, n_words, response_content",
        [
            pytest.param(2, 2, "\n* * *\n".join(["Cats sleep. Dogs run."] * 3), id="divider-from-instruction"),
            pytest.param(2, 2, "\n* * *\n".join(["Cats, sleep! Dogs run?"] * 3), id="punctuation-is-not-a-word"),
            pytest.param(
                3,
                3,
                "\n\n* * *\n\n".join(["Cats sleep soundly. Dogs run fast. Birds sing loudly."] * 3),
                id="blank-lines-around-divider",
            ),
            pytest.param(
                2,
                2,
                "Cats sleep. Dogs run.\r\n  * * *  \r\nBirds sing. Fish swim.\r\n* * *\r\nCows graze. Owls hoot.",
                id="crlf-and-padded-divider",
            ),
            pytest.param(
                2,
                2,
                "* * *\n" + "\n* * *\n".join(["Cats sleep. Dogs run."] * 3) + "\n* * *",
                id="outer-dividers",
            ),
        ],
    )
    def test_counting_composition_positive(self, n_sent, n_words, response_content):
        """Test count:counting_composition instruction - should pass when the response follows it."""
        real_request = self._create_counting_composition_request(n_sent, n_words, response_content)
        self._run_verify_test(real_request, True, 1.0, [True])

    @pytest.mark.parametrize(
        "response_content",
        [
            # The instruction names the spaced "* * *" divider; "***" is what paragraphs:paragraphs asks for.
            pytest.param("\n***\n".join(["Cats sleep. Dogs run."] * 3), id="unspaced-divider"),
            pytest.param("\n***\n".join(["Sleep. Run."] * 3), id="unspaced-divider-one-word-sentences"),
            pytest.param("\n* * *\n".join(["Sleep. Run."] * 3), id="one-word-sentences"),
            pytest.param("\n* * *\n".join(["Cats sleep soundly. Dogs run."] * 3), id="three-word-sentence"),
            pytest.param("\n* * *\n".join(["Cats sleep. Dogs run. Birds sing."] * 3), id="three-sentences"),
            pytest.param("\n* * *\n".join(["Cats sleep. Dogs run."] * 2), id="two-paragraphs"),
            pytest.param("\n* * *\n".join(["Cats sleep. Dogs run."] * 4), id="four-paragraphs"),
            pytest.param(" * * * ".join(["Cats sleep. Dogs run."] * 3), id="inline-divider"),
            pytest.param(
                "Cats sleep. Dogs run.\n* * *\n\n* * *\nCats sleep. Dogs run.\n* * *\nCats sleep. Dogs run.",
                id="empty-middle-paragraph",
            ),
        ],
    )
    def test_counting_composition_negative(self, response_content):
        """Test count:counting_composition instruction (2 sentences of 2 words) - should fail."""
        real_request = self._create_counting_composition_request(2, 2, response_content)
        self._run_verify_test(real_request, False, 0.0, [False])

    def _create_count_unique_request(self, response_content, kwargs=None):
        """Helper to create a count:count_unique request with the instruction text it is graded against.

        The published rows store `null` as this instruction's kwargs, so the default mirrors them.
        """
        return self._create_real_request(
            instruction_ids=["count:count_unique"],
            prompt="Only use unique words in your response, no word should be repeated!",
            kwargs=[kwargs],
            response_content=response_content,
        )

    @pytest.mark.parametrize(
        "response_content",
        [
            pytest.param("Cats purr. Dogs bark.", id="two-sentences-only-the-period-repeats"),
            pytest.param("Cats purr, dogs bark, birds sing.", id="repeated-commas"),
            pytest.param("Really? Why run? Because cats chase!", id="repeated-question-marks"),
            pytest.param("“Cats purr,” she said. “Dogs bark,” he replied.", id="curly-quotes-also-dropped"),
            pytest.param("- cats purr\n- dogs bark\n- birds sing", id="markdown-bullets-dropped"),
            pytest.param("I paid $5 and $6.", id="symbols-dropped-digits-are-distinct-words"),
            pytest.param("Hello world.", id="one-sentence-passed-upstream-too"),
            pytest.param("Dogs bark. dogs run.", id="case-kept-as-upstream"),
        ],
    )
    def test_count_unique_positive(self, response_content):
        """Test count:count_unique instruction - should pass when no word repeats, whatever the punctuation."""
        real_request = self._create_count_unique_request(response_content)
        self._run_verify_test(real_request, True, 1.0, [True])

    @pytest.mark.parametrize(
        "response_content",
        [
            pytest.param("Cats purr. Cats bark.", id="word-repeated-across-sentences"),
            pytest.param("run run", id="word-repeated-without-punctuation"),
            pytest.param("Dogs bark, dogs bark.", id="bark-repeats-while-Dogs-and-dogs-differ"),
            pytest.param("Don't stop. Can't wait.", id="contractions-split-as-upstream"),
        ],
    )
    def test_count_unique_negative(self, response_content):
        """Test count:count_unique instruction - should fail when a word repeats."""
        real_request = self._create_count_unique_request(response_content)
        self._run_verify_test(real_request, False, 0.0, [False])

    def test_count_unique_checker_keeps_upstream_instruction_text(self):
        """Only `check_following` is overridden, so the published prompts still describe what is graded."""
        from verifiable_instructions import instructions_registry

        from resources_servers.instruction_following.count_unique import CountUniqueChecker

        upstream = instructions_registry.INSTRUCTION_DICT["count:count_unique"]("count:count_unique")
        local = CountUniqueChecker("count:count_unique")
        assert local.build_description() == upstream.build_description()
        assert local.get_instruction_args() == upstream.get_instruction_args()
        assert local.get_instruction_args_keys() == upstream.get_instruction_args_keys()

    def test_only_local_checkers_replace_upstream(self):
        """Only the two local count: checkers are overridden; every other instruction uses the upstream checker."""
        from verifiable_instructions import instructions_registry

        from resources_servers.instruction_following.app import INSTRUCTION_DICT
        from resources_servers.instruction_following.count_unique import CountUniqueChecker
        from resources_servers.instruction_following.counting_composition import CountingCompositionChecker

        upstream = instructions_registry.INSTRUCTION_DICT
        assert INSTRUCTION_DICT.keys() == upstream.keys()
        assert {key for key, cls in INSTRUCTION_DICT.items() if cls is not upstream[key]} == {
            "count:counting_composition",
            "count:count_unique",
        }
        assert INSTRUCTION_DICT["count:counting_composition"] is CountingCompositionChecker
        assert INSTRUCTION_DICT["count:count_unique"] is CountUniqueChecker

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

    def test_missing_verifier_metadata_fields_raises(self):
        """verifier_metadata missing required fields should raise at parse time."""
        with pytest.raises(ValidationError, match="verifier_metadata is missing required fields"):
            InstructionFollowingVerifyRequest(
                id=1,
                responses_create_params={"input": []},
                response=NeMoGymResponse(
                    id="r",
                    created_at=0.0,
                    model="m",
                    object="response",
                    output=[],
                    parallel_tool_calls=True,
                    tool_choice="auto",
                    tools=[],
                ),
                verifier_metadata={"prompt": "some prompt"},  # missing instruction_id_list and kwargs
            )

    def _make_response(self, response_content, request_id=1):
        """Helper to build a NeMoGymResponse carrying a single assistant text output."""
        return NeMoGymResponse(
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

    def test_legacy_top_level_format(self):
        """Backward compat: rows with verifier fields at the top level (no verifier_metadata)
        should still parse and verify, matching published HF blends."""
        request = InstructionFollowingVerifyRequest(
            id=1,
            instruction_id_list=["detectable_format:title"],
            prompt="Write the entire response with a title.",
            kwargs=[{}],
            responses_create_params={"input": [{"role": "user", "content": "prompt"}]},
            response=self._make_response("<<My Title>>\n\nThis is the content of my response."),
        )
        # Legacy top-level fields are migrated into verifier_metadata.
        assert request.verifier_metadata["instruction_id_list"] == ["detectable_format:title"]
        assert request.verifier_metadata["kwargs"] == [{}]
        self._run_verify_test(request, True, 1.0, [True])

    def test_legacy_grading_mode_migrated(self):
        """Backward compat: a top-level grading_mode is migrated into verifier_metadata."""
        request = InstructionFollowingVerifyRequest(
            id=1,
            instruction_id_list=["detectable_format:title", "punctuation:no_comma"],
            prompt="Write a response with a title and no commas.",
            kwargs=[{}, {}],
            grading_mode="fraction",
            responses_create_params={"input": [{"role": "user", "content": "prompt"}]},
            response=self._make_response("<<My Great Title>>\n\n with a comma here, okay."),
        )
        assert request.verifier_metadata["grading_mode"] == "fraction"
        self._run_verify_test(request, False, 0.5, [True, False])

    def test_explicit_verifier_metadata_takes_precedence(self):
        """When both top-level fields and verifier_metadata are present, verifier_metadata wins
        and the request parses without error."""
        request = InstructionFollowingVerifyRequest(
            id=1,
            # Legacy top-level fields point at a different instruction; they must be ignored.
            instruction_id_list=["punctuation:no_comma"],
            prompt="stale top-level prompt",
            kwargs=[{}],
            responses_create_params={"input": [{"role": "user", "content": "prompt"}]},
            response=self._make_response("<<My Title>>\n\nThis is the content of my response."),
            verifier_metadata={
                "instruction_id_list": ["detectable_format:title"],
                "prompt": "Write the entire response with a title.",
                "kwargs": [{}],
            },
        )
        assert request.verifier_metadata["instruction_id_list"] == ["detectable_format:title"]
        self._run_verify_test(request, True, 1.0, [True])
