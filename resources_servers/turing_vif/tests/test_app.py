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

"""Unit tests for the Turing VIF Resource Server."""

import asyncio
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.turing_vif.app import (
    TuringVIFResourcesServer,
    TuringVIFResourcesServerConfig,
    TuringVIFVerifyRequest,
)


def _normalize_instructions(instructions):
    """Ensure each instruction has uid, source, is_misalignment_check for schema validation."""
    normalized = []
    for i, inst in enumerate(instructions):
        if isinstance(inst, dict):
            inst = dict(inst)
            inst.setdefault("uid", i + 1)
            inst.setdefault("source", "system")
            inst.setdefault("is_misalignment_check", False)
        normalized.append(inst)
    return normalized


class TestTuringVIFApp:
    """Test suite for the Turing VIF Resource Server."""

    def _create_server(self):
        """Helper to create server instance."""
        config = TuringVIFResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return TuringVIFResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_real_request(self, instructions, response_content, request_id=1, llm_judge=None):
        """Helper to create real request with NeMoGymResponse.

        Normalizes instructions so each has required schema fields (uid, source,
        is_misalignment_check) for validate_instructions_schema.
        """
        # Ensure each instruction has uid, source, is_misalignment_check so schema validation passes
        normalized = _normalize_instructions(instructions)

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

        return TuringVIFVerifyRequest(
            id=request_id,
            instructions=normalized,
            llm_judge=llm_judge or [],
            responses_create_params={"input": []},
            response=response,
        )

    def _run_verify_test(self, request, expected_follow_all, expected_reward, expected_follow_list=None):
        """Helper to run verify method and check results."""
        server = self._create_server()
        result = asyncio.run(server.verify(request))

        assert result.follow_all_instructions == expected_follow_all
        assert result.reward == expected_reward
        if expected_follow_list is not None:
            assert result.follow_instruction_list == expected_follow_list

    # ========================================================================
    # Sanity Tests
    # ========================================================================

    def test_sanity(self) -> None:
        """Test that the server can be instantiated."""
        config = TuringVIFResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        TuringVIFResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_validator_imports(self) -> None:
        """Test that VIF validators can be imported."""
        from resources_servers.turing_vif.vif_validators.data_loader import (
            EXPECTED_ARGUMENTS,
            LLM_INSTRUCTIONS,
        )

        assert len(LLM_INSTRUCTIONS) > 0
        assert len(EXPECTED_ARGUMENTS) > 0

    # ========================================================================
    # Word Count Tests
    # ========================================================================

    def test_word_count_positive(self):
        """Test length_constraints:number_words instruction - should pass."""
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 5}
            ],
            response_content="This response has more than five words in it.",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_word_count_negative(self):
        """Test length_constraints:number_words instruction - should fail."""
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 100}
            ],
            response_content="Too short.",
        )
        self._run_verify_test(request, False, 0.0, [False])

    def test_word_count_equal_to(self):
        """Test length_constraints:number_words with equal to relation."""
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "length_constraints:number_words", "relation": "equal to", "num_words": 4}
            ],
            response_content="Exactly four words here.",
        )
        self._run_verify_test(request, True, 1.0, [True])

    # ========================================================================
    # Keywords Tests
    # ========================================================================

    def test_keywords_existence_positive(self):
        """Test keywords:existence instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "keywords:existence", "keywords": ["hello", "world"]}],
            response_content="Hello there! Welcome to the world of programming.",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_keywords_existence_negative(self):
        """Test keywords:existence instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "keywords:existence", "keywords": ["python", "java"]}],
            response_content="Hello there! Welcome to the world.",
        )
        self._run_verify_test(request, False, 0.0, [False])

    def test_keywords_forbidden_positive(self):
        """Test keywords:forbidden_words instruction - should pass (no forbidden words)."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "keywords:forbidden_words", "forbidden_words": ["secret", "hidden"]}],
            response_content="This is a public message with no restricted content.",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_keywords_forbidden_negative(self):
        """Test keywords:forbidden_words instruction - should fail (contains forbidden word)."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "keywords:forbidden_words", "forbidden_words": ["secret", "hidden"]}],
            response_content="This is a secret message.",
        )
        self._run_verify_test(request, False, 0.0, [False])

    # ========================================================================
    # Title Tests
    # ========================================================================

    def test_title_positive(self):
        """Test detectable_format:title instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "detectable_format:title"}],
            response_content="<<My Great Title>>\n\nThis is the content of my response.",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_title_negative(self):
        """Test detectable_format:title instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "detectable_format:title"}],
            response_content="This response has no title.",
        )
        self._run_verify_test(request, False, 0.0, [False])

    # ========================================================================
    # Punctuation Tests
    # ========================================================================

    def test_no_comma_positive(self):
        """Test punctuation:no_comma instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "punctuation:no_comma"}],
            response_content="Hello world without commas at all",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_no_comma_negative(self):
        """Test punctuation:no_comma instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "punctuation:no_comma"}],
            response_content="Hello, world with commas",
        )
        self._run_verify_test(request, False, 0.0, [False])

    def test_no_period_positive(self):
        """Test punctuation:no_period instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "punctuation:no_period"}],
            response_content="Hello world without periods!",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_no_period_negative(self):
        """Test punctuation:no_period instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "punctuation:no_period"}],
            response_content="Hello world. With periods.",
        )
        self._run_verify_test(request, False, 0.0, [False])

    # ========================================================================
    # Format Tests
    # ========================================================================

    def test_bullet_list_positive(self):
        """Test detectable_format:number_bullet_lists instruction - should pass."""
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "detectable_format:number_bullet_lists", "relation": "at least", "num_bullets": 3}
            ],
            response_content="Here are some items:\n* First item\n* Second item\n* Third item",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_bullet_list_negative(self):
        """Test detectable_format:number_bullet_lists instruction - should fail."""
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "detectable_format:number_bullet_lists", "relation": "at least", "num_bullets": 5}
            ],
            response_content="Here are some items:\n* First item\n* Second item",
        )
        self._run_verify_test(request, False, 0.0, [False])

    def test_numbered_list_positive(self):
        """Test detectable_format:numbered_list instruction - should pass."""
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "detectable_format:numbered_list", "relation": "at least", "num_numbered_items": 2}
            ],
            response_content="Here are the steps:\n1. First step\n2. Second step\n3. Third step",
        )
        self._run_verify_test(request, True, 1.0, [True])

    # ========================================================================
    # Start/End Tests
    # ========================================================================

    def test_start_checker_positive(self):
        """Test startend:start_checker instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "startend:start_checker", "start_phrase": "Hello"}],
            response_content="Hello, this is a response that starts correctly.",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_start_checker_negative(self):
        """Test startend:start_checker instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "startend:start_checker", "start_phrase": "Hello"}],
            response_content="This response does not start with hello.",
        )
        self._run_verify_test(request, False, 0.0, [False])

    def test_quotation_positive(self):
        """Test startend:quotation instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "startend:quotation"}],
            response_content='"This entire response is wrapped in double quotes."',
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_quotation_negative(self):
        """Test startend:quotation instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "startend:quotation"}],
            response_content="This response is not wrapped in quotes.",
        )
        self._run_verify_test(request, False, 0.0, [False])

    # ========================================================================
    # Postscript Tests
    # ========================================================================

    def test_postscript_positive(self):
        """Test detectable_content:postscript instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "detectable_content:postscript", "postscript_marker": "P.S."}],
            response_content="Main content here.\n\nP.S. Don't forget to check the appendix.",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_postscript_negative(self):
        """Test detectable_content:postscript instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "detectable_content:postscript", "postscript_marker": "P.S."}],
            response_content="Main content here with no postscript.",
        )
        self._run_verify_test(request, False, 0.0, [False])

    # ========================================================================
    # Multiple Constraints Tests
    # ========================================================================

    def test_multiple_constraints_all_pass(self):
        """Test multiple constraints together - all should pass."""
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "detectable_format:title"},
                {"instruction_id": "punctuation:no_comma"},
                {"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 5},
            ],
            response_content="<<My Title>>\n\nThis response has no commas and enough words.",
        )
        self._run_verify_test(request, True, 1.0, [True, True, True])

    def test_multiple_constraints_partial_fail(self):
        """Test multiple constraints together - one should fail."""
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "detectable_format:title"},
                {"instruction_id": "punctuation:no_comma"},
                {"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 5},
            ],
            response_content="<<My Title>>\n\nThis response has commas, which should fail.",
        )
        self._run_verify_test(request, False, 0.0, [True, False, True])

    # ========================================================================
    # Case Tests
    # ========================================================================

    def test_all_caps_positive(self):
        """Test change_case:all_caps instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "change_case:all_caps"}],
            response_content="THIS IS ALL UPPERCASE TEXT!",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_all_caps_negative(self):
        """Test change_case:all_caps instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "change_case:all_caps"}],
            response_content="This is Not All Caps",
        )
        self._run_verify_test(request, False, 0.0, [False])

    def test_lowercase_positive(self):
        """Test change_case:lowercase instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "change_case:lowercase"}],
            response_content="this is all lowercase text",
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_lowercase_negative(self):
        """Test change_case:lowercase instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "change_case:lowercase"}],
            response_content="This Has Capital Letters",
        )
        self._run_verify_test(request, False, 0.0, [False])

    # ========================================================================
    # JSON Format Tests
    # ========================================================================

    def test_json_format_positive(self):
        """Test detectable_format:json_format instruction - should pass."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "detectable_format:json_format"}],
            response_content='{"name": "John", "age": 30}',
        )
        self._run_verify_test(request, True, 1.0, [True])

    def test_json_format_negative(self):
        """Test detectable_format:json_format instruction - should fail."""
        request = self._create_real_request(
            instructions=[{"instruction_id": "detectable_format:json_format"}],
            response_content="This is not JSON format.",
        )
        self._run_verify_test(request, False, 0.0, [False])


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def _create_server(self):
        """Helper to create server instance."""
        config = TuringVIFResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return TuringVIFResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_real_request(self, instructions, response_content, request_id=1):
        """Helper to create real request with NeMoGymResponse. Normalizes instructions for schema."""
        normalized = _normalize_instructions(instructions)
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

        return TuringVIFVerifyRequest(
            id=request_id,
            instructions=normalized,
            llm_judge=[],
            responses_create_params={"input": []},
            response=response,
        )

    def test_empty_instructions(self):
        """Test with empty instructions list - should pass with reward 1.0."""
        server = self._create_server()
        request = self._create_real_request(
            instructions=[],
            response_content="Any response should pass with no constraints.",
        )
        result = asyncio.run(server.verify(request))
        assert result.follow_all_instructions is True
        assert result.reward == 1.0
        assert result.follow_instruction_list == []

    def test_empty_response(self):
        """Test with empty response content."""
        server = self._create_server()
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 1}
            ],
            response_content="",
        )
        result = asyncio.run(server.verify(request))
        assert result.follow_all_instructions is False
        assert result.reward == 0.0

    def test_unicode_content(self):
        """Test with unicode characters in response - validators should work with ASCII keywords."""
        server = self._create_server()
        request = self._create_real_request(
            instructions=[{"instruction_id": "keywords:existence", "keywords": ["contains", "characters"]}],
            response_content="This response contains 日本語 and emoji 🎉 characters.",
        )
        result = asyncio.run(server.verify(request))
        # Should find the ASCII keywords in a response that contains unicode
        assert result.follow_instruction_list[0] is True

    def test_very_long_response(self):
        """Test with a very long response."""
        server = self._create_server()
        long_content = "word " * 5000  # 5000 words
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 4000}
            ],
            response_content=long_content,
        )
        result = asyncio.run(server.verify(request))
        assert result.follow_all_instructions is True
        assert result.reward == 1.0

    def test_unknown_instruction_id(self):
        """Test with unknown instruction ID - should fail gracefully."""
        server = self._create_server()
        request = self._create_real_request(
            instructions=[{"instruction_id": "unknown:fake_instruction"}],
            response_content="This should handle unknown instructions.",
        )
        result = asyncio.run(server.verify(request))
        # Should fail but not crash
        assert result.follow_all_instructions is False


class TestAgentPipelineFlow:
    """Tests simulating the full agent pipeline flow."""

    def _create_server(self):
        """Helper to create server instance."""
        config = TuringVIFResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return TuringVIFResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_real_request(self, instructions, response_content, request_id=1, prompt=None):
        """Helper to create real request with NeMoGymResponse. Normalizes instructions for schema."""
        normalized = _normalize_instructions(instructions)
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

        return TuringVIFVerifyRequest(
            id=request_id,
            instructions=normalized,
            llm_judge=[],
            prompt=prompt,
            responses_create_params={"input": [{"role": "user", "content": prompt or "Test prompt"}]},
            response=response,
        )

    def test_seed_session_endpoint(self):
        """Test that the seed_session endpoint works."""
        server = self._create_server()
        # The default seed_session should return successfully
        from nemo_gym.base_resources_server import BaseSeedSessionRequest

        request = BaseSeedSessionRequest()
        result = asyncio.run(server.seed_session(request))
        assert result is not None

    def test_full_verify_flow_success(self):
        """Test a complete verify flow with passing constraints."""
        server = self._create_server()
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 10},
                {"instruction_id": "keywords:existence", "keywords": ["renewable", "energy"]},
                {"instruction_id": "detectable_format:title"},
            ],
            response_content="<<The Future of Renewable Energy>>\n\nRenewable energy sources like solar and wind power are becoming increasingly important for sustainable development and reducing carbon emissions.",
            prompt="Write about renewable energy with a title.",
        )
        result = asyncio.run(server.verify(request))
        assert result.follow_all_instructions is True
        assert result.reward == 1.0
        assert len(result.validation_results) == 3
        assert all(r.status == "Passed" for r in result.validation_results)

    def test_full_verify_flow_partial_failure(self):
        """Test a complete verify flow with some failing constraints."""
        server = self._create_server()
        request = self._create_real_request(
            instructions=[
                {"instruction_id": "length_constraints:number_words", "relation": "at least", "num_words": 10},
                {"instruction_id": "keywords:existence", "keywords": ["renewable", "energy"]},
                {"instruction_id": "detectable_format:title"},  # Missing title
            ],
            response_content="Renewable energy sources like solar and wind power are becoming increasingly important for sustainable development.",
            prompt="Write about renewable energy with a title.",
        )
        result = asyncio.run(server.verify(request))
        assert result.follow_all_instructions is False
        assert result.reward == 0.0
        assert len(result.validation_results) == 3
        # Title should fail
        title_result = next(r for r in result.validation_results if r.instruction == "detectable_format:title")
        assert title_result.status == "Failed"

    def test_validation_results_contain_messages(self):
        """Test that validation results contain helpful messages."""
        server = self._create_server()
        request = self._create_real_request(
            instructions=[{"instruction_id": "keywords:existence", "keywords": ["missing_keyword"]}],
            response_content="This response does not contain the required word.",
        )
        result = asyncio.run(server.verify(request))
        assert len(result.validation_results) == 1
        assert result.validation_results[0].status == "Failed"
        assert (
            "missing" in result.validation_results[0].message.lower()
            or "keyword" in result.validation_results[0].message.lower()
        )
