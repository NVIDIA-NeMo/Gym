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

from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from responses_api_agents.cobol_eval_agent.app import (
    CobolEvalAgent,
    CobolEvalAgentConfig,
    ModelServerRef,
    ResourcesServerRef,
    _build_error_feedback,
    _extract_problem_description,
)


class TestApp:
    def test_sanity(self) -> None:
        """Agent can be instantiated with valid config."""
        config = CobolEvalAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        CobolEvalAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_config_defaults(self) -> None:
        """Test that config has correct default values."""
        config = CobolEvalAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="cobol_compiler",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
        )
        assert config.max_correction_turns == 3
        assert config.include_all_attempts is True

    def test_config_custom_values(self) -> None:
        """Test that config accepts custom values."""
        config = CobolEvalAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="cobol_compiler",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
            max_correction_turns=5,
            include_all_attempts=False,
        )
        assert config.max_correction_turns == 5
        assert config.include_all_attempts is False


class TestBuildErrorFeedback:
    def test_compilation_errors(self) -> None:
        result = _build_error_feedback(
            {
                "compilation_success": False,
                "compilation_errors": ["program.cob:5: error: syntax error", "program.cob:10: error: undefined"],
            }
        )
        assert "COMPILATION ERRORS:" in result
        assert "syntax error" in result
        assert "undefined" in result

    def test_test_failures(self) -> None:
        result = _build_error_feedback(
            {
                "compilation_success": True,
                "test_results": [
                    {"test_id": 0, "passed": True, "expected": "9", "actual": "9"},
                    {"test_id": 1, "passed": False, "expected": "25", "actual": "26", "error": None},
                    {"test_id": 2, "passed": False, "expected": "100", "actual": "", "error": "Execution timed out"},
                ],
            }
        )
        assert "TEST FAILURES:" in result
        assert "Test 1:" in result
        assert "Expected: 25" in result
        assert "Got:      26" in result
        assert "Test 2:" in result
        assert "Execution timed out" in result
        # Passing test should not appear
        assert "Test 0:" not in result

    def test_compilation_errors_and_test_failures(self) -> None:
        result = _build_error_feedback(
            {
                "compilation_success": False,
                "compilation_errors": ["some error"],
                "test_results": [{"test_id": 0, "passed": False, "expected": "X", "actual": "Y"}],
            }
        )
        assert "COMPILATION ERRORS:" in result
        assert "TEST FAILURES:" in result

    def test_no_code_extracted(self) -> None:
        result = _build_error_feedback({"extracted_code": None})
        assert "No COBOL code could be extracted" in result
        assert "IDENTIFICATION DIVISION" in result

    def test_no_code_but_has_other_errors(self) -> None:
        """If there are compilation errors, don't also show the no-code message."""
        result = _build_error_feedback(
            {
                "extracted_code": None,
                "compilation_success": False,
                "compilation_errors": ["some error"],
            }
        )
        assert "COMPILATION ERRORS:" in result
        assert "No COBOL code could be extracted" not in result

    def test_all_tests_passed(self) -> None:
        result = _build_error_feedback(
            {
                "compilation_success": True,
                "extracted_code": "some code",
                "test_results": [{"test_id": 0, "passed": True}],
            }
        )
        assert result == ""

    def test_empty_result(self) -> None:
        """Empty dict has no extracted_code key, so defaults to no-code message."""
        result = _build_error_feedback({})
        assert "No COBOL code could be extracted" in result


class TestExtractProblemDescription:
    def test_dict_input(self) -> None:
        params = {
            "input": [
                {"role": "system", "content": "You are a COBOL expert."},
                {"role": "user", "content": "Write a program that computes factorial."},
            ]
        }
        assert _extract_problem_description(params) == "Write a program that computes factorial."

    def test_dict_no_user_message(self) -> None:
        params = {"input": [{"role": "system", "content": "System prompt."}]}
        assert _extract_problem_description(params) == "(Problem description not available)"

    def test_dict_empty_input(self) -> None:
        params = {"input": []}
        assert _extract_problem_description(params) == "(Problem description not available)"

    def test_dict_no_input_key(self) -> None:
        params = {"model": "some-model"}
        assert _extract_problem_description(params) == "(Problem description not available)"

    def test_pydantic_model_input(self) -> None:
        params = MagicMock()
        params.input = [
            MagicMock(role="system", content="System prompt."),
            MagicMock(role="user", content="Write hello world."),
        ]
        assert _extract_problem_description(params) == "Write hello world."

    def test_non_dict_non_pydantic(self) -> None:
        assert _extract_problem_description(42) == "(Problem description not available)"
