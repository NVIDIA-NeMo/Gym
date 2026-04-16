# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from app import (
    PalantirFdeEvalConfig,
    PalantirFdeEvalResourcesServer,
    PalantirFdeEvalVerifyRequest,
)

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient


def _make_config():
    return PalantirFdeEvalConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="palantir_fde_eval",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_template_fpath=str(Path(__file__).resolve().parents[1] / "prompt_templates/tool_call_judge.txt"),
    )


def _make_server(server_client=None):
    if server_client is None:
        server_client = MagicMock(spec=ServerClient)
    return PalantirFdeEvalResourcesServer(
        config=_make_config(),
        server_client=server_client,
    )


def _make_response(output_items):
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=output_items,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _judge_response_json(text):
    """Create a JSON string mimicking judge model response with the given verdict text."""
    return NeMoGymResponse(
        id="judge_resp",
        created_at=0.0,
        model="judge_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_judge",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump_json()


def _mock_judge_post(verdict_text):
    """Create a mock server_client with post() returning a judge verdict."""
    server_mock = MagicMock(spec=ServerClient)
    post_mock = MagicMock()
    post_mock.read = AsyncMock(return_value=_judge_response_json(verdict_text))
    server_mock.post = AsyncMock(return_value=post_mock)
    return server_mock


def _make_config_with_schemas():
    return PalantirFdeEvalConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="palantir_fde_eval",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_template_fpath=str(Path(__file__).resolve().parents[1] / "prompt_templates/tool_call_judge.txt"),
        tool_schemas_fpath=str(Path(__file__).resolve().parents[1] / "data/tool_definitions.json"),
    )


def _make_server_with_schemas(server_client=None):
    if server_client is None:
        server_client = MagicMock(spec=ServerClient)
    return PalantirFdeEvalResourcesServer(
        config=_make_config_with_schemas(),
        server_client=server_client,
    )


class TestApp:
    # --- Sanity ---

    def test_sanity(self) -> None:
        _make_server()

    # --- EVAL-01: Tool call extraction ---

    async def test_extract_function_calls(self) -> None:
        """function_call output items are extracted with parsed arguments."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "dataset_sql_query",
                    "arguments": json.dumps({"queries": ["SELECT * FROM table"], "branch": {"mainBranch": True}}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {"name": "dataset_sql_query", "arguments": {"queries": ["SELECT * FROM table"]}}
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.num_predicted == 1
        assert result.predicted_calls[0]["name"] == "dataset_sql_query"
        assert result.predicted_calls[0]["arguments"]["queries"] == ["SELECT * FROM table"]

    async def test_non_function_call_items_ignored(self) -> None:
        """Non-function_call output items (type 'message') are ignored."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "id": "msg_1",
                    "content": [{"annotations": [], "text": "Some text.", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
                {
                    "call_id": "call_1",
                    "name": "get_data",
                    "arguments": json.dumps({"key": "value"}),
                    "type": "function_call",
                },
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "get_data", "arguments": {"key": "value"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.num_predicted == 1
        assert result.predicted_calls[0]["name"] == "get_data"

    async def test_malformed_json_arguments(self) -> None:
        """Malformed JSON arguments produce empty dict, not crash."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "some_tool",
                    "arguments": "{not valid json",
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "some_tool", "arguments": {}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.num_predicted == 1
        assert result.predicted_calls[0]["arguments"] == {}

    # --- EVAL-02: Stringified parameter detection ---

    async def test_stringified_array_reward_0(self) -> None:
        """Stringified array in arguments -> structure_valid=False, reward=0.0."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "dataset_sql_query",
                    "arguments": json.dumps(
                        {
                            "queries": '["SELECT * FROM table"]',
                            "branch": {"mainBranch": True},
                        }
                    ),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {
                        "name": "dataset_sql_query",
                        "arguments": {"queries": ["SELECT * FROM table"], "branch": {"mainBranch": True}},
                    }
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.structure_valid is False

    async def test_stringified_object_reward_0(self) -> None:
        """Stringified object in arguments -> structure_valid=False, reward=0.0."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "dataset_sql_query",
                    "arguments": json.dumps(
                        {
                            "queries": ["SELECT * FROM table"],
                            "branch": '{"mainBranch": true}',
                        }
                    ),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {
                        "name": "dataset_sql_query",
                        "arguments": {"queries": ["SELECT * FROM table"], "branch": {"mainBranch": True}},
                    }
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.structure_valid is False

    async def test_plain_string_json_primitives_no_false_positive(self) -> None:
        """String values that parse to JSON primitives ('true', '123') -> structure_valid=True."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "some_tool",
                    "arguments": json.dumps(
                        {
                            "flag": "true",
                            "count": "123",
                            "label": "null",
                        }
                    ),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "some_tool", "arguments": {"flag": "true", "count": "123"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True

    async def test_nested_dict_stringification_detected(self) -> None:
        """Nested dict values are recursively checked for stringification."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "some_tool",
                    "arguments": json.dumps(
                        {
                            "config": {
                                "nested_list": "[1, 2, 3]",
                            },
                        }
                    ),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "some_tool", "arguments": {"config": {"nested_list": [1, 2, 3]}}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is False
        assert result.reward == 0.0

    # --- EVAL-04: Tool name matching ---

    async def test_structure_valid_reward_1(self) -> None:
        """Matching tool name + valid structure + judge pass -> reward=1.0."""
        server_mock = _mock_judge_post("The parameters are equivalent. [[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "dataset_sql_query",
                    "arguments": json.dumps(
                        {
                            "queries": ["SELECT * FROM table"],
                            "branch": {"mainBranch": True},
                        }
                    ),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {
                        "name": "dataset_sql_query",
                        "arguments": {"queries": ["SELECT * FROM table"], "branch": {"mainBranch": True}},
                    }
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 1.0
        assert result.structure_valid is True
        assert result.tool_name_match is True

    async def test_wrong_tool_name_reward_0(self) -> None:
        """Wrong tool name -> tool_name_match=False, reward=0.0."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "wrong_tool",
                    "arguments": json.dumps({"key": "value"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "correct_tool", "arguments": {"key": "value"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.tool_name_match is False

    async def test_predicted_subset_fewer_tools_passes_name_check(self) -> None:
        """Predicted subset (fewer tools) -> tool_name_match=True, proceeds to judge.

        With subset semantics, predicted={tool_a} is a valid subset of
        expected={tool_a, tool_b}, so tool_name_match=True and we reach the judge.
        """
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "tool_a",
                    "arguments": json.dumps({"key": "value"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {"name": "tool_a", "arguments": {"key": "value"}},
                    {"name": "tool_b", "arguments": {"key": "value2"}},
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.tool_name_match is True
        assert result.reward == 1.0

    async def test_multi_tool_call_name_match(self) -> None:
        """Multi-tool-call with all names matching + judge pass -> reward=1.0."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "tool_b",
                    "arguments": json.dumps({"key": "value2"}),
                    "type": "function_call",
                },
                {
                    "call_id": "call_2",
                    "name": "tool_a",
                    "arguments": json.dumps({"key": "value1"}),
                    "type": "function_call",
                },
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {"name": "tool_a", "arguments": {"key": "value1"}},
                    {"name": "tool_b", "arguments": {"key": "value2"}},
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 1.0
        assert result.tool_name_match is True

    # --- Edge cases ---

    async def test_no_tool_call_expected_none_predicted_reward_1(self) -> None:
        """No tool call expected (has_tool_call=False) and none predicted -> reward=1.0."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "id": "msg_1",
                    "content": [{"annotations": [], "text": "A diagram.", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "make diagram"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [],
                "has_tool_call": False,
            },
        )
        result = await server.verify(request)
        assert result.reward == 1.0

    async def test_no_tool_call_expected_but_predicted_reward_0(self) -> None:
        """No tool call expected but model predicts one -> reward=0.0."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "some_tool",
                    "arguments": json.dumps({"key": "value"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "make diagram"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [],
                "has_tool_call": False,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0

    async def test_tool_call_expected_but_none_predicted_reward_0(self) -> None:
        """Tool call expected but model predicts none -> reward=0.0."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "id": "msg_1",
                    "content": [{"annotations": [], "text": "No tool needed.", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query data"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "dataset_sql_query", "arguments": {"queries": ["SELECT 1"]}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0

    # --- JUDGE-03: Config with judge and oracle model servers ---

    def test_judge_config_model_server(self) -> None:
        """Config accepts judge_model_server and related judge fields."""
        config = _make_config()
        assert config.judge_model_server.name == "judge"
        assert config.judge_model_server.type == "responses_api_models"
        assert config.judge_equal_label == "[[PASS]]"
        assert config.judge_not_equal_label == "[[FAIL]]"
        assert config.judge_endpoint_max_concurrency == 64

    def test_oracle_config(self) -> None:
        """Config accepts oracle_model_server as Optional (defaults to None)."""
        config = _make_config()
        assert config.oracle_model_server is None

    def test_independent_judge_oracle(self) -> None:
        """judge_model_server and oracle_model_server can have different names."""
        config = PalantirFdeEvalConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="palantir_fde_eval",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_llm"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=str(
                Path(__file__).resolve().parents[1] / "prompt_templates/tool_call_judge.txt"
            ),
            oracle_model_server=ModelServerRef(type="responses_api_models", name="oracle_llm"),
        )
        assert config.judge_model_server.name == "judge_llm"
        assert config.oracle_model_server.name == "oracle_llm"
        assert config.judge_model_server.name != config.oracle_model_server.name

    # --- JUDGE-02: Prompt template formatting ---

    def test_judge_prompt_formatting(self) -> None:
        """Prompt template has {expected_params} and {predicted_params} placeholders."""
        template_path = Path(__file__).resolve().parents[1] / "prompt_templates/tool_call_judge.txt"
        template = template_path.read_text().strip()
        assert "{expected_params}" in template
        assert "{predicted_params}" in template
        # Verify formatting works with json.dumps
        formatted = template.format(
            expected_params=json.dumps({"key": "value"}, indent=2),
            predicted_params=json.dumps({"key": "value"}, indent=2),
        )
        assert '"key": "value"' in formatted

    # --- JUDGE-01: Judge pass/fail rewards ---

    async def test_judge_pass_reward_1(self) -> None:
        """Judge returns [[PASS]] -> reward=1.0, judge_score=True."""
        server_mock = _mock_judge_post("The parameters match perfectly. [[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "dataset_sql_query",
                    "arguments": json.dumps({"queries": ["SELECT * FROM t"]}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "dataset_sql_query", "arguments": {"queries": ["SELECT * FROM t"]}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 1.0
        assert result.judge_score is True
        server_mock.post.assert_called_once()

    async def test_judge_fail_reward_0(self) -> None:
        """Judge returns [[FAIL]] -> reward=0.0, judge_score=False."""
        server_mock = _mock_judge_post("The parameters differ significantly. [[FAIL]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "dataset_sql_query",
                    "arguments": json.dumps({"queries": ["SELECT wrong FROM t"]}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "dataset_sql_query", "arguments": {"queries": ["SELECT * FROM t"]}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.judge_score is False
        server_mock.post.assert_called_once()

    # --- EVAL-05: Composite reward ---

    async def test_composite_all_pass(self) -> None:
        """Structure valid + name match + judge pass -> reward=1.0."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "get_data",
                    "arguments": json.dumps({"rid": "ri.foundry.main.dataset.abc123"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "get data"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "get_data", "arguments": {"rid": "ri.foundry.main.dataset.abc123"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 1.0
        assert result.structure_valid is True
        assert result.tool_name_match is True
        assert result.judge_score is True

    async def test_structure_fail_skips_judge(self) -> None:
        """Structure invalid -> reward=0.0, server_client.post NOT called."""
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock()
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "get_data",
                    "arguments": json.dumps({"items": '["a", "b"]'}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "get data"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "get_data", "arguments": {"items": ["a", "b"]}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.structure_valid is False
        assert result.judge_score is None
        server_mock.post.assert_not_called()

    async def test_name_fail_skips_judge(self) -> None:
        """Name mismatch -> reward=0.0, server_client.post NOT called."""
        server_mock = MagicMock(spec=ServerClient)
        server_mock.post = AsyncMock()
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "wrong_tool",
                    "arguments": json.dumps({"key": "value"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "get data"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "correct_tool", "arguments": {"key": "value"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.tool_name_match is False
        assert result.judge_score is None
        server_mock.post.assert_not_called()

    async def test_judge_fail_composite_0(self) -> None:
        """All deterministic pass but judge fails -> reward=0.0."""
        server_mock = _mock_judge_post("Parameters do not match. [[FAIL]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "get_data",
                    "arguments": json.dumps({"rid": "wrong_rid"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "get data"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "get_data", "arguments": {"rid": "ri.foundry.main.dataset.abc123"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.structure_valid is True
        assert result.tool_name_match is True
        assert result.judge_score is False

    # --- DATA-05: Schema data setup ---

    def test_schema_file_exists(self) -> None:
        """tool_definitions.json exists in data/ directory and contains 105 tools."""
        schema_path = Path(__file__).resolve().parents[1] / "data/tool_definitions.json"
        assert schema_path.exists(), f"Schema file not found: {schema_path}"
        with open(schema_path) as f:
            tools = json.load(f)
        assert len(tools) == 105, f"Expected 105 tools, got {len(tools)}"
        # Verify structure: each tool has name and parameters
        for tool in tools:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "parameters" in tool, f"Tool missing 'parameters': {tool}"

    def test_schema_loading(self) -> None:
        """Server with tool_schemas_fpath loads schemas into self._tool_schemas dict, keyed by tool name."""
        server = _make_server_with_schemas()
        assert hasattr(server, "_tool_schemas")
        assert isinstance(server._tool_schemas, dict)
        assert len(server._tool_schemas) == 105
        # Spot-check a known tool
        assert "search_object_types" in server._tool_schemas
        assert "properties" in server._tool_schemas["search_object_types"]

    def test_no_schema_config_skips(self) -> None:
        """Server without tool_schemas_fpath has empty _tool_schemas dict, verify() works as before."""
        server = _make_server()
        assert hasattr(server, "_tool_schemas")
        assert isinstance(server._tool_schemas, dict)
        assert len(server._tool_schemas) == 0

    # --- EVAL-03: Schema-aware anyOf validation ---

    async def test_anyof_union_violation_reward_0(self) -> None:
        """branch param as string 'main' instead of object -> structure_valid=False, reward=0.0."""
        server = _make_server_with_schemas()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "dataset_sql_query",
                    "arguments": json.dumps(
                        {
                            "queries": ["SELECT * FROM table"],
                            "branch": "main",  # Should be an object, not a string
                        }
                    ),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {
                        "name": "dataset_sql_query",
                        "arguments": {"queries": ["SELECT * FROM table"], "branch": {"mainBranch": True}},
                    }
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is False
        assert result.reward == 0.0

    async def test_valid_anyof_union_passes(self) -> None:
        """branch param as valid object -> structure_valid=True, judge called, returns PASS -> reward=1.0."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server_with_schemas(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "dataset_sql_query",
                    "arguments": json.dumps(
                        {
                            "queries": ["SELECT * FROM table"],
                            "branch": {"mainBranch": True},
                        }
                    ),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {
                        "name": "dataset_sql_query",
                        "arguments": {"queries": ["SELECT * FROM table"], "branch": {"mainBranch": True}},
                    }
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True
        assert result.reward == 1.0

    async def test_nullable_anyof_null_passes(self) -> None:
        """numResults param as None for search_object_types (nullable anyOf [number, null]) -> structure_valid=True."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server_with_schemas(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "search_object_types",
                    "arguments": json.dumps(
                        {
                            "query": "find objects",
                            "numResults": None,
                        }
                    ),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "search"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {
                        "name": "search_object_types",
                        "arguments": {"query": "find objects", "numResults": None},
                    }
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True

    async def test_unknown_tool_skips_schema(self) -> None:
        """Tool name not in catalog -> schema validation skipped, structure_valid=True."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server_with_schemas(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "totally_unknown_tool",
                    "arguments": json.dumps({"anything": "goes"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "do something"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {"name": "totally_unknown_tool", "arguments": {"anything": "goes"}},
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True

    async def test_bool_not_matching_integer(self) -> None:
        """Boolean True where anyOf has only [integer, null] -> violation detected."""
        server = _make_server_with_schemas()
        # Manually inject a schema with anyOf [integer, null] for testing
        server._tool_schemas["test_int_tool"] = {
            "type": "object",
            "properties": {
                "count": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                },
            },
        }
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "test_int_tool",
                    "arguments": json.dumps({"count": True}),  # bool, not int
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "count"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {"name": "test_int_tool", "arguments": {"count": 5}},
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is False
        assert result.reward == 0.0

    # --- TEST-01/TEST-02: Coverage gap tests ---

    def test_nullcontext_semaphore(self) -> None:
        """Config with judge_endpoint_max_concurrency=None uses nullcontext (line 75)."""
        config = PalantirFdeEvalConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="palantir_fde_eval",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath=str(
                Path(__file__).resolve().parents[1] / "prompt_templates/tool_call_judge.txt"
            ),
            judge_endpoint_max_concurrency=None,
        )
        server = PalantirFdeEvalResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        assert isinstance(server._judge_semaphore, contextlib.nullcontext)

    async def test_stringified_dict_in_list(self) -> None:
        """List item containing stringified dict -> structure_valid=False (lines 127-128, 132-133)."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "some_tool",
                    "arguments": json.dumps({"items": ["normal_string", '{"key": "value"}']}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {"name": "some_tool", "arguments": {"items": ["normal_string", {"key": "value"}]}},
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is False
        assert result.reward == 0.0

    async def test_value_matches_number(self) -> None:
        """anyOf with 'number' type correctly matches float value (line 144/149)."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server_with_schemas(server_client=server_mock)
        server._tool_schemas["test_num_tool"] = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                },
            },
        }
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "test_num_tool",
                    "arguments": json.dumps({"value": 3.14}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "test_num_tool", "arguments": {"value": 3.14}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True
        assert result.reward == 1.0

    async def test_value_matches_array(self) -> None:
        """anyOf with 'array' type correctly matches list value (line 151/152)."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server_with_schemas(server_client=server_mock)
        server._tool_schemas["test_arr_tool"] = {
            "type": "object",
            "properties": {
                "items": {
                    "anyOf": [{"type": "array"}, {"type": "null"}],
                },
            },
        }
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "test_arr_tool",
                    "arguments": json.dumps({"items": [1, 2, 3]}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "test_arr_tool", "arguments": {"items": [1, 2, 3]}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True
        assert result.reward == 1.0

    async def test_value_matches_object(self) -> None:
        """anyOf with 'object' type correctly matches dict value (line 153/154)."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server_with_schemas(server_client=server_mock)
        server._tool_schemas["test_obj_tool"] = {
            "type": "object",
            "properties": {
                "config": {
                    "anyOf": [{"type": "object"}, {"type": "null"}],
                },
            },
        }
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "test_obj_tool",
                    "arguments": json.dumps({"config": {"key": "value"}}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "test_obj_tool", "arguments": {"config": {"key": "value"}}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True
        assert result.reward == 1.0

    async def test_value_matches_unknown_type(self) -> None:
        """anyOf with unknown type returns True (line 157)."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server_with_schemas(server_client=server_mock)
        server._tool_schemas["test_unk_tool"] = {
            "type": "object",
            "properties": {
                "data": {
                    "anyOf": [{"type": "custom_unknown"}],
                },
            },
        }
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "test_unk_tool",
                    "arguments": json.dumps({"data": "anything"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "test_unk_tool", "arguments": {"data": "anything"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True
        assert result.reward == 1.0

    async def test_judge_non_message_output(self) -> None:
        """Judge response with non-message output type returns reward 0.0 (line 232)."""
        server_mock = MagicMock(spec=ServerClient)
        # Build a judge response where the last output item is NOT a message type
        judge_resp = NeMoGymResponse(
            id="judge_resp",
            created_at=0.0,
            model="judge_model",
            object="response",
            output=[
                {
                    "call_id": "call_judge",
                    "name": "some_function",
                    "arguments": "{}",
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump_json()
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=judge_resp)
        server_mock.post = AsyncMock(return_value=post_mock)

        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "get_data",
                    "arguments": json.dumps({"key": "value"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "get_data", "arguments": {"key": "value"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.judge_score is False

    async def test_judge_exception_during_parsing(self) -> None:
        """Judge response that throws exception during parsing returns reward 0.0 (lines 235-236)."""
        server_mock = MagicMock(spec=ServerClient)
        # Build a judge response where the message has empty content list -> IndexError on content[-1]
        judge_resp = NeMoGymResponse(
            id="judge_resp",
            created_at=0.0,
            model="judge_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_judge",
                    content=[],  # Empty content -> content[-1] raises IndexError
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump_json()
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=judge_resp)
        server_mock.post = AsyncMock(return_value=post_mock)

        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "get_data",
                    "arguments": json.dumps({"key": "value"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "get_data", "arguments": {"key": "value"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.judge_score is False

    async def test_judge_fail_before_pass(self) -> None:
        """Judge text with [[FAIL]] before [[PASS]] returns reward 0.0 (line 246)."""
        server_mock = _mock_judge_post("The answer is [[FAIL]] but also [[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "get_data",
                    "arguments": json.dumps({"key": "value"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "get_data", "arguments": {"key": "value"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.judge_score is False

    async def test_judge_no_verdict_label(self) -> None:
        """Judge text with neither [[PASS]] nor [[FAIL]] returns reward 0.0 (line 243)."""
        server_mock = _mock_judge_post("The parameters look reasonable but I am unsure.")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "get_data",
                    "arguments": json.dumps({"key": "value"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "get_data", "arguments": {"key": "value"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.judge_score is False

    async def test_stringified_params_in_dict_inside_list(self) -> None:
        """Dict item inside list with stringified params -> structure_valid=False (lines 127-128)."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "some_tool",
                    "arguments": json.dumps(
                        {
                            "steps": [
                                {"action": "run", "params": '{"nested": true}'},
                            ]
                        }
                    ),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {
                        "name": "some_tool",
                        "arguments": {"steps": [{"action": "run", "params": {"nested": True}}]},
                    },
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is False
        assert result.reward == 0.0

    async def test_value_matches_boolean_variant(self) -> None:
        """anyOf with 'boolean' type correctly matches bool value (line 144)."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server_with_schemas(server_client=server_mock)
        server._tool_schemas["test_bool_tool"] = {
            "type": "object",
            "properties": {
                "flag": {
                    "anyOf": [{"type": "boolean"}, {"type": "null"}],
                },
            },
        }
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "test_bool_tool",
                    "arguments": json.dumps({"flag": True}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "test_bool_tool", "arguments": {"flag": True}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True
        assert result.reward == 1.0

    async def test_value_matches_string_variant(self) -> None:
        """anyOf with 'string' type correctly matches string value (line 151)."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server_with_schemas(server_client=server_mock)
        server._tool_schemas["test_str_tool"] = {
            "type": "object",
            "properties": {
                "label": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                },
            },
        }
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "test_str_tool",
                    "arguments": json.dumps({"label": "hello"}),
                    "type": "function_call",
                }
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [{"name": "test_str_tool", "arguments": {"label": "hello"}}],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.structure_valid is True
        assert result.reward == 1.0

    # --- Subset tool name matching (EVAL-04) ---

    async def test_subset_tool_names_match(self) -> None:
        """Predicted tools are a subset of expected tools -> tool_name_match=True, proceeds to judge."""
        server_mock = _mock_judge_post("[[PASS]]")
        server = _make_server(server_client=server_mock)
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "tool_a",
                    "arguments": json.dumps({"key": "value1"}),
                    "type": "function_call",
                },
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {"name": "tool_a", "arguments": {"key": "value1"}},
                    {"name": "tool_b", "arguments": {"key": "value2"}},
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.tool_name_match is True
        assert result.reward == 1.0

    async def test_superset_tool_names_fail(self) -> None:
        """Predicted tools contain a tool NOT in expected -> tool_name_match=False, reward=0.0."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "tool_a",
                    "arguments": json.dumps({"key": "value1"}),
                    "type": "function_call",
                },
                {
                    "call_id": "call_2",
                    "name": "tool_c",
                    "arguments": json.dumps({"key": "value3"}),
                    "type": "function_call",
                },
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {"name": "tool_a", "arguments": {"key": "value1"}},
                    {"name": "tool_b", "arguments": {"key": "value2"}},
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.tool_name_match is False

    async def test_predicted_superset_of_expected_fails(self) -> None:
        """Predicted calls {tool_a, tool_b} with expected {tool_a} -> reward=0.0 (superset fails)."""
        server = _make_server()
        response = _make_response(
            [
                {
                    "call_id": "call_1",
                    "name": "tool_a",
                    "arguments": json.dumps({"key": "value1"}),
                    "type": "function_call",
                },
                {
                    "call_id": "call_2",
                    "name": "tool_b",
                    "arguments": json.dumps({"key": "value2"}),
                    "type": "function_call",
                },
            ]
        )
        request = PalantirFdeEvalVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "query"}]},
            response=response,
            verifier_metadata={
                "expected_tool_calls": [
                    {"name": "tool_a", "arguments": {"key": "value1"}},
                ],
                "has_tool_call": True,
            },
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.tool_name_match is False
