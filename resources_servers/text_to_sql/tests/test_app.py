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
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.text_to_sql.app import (
    FailureCode,
    TextToSqlResourcesServer,
    TextToSqlResourcesServerConfig,
    TextToSqlVerifyRequest,
    _extract_last_assistant_text,
    extract_sql_from_response,
)


class TestExtractSqlFromResponse:
    """Tests for the extract_sql_from_response function."""

    def test_extract_from_sql_code_block(self):
        """Test extracting SQL from ```sql ... ``` code block."""
        text = "Here's the query:\n\n```sql\nSELECT * FROM users;\n```"
        result = extract_sql_from_response(text)
        assert result == "SELECT * FROM users;"

    def test_extract_from_generic_code_block(self):
        """Test extracting SQL from ``` ... ``` code block."""
        text = "Here's the query:\n\n```\nSELECT name FROM employees;\n```"
        result = extract_sql_from_response(text)
        assert result == "SELECT name FROM employees;"

    def test_extract_raw_sql_select(self):
        """Test extracting raw SELECT statement."""
        text = "The answer is SELECT id, name FROM products WHERE price > 100;"
        result = extract_sql_from_response(text)
        assert "SELECT id, name FROM products WHERE price > 100" in result

    def test_extract_raw_sql_with_cte(self):
        """Test extracting SQL with WITH clause (CTE)."""
        text = "Answer: WITH cte AS (SELECT * FROM t) SELECT * FROM cte;"
        result = extract_sql_from_response(text)
        assert result is not None
        assert "WITH cte" in result

    def test_no_sql_found(self):
        """Test when no SQL is found."""
        text = "I don't know how to write this query."
        result = extract_sql_from_response(text)
        assert result is None

    def test_empty_text(self):
        """Test with empty text."""
        result = extract_sql_from_response("")
        assert result is None

    def test_multiple_code_blocks_returns_last(self):
        """Test that the last SQL code block is returned."""
        text = "First:\n```sql\nSELECT 1;\n```\n\nSecond:\n```sql\nSELECT 2;\n```"
        result = extract_sql_from_response(text)
        assert result == "SELECT 2;"

    def test_extract_multiline_sql(self):
        """Test extracting multiline SQL query."""
        text = """```sql
SELECT
    u.name,
    COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY order_count DESC;
```"""
        result = extract_sql_from_response(text)
        assert result is not None
        assert "SELECT" in result
        assert "LEFT JOIN" in result
        assert "GROUP BY" in result


class TestExtractLastAssistantText:
    """Tests for the _extract_last_assistant_text function."""

    def _create_verify_request_with_output(self, output_items: list) -> TextToSqlVerifyRequest:
        """Helper to create a TextToSqlVerifyRequest with specified output items."""
        response = NeMoGymResponse(
            id="test_response",
            created_at=1000,
            model="test_model",
            object="response",
            output=output_items,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        return TextToSqlVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            response=response,
            expected_answer="SELECT 1;",
            sql_dialect="postgresql",
            sql_context="CREATE TABLE test (id INT);",
            sql_prompt="Select one",
        )

    def test_extract_single_assistant_message(self):
        """Test extracting text from a single assistant message."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text="SELECT * FROM users;")],
        )
        body = self._create_verify_request_with_output([output_message])
        result = _extract_last_assistant_text(body)
        assert result == "SELECT * FROM users;"

    def test_extract_empty_output(self):
        """Test extracting from empty output."""
        body = self._create_verify_request_with_output([])
        result = _extract_last_assistant_text(body)
        assert result == ""


class TestTextToSqlResourcesServerVerify:
    """Tests for the TextToSqlResourcesServer.verify method."""

    @pytest.fixture
    def resources_server(self) -> TextToSqlResourcesServer:
        """Create a TextToSqlResourcesServer instance for testing."""
        config = TextToSqlResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="text_to_sql_test_server",
            judge_model_server={"name": "test_judge", "type": "responses_api_models"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
        )

        server = TextToSqlResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        return server

    def _create_verify_request(
        self,
        model_output: str,
        expected_answer: str,
        sql_dialect: str = "postgresql",
        sql_context: str = "CREATE TABLE users (id INT, name VARCHAR(100));",
        sql_prompt: str = "List all users",
    ) -> TextToSqlVerifyRequest:
        """Helper to create a TextToSqlVerifyRequest."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text=model_output)],
        )
        response = NeMoGymResponse(
            id="test_response",
            created_at=1000,
            model="test_model",
            object="response",
            output=[output_message],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        return TextToSqlVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content=sql_prompt)]
            ),
            response=response,
            expected_answer=expected_answer,
            sql_dialect=sql_dialect,
            sql_context=sql_context,
            sql_prompt=sql_prompt,
        )

    @pytest.mark.asyncio
    async def test_verify_no_sql_extracted(self, resources_server: TextToSqlResourcesServer):
        """Test verify returns reward=0.0 when no SQL is found."""
        request = self._create_verify_request(
            model_output="I don't know how to write SQL",
            expected_answer="SELECT * FROM users;",
        )

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.failure_reason == FailureCode.NO_SQL_EXTRACTED
        assert response.extracted_sql is None

    @pytest.mark.asyncio
    async def test_verify_judge_passes(self, resources_server: TextToSqlResourcesServer):
        """Test verify returns reward=1.0 when judge passes."""
        resources_server.config.check_twice_swap = False

        request = self._create_verify_request(
            model_output="```sql\nSELECT * FROM users;\n```",
            expected_answer="SELECT * FROM users;",
        )

        # Mock judge to return equal
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.judge_passed is True
        assert response.failure_reason == FailureCode.NONE
        assert response.extracted_sql == "SELECT * FROM users;"

    @pytest.mark.asyncio
    async def test_verify_judge_fails(self, resources_server: TextToSqlResourcesServer):
        """Test verify returns reward=0.0 when judge fails."""
        request = self._create_verify_request(
            model_output="```sql\nSELECT id FROM users;\n```",
            expected_answer="SELECT * FROM users;",
        )

        # Mock judge to return not equal
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A!=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        response = await resources_server.verify(request)

        assert response.reward == 0.0
        assert response.judge_passed is False
        assert response.failure_reason == FailureCode.JUDGE_EVALUATION_FAILED

    @pytest.mark.asyncio
    async def test_verify_with_swap_check(self, resources_server: TextToSqlResourcesServer):
        """Test verify with swap check enabled."""
        resources_server.config.check_twice_swap = True

        request = self._create_verify_request(
            model_output="```sql\nSELECT * FROM users;\n```",
            expected_answer="SELECT * FROM users;",
        )

        # Mock judge to return equal for both calls
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert response.judge_passed is True
        assert len(response.judge_evaluations) == 2

    @pytest.mark.asyncio
    async def test_verify_with_think_tags(self, resources_server: TextToSqlResourcesServer):
        """Test verify handles </think> tags correctly."""
        resources_server.config.check_twice_swap = False

        request = self._create_verify_request(
            model_output="<think>Let me think...</think>```sql\nSELECT * FROM users;\n```",
            expected_answer="SELECT * FROM users;",
        )

        # Mock judge to return equal
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        response = await resources_server.verify(request)

        assert response.reward == 1.0
        assert "SELECT * FROM users;" in response.extracted_sql

    @pytest.mark.asyncio
    async def test_verify_missing_expected_answer(self, resources_server: TextToSqlResourcesServer):
        """Test verify raises error when expected answer is missing."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text="SELECT 1;")],
        )
        response = NeMoGymResponse(
            id="test_response",
            created_at=1000,
            model="test_model",
            object="response",
            output=[output_message],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        request = TextToSqlVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            response=response,
        )

        with pytest.raises(ValueError, match="Expected answer is required"):
            await resources_server.verify(request)
