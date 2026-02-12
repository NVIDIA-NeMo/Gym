# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for comparison strategies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from resources_servers.genrm_compare.comparison_strategies import (
    ComparisonStrategy,
    GenRMStrategy,
    GenRMStrategyConfig,
    extract_conversation_history,
    extract_generated_text,
    generate_response,
    get_prompt_key,
)


class TestGetPromptKey:
    """Test get_prompt_key function."""

    def test_with_prompt_id_no_principle(self):
        """Test prompt_key with prompt_id, no principle."""
        example = {"prompt_id": "123"}
        key = get_prompt_key(example)
        assert key == "123"

    def test_with_prompt_id_and_principle(self):
        """Test prompt_key with prompt_id and principle."""
        example = {"prompt_id": "123", "principle": "Be concise"}
        key = get_prompt_key(example)
        assert key == "123::Be concise"

    def test_without_prompt_id(self):
        """Test prompt_key without prompt_id - hashes conversation + principle."""
        example = {
            "responses_create_params": {"input": [{"role": "user", "content": "What is 2+2?"}]},
            "principle": "Be accurate",
        }
        key = get_prompt_key(example)
        # Should be a SHA256 hash
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_principles_different_keys(self):
        """Test that different principles produce different keys."""
        base_example = {"responses_create_params": {"input": [{"role": "user", "content": "Hello"}]}}
        example1 = {**base_example, "principle": "Be helpful"}
        example2 = {**base_example, "principle": "Be concise"}

        key1 = get_prompt_key(example1)
        key2 = get_prompt_key(example2)

        assert key1 != key2


class TestExtractConversationHistory:
    """Test extract_conversation_history function."""

    def test_extract_valid_history(self):
        """Test extracting valid conversation history."""
        example = {
            "responses_create_params": {
                "input": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        }
        history = extract_conversation_history(example)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_missing_responses_create_params(self):
        """Test error when responses_create_params is missing."""
        example = {"other_field": "value"}
        with pytest.raises(ValueError, match="missing 'responses_create_params'"):
            extract_conversation_history(example)

    def test_missing_input_field(self):
        """Test error when input field is missing."""
        example = {"responses_create_params": {}}
        with pytest.raises(ValueError, match="missing 'input'"):
            extract_conversation_history(example)


class TestExtractGeneratedText:
    """Test extract_generated_text function."""

    def test_extract_from_output_list(self):
        """Test extracting text from output list."""
        gen_result = {"output": [{"content": "Hello world", "type": "text"}]}
        text = extract_generated_text(gen_result)
        assert text == "Hello world"

    def test_extract_from_output_string(self):
        """Test extracting text from output string."""
        gen_result = {"output": "Hello world"}
        text = extract_generated_text(gen_result)
        assert text == "Hello world"

    def test_extract_from_response_field(self):
        """Test extracting text from response field."""
        gen_result = {"response": "Hello world"}
        text = extract_generated_text(gen_result)
        assert text == "Hello world"

    def test_invalid_type_raises(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Expected dict"):
            extract_generated_text("not a dict")

    def test_no_extractable_field_raises(self):
        """Test that missing fields raises ValueError."""
        gen_result = {"other_field": "value"}
        with pytest.raises(ValueError, match="Cannot extract generated text"):
            extract_generated_text(gen_result)


class TestGenRMStrategy:
    """Test GenRMStrategy class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return GenRMStrategyConfig(
            agent_names=["genrm_simple_agent"],
            num_generations_per_prompt=4,
            genrm_compare_server_name="genrm_compare",
            policy_model_server_name="policy_model",
        )

    @pytest.fixture
    def strategy(self, config):
        """Create a GenRMStrategy instance."""
        return GenRMStrategy(config)

    def test_strategy_initialization(self, strategy, config):
        """Test that strategy is initialized correctly."""
        assert strategy.agent_names == config.agent_names
        assert strategy.num_generations_per_prompt == config.num_generations_per_prompt
        assert strategy.policy_model_server_name == config.policy_model_server_name

    def test_strategy_implements_protocol(self, strategy):
        """Test that GenRMStrategy implements ComparisonStrategy protocol."""
        assert isinstance(strategy, ComparisonStrategy)

    @pytest.mark.asyncio
    async def test_compare_calls_genrm_compare_server(self, strategy):
        """Test that compare() calls the genrm_compare server correctly."""
        # Mock server client
        server_client = MagicMock()
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(
            return_value={"rewards": [4.5, 3.2, 4.0], "metrics": {"mean_individual_score": 3.9}}
        )
        server_client.post = AsyncMock(return_value=mock_response)

        # Test data
        conversation_history = [{"role": "user", "content": "What is 2+2?"}]
        response_objs = [
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "4"}]}]},
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Four"}]}]},
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "2+2=4"}]}]},
        ]
        principle = "Be concise"

        # Call compare
        rewards, metrics = await strategy.compare(
            conversation_history, response_objs, server_client, principle=principle
        )

        # Verify server call
        server_client.post.assert_called_once()
        call_args = server_client.post.call_args
        assert call_args.kwargs["server_name"] == "genrm_compare"
        assert call_args.kwargs["url_path"] == "/compare"

        payload = call_args.kwargs["json"]
        assert payload["conversation_history"] == conversation_history
        assert payload["response_objs"] == response_objs
        assert payload["principle"] == principle

        # Verify results
        assert rewards == [4.5, 3.2, 4.0]
        assert metrics == {"mean_individual_score": 3.9}

    @pytest.mark.asyncio
    async def test_compare_without_principle(self, strategy):
        """Test compare() without principle."""
        server_client = MagicMock()
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"rewards": [3.0, 3.0], "metrics": {}})
        server_client.post = AsyncMock(return_value=mock_response)

        conversation_history = [{"role": "user", "content": "Hello"}]
        response_objs = [{"output": []}, {"output": []}]

        rewards, metrics = await strategy.compare(conversation_history, response_objs, server_client, principle=None)

        # Verify principle was not included in payload
        payload = server_client.post.call_args.kwargs["json"]
        assert "principle" not in payload

        assert rewards == [3.0, 3.0]
        assert metrics == {}


class TestGenerateResponse:
    """Test generate_response function."""

    @pytest.mark.asyncio
    async def test_generate_response_success(self):
        """Test successful response generation."""
        server_client = MagicMock()
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(
            return_value={
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Generated response"}]}]
            }
        )
        server_client.post = AsyncMock(return_value=mock_response)

        example = {
            "responses_create_params": {"input": [{"role": "user", "content": "Hello"}], "max_output_tokens": 100}
        }

        result = await generate_response(example, server_client, "policy_model")

        # Verify server call
        server_client.post.assert_called_once_with(
            server_name="policy_model", url_path="/v1/responses", json=example["responses_create_params"]
        )

        assert result["output"][0]["content"][0]["text"] == "Generated response"

    @pytest.mark.asyncio
    async def test_generate_response_missing_params(self):
        """Test error when responses_create_params is missing."""
        server_client = MagicMock()
        example = {"other_field": "value"}

        with pytest.raises(ValueError, match="missing 'responses_create_params'"):
            await generate_response(example, server_client, "policy_model")
