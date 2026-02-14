# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GenRM remote model message formatting."""

from responses_api_models.genrm_model.remote.app import GenRMConverter, GenRMModelConfig
from responses_api_models.vllm_model.app import VLLMConverterResponsesToChatCompletionsState


class TestGenRMConverter:
    """Test GenRM message formatting with custom roles."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = GenRMConverter(return_token_id_information=False, supports_principle_role=True)

    def test_response_1_role_formatting(self):
        """Test formatting of response_1 custom role."""
        state = VLLMConverterResponsesToChatCompletionsState(return_token_id_information=False)

        message = {"role": "response_1", "content": "The capital of France is Paris.", "type": "message"}

        self.converter._format_message(message, state)

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "response_1"
        assert state.messages[0]["content"] == "The capital of France is Paris."

    def test_response_2_role_formatting(self):
        """Test formatting of response_2 custom role."""
        state = VLLMConverterResponsesToChatCompletionsState(return_token_id_information=False)

        message = {"role": "response_2", "content": "Paris is the capital city of France.", "type": "message"}

        self.converter._format_message(message, state)

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "response_2"
        assert state.messages[0]["content"] == "Paris is the capital city of France."

    def test_principle_role_formatting(self):
        """Test formatting of principle custom role."""
        state = VLLMConverterResponsesToChatCompletionsState(return_token_id_information=False)

        message = {"role": "principle", "content": "Please act as an impartial judge...", "type": "message"}

        self.converter._format_message(message, state)

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "principle"
        assert "impartial judge" in state.messages[0]["content"]

    def test_mixed_standard_and_custom_roles(self):
        """Test that standard roles still work alongside custom roles."""
        state = VLLMConverterResponsesToChatCompletionsState(return_token_id_information=False)

        user_msg = {"role": "user", "content": "What is the capital of France?", "type": "message"}
        self.converter._format_message(user_msg, state)

        principle_msg = {"role": "principle", "content": "Be concise", "type": "message"}
        self.converter._format_message(principle_msg, state)

        resp1_msg = {"role": "response_1", "content": "Paris", "type": "message"}
        self.converter._format_message(resp1_msg, state)

        resp2_msg = {"role": "response_2", "content": "The answer is Paris.", "type": "message"}
        self.converter._format_message(resp2_msg, state)

        assert len(state.messages) == 4
        assert state.messages[0]["role"] == "user"
        assert state.messages[1]["role"] == "principle"
        assert state.messages[2]["role"] == "response_1"
        assert state.messages[3]["role"] == "response_2"

    def test_custom_role_with_list_content(self):
        """Test custom role formatting when content is a list."""
        state = VLLMConverterResponsesToChatCompletionsState(return_token_id_information=False)

        message = {
            "role": "response_1",
            "content": [{"type": "text", "text": "Part 1 "}, {"type": "text", "text": "Part 2"}],
            "type": "message",
        }

        self.converter._format_message(message, state)

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "response_1"
        assert state.messages[0]["content"] == "Part 1 Part 2"


class TestGenRMModelConfig:
    """Test GenRM remote model configuration."""

    def test_config_defaults(self):
        """Test that GenRMModelConfig has correct defaults."""
        config = GenRMModelConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            name="test_genrm_remote_model",
            base_url="http://localhost:8000",
            api_key="test",
            model="test-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
        )

        assert config.supports_principle_role is True

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = GenRMModelConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            name="test_genrm_remote_model",
            base_url="http://localhost:8000",
            api_key="test",
            model="test-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            supports_principle_role=False,
        )

        assert config.supports_principle_role is False
