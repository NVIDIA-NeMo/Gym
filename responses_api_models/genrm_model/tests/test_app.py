# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GenRM model."""

from responses_api_models.genrm_model.app import GenRMModelConfig, GenRMModelMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeBase:
    """Stand-in for VLLMModel that performs a no-op base preprocessing."""

    def _preprocess_chat_completion_create_params(self, request, body_dict):
        return body_dict


class _FakeConfig:
    supports_principle_role: bool = True


class _TestServer(GenRMModelMixin, _FakeBase):
    """Minimal concrete class that exercises GenRMModelMixin in isolation
    (no vLLM / Ray dependency)."""

    def __init__(self, supports_principle_role: bool = True):
        cfg = _FakeConfig()
        cfg.supports_principle_role = supports_principle_role
        self.config = cfg


def _messages(*roles_and_contents):
    return [{"role": role, "content": content} for role, content in roles_and_contents]


# ---------------------------------------------------------------------------
# TestGenRMModelConfig
# ---------------------------------------------------------------------------


class TestGenRMModelConfig:
    """Test GenRM model configuration."""

    def test_config_defaults(self):
        """GenRMModelConfig has supports_principle_role and inherits local vLLM fields."""
        config = GenRMModelConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            name="test_genrm_model",
            model="test-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            vllm_serve_kwargs={"tensor_parallel_size": 1, "data_parallel_size": 2},
            vllm_serve_env_vars={},
        )
        assert config.supports_principle_role is True

    def test_config_supports_principle_role_override(self):
        """supports_principle_role can be set to False."""
        config = GenRMModelConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            name="test_genrm_model",
            model="test-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            vllm_serve_kwargs={"tensor_parallel_size": 1, "data_parallel_size": 2},
            vllm_serve_env_vars={},
            supports_principle_role=False,
        )
        assert config.supports_principle_role is False


# ---------------------------------------------------------------------------
# TestGenRMPreprocess
# ---------------------------------------------------------------------------


class TestGenRMPreprocess:
    """Unit tests for GenRMModelMixin._preprocess_chat_completion_create_params."""

    def test_last_two_user_messages_become_response_roles(self):
        """The last two messages (user) are remapped to response_1 / response_2."""
        server = _TestServer(supports_principle_role=True)
        body_dict = {
            "messages": _messages(
                ("user", "What is the capital of France?"),
                ("assistant", "Paris."),
                ("user", "Response A"),
                ("user", "Response B"),
            )
        }
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        assert msgs[-2]["role"] == "response_1"
        assert msgs[-2]["content"] == "Response A"
        assert msgs[-1]["role"] == "response_2"
        assert msgs[-1]["content"] == "Response B"
        # Conversation history is untouched
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_system_message_before_responses_becomes_principle(self):
        """A system message immediately before the two response messages becomes 'principle'."""
        server = _TestServer(supports_principle_role=True)
        body_dict = {
            "messages": _messages(
                ("user", "What is the capital of France?"),
                ("assistant", "Paris."),
                ("system", "Judge impartially."),
                ("user", "Response A"),
                ("user", "Response B"),
            )
        }
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        assert msgs[-3]["role"] == "principle"
        assert msgs[-3]["content"] == "Judge impartially."
        assert msgs[-2]["role"] == "response_1"
        assert msgs[-1]["role"] == "response_2"

    def test_non_system_message_before_responses_is_not_converted(self):
        """Only a 'system' role at position -3 triggers principle conversion."""
        server = _TestServer(supports_principle_role=True)
        body_dict = {
            "messages": _messages(
                ("user", "What is the capital of France?"),
                ("assistant", "Paris."),  # assistant, not system
                ("user", "Response A"),
                ("user", "Response B"),
            )
        }
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        assert msgs[-3]["role"] == "assistant"  # unchanged
        assert msgs[-2]["role"] == "response_1"
        assert msgs[-1]["role"] == "response_2"

    def test_supports_principle_role_false_skips_principle_conversion(self):
        """When supports_principle_role=False, the system message is left as-is."""
        server = _TestServer(supports_principle_role=False)
        body_dict = {
            "messages": _messages(
                ("user", "Prompt"),
                ("system", "Judge impartially."),
                ("user", "Response A"),
                ("user", "Response B"),
            )
        }
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        assert msgs[-3]["role"] == "system"  # NOT converted to principle
        assert msgs[-2]["role"] == "response_1"
        assert msgs[-1]["role"] == "response_2"

    def test_minimal_two_message_input(self):
        """Works correctly when the input contains exactly two messages."""
        server = _TestServer()
        body_dict = {
            "messages": _messages(
                ("user", "Response A"),
                ("user", "Response B"),
            )
        }
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        assert msgs[-2]["role"] == "response_1"
        assert msgs[-1]["role"] == "response_2"

    def test_single_message_input_is_not_modified(self):
        """With fewer than two messages no role remapping is performed."""
        server = _TestServer()
        body_dict = {"messages": _messages(("user", "Only one message"))}
        result = server._preprocess_chat_completion_create_params(None, body_dict)

        assert result["messages"][0]["role"] == "user"

    def test_messages_mutated_in_place(self):
        """The method mutates the list in-place and also returns it."""
        server = _TestServer()
        original_messages = _messages(("user", "A"), ("user", "B"))
        body_dict = {"messages": original_messages}
        result = server._preprocess_chat_completion_create_params(None, body_dict)

        assert result["messages"] is original_messages
        assert original_messages[-2]["role"] == "response_1"
        assert original_messages[-1]["role"] == "response_2"
