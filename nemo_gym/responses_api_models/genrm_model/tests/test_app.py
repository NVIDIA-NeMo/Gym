# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GenRM model."""

from nemo_gym.responses_api_models.genrm_model.app import GenRMModelConfig, GenRMModelMixin


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

    def _body(self, conversation, metadata=None):
        return {
            "messages": _messages(*conversation),
            **({"metadata": metadata} if metadata is not None else {}),
        }

    def test_response_messages_appended_from_metadata(self):
        """response_1 and response_2 from metadata are appended as custom-role messages."""
        server = _TestServer(supports_principle_role=True)
        body_dict = self._body(
            conversation=[
                ("user", "What is the capital of France?"),
                ("assistant", "Paris."),
            ],
            metadata={"response_1": "Response A", "response_2": "Response B"},
        )
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        assert msgs[-2] == {"role": "response_1", "content": "Response A"}
        assert msgs[-1] == {"role": "response_2", "content": "Response B"}
        # Conversation history is untouched
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_principle_from_metadata_inserted_before_responses(self):
        """metadata['principle'] is appended as a 'principle' message before the responses."""
        server = _TestServer(supports_principle_role=True)
        body_dict = self._body(
            conversation=[("user", "What is the capital of France?")],
            metadata={
                "principle": "Judge impartially.",
                "response_1": "Response A",
                "response_2": "Response B",
            },
        )
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        assert msgs[-3] == {"role": "principle", "content": "Judge impartially."}
        assert msgs[-2] == {"role": "response_1", "content": "Response A"}
        assert msgs[-1] == {"role": "response_2", "content": "Response B"}

    def test_supports_principle_role_false_skips_principle(self):
        """When supports_principle_role=False, metadata['principle'] is silently ignored."""
        server = _TestServer(supports_principle_role=False)
        body_dict = self._body(
            conversation=[("user", "Prompt")],
            metadata={
                "principle": "Judge impartially.",
                "response_1": "Response A",
                "response_2": "Response B",
            },
        )
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        roles = [m["role"] for m in msgs]
        assert "principle" not in roles
        assert msgs[-2]["role"] == "response_1"
        assert msgs[-1]["role"] == "response_2"

    def test_metadata_is_consumed_and_not_forwarded(self):
        """metadata is popped from body_dict so it is not forwarded to vLLM."""
        server = _TestServer()
        body_dict = self._body(
            conversation=[("user", "Prompt")],
            metadata={"response_1": "A", "response_2": "B"},
        )
        result = server._preprocess_chat_completion_create_params(None, body_dict)

        assert "metadata" not in result

    def test_no_metadata_leaves_messages_unchanged(self):
        """When no metadata is present no custom-role messages are appended."""
        server = _TestServer()
        body_dict = self._body(
            conversation=[("user", "Prompt"), ("assistant", "Answer.")],
        )
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_empty_metadata_leaves_messages_unchanged(self):
        """An empty metadata dict is treated the same as no metadata."""
        server = _TestServer()
        body_dict = self._body(
            conversation=[("user", "Prompt")],
            metadata={},
        )
        result = server._preprocess_chat_completion_create_params(None, body_dict)

        assert len(result["messages"]) == 1
        assert "metadata" not in result

    def test_conversation_system_messages_are_not_misidentified(self):
        """A system message in the conversation history is never converted to 'principle'."""
        server = _TestServer(supports_principle_role=True)
        body_dict = self._body(
            conversation=[("system", "You are a helpful assistant."), ("user", "Hi")],
            metadata={"response_1": "A", "response_2": "B"},
        )
        result = server._preprocess_chat_completion_create_params(None, body_dict)
        msgs = result["messages"]

        assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert msgs[1] == {"role": "user", "content": "Hi"}
        assert msgs[2] == {"role": "response_1", "content": "A"}
        assert msgs[3] == {"role": "response_2", "content": "B"}
