# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GenRM model."""

from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from responses_api_models.genrm_model.app import GenRMModel, GenRMModelConfig
from responses_api_models.genrm_model.common import append_genrm_roles


def _config(**overrides) -> GenRMModelConfig:
    fields = dict(
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
    return GenRMModelConfig(**(fields | overrides))


def _messages(*roles_and_contents):
    return [{"role": role, "content": content} for role, content in roles_and_contents]


def _body(conversation, metadata=None):
    return {
        "messages": _messages(*conversation),
        **({"metadata": metadata} if metadata is not None else {}),
    }


# ---------------------------------------------------------------------------
# TestGenRMModelConfig
# ---------------------------------------------------------------------------


class TestGenRMModelConfig:
    """Test GenRM model configuration."""

    def test_config_defaults(self):
        """GenRMModelConfig has supports_principle_role and inherits local vLLM fields."""
        assert _config().supports_principle_role is True

    def test_config_supports_principle_role_override(self):
        """supports_principle_role can be set to False."""
        assert _config(supports_principle_role=False).supports_principle_role is False


# ---------------------------------------------------------------------------
# TestAppendGenRMRoles
# ---------------------------------------------------------------------------


class TestAppendGenRMRoles:
    """Unit tests for append_genrm_roles."""

    def test_response_messages_appended_from_metadata(self):
        """response_1 and response_2 from metadata are appended as custom-role messages."""
        body_dict = _body(
            conversation=[
                ("user", "What is the capital of France?"),
                ("assistant", "Paris."),
            ],
            metadata={"response_1": "Response A", "response_2": "Response B"},
        )
        msgs = append_genrm_roles(body_dict, supports_principle_role=True)["messages"]

        assert msgs[-2] == {"role": "response_1", "content": "Response A"}
        assert msgs[-1] == {"role": "response_2", "content": "Response B"}
        # Conversation history is untouched
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_principle_from_metadata_inserted_before_responses(self):
        """metadata['principle'] is appended as a 'principle' message before the responses."""
        body_dict = _body(
            conversation=[("user", "What is the capital of France?")],
            metadata={
                "principle": "Judge impartially.",
                "response_1": "Response A",
                "response_2": "Response B",
            },
        )
        msgs = append_genrm_roles(body_dict, supports_principle_role=True)["messages"]

        assert msgs[-3] == {"role": "principle", "content": "Judge impartially."}
        assert msgs[-2] == {"role": "response_1", "content": "Response A"}
        assert msgs[-1] == {"role": "response_2", "content": "Response B"}

    def test_supports_principle_role_false_skips_principle(self):
        """When supports_principle_role=False, metadata['principle'] is silently ignored."""
        body_dict = _body(
            conversation=[("user", "Prompt")],
            metadata={
                "principle": "Judge impartially.",
                "response_1": "Response A",
                "response_2": "Response B",
            },
        )
        msgs = append_genrm_roles(body_dict, supports_principle_role=False)["messages"]

        roles = [m["role"] for m in msgs]
        assert "principle" not in roles
        assert msgs[-2]["role"] == "response_1"
        assert msgs[-1]["role"] == "response_2"

    def test_metadata_is_consumed_and_not_forwarded(self):
        """metadata is popped from body_dict so it is not forwarded to vLLM."""
        body_dict = _body(
            conversation=[("user", "Prompt")],
            metadata={"response_1": "A", "response_2": "B"},
        )
        result = append_genrm_roles(body_dict, supports_principle_role=True)

        assert "metadata" not in result

    def test_no_metadata_leaves_messages_unchanged(self):
        """When no metadata is present no custom-role messages are appended."""
        body_dict = _body(
            conversation=[("user", "Prompt"), ("assistant", "Answer.")],
        )
        msgs = append_genrm_roles(body_dict, supports_principle_role=True)["messages"]

        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_empty_metadata_leaves_messages_unchanged(self):
        """An empty metadata dict is treated the same as no metadata."""
        body_dict = _body(
            conversation=[("user", "Prompt")],
            metadata={},
        )
        result = append_genrm_roles(body_dict, supports_principle_role=True)

        assert len(result["messages"]) == 1
        assert "metadata" not in result

    def test_conversation_system_messages_are_not_misidentified(self):
        """A system message in the conversation history is never converted to 'principle'."""
        body_dict = _body(
            conversation=[("system", "You are a helpful assistant."), ("user", "Hi")],
            metadata={"response_1": "A", "response_2": "B"},
        )
        msgs = append_genrm_roles(body_dict, supports_principle_role=True)["messages"]

        assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert msgs[1] == {"role": "user", "content": "Hi"}
        assert msgs[2] == {"role": "response_1", "content": "A"}
        assert msgs[3] == {"role": "response_2", "content": "B"}


@pytest.mark.parametrize("supports_principle_role", [True, False])
def test_server_appends_genrm_roles_from_metadata(supports_principle_role: bool) -> None:
    """GenRMModel runs the base preprocessing and then appends the GenRM roles."""
    server = GenRMModel(
        config=_config(supports_principle_role=supports_principle_role),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    body = {
        "messages": [{"role": "user", "content": "Which is better?"}],
        "metadata": {"response_1": "A", "response_2": "B", "principle": "Be concise."},
    }

    body = server._preprocess_chat_completion_create_params(MagicMock(), body)

    principle = [{"role": "principle", "content": "Be concise."}] if supports_principle_role else []
    assert body["messages"] == [
        {"role": "user", "content": "Which is better?"},
        *principle,
        {"role": "response_1", "content": "A"},
        {"role": "response_2", "content": "B"},
    ]
    assert "metadata" not in body
    assert body["model"] == "test-model"
