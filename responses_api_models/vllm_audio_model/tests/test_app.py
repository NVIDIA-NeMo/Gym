# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_audio_model — the metadata→audio_url splicer.

Exercises ``_preprocess_chat_completion_create_params`` directly with
synthetic body_dicts so the tests don't require a running vLLM endpoint.
"""

from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_audio_model.app import (
    METADATA_AUDIO_URL_KEY,
    VLLMAudioModel,
)
from responses_api_models.vllm_model.app import VLLMModelConfig


def _make_model() -> VLLMAudioModel:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vllm_audio_model",
        base_url="http://localhost:9999/v1",
        api_key="dummy",
        model="dummy-model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        uses_interleaved_reasoning=False,
    )
    return VLLMAudioModel(config=config, server_client=MagicMock(spec=ServerClient))


_AUDIO_URL = "data:audio/wav;base64,QUFB"  # placeholder — bytes don't matter for these tests


class TestVLLMAudioModelSplice:
    def test_no_metadata_passthrough(self) -> None:
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)
        # No audio in metadata → user message content stays a plain string.
        assert result["messages"][1]["content"] == "hello"

    def test_splice_into_string_user_content(self) -> None:
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Transcribe please."},
            ],
            "metadata": {METADATA_AUDIO_URL_KEY: _AUDIO_URL},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        # Audio_url removed from metadata; metadata cleared since it's now empty.
        assert "metadata" not in result

        user_content = result["messages"][1]["content"]
        assert isinstance(user_content, list)
        # Audio block must come BEFORE the text part (some audio models care).
        assert user_content[0] == {"type": "audio_url", "audio_url": {"url": _AUDIO_URL}}
        assert user_content[1] == {"type": "text", "text": "Transcribe please."}

    def test_splice_into_list_user_content(self) -> None:
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Transcribe please."}],
                }
            ],
            "metadata": {METADATA_AUDIO_URL_KEY: _AUDIO_URL, "other": "keep"},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        # audio_url stripped, but the "other" key is kept.
        assert result["metadata"] == {"other": "keep"}

        user_content = result["messages"][0]["content"]
        assert user_content[0]["type"] == "audio_url"
        assert user_content[1]["type"] == "text"

    def test_splice_targets_most_recent_user(self) -> None:
        """Multi-turn: audio attaches to the LATEST user message."""
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second turn"},
            ],
            "metadata": {METADATA_AUDIO_URL_KEY: _AUDIO_URL},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        # First user message untouched
        assert result["messages"][0]["content"] == "first turn"
        # Second user message — audio spliced in
        assert isinstance(result["messages"][2]["content"], list)
        assert result["messages"][2]["content"][0]["type"] == "audio_url"

    def test_no_user_message_creates_one(self) -> None:
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "system", "content": "sys"}],
            "metadata": {METADATA_AUDIO_URL_KEY: _AUDIO_URL},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)

        assert len(result["messages"]) == 2
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"][0]["type"] == "audio_url"

    def test_empty_audio_url_is_noop(self) -> None:
        """Empty/falsy audio_url string skips the splice (safety check)."""
        model = _make_model()
        body = {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {METADATA_AUDIO_URL_KEY: ""},
        }
        result = model._preprocess_chat_completion_create_params(MagicMock(), body)
        # Plain string content untouched; audio_url not removed because we returned early.
        assert result["messages"][0]["content"] == "hi"
