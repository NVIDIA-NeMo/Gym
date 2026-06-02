# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from responses_api_models.sglang_model.app import SGLangModel, SGLangModelConfig


class TestApp:
    def test_sanity_config_init(self) -> None:
        config = SGLangModelConfig(
            host="",
            port=0,
            entrypoint="",
            name="test name",
            base_url="http://127.0.0.1:30000/v1",
            api_key="dummy",
            model="test model",
            uses_reasoning_parser=False,
            return_token_id_information=False,
        )

        assert config.base_url == ["http://127.0.0.1:30000/v1"]
        assert not config.is_responses_native
        assert config.replace_developer_role_with_system

    def test_return_token_id_information_rejected(self) -> None:
        config = SGLangModelConfig(
            host="",
            port=0,
            entrypoint="",
            name="test name",
            base_url="http://127.0.0.1:30000/v1",
            api_key="dummy",
            model="test model",
            uses_reasoning_parser=False,
            return_token_id_information=True,
        )
        server = SGLangModel(config=config, server_client=MagicMock(spec=ServerClient))

        with pytest.raises(NotImplementedError, match="return_token_id_information"):
            server._preprocess_chat_completion_create_params(
                MagicMock(),
                {"messages": [{"role": "user", "content": "hello"}]},
            )
