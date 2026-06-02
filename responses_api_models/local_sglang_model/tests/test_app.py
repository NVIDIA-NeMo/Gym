# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import responses_api_models.local_sglang_model.app
from nemo_gym.global_config import DISALLOWED_PORTS_KEY_NAME, DictConfig
from responses_api_models.local_sglang_model.app import LocalSGLangModel, LocalSGLangModelConfig


class TestApp:
    def test_sanity_config_init(self) -> None:
        LocalSGLangModelConfig(
            host="",
            port=0,
            entrypoint="",
            name="test name",
            model="test model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            sglang_serve_kwargs=dict(),
        )

    def test_kwargs_to_cli_args(self) -> None:
        assert LocalSGLangModel._kwargs_to_cli_args(
            {
                "model_path": "Qwen/Qwen3",
                "tp_size": 4,
                "trust_remote_code": True,
                "skip_tokenizer_init": False,
                "json_arg": {"a": 1},
                "none_arg": None,
            }
        ) == [
            "--model-path",
            "Qwen/Qwen3",
            "--tp-size",
            "4",
            "--trust-remote-code",
            "--json-arg",
            '{"a": 1}',
        ]

    def test_normalize_server_args(self) -> None:
        assert LocalSGLangModel._normalize_server_args(
            {
                "tp_size": 4,
                "pp_size": 2,
                "dp_size": 3,
                "ep_size": 5,
                "tensor_parallel_size": 8,
            }
        ) == {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 3,
            "expert_parallel_size": 5,
        }

    def test_configure_sglang_serve(self, monkeypatch) -> None:
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = DictConfig({DISALLOWED_PORTS_KEY_NAME: []})
        monkeypatch.setattr(
            responses_api_models.local_sglang_model.app,
            "get_global_config_dict",
            get_global_config_dict_mock,
        )
        monkeypatch.setattr(responses_api_models.local_sglang_model.app, "find_open_port", lambda disallowed_ports: 12345)
        monkeypatch.setattr(responses_api_models.local_sglang_model.app, "get_hf_token", lambda: None)

        class DummyLocalSGLangModel:
            config = LocalSGLangModelConfig(
                host="",
                port=0,
                entrypoint="",
                name="test name",
                model="served name",
                model_path="model path",
                return_token_id_information=False,
                uses_reasoning_parser=False,
                sglang_env_prefix="/opt/sglang",
                sglang_serve_env_vars={"A": "B"},
                sglang_serve_kwargs={"tensor_parallel_size": 2, "trust_remote_code": True},
            )

        cmd, env_vars, port = LocalSGLangModel._configure_sglang_serve(DummyLocalSGLangModel())

        assert cmd == [
            "/opt/sglang/bin/python",
            "-m",
            "sglang.launch_server",
            "--tensor-parallel-size",
            "2",
            "--trust-remote-code",
            "--model-path",
            "model path",
            "--served-model-name",
            "served name",
            "--host",
            "0.0.0.0",
            "--port",
            "12345",
        ]
        assert env_vars["A"] == "B"
        assert env_vars["PATH"].startswith("/opt/sglang/bin:")
        assert "/opt/sglang/lib" in env_vars["LD_LIBRARY_PATH"]
        assert env_vars["TOKENIZERS_PARALLELISM"] == "false"
        assert port == 12345
