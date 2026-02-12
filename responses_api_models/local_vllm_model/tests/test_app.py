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

import sys
from unittest.mock import MagicMock

from vllm import platforms
from vllm.platforms import resolve_obj_by_qualname

import responses_api_models.local_vllm_model.app
from nemo_gym.global_config import DISALLOWED_PORTS_KEY_NAME, DictConfig
from responses_api_models.local_vllm_model.app import LocalVLLMModel, LocalVLLMModelConfig


class TestApp:
    def test_sanity_vllm_import(self) -> None:
        import vllm

        print(f"Found vLLM version: {vllm.__version__}")
        assert vllm.__version__

    def test_sanity_config_init(self) -> None:
        LocalVLLMModelConfig(
            host="",
            port=0,
            entrypoint="",
            name="test name",
            model="test model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            vllm_serve_env_vars=dict(),
            vllm_serve_kwargs=dict(),
        )

    def test_sanity_start_vllm_server(self, monkeypatch) -> None:
        get_global_config_dict_mock = MagicMock()
        get_global_config_dict_mock.return_value = DictConfig({DISALLOWED_PORTS_KEY_NAME: []})
        monkeypatch.setattr(
            responses_api_models.local_vllm_model.app,
            "get_global_config_dict",
            get_global_config_dict_mock,
        )

        cpu_platform = resolve_obj_by_qualname("vllm.platforms.cpu.CpuPlatform")()
        monkeypatch.setattr(platforms, "_current_platform", cpu_platform)

        monkeypatch.setattr(sys, "argv", ["dummy"])

        class DummyLocalVLLMModel:
            config = LocalVLLMModelConfig(
                host="",
                port=0,
                entrypoint="",
                name="test name",
                model="test model",
                return_token_id_information=False,
                uses_reasoning_parser=False,
                vllm_serve_env_vars=dict(),
                vllm_serve_kwargs=dict(),
            )

            get_cache_dir = LocalVLLMModel.get_cache_dir
            get_hf_token = LocalVLLMModel.get_hf_token

        LocalVLLMModel._configure_vllm_serve(DummyLocalVLLMModel())
