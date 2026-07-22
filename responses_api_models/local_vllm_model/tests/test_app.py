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
from argparse import Namespace
from unittest.mock import MagicMock, call

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
                vllm_serve_env_vars={"VLLM_RAY_DP_PACK_STRATEGY": "strict"},
                vllm_serve_kwargs={"data_parallel_size": 1, "tensor_parallel_size": 1, "pipeline_parallel_size": 1},
            )

            get_cache_dir = LocalVLLMModel.get_cache_dir

        LocalVLLMModel._configure_vllm_serve(DummyLocalVLLMModel())

    def test_start_vllm_server_launches_independent_replicas(self, monkeypatch) -> None:
        server_args = Namespace(data_parallel_size=1, port=1234)
        env_vars = {"VLLM_RAY_DP_PACK_STRATEGY": "strict"}
        placement_groups = [MagicMock(name="placement_group_0"), MagicMock(name="placement_group_1")]

        model = MagicMock()
        model.config = LocalVLLMModelConfig(
            host="",
            port=0,
            entrypoint="",
            name="test_model",
            model="test model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            vllm_serve_env_vars=env_vars,
            vllm_serve_kwargs={
                "data_parallel_size": 1,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            num_replicas=2,
        )
        model._configure_vllm_serve.return_value = (server_args, env_vars)
        model._select_vllm_server_head_node.side_effect = placement_groups

        base_url_refs = [object(), object()]
        actors = [MagicMock(name="actor_0"), MagicMock(name="actor_1")]
        for actor, base_url_ref in zip(actors, base_url_refs):
            actor.base_url.remote.return_value = base_url_ref

        actor_builder = MagicMock()
        actor_builder.remote.side_effect = actors
        local_vllm_model_actor_mock = MagicMock()
        local_vllm_model_actor_mock.options.return_value = actor_builder
        monkeypatch.setattr(
            responses_api_models.local_vllm_model.app,
            "LocalVLLMModelActor",
            local_vllm_model_actor_mock,
        )
        monkeypatch.setattr(
            responses_api_models.local_vllm_model.app,
            "PlacementGroupSchedulingStrategy",
            lambda **kwargs: kwargs,
        )
        monkeypatch.setattr(
            responses_api_models.local_vllm_model.app,
            "get_global_config_dict",
            lambda: DictConfig({DISALLOWED_PORTS_KEY_NAME: []}),
        )
        monkeypatch.setattr(
            responses_api_models.local_vllm_model.app,
            "find_open_port",
            lambda disallowed_ports: 1235,
        )

        def fake_ray_get(refs):
            assert refs == base_url_refs
            return ["http://replica-0/v1", "http://replica-1/v1"]

        monkeypatch.setattr(responses_api_models.local_vllm_model.app.ray, "get", fake_ray_get)

        LocalVLLMModel.start_vllm_server(model)

        assert model._local_vllm_model_actors == actors
        assert model._local_vllm_model_actor is actors[0]
        assert model.config.base_url == ["http://replica-0/v1", "http://replica-1/v1"]
        assert model._select_vllm_server_head_node.call_args_list == [
            call(server_args, env_vars, replica_idx=0),
            call(Namespace(data_parallel_size=1, port=1235), env_vars, replica_idx=1),
        ]
        assert [remote_call.kwargs["server_args"].port for remote_call in actor_builder.remote.call_args_list] == [
            1234,
            1235,
        ]
        assert [remote_call.kwargs["server_name"] for remote_call in actor_builder.remote.call_args_list] == [
            "test_model_replica_0",
            "test_model_replica_1",
        ]
        model._post_init.assert_called_once_with()
        model.await_server_ready.assert_called_once_with()
