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

import os
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, call

from vllm import platforms
from vllm.platforms import resolve_obj_by_qualname

import responses_api_models.local_vllm_model.app
from nemo_gym.global_config import DISALLOWED_PORTS_KEY_NAME, DictConfig
from responses_api_models.local_vllm_model.app import LocalVLLMModel, LocalVLLMModelConfig
from responses_api_models.local_vllm_model.local_vllm_model_actor import _get_local_dp_ranks


class TestApp:
    def test_local_dp_ranks_increment_per_worker_node(self, monkeypatch) -> None:
        placement_groups = [object(), object(), object(), object()]
        node_ids = ["node-a", "node-b", "node-a", "node-b"]
        placement_group_data = {
            placement_group: {"bundles_to_node_id": {0: node_id}}
            for placement_group, node_id in zip(placement_groups, node_ids)
        }
        monkeypatch.setattr(
            responses_api_models.local_vllm_model.local_vllm_model_actor.ray.util,
            "placement_group_table",
            placement_group_data.__getitem__,
        )

        assert _get_local_dp_ranks(placement_groups) == [0, 0, 1, 1]

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

    def test_completions_api_fields_inherited_from_vllm_model_config(self) -> None:
        """LocalVLLMModelConfig must inherit use_completions_api / render_chat_template /
        tokenizer from VLLMModelConfig so the same YAML flag flips behavior for a
        local-vLLM deployment."""
        cfg = LocalVLLMModelConfig(
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
        # Defaults match VLLMModelConfig's.
        assert cfg.use_completions_api is False
        assert cfg.render_chat_template is False
        assert cfg.tokenizer is None

        # All three fields are settable through the same constructor surface.
        cfg = LocalVLLMModelConfig(
            host="",
            port=0,
            entrypoint="",
            name="test name",
            model="test model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            vllm_serve_env_vars=dict(),
            vllm_serve_kwargs=dict(),
            use_completions_api=True,
            render_chat_template=True,
            tokenizer="some-other-model",
        )
        assert cfg.use_completions_api is True
        assert cfg.render_chat_template is True
        assert cfg.tokenizer == "some-other-model"

    @staticmethod
    def _config_with_py_executable(py_executable: str) -> LocalVLLMModelConfig:
        return LocalVLLMModelConfig(
            host="",
            port=0,
            entrypoint="",
            name="test name",
            model="test model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            vllm_serve_env_vars=dict(),
            vllm_serve_kwargs=dict(),
            ray_worker_py_executable=py_executable,
        )

    def test_ray_actor_path_leads_with_the_py_executable_directory(self, monkeypatch, tmp_path) -> None:
        """The actor resolves console scripts from the same venv as its interpreter.

        runtime_env.py_executable only sets the interpreter, so without this the binaries
        installed beside it (vLLM's ninja) are not on the actor's PATH.
        """
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        py_executable = venv_bin / "python"
        py_executable.touch()

        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        class DummyLocalVLLMModel:
            config = self._config_with_py_executable(str(py_executable))

        actor_path = LocalVLLMModel._ray_actor_path(DummyLocalVLLMModel())

        # The venv's bin directory wins over anything inherited...
        assert actor_path.split(os.pathsep)[0] == str(venv_bin)
        # ...but the inherited PATH is preserved as a fallback.
        assert actor_path == os.pathsep.join([str(venv_bin), "/usr/bin", "/bin"])

    def test_ray_actor_path_without_an_inherited_path(self, monkeypatch, tmp_path) -> None:
        """An unset/empty PATH must not produce a stray empty entry, which resolves to cwd."""
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        py_executable = venv_bin / "python"
        py_executable.touch()

        monkeypatch.delenv("PATH", raising=False)

        class DummyLocalVLLMModel:
            config = self._config_with_py_executable(str(py_executable))

        actor_path = LocalVLLMModel._ray_actor_path(DummyLocalVLLMModel())

        assert actor_path == str(venv_bin)
        assert "" not in actor_path.split(os.pathsep)

    def test_ray_actor_py_executable_defaults_to_the_running_interpreter(self) -> None:
        """Gym activates the server venv before launching, so sys.executable is that venv's
        python and the actor PATH follows it without any per-server configuration."""

        class DummyLocalVLLMModel:
            config = self._config_with_py_executable(sys.executable)

        assert DummyLocalVLLMModel.config.ray_worker_py_executable == sys.executable
        actor_path = LocalVLLMModel._ray_actor_path(DummyLocalVLLMModel())
        assert actor_path.split(os.pathsep)[0] == str(Path(sys.executable).resolve().parent)

    def test_start_vllm_server_passes_path_into_the_actor_runtime_env(self, monkeypatch, tmp_path) -> None:
        """The computed PATH reaches the actor's runtime_env, and a server-supplied PATH wins."""
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        py_executable = venv_bin / "python"
        py_executable.touch()

        captured = {}

        class FakeActorClass:
            @staticmethod
            def options(**kwargs):
                captured.update(kwargs)
                return MagicMock()

        monkeypatch.setattr(responses_api_models.local_vllm_model.app, "LocalVLLMModelActor", FakeActorClass)
        monkeypatch.setattr(
            responses_api_models.local_vllm_model.app.ray, "get", lambda _: ["http://localhost:1234/v1"]
        )

        def build_dummy(vllm_serve_env_vars):
            config = self._config_with_py_executable(str(py_executable))
            config.vllm_serve_env_vars = vllm_serve_env_vars
            config.base_url = None

            class DummyLocalVLLMModel:
                pass

            dummy = DummyLocalVLLMModel()
            dummy.config = config
            dummy._configure_vllm_serve = lambda: (
                Namespace(port=1234, data_parallel_size=1),
                dict(vllm_serve_env_vars),
            )
            dummy._select_vllm_server_head_node = lambda *args, **kwargs: MagicMock()
            dummy._ray_actor_path = lambda: LocalVLLMModel._ray_actor_path(dummy)
            dummy._post_init = lambda: None
            dummy.await_server_ready = lambda: None
            return dummy

        # Default: the venv's bin directory is on the actor's PATH.
        LocalVLLMModel.start_vllm_server(build_dummy({}))
        env_vars = captured["runtime_env"]["env_vars"]
        assert env_vars["PATH"].split(os.pathsep)[0] == str(venv_bin)
        assert captured["runtime_env"]["py_executable"] == str(py_executable)

        # A server config can still override it.
        captured.clear()
        LocalVLLMModel.start_vllm_server(build_dummy({"PATH": "/custom/bin"}))
        assert captured["runtime_env"]["env_vars"]["PATH"] == "/custom/bin"

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


def test_replica_readiness_waits_for_every_endpoint(monkeypatch) -> None:
    actors = [MagicMock(), MagicMock()]
    references = [object(), object()]
    for actor, reference in zip(actors, references):
        actor.is_alive.remote.return_value = reference
    model = Namespace(
        config=Namespace(name="replicas", base_url=["http://first/v1", "http://second/v1"]),
        _local_vllm_model_actors=actors,
    )
    alive_batches = []
    probes = []

    def alive(refs):
        alive_batches.append(refs)
        return [True] * len(refs)

    def probe(*, url):
        probes.append(url)
        if url == "http://second/v1/models" and probes.count(url) == 1:
            raise responses_api_models.local_vllm_model.app.ConnectionError("still starting")
        return MagicMock()

    monkeypatch.setattr(responses_api_models.local_vllm_model.app.ray, "get", alive)
    monkeypatch.setattr(responses_api_models.local_vllm_model.app.requests, "get", probe)
    monkeypatch.setattr(responses_api_models.local_vllm_model.app, "sleep", lambda _: None)
    LocalVLLMModel.await_server_ready(model)
    assert alive_batches == [references, [references[1]]]
    assert probes == ["http://first/v1/models", "http://second/v1/models", "http://second/v1/models"]


def test_independent_replicas_reject_nested_data_parallelism_before_allocating(monkeypatch) -> None:
    from pytest import raises

    model = MagicMock()
    model.config = Namespace(base_url=[], debug=False, num_replicas=2)
    model._configure_vllm_serve.return_value = (Namespace(data_parallel_size=2), {})
    with raises(AssertionError, match="data_parallel_size=1"):
        LocalVLLMModel.start_vllm_server(model)
    model._select_vllm_server_head_node.assert_not_called()
