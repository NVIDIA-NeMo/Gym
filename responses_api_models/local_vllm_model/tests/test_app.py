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
from typing import Dict, Tuple
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from requests.exceptions import ConnectionError
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
        monkeypatch.setattr(responses_api_models.local_vllm_model.app.ray, "get", lambda _: "http://localhost:1234/v1")

        def build_dummy(vllm_serve_env_vars):
            config = self._config_with_py_executable(str(py_executable))
            config.vllm_serve_env_vars = vllm_serve_env_vars
            config.base_url = None

            class DummyLocalVLLMModel:
                pass

            dummy = DummyLocalVLLMModel()
            dummy.config = config
            dummy._configure_vllm_serve = lambda taken_ports: (MagicMock(), dict(vllm_serve_env_vars))
            dummy._select_vllm_server_head_node = lambda *args, **kwargs: MagicMock()
            dummy._replica_name = lambda replica_idx: LocalVLLMModel._replica_name(dummy, replica_idx)
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

    def test_num_replicas_defaults_to_one_and_must_be_positive(self) -> None:
        assert self._config_with_py_executable(sys.executable).num_replicas == 1

        with pytest.raises(ValidationError):
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
                num_replicas=0,
            )

    @staticmethod
    def _fake_replica_actor(base_url: str, alive: bool = True) -> MagicMock:
        actor = MagicMock()
        actor.base_url.remote.return_value = base_url
        actor.is_alive.remote.return_value = alive
        return actor

    def _start_replicas(self, monkeypatch, num_replicas: int) -> Tuple[MagicMock, Dict[str, list]]:
        """Run start_vllm_server with Ray, port selection and readiness stubbed out, and record what it launched."""
        launched = {"configure_taken_ports": [], "head_node_names": [], "actor_options": [], "actor_kwargs": []}
        next_port = iter(range(5000, 5000 + num_replicas))

        def fake_configure(taken_ports):
            launched["configure_taken_ports"].append(list(taken_ports))
            return Namespace(port=next(next_port)), {"VLLM_RAY_DP_PACK_STRATEGY": "strict"}

        def fake_remote(**kwargs):
            launched["actor_kwargs"].append(kwargs)
            return self._fake_replica_actor(f"http://node-{kwargs['server_args'].port}:{kwargs['server_args'].port}/v1")

        class FakeActorClass:
            @staticmethod
            def options(**kwargs):
                launched["actor_options"].append(kwargs)
                return MagicMock(remote=fake_remote)

        monkeypatch.setattr(responses_api_models.local_vllm_model.app, "LocalVLLMModelActor", FakeActorClass)
        monkeypatch.setattr(responses_api_models.local_vllm_model.app.ray, "get", lambda refs: refs)

        config = self._config_with_py_executable(sys.executable)
        config.base_url = []
        config.num_replicas = num_replicas

        class DummyLocalVLLMModel:
            pass

        dummy = DummyLocalVLLMModel()
        dummy.config = config
        dummy._configure_vllm_serve = fake_configure
        dummy._select_vllm_server_head_node = lambda server_args, env_vars, replica_name: launched[
            "head_node_names"
        ].append(replica_name) or MagicMock(name=replica_name)
        dummy._replica_name = lambda replica_idx: LocalVLLMModel._replica_name(dummy, replica_idx)
        dummy._ray_actor_path = lambda: "/venv/bin"
        dummy._post_init = MagicMock()
        dummy.await_server_ready = MagicMock()

        LocalVLLMModel.start_vllm_server(dummy)
        return dummy, launched

    def test_start_vllm_server_launches_one_server_per_replica(self, monkeypatch) -> None:
        dummy, launched = self._start_replicas(monkeypatch, num_replicas=3)

        assert len(launched["actor_kwargs"]) == 3
        assert [kwargs["server_name"] for kwargs in launched["actor_kwargs"]] == [
            "test name_replica_0",
            "test name_replica_1",
            "test name_replica_2",
        ]
        # Each replica gets its own head node placement group, named after it.
        assert launched["head_node_names"] == ["test name_replica_0", "test name_replica_1", "test name_replica_2"]
        # Every replica's URL is served, in launch order, and the clients are rebuilt from them before waiting.
        assert dummy.config.base_url == [
            "http://node-5000:5000/v1",
            "http://node-5001:5001/v1",
            "http://node-5002:5002/v1",
        ]
        dummy._post_init.assert_called_once()
        dummy.await_server_ready.assert_called_once()

    def test_start_vllm_server_does_not_reuse_an_earlier_replicas_port(self, monkeypatch) -> None:
        _, launched = self._start_replicas(monkeypatch, num_replicas=3)

        assert launched["configure_taken_ports"] == [[], [5000], [5000, 5001]]

    def test_single_replica_keeps_the_server_name(self, monkeypatch) -> None:
        dummy, launched = self._start_replicas(monkeypatch, num_replicas=1)

        assert [kwargs["server_name"] for kwargs in launched["actor_kwargs"]] == ["test name"]
        assert launched["head_node_names"] == ["test name"]
        assert dummy.config.base_url == ["http://node-5000:5000/v1"]

    @staticmethod
    def _dummy_awaiting(actors_and_urls) -> object:
        class DummyLocalVLLMModel:
            pass

        dummy = DummyLocalVLLMModel()
        dummy.config = TestApp._config_with_py_executable(sys.executable)
        dummy._local_vllm_model_actors = [actor for actor, _ in actors_and_urls]
        dummy.config.base_url = [url for _, url in actors_and_urls]
        return dummy

    def test_await_server_ready_waits_for_every_replica(self, monkeypatch) -> None:
        url_0, url_1 = "http://node-0:5000/v1", "http://node-1:5001/v1"
        dummy = self._dummy_awaiting([(self._fake_replica_actor(url_0), url_0), (self._fake_replica_actor(url_1), url_1)])

        # Replica 0 is up at once; replica 1 refuses connections for two polls.
        refusals = {url_1: 2}
        polled = []

        def fake_get(url):
            polled.append(url)
            base_url = url.removesuffix("/models")
            if refusals.get(base_url, 0) > 0:
                refusals[base_url] -= 1
                raise ConnectionError()

        monkeypatch.setattr(responses_api_models.local_vllm_model.app.ray, "get", lambda refs: refs)
        monkeypatch.setattr(responses_api_models.local_vllm_model.app.requests, "get", lambda url: fake_get(url))
        sleep = MagicMock()
        monkeypatch.setattr(responses_api_models.local_vllm_model.app, "sleep", sleep)

        LocalVLLMModel.await_server_ready(dummy)

        # A ready replica is not polled again while the others are still starting.
        assert polled == [f"{url_0}/models", f"{url_1}/models", f"{url_1}/models", f"{url_1}/models"]
        assert sleep.call_count == 2

    def test_await_server_ready_fails_when_any_replica_dies(self, monkeypatch) -> None:
        url_0, url_1 = "http://node-0:5000/v1", "http://node-1:5001/v1"
        dummy = self._dummy_awaiting(
            [(self._fake_replica_actor(url_0), url_0), (self._fake_replica_actor(url_1, alive=False), url_1)]
        )

        monkeypatch.setattr(responses_api_models.local_vllm_model.app.ray, "get", lambda refs: refs)
        monkeypatch.setattr(responses_api_models.local_vllm_model.app.requests, "get", MagicMock())

        with pytest.raises(AssertionError, match=url_1):
            LocalVLLMModel.await_server_ready(dummy)

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
