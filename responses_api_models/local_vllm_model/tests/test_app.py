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
from pathlib import Path
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
            dummy._configure_vllm_serve = lambda: (MagicMock(), dict(vllm_serve_env_vars))
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
