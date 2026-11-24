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

import pytest
from fastapi.testclient import TestClient
from vllm import platforms
from vllm.platforms import resolve_obj_by_qualname

import responses_api_models.local_vllm_model.app
from nemo_gym.client import ServerClient
from nemo_gym.config import DISALLOWED_PORTS_KEY_NAME
from nemo_gym.config.global_config import DictConfig
from nemo_gym.models.token_capture import TokenCaptureStore
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

    def test_inner_config_route_is_non_generating(self) -> None:
        assert ("GET", "/get_inner_vllm_config") in LocalVLLMModel.non_generating_model_routes

    @pytest.mark.parametrize(
        ("method", "path", "incomplete"),
        [("GET", "/get_inner_vllm_config", False), ("POST", "/context/measure", False), ("POST", "/unlisted", True)],
    )
    def test_declared_and_inherited_routes_keep_captured_rollout_complete(
        self, monkeypatch, tmp_path, method: str, path: str, incomplete: bool
    ) -> None:
        """The inner-config declaration extends VLLMModel's routes, so a context measurement is not a generation."""
        monkeypatch.setattr(LocalVLLMModel, "start_vllm_server", lambda self: None)
        model = LocalVLLMModel(
            config=LocalVLLMModelConfig(
                host="",
                port=0,
                entrypoint="",
                name="test name",
                model="test model",
                return_token_id_information=False,
                uses_reasoning_parser=False,
                vllm_serve_env_vars=dict(),
                vllm_serve_kwargs=dict(),
            ),
            server_client=MagicMock(
                spec=ServerClient, global_config_dict={"token_id_capture": {"enabled": True, "dir": str(tmp_path)}}
            ),
        )
        app = model.setup_webserver()
        if path != "/get_inner_vllm_config":
            # /context/measure is deployed only under source_history capture; a stub isolates the classification.
            app.add_api_route(path, lambda: {"prompt_token_count": 1}, methods=[method])

        response = TestClient(app).request(method, f"/ng-rollout/measure-roll0/training-token-capture{path}")

        assert response.status_code == 200
        assert TokenCaptureStore(tmp_path).freeze_now("measure-roll0").incomplete is incomplete

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
            dummy.await_server_ready = lambda deadline=None: None
            dummy._get_within = lambda ref, deadline: LocalVLLMModel._get_within(dummy, ref, deadline)
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


class TestAwaitServerReady:
    """`await_server_ready` waits for the launched vLLM server, for as long as its actor is alive unless limited."""

    def _server(self, monkeypatch, *, timeout_s, answers: list, ray_get=None):
        """A stand-in server. `answers` is what each poll of `/models` does: an exception to raise, or a reply.

        `ray_get(reference, timeout, clock)` stands in for the actor's health check, which is alive by default; the
        fake clock is a one-element list.
        """
        import requests

        module = responses_api_models.local_vllm_model.app
        now = [0.0]
        monkeypatch.setattr(module, "monotonic", lambda: now[0])
        monkeypatch.setattr(module, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
        monkeypatch.setattr(
            module.ray,
            "get",
            lambda reference, timeout=None: True if ray_get is None else ray_get(reference, timeout, now),
        )
        calls: list = []

        def get(url, **kwargs):
            calls.append(kwargs)
            answer = answers[min(len(calls), len(answers)) - 1]
            if isinstance(answer, Exception):
                raise answer
            return answer

        monkeypatch.setattr(module.requests, "get", get)
        # A wait with a limit probes through `get_within_deadline`, so it answers from the same script.
        monkeypatch.setattr(
            module, "get_within_deadline", lambda url, *, timeout_s, **kwargs: get(url, timeout=timeout_s)
        )

        class Server:
            config = MagicMock(server_ready_timeout_seconds=timeout_s, base_url=["http://engine/v1"])
            _local_vllm_model_actor = MagicMock()

            @classmethod
            def _startup_timeout_error(cls):
                return LocalVLLMModel._startup_timeout_error(cls)

        Server.config.name = "model"
        return Server, calls, requests

    def test_the_limit_is_off_by_default_and_must_be_positive(self) -> None:
        assert LocalVLLMModelConfig.model_fields["server_ready_timeout_seconds"].default is None
        with pytest.raises(ValueError):
            LocalVLLMModelConfig(
                host="",
                port=0,
                entrypoint="",
                name="test name",
                model="test model",
                return_token_id_information=False,
                uses_reasoning_parser=False,
                vllm_serve_env_vars={},
                vllm_serve_kwargs={},
                server_ready_timeout_seconds=0,
            )

    def test_a_server_that_is_not_ready_in_time_fails_startup(self, monkeypatch) -> None:
        import requests

        # Refuses 10 polls (30 s on the fake clock) and would answer on the 11th, so a limit that is ignored returns
        # normally and fails the raises below.
        server, calls, _ = self._server(
            monkeypatch, timeout_s=10, answers=[requests.exceptions.ConnectionError()] * 10 + [MagicMock()]
        )

        with pytest.raises(TimeoutError, match="was not ready within 10"):
            LocalVLLMModel.await_server_ready(server)

        assert len(calls) < 10

    def test_a_server_that_accepts_but_never_answers_is_polled_with_a_time_limit(self, monkeypatch) -> None:
        import requests

        server, calls, _ = self._server(
            monkeypatch, timeout_s=10, answers=[requests.exceptions.ReadTimeout()] * 10 + [MagicMock()]
        )

        with pytest.raises(TimeoutError):
            LocalVLLMModel.await_server_ready(server)

        assert all(0 < call["timeout"] <= 10 for call in calls)

    def test_a_probe_that_answers_after_the_deadline_is_not_accepted(self, monkeypatch) -> None:
        import requests

        clocks = []

        def healthy(reference, timeout, clock):
            clocks.append(clock)
            return True

        server, calls, _ = self._server(monkeypatch, timeout_s=1, answers=[MagicMock()], ray_get=healthy)
        answer = requests.get

        def slow_get(url, **kwargs):
            clocks[0][0] += 5  # the answer arrives after the 1 s limit
            return answer(url, **kwargs)

        monkeypatch.setattr(
            responses_api_models.local_vllm_model.app,
            "get_within_deadline",
            lambda url, *, timeout_s, **kwargs: slow_get(url, timeout=timeout_s),
        )

        with pytest.raises(TimeoutError, match="was not ready within 1"):
            LocalVLLMModel.await_server_ready(server)

    def test_a_health_check_that_returns_after_the_deadline_does_not_start_another_probe(self, monkeypatch) -> None:
        def late_reply(reference, timeout, clock):
            clock[0] += 5  # the check comes back after the 1 s limit
            return True

        server, calls, _ = self._server(monkeypatch, timeout_s=1, answers=[MagicMock()], ray_get=late_reply)

        with pytest.raises(TimeoutError, match="was not ready within 1"):
            LocalVLLMModel.await_server_ready(server)

        assert calls == []  # the server was never probed: it would have answered, so an accepted late check starts up

    def test_a_health_check_that_stalls_is_bounded_by_the_remaining_time(self, monkeypatch) -> None:
        from ray.exceptions import GetTimeoutError

        timeouts = []

        def stalled(reference, timeout, clock):
            timeouts.append(timeout)
            raise GetTimeoutError("the actor does not answer")

        server, calls, _ = self._server(monkeypatch, timeout_s=10, answers=[MagicMock()], ray_get=stalled)

        with pytest.raises(TimeoutError, match="was not ready within 10"):
            LocalVLLMModel.await_server_ready(server)

        assert timeouts == [10] and calls == []

    def test_without_a_limit_the_health_check_is_not_given_one(self, monkeypatch) -> None:
        timeouts = []

        def alive(reference, timeout, clock):
            timeouts.append(timeout)
            return True

        server, _, _ = self._server(monkeypatch, timeout_s=None, answers=[MagicMock()], ray_get=alive)

        LocalVLLMModel.await_server_ready(server)

        assert timeouts == [None]

    def test_a_server_that_is_ready_in_time_ends_the_wait(self, monkeypatch) -> None:
        import requests

        server, calls, _ = self._server(
            monkeypatch, timeout_s=60, answers=[requests.exceptions.ConnectionError(), MagicMock()]
        )

        LocalVLLMModel.await_server_ready(server)

        assert len(calls) == 2

    def test_without_a_limit_the_wait_has_no_time_limit_on_a_poll_and_goes_on_until_ready(self, monkeypatch) -> None:
        import requests

        server, calls, _ = self._server(
            monkeypatch, timeout_s=None, answers=[requests.exceptions.ConnectionError()] * 40 + [MagicMock()]
        )

        LocalVLLMModel.await_server_ready(server)

        assert len(calls) == 41
        assert all("timeout" not in call for call in calls)


class TestStartupDeadline:
    """One `server_ready_timeout_seconds` deadline covers the placement group, the actor address and the readiness wait."""

    def _launch(self, monkeypatch, *, timeout_s, stage_costs: dict, health_timeouts: list | None = None):
        """Run `start_vllm_server` on a fake clock. Each stage advances it by its cost, or never returns if None."""
        from ray.exceptions import GetTimeoutError

        module = responses_api_models.local_vllm_model.app
        now = [0.0]
        monkeypatch.setattr(module, "monotonic", lambda: now[0])
        polled: list = []

        def ray_get(reference, timeout=None):
            cost = stage_costs[reference]
            if cost is None or (timeout is not None and cost > timeout):
                now[0] += timeout if timeout is not None else 1e9
                raise GetTimeoutError("stalled")
            now[0] += cost
            if reference == "alive":
                health_timeouts.append(timeout)
                return True
            return "http://engine/v1"

        monkeypatch.setattr(module.ray, "get", ray_get)
        monkeypatch.setattr(module.requests, "get", lambda url, **kwargs: polled.append(url) or MagicMock())
        monkeypatch.setattr(
            module, "get_within_deadline", lambda url, *, timeout_s, **kwargs: polled.append(url) or MagicMock()
        )

        actor = MagicMock()
        actor.base_url.remote.return_value = "address"
        actor.is_alive.remote.return_value = "alive"

        class FakeActorClass:
            @staticmethod
            def options(**kwargs):
                return MagicMock(remote=lambda **_: actor)

        monkeypatch.setattr(module, "LocalVLLMModelActor", FakeActorClass)

        class Server:
            config = MagicMock(
                server_ready_timeout_seconds=timeout_s, base_url=None, debug=False, vllm_serve_env_vars={}
            )
            _post_init = lambda self: None  # noqa: E731
            _ray_actor_path = lambda self: ""  # noqa: E731
            _configure_vllm_serve = lambda self: (MagicMock(), {})  # noqa: E731
            _startup_timeout_error = LocalVLLMModel._startup_timeout_error
            _get_within = LocalVLLMModel._get_within
            await_server_ready = LocalVLLMModel.await_server_ready

            def _select_vllm_server_head_node(self, server_args, env_vars, deadline=None):
                return self._get_within(placement_group_ready, deadline)

        Server.config.name = "model"
        placement_group_ready = "placement"
        return Server(), polled, now

    def test_a_stalled_placement_group_ends_at_the_deadline_without_polling(self, monkeypatch) -> None:
        server, polled, now = self._launch(monkeypatch, timeout_s=1, stage_costs={"placement": None})

        with pytest.raises(TimeoutError, match="not ready within 1 s"):
            LocalVLLMModel.start_vllm_server(server)

        assert polled == [] and now[0] == 1

    def test_a_stalled_actor_address_ends_at_the_deadline_without_polling(self, monkeypatch) -> None:
        server, polled, now = self._launch(monkeypatch, timeout_s=1, stage_costs={"placement": 0.4, "address": None})

        with pytest.raises(TimeoutError, match="not ready within 1 s"):
            LocalVLLMModel.start_vllm_server(server)

        assert polled == [] and now[0] == 1

    def test_the_health_check_gets_only_what_the_earlier_stages_left(self, monkeypatch) -> None:
        health_timeouts: list = []
        server, polled, _ = self._launch(
            monkeypatch,
            timeout_s=100,
            stage_costs={"placement": 30, "address": 20, "alive": 0},
            health_timeouts=health_timeouts,
        )

        LocalVLLMModel.start_vllm_server(server)

        assert health_timeouts == [50] and len(polled) == 1

    def test_without_a_limit_no_launch_wait_has_a_time_limit(self, monkeypatch) -> None:
        seen: list = []
        server, _, _ = self._launch(
            monkeypatch, timeout_s=None, stage_costs={"placement": 5, "address": 5, "alive": 0}, health_timeouts=seen
        )

        LocalVLLMModel.start_vllm_server(server)

        assert seen == [None]
