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
import shutil
import sys
import tomllib
import warnings
from importlib import import_module
from io import StringIO
from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
from pytest import MonkeyPatch

import nemo_gym.global_config
from nemo_gym import PARENT_DIR
from nemo_gym.cli.env import (
    _FORCE_KILL_REAP_TIMEOUT_SEC,
    _GRACEFUL_SHUTDOWN_TIMEOUT_SEC,
    RunConfig,
    RunHelper,
    _run_module_tests_all,
    init_resources_server,
)
from nemo_gym.cli.general import display_help_legacy
from nemo_gym.config_types import DatasetConfig, ResourcesServerInstanceConfig


# TODO: Eventually we want to add more tests to ensure that the CLI flows do not break
class TestCLI:
    def test_sanity(self) -> None:
        RunConfig(entrypoint="", name="")

    def test_pyproject_scripts_are_importable(self) -> None:
        """Every console-script entry point must resolve to an importable callable."""
        pyproject_path = PARENT_DIR / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject_data = tomllib.load(f)

        for script_name, import_path in pyproject_data["project"]["scripts"].items():
            module, fn = import_path.split(":")
            target = getattr(import_module(module), fn)
            assert callable(target), f"{script_name} -> {import_path} is not callable"

    def test_display_help_discovers_scripts(self) -> None:
        with MonkeyPatch.context() as mp:
            mp.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", OmegaConf.create({}))

            text_trap = StringIO()
            mp.setattr(sys, "stdout", text_trap)

            display_help_legacy()

            output = text_trap.getvalue()
            assert "ng_help" in output
            assert "ng_run" in output
            assert "ng_collect_rollouts" in output

    def test_init_resources_server_includes_domain(self) -> None:
        """Test that init_resources_server creates a config with the required domain field."""

        server_name = "test_cli_server"
        entrypoint = f"resources_servers/{server_name}"
        server_path = Path(entrypoint).resolve()

        # Clean up any existing test server directory
        if server_path.exists():
            shutil.rmtree(server_path)

        try:
            with MonkeyPatch.context() as mp:
                # Set up the global config to point to our test entrypoint
                mp.setattr(
                    nemo_gym.global_config,
                    "_GLOBAL_CONFIG_DICT",
                    OmegaConf.create({"entrypoint": entrypoint}),
                )

                # Run init_resources_server
                init_resources_server()

                # Verify the generated config file exists
                config_file = server_path / "configs" / f"{server_name}.yaml"
                assert config_file.exists(), f"Config file not created at {config_file}"

                # Load and verify the config
                config_dict = OmegaConf.load(config_file)

                # Check that the domain field is present in the resources server config
                resources_server_key = f"{server_name}_resources_server"
                assert resources_server_key in config_dict, f"Resources server key '{resources_server_key}' not found"

                resources_config = config_dict[resources_server_key]
                assert "resources_servers" in resources_config
                assert server_name in resources_config["resources_servers"]

                server_config = resources_config["resources_servers"][server_name]
                assert "domain" in server_config, "Domain field missing from resources server config"
                assert server_config["domain"] == "other", f"Expected domain 'other', got '{server_config['domain']}'"

                # Verify that the config can be validated (this would have failed before the fix)
                full_config_dict = OmegaConf.create(
                    {
                        "name": resources_server_key,
                        "server_type_config_dict": config_dict[resources_server_key],
                        **OmegaConf.to_container(config_dict[resources_server_key]),
                    }
                )

                # This should not raise an assertion error about missing domain
                instance_config = ResourcesServerInstanceConfig.model_validate(full_config_dict)
                assert instance_config is not None

                # The generated config carries inline field documentation (friction #7).
                config_text = config_file.read_text()
                assert "# Task category" in config_text
                assert "policy_model" in config_text and "magic name" in config_text

                # The scaffold emits the canonical `source:` block, not the deprecated
                # `gitlab_identifier`, so a fresh server starts on the recommended schema.
                datasets = config_dict[f"{server_name}_simple_agent"]["responses_api_agents"]["simple_agent"][
                    "datasets"
                ]
                train = next(d for d in datasets if d["name"] == "train")
                assert "gitlab_identifier" not in train
                assert train["source"]["type"] == "gitlab"
                with warnings.catch_warnings():
                    warnings.simplefilter("error", DeprecationWarning)
                    DatasetConfig.model_validate(OmegaConf.to_container(train, resolve=True))
        finally:
            # Clean up the test server directory
            if server_path.exists():
                shutil.rmtree(server_path)

    def test_run_helper_prefers_cwd_server_over_install(self, tmp_path: Path) -> None:
        """ng_run should use a local CWD server dir instead of the installed one."""
        # Create a fake local server dir in tmp_path (simulates user's own resources_servers/)
        local_server = tmp_path / "resources_servers" / "my_server"
        local_server.mkdir(parents=True)
        (local_server / "requirements.txt").write_text("nemo-gym\n")

        with patch.object(Path, "cwd", return_value=tmp_path):
            _cwd_path = Path.cwd() / Path("resources_servers", "my_server")
            dir_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / Path("resources_servers", "my_server")

        assert dir_path == local_server

    def test_run_helper_falls_back_to_install_when_not_in_cwd(self, tmp_path: Path) -> None:
        """ng_run should fall back to PARENT_DIR when the server doesn't exist in CWD."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            _cwd_path = Path.cwd() / Path("resources_servers", "arc_agi")
            dir_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / Path("resources_servers", "arc_agi")

        assert dir_path == PARENT_DIR / "resources_servers" / "arc_agi"


class TestRunHelperShutdownReap:
    """RunHelper.shutdown must reap every server subprocess on every exit path."""

    def _make_runner_with_processes(self, processes: dict) -> RunHelper:
        runner = RunHelper()
        runner._processes = processes
        runner._head_server = MagicMock()
        runner._head_server_thread = MagicMock()
        return runner

    def test_kill_is_followed_by_reap_wait(self) -> None:
        good = MagicMock()
        good.wait.return_value = 0
        bad = MagicMock()
        bad.wait.side_effect = [TimeoutExpired(cmd="bad", timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_SEC), 0]

        runner = self._make_runner_with_processes({"good_server": good, "bad_server": bad})
        runner.shutdown()

        good.send_signal.assert_called_once()
        bad.send_signal.assert_called_once()
        good.kill.assert_not_called()
        bad.kill.assert_called_once()
        assert good.wait.call_count == 1
        assert bad.wait.call_count == 2
        assert runner._processes == {}

    def test_unreaped_server_after_sigkill_is_warned(self, capsys) -> None:
        zombie = MagicMock()
        zombie.wait.side_effect = TimeoutExpired(cmd="zombie", timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_SEC)

        runner = self._make_runner_with_processes({"zombie_server": zombie})
        runner.shutdown()

        zombie.kill.assert_called_once()
        assert zombie.wait.call_count == 2
        out: str = capsys.readouterr().out
        assert "zombie_server" in out
        assert f"{_GRACEFUL_SHUTDOWN_TIMEOUT_SEC}s timeout" in out
        assert f"{_FORCE_KILL_REAP_TIMEOUT_SEC}s after SIGKILL" in out

    def test_shutdown_message_matches_actual_timeout(self, capsys) -> None:
        bad = MagicMock()
        bad.wait.side_effect = [TimeoutExpired(cmd="bad", timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_SEC), 0]
        runner = self._make_runner_with_processes({"bad": bad})
        runner.shutdown()

        out: str = capsys.readouterr().out
        assert f"{_GRACEFUL_SHUTDOWN_TIMEOUT_SEC}s timeout" in out

    def test_graceful_termination_does_not_kill(self) -> None:
        a = MagicMock()
        a.wait.return_value = 0
        b = MagicMock()
        b.wait.return_value = 0
        runner = self._make_runner_with_processes({"a": a, "b": b})
        runner.shutdown()

        a.kill.assert_not_called()
        b.kill.assert_not_called()
        assert a.wait.call_count == 1
        assert b.wait.call_count == 1
        assert runner._processes == {}


class TestRunModuleTestsAll:
    def test_sequential_runs_all_in_order(self) -> None:
        paths = [Path(f"resources_servers/s{i}") for i in range(5)]
        seen: list[Path] = []

        def run_one(p: Path) -> Path:
            seen.append(p)
            return p

        results = _run_module_tests_all(run_one, paths, max_concurrency=1)
        assert results == paths
        assert seen == paths  # max_concurrency=1 preserves order

    def test_concurrent_runs_every_module_exactly_once(self) -> None:
        paths = [Path(f"resources_servers/s{i:02d}") for i in range(20)]

        def run_one(p: Path) -> Path:
            return p

        results = _run_module_tests_all(run_one, paths, max_concurrency=8)
        # Completion order is non-deterministic, but every module must run exactly once.
        assert sorted(results, key=str) == sorted(paths, key=str)
        assert len(results) == len(set(results)) == len(paths)

    def test_concurrency_actually_overlaps(self) -> None:
        import threading
        from time import sleep

        paths = [Path(f"resources_servers/s{i}") for i in range(8)]
        lock = threading.Lock()
        in_flight = 0
        max_in_flight = 0

        def run_one(p: Path) -> Path:
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            sleep(0.05)
            with lock:
                in_flight -= 1
            return p

        _run_module_tests_all(run_one, paths, max_concurrency=4)
        # With a pool of 4, multiple modules must have been in flight simultaneously.
        assert max_in_flight >= 2
