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
import select
import shlex
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from pytest import MonkeyPatch, raises

import nemo_gym.cli._venv_setup
import nemo_gym.cli.setup_command
from nemo_gym.cli._venv_setup import SETUP_COMPLETE_MARKER, setup_environment
from nemo_gym.cli.setup_command import (
    get_venv_path,
    run_command,
    setup_env_command,
)
from nemo_gym.global_config import UV_VENV_DIR_KEY_NAME
from tests.unit_tests.test_global_config import TestGlobalConfig as _TestGlobalConfig


class TestCLISetupCommandSetupEnvCommand:
    def _setup(self, tmp_path: Path) -> tuple[Path, dict]:
        server = tmp_path / "resources_servers" / "sample"
        server.mkdir(parents=True)
        (server / "pyproject.toml").write_text('[project]\nname="sample"\nversion="0"\n')
        (server / "uv.lock").write_text("version=1\nrevision=3\n")
        config = _TestGlobalConfig._default_global_config_dict_values.fget(None) | {
            UV_VENV_DIR_KEY_NAME: str(tmp_path / "custom venvs")
        }
        return server, config

    def test_locked_setup_targets_custom_venv_and_preserves_parent_contracts(self, tmp_path: Path) -> None:
        server, config = self._setup(tmp_path)
        command = setup_env_command(server, config, "sample")
        args = shlex.split(command)
        installer = shlex.split(args[args.index("--command") + 1].split(" > >(")[0])
        assert installer[1].endswith("/_locked_env.py")
        assert installer[installer.index("--project") + 1] == str(server)
        assert installer[installer.index("--venv") + 1] == str(get_venv_path(server, config))
        assert installer[installer.index("--python") + 1] == config["python_version"]
        assert installer[installer.index("--requirement") + 1] == "openai==test openai version"
        assert installer[installer.index("--constraint") + 1] == "ray==test ray version"
        assert "uv pip install" not in command
        assert command.endswith(f"source {shlex.quote(str(get_venv_path(server, config) / 'bin/activate'))}")

    def test_component_extras_are_scoped_and_forwarded(self, tmp_path: Path) -> None:
        server, config = self._setup(tmp_path)
        config["component_extras"] = {
            "resources_servers/sample": ["lexmount"],
            "responses_api_agents/other": ["example-tools"],
        }
        command = setup_env_command(server, config, "sample")
        args = shlex.split(command)
        installer = shlex.split(args[args.index("--command") + 1].split(" > >(")[0])
        assert installer[installer.index("--extra") + 1] == "lexmount"
        assert "example-tools" not in installer

    @pytest.mark.parametrize("extras", ["lexmount", [1], [""]])
    def test_invalid_component_extras_fail_before_install(self, tmp_path: Path, extras: object) -> None:
        server, config = self._setup(tmp_path)
        config["component_extras"] = {"resources_servers/sample": extras}
        with raises(RuntimeError, match="must be a list"):
            setup_env_command(server, config, "sample")

    @pytest.mark.parametrize("missing", ["pyproject.toml", "uv.lock"])
    def test_missing_metadata_fails_before_install(self, tmp_path: Path, missing: str) -> None:
        server, config = self._setup(tmp_path)
        (server / missing).unlink()
        with raises(RuntimeError, match=f"Missing {missing}"):
            setup_env_command(server, config, "sample")

    def test_verbosity_is_forwarded(self, tmp_path: Path) -> None:
        server, config = self._setup(tmp_path)
        command = setup_env_command(server, config | {"pip_install_verbose": True}, "sample")
        assert "--verbose" in command


@pytest.fixture
def setup_component(tmp_path: Path) -> tuple[Path, dict, Path]:
    server_dir = tmp_path / "source with spaces" / "models" / "example"
    server_dir.mkdir(parents=True)
    (server_dir.parent.parent / "pyproject.toml").touch()
    (server_dir / "pyproject.toml").touch()
    (server_dir / "uv.lock").touch()
    config = _TestGlobalConfig._default_global_config_dict_values.fget(None) | {
        "uv_venv_dir": str(tmp_path / "venvs with spaces"),
        "skip_venv_if_present": True,
    }
    venv = get_venv_path(server_dir, config)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").touch()
    (venv / "bin/activate").write_text("export SETUP_TEST_ACTIVATED=1\n")
    return server_dir, config, venv


def _install_command(venv: Path) -> str:
    return f"cd {shlex.quote(str(venv))} && touch bin/python bin/activate && echo install >> attempts"


def _runner_command(venv: Path, command: str) -> list[str]:
    return [
        sys.executable,
        nemo_gym.cli._venv_setup.__file__,
        "--venv",
        str(venv),
        "--command",
        command,
        "--skip-if-ready",
    ]


@contextmanager
def _setup_process(command: str | list[str]):
    process = subprocess.Popen(
        ["/bin/bash", "-c", command] if isinstance(command, str) else command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        yield process
    finally:
        # Include surviving installers when a test kills their wrapper.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=10)


def _expect_output(process: subprocess.Popen, text: str) -> None:
    assert select.select([process.stdout], [], [], 10)[0], f"process did not report {text!r}"
    assert text in process.stdout.readline()


def _expect_success(process: subprocess.Popen) -> None:
    output, _ = process.communicate(timeout=10)
    assert process.returncode == 0, output


@pytest.mark.parametrize("interrupt", [False, True])
def test_shared_venv_waits_for_installer_even_if_wrapper_dies(setup_component, interrupt: bool) -> None:
    _, _, venv = setup_component
    marker = venv / SETUP_COMPLETE_MARKER
    command = _install_command(venv)
    # bin/python and bin/activate already exist, as they do during uv pip install.
    with _setup_process(_runner_command(venv, f"echo INSTALLING; read -r release; {command}")) as first:
        _expect_output(first, "INSTALLING")
        assert not marker.exists()
        if interrupt:
            first.kill()
            assert first.wait(timeout=10) == -signal.SIGKILL
        with _setup_process(_runner_command(venv, command)) as second:
            _expect_output(second, "Waiting for virtual environment setup:")
            assert second.poll() is None
            assert not marker.exists()
            first.stdin.write("continue\n")
            first.stdin.flush()
            _expect_success(second)
        if not interrupt:
            _expect_success(first)
    assert (venv / "attempts").read_text() == "install\n" * (2 if interrupt else 1)
    assert marker.is_file()


@pytest.mark.parametrize("command,code", [("exit 7", 7), ("kill -TERM $$", 128 + signal.SIGTERM)])
def test_failed_forced_setup_invalidates_marker_and_can_retry(setup_component, command: str, code: int) -> None:
    _, _, venv = setup_component
    marker = venv / SETUP_COMPLETE_MARKER
    marker.touch()
    assert setup_environment(venv, command, skip_if_ready=False) == code
    assert not marker.exists()
    assert setup_environment(venv, _install_command(venv), skip_if_ready=True) == 0
    assert marker.is_file()


@pytest.mark.parametrize("has_marker", [False, True])
@pytest.mark.parametrize("manifest", ["missing", "conflicting"])
def test_existing_venv_activates_without_setup_or_manifest_validation(setup_component, has_marker, manifest) -> None:
    server_dir, config, venv = setup_component
    if manifest == "missing":
        (server_dir / "pyproject.toml").unlink()
    else:
        (server_dir / "pyproject.toml").touch()
    marker = venv / SETUP_COMPLETE_MARKER
    if has_marker:
        marker.touch()
    command = setup_env_command(server_dir, config, "server") + ' && test "$SETUP_TEST_ACTIVATED" = 1 && echo STARTED'
    with _setup_process(command) as process:
        output, _ = process.communicate(timeout=10)
        assert process.returncode == 0, output
        assert "STARTED" in output
    assert marker.exists() == has_marker
    assert not venv.with_name(f"{venv.name}.setup.lock").exists()


@pytest.mark.parametrize("missing_file", [None, "python", "activate"])
def test_generated_setup_installs_and_activates_before_starting_server(setup_component, missing_file) -> None:
    server_dir, config, venv = setup_component
    if missing_file:
        (venv / "bin" / missing_file).unlink()
    else:
        config["skip_venv_if_present"] = False
    # Substitute only the sync helper; keep the real serialization and activation wrapper.
    (venv / "bin").mkdir(exist_ok=True)
    fake = server_dir / "fake_sync.py"
    fake.write_text(
        "from pathlib import Path\n"
        f"venv = Path({str(venv)!r})\n"
        "(venv / 'bin/python').touch()\n"
        "(venv / 'bin/activate').write_text('export SETUP_TEST_ACTIVATED=1\\n')\n"
        "(venv / 'attempts').write_text('install\\n')\n"
    )
    directory_command, _, rest = setup_env_command(server_dir, config, "server").partition(" && ")
    runner, _, activation = rest.rpartition(" && ")
    args = shlex.split(runner)
    install_index = args.index("--command") + 1
    args[install_index] = args[install_index].replace(
        shlex.quote(str(Path(nemo_gym.cli.setup_command.__file__).with_name("_locked_env.py"))),
        shlex.quote(str(fake)),
    )
    command = f"{directory_command} && {shlex.join(args)} && {activation}"
    command += f' && test "$SETUP_TEST_ACTIVATED" = 1 && test -f {shlex.quote(str(venv / "attempts"))} && echo STARTED'
    with _setup_process(command) as process:
        _expect_success(process)
    assert (venv / "attempts").read_text() == "install\n"
    assert (venv / SETUP_COMPLETE_MARKER).is_file()


class TestCLISetupCommandRunCommand:
    def _setup(self, monkeypatch: MonkeyPatch) -> tuple[MagicMock, MagicMock]:
        Popen_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.cli.setup_command, "Popen", Popen_mock)

        get_global_config_dict_mock = MagicMock(return_value={"uv_cache_dir": "default uv cache dir"})
        monkeypatch.setattr(nemo_gym.cli.setup_command, "get_global_config_dict", get_global_config_dict_mock)

        monkeypatch.setattr(nemo_gym.cli.setup_command, "environ", dict())

        monkeypatch.setattr(nemo_gym.cli.setup_command, "stdout", "stdout")
        monkeypatch.setattr(nemo_gym.cli.setup_command, "stderr", "stderr")

        return Popen_mock, get_global_config_dict_mock

    def test_sanity(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            # Default (no project_root): only the server dir is on PYTHONPATH.
            env={"PYTHONPATH": "/my path", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_custom_pythonpath(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)
        monkeypatch.setattr(nemo_gym.cli.setup_command, "environ", {"PYTHONPATH": "existing pythonpath"})

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/my path:existing pythonpath", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_project_root_added_to_pythonpath(self, monkeypatch: MonkeyPatch) -> None:
        # Opt-in: callers that need `resources_servers.<name>`-style imports (e.g. gym env test) pass
        # the project root, which is appended after the server dir.
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
            project_root=Path("/root"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server:/root", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_custom_uv_cache_dir(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {"uv_cache_dir": "my uv cache dir"}

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/my path", "UV_CACHE_DIR": "my uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_supplied_config_and_streams_avoid_global_config(self, monkeypatch: MonkeyPatch) -> None:
        popen, get_global_config = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
            global_config_dict={"uv_cache_dir": "isolated cache"},
            stdout_target="isolated stdout",
            stderr_target="isolated stderr",
        )

        get_global_config.assert_not_called()
        assert popen.call_args.kwargs["env"]["UV_CACHE_DIR"] == "isolated cache"
        assert popen.call_args.kwargs["stdout"] == "isolated stdout"
        assert popen.call_args.kwargs["stderr"] == "isolated stderr"

    def test_uv_lock_timeout_is_propagated_to_server_processes(self, monkeypatch: MonkeyPatch) -> None:
        popen, _ = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
            global_config_dict={"uv_cache_dir": "shared cache", "uv_lock_timeout_seconds": 1800},
        )

        assert popen.call_args.kwargs["env"]["UV_LOCK_TIMEOUT"] == "1800"

    def test_uv_lock_timeout_absent_when_unconfigured(self, monkeypatch: MonkeyPatch) -> None:
        popen, _ = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
            global_config_dict={"uv_cache_dir": "shared cache"},
        )

        assert "UV_LOCK_TIMEOUT" not in popen.call_args.kwargs["env"]

    def test_extra_env_passed_to_process_environment(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, _ = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
            extra_env={"SECRET_KEY": "supersecret", "NEMO_GYM_CONFIG_PATH": "my_path"},
        )

        env = Popen_mock.call_args.kwargs["env"]
        assert env["SECRET_KEY"] == "supersecret"
        assert env["NEMO_GYM_CONFIG_PATH"] == "my_path"
        assert env["PYTHONPATH"] == "/my path"


class TestCLISetupCommandRunCommandTeeLog(TestCLISetupCommandRunCommand):
    def test_tee_logs_with_server_name(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {
            "uv_cache_dir": "default uv cache dir",
            "nemo_gym_log_dir": "/tmp/gym_logs",
        }

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
            server_name="my_resources/my_server",
        )

        expected_args = call(
            "set -o pipefail; (my command) 2>&1 | tee -a /tmp/gym_logs/my_resources_my_server.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_tee_logs_falls_back_to_dir_name(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {
            "uv_cache_dir": "default uv cache dir",
            "nemo_gym_log_dir": "/tmp/gym_logs",
        }

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
        )

        expected_args = call(
            "set -o pipefail; (my command) 2>&1 | tee -a /tmp/gym_logs/my_server.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args
