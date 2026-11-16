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
import importlib.metadata
import json
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from pytest import MonkeyPatch, raises

import nemo_gym.server_runtime.supervisor.server_venv
import nemo_gym.utils.venv_setup
from nemo_gym import PARENT_DIR
from nemo_gym.config.global_config import UV_VENV_DIR_KEY_NAME
from nemo_gym.server_runtime.supervisor.server_venv import (
    _NEMO_GYM_REQUIREMENT_ERE,
    _NEMO_GYM_REQUIREMENT_RE,
    _get_nemo_gym_install_flags,
    _get_nemo_gym_version_spec,
    _is_gym_checkout,
    _local_nemo_gym_extras,
    _python_request_admits,
    get_venv_path,
    run_command,
    setup_env_command,
)
from nemo_gym.utils.venv_setup import SETUP_COMPLETE_MARKER, setup_environment
from tests.unit_tests.config.test_global_config import TestGlobalConfig as _TestGlobalConfig


# The filter for a server outside the install root: only its Gym requirement lines are replaced.
_OUTSIDE_FILTER = shlex.quote(_NEMO_GYM_REQUIREMENT_ERE)


class TestCLISetupCommandSetupEnvCommand:
    def _setup(self, tmp_path: Path) -> tuple[Path, dict]:
        server = tmp_path / "resources_servers" / "sample"
        server.mkdir(parents=True)
        (server / "pyproject.toml").write_text('[project]\nname="sample"\nversion="0"\n')
        (server / "uv.lock").write_text("version=1\nrevision=3\n")
        # Gym's own locked server: the root is a Gym checkout.
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "nemo-gym"\n')
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

    def test_missing_install_manifest_fails_before_install(self, tmp_path: Path) -> None:
        server, config = self._setup(tmp_path)
        (server / "pyproject.toml").unlink()
        with raises(RuntimeError, match="Missing pyproject.toml or requirements.txt"):
            setup_env_command(server, config, "sample")

    @pytest.mark.parametrize("manifest", ["requirements.txt", "pyproject.toml"])
    def test_custom_component_keeps_legacy_setup(self, tmp_path: Path, manifest: str) -> None:
        server, config = self._setup(tmp_path)
        (server / "uv.lock").unlink()
        if manifest == "requirements.txt":
            (server / "pyproject.toml").unlink()
            (server / manifest).write_text("example-dependency==1\n")
        command = setup_env_command(server, config, "sample")
        assert "uv pip install" in command
        assert "openai==test openai version" in command
        assert "--overrides" in command and "ray==test ray version" in command
        assert "_locked_env.py" not in command

    def test_verbosity_is_forwarded(self, tmp_path: Path) -> None:
        server, config = self._setup(tmp_path)
        command = setup_env_command(server, config | {"pip_install_verbose": True}, "sample")
        assert "--verbose" in command

    def test_rejects_empty_server_python_version(self, tmp_path: Path) -> None:
        server_dir, config = self._setup(tmp_path)
        (server_dir / ".python-version").write_text("\n# no request here\n")

        with raises(RuntimeError, match="Empty Python version"):
            setup_env_command(
                dir_path=server_dir,
                global_config_dict=config,
                prefix="osworld agent",
            )

    @pytest.mark.parametrize(
        "requested, expected",
        [
            # The parent's patch release satisfies the range, so uv must not pick another 3.13.x (Ray rejects that).
            (">=3.13.14,<3.14", "3.13.14"),
            ("3.13", "3.13.14"),
            ("3.13.14", "3.13.14"),
            ("cpython@3.13", "3.13.14"),
            # A request that excludes the parent is the component's to make.
            (">=3.12,<3.13", ">=3.12,<3.13"),
            ("3.12", "3.12"),
            ("pypy@3.10", "pypy@3.10"),
            # A free-threaded build is a different interpreter, even at the parent's version.
            ("3.13+freethreaded", "3.13+freethreaded"),
            # uv reads the first request and skips comments.
            ("# pinned for the model server\n>=3.13.14,<3.14\n3.12", "3.13.14"),
        ],
    )
    def test_server_python_version_keeps_the_parent_patch_release_when_admitted(
        self, tmp_path: Path, requested: str, expected: str
    ) -> None:
        server_dir, config = self._setup(tmp_path)
        (server_dir / ".python-version").write_text(f"{requested}\n")

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=config | {"python_version": "3.13.14"},
            prefix="my server name",
        )

        assert (
            f"--python {shlex.quote(expected)} "
            in shlex.split(actual_command)[shlex.split(actual_command).index("--command") + 1]
        )

    @pytest.mark.parametrize(
        "request_, version, admitted",
        [
            (">=3.13.14,<3.14", "3.13.15", True),
            (">=3.13.14,<3.14", "3.13.13", False),
            ("3.13", "3.13.2", True),
            ("3.13", "3.12.9", False),
            ("cpython@3.13", "3.13.14", True),
            ("cpython3.13", "3.13.14", True),
            ("cpython@>=3.13.14,<3.14", "3.13.14", True),
            ("pypy@3.10", "3.10.14", False),
            ("3.13+freethreaded", "3.13.14", False),
            ("3.13.14+debug", "3.13.14", False),
            ("3.13.14rc1", "3.13.14", False),
            (">=3.13", "test python version", False),
        ],
    )
    def test_python_request_admits(self, request_: str, version: str, admitted: bool) -> None:
        assert _python_request_admits(request_, version) is admitted


@pytest.fixture
def setup_component(tmp_path: Path) -> tuple[Path, dict, Path]:
    server_dir = tmp_path / "source with spaces" / "models" / "example"
    server_dir.mkdir(parents=True)
    # Gym's own locked server: the root is a Gym checkout.
    (server_dir.parent.parent / "pyproject.toml").write_text('[project]\nname = "nemo-gym"\n')
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
        nemo_gym.utils.venv_setup.__file__,
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


@pytest.mark.parametrize("manifest", ["missing", "conflicting"])
def test_complete_venv_activates_without_setup_or_manifest_validation(setup_component, manifest) -> None:
    server_dir, config, venv = setup_component
    if manifest == "missing":
        (server_dir / "pyproject.toml").unlink()
    else:
        (server_dir / "pyproject.toml").touch()
    marker = venv / SETUP_COMPLETE_MARKER
    marker.touch()
    command = setup_env_command(server_dir, config, "server") + ' && test "$SETUP_TEST_ACTIVATED" = 1 && echo STARTED'
    with _setup_process(command) as process:
        output, _ = process.communicate(timeout=10)
        assert process.returncode == 0, output
        assert "STARTED" in output
    assert marker.exists()
    assert not venv.with_name(f"{venv.name}.setup.lock").exists()


@pytest.mark.parametrize("missing_file", [None, "python", "activate", SETUP_COMPLETE_MARKER])
def test_generated_setup_installs_and_activates_before_starting_server(setup_component, missing_file) -> None:
    server_dir, config, venv = setup_component
    if missing_file in ("python", "activate"):
        (venv / "bin" / missing_file).unlink()
    elif missing_file is None:
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
        shlex.quote(str(Path(nemo_gym.server_runtime.supervisor.server_venv.__file__).with_name("_locked_env.py"))),
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
        monkeypatch.setattr(nemo_gym.server_runtime.supervisor.server_venv, "Popen", Popen_mock)

        get_global_config_dict_mock = MagicMock(return_value={"uv_cache_dir": "default uv cache dir"})
        monkeypatch.setattr(
            nemo_gym.server_runtime.supervisor.server_venv, "get_global_config_dict", get_global_config_dict_mock
        )

        monkeypatch.setattr(nemo_gym.server_runtime.supervisor.server_venv, "environ", dict())

        monkeypatch.setattr(nemo_gym.server_runtime.supervisor.server_venv, "stdout", "stdout")
        monkeypatch.setattr(nemo_gym.server_runtime.supervisor.server_venv, "stderr", "stderr")

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
            start_new_session=False,
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_custom_pythonpath(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)
        monkeypatch.setattr(
            nemo_gym.server_runtime.supervisor.server_venv, "environ", {"PYTHONPATH": "existing pythonpath"}
        )

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
            start_new_session=False,
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_scheduler_component_root_gate_accepts_pinned_tree(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        Popen_mock, _ = self._setup(monkeypatch)
        allowed_root = tmp_path / "paired-gym"
        server_dir = allowed_root / "responses_api_agents" / "agent"
        server_dir.mkdir(parents=True)
        monkeypatch.setattr(
            nemo_gym.server_runtime.supervisor.server_venv,
            "environ",
            {"NEMO_GYM_ALLOWED_COMPONENT_ROOTS": str(allowed_root)},
        )

        run_command(command="my command", working_dir_path=server_dir, server_name="agent")

        assert Popen_mock.call_args.kwargs["env"]["NEMO_GYM_ALLOWED_COMPONENT_ROOTS"] == str(allowed_root)

    def test_scheduler_component_root_gate_rejects_stale_tree(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        Popen_mock, _ = self._setup(monkeypatch)
        allowed_root = tmp_path / "paired-gym"
        stale_server_dir = tmp_path / "image-gym" / "responses_api_agents" / "agent"
        allowed_root.mkdir()
        stale_server_dir.mkdir(parents=True)
        monkeypatch.setattr(
            nemo_gym.server_runtime.supervisor.server_venv,
            "environ",
            {"NEMO_GYM_ALLOWED_COMPONENT_ROOTS": str(allowed_root)},
        )

        with pytest.raises(RuntimeError, match="potentially stale server"):
            run_command(
                command="my command",
                working_dir_path=stale_server_dir,
                server_name="agent",
            )

        Popen_mock.assert_not_called()

    def test_project_root_added_to_pythonpath(self, monkeypatch: MonkeyPatch) -> None:
        # Opt-in: callers that need `resources_servers.<name>`-style imports (e.g. gym env test) pass
        # the project root, which is appended after the server dir.
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
            project_root=Path("/root"),
            extra_env={"NEMO_GYM_CONFIG_DICT": "api_key: synthetic-test-secret"},
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={
                "NEMO_GYM_EXTRA_ROOTS": "/root",
                "NEMO_GYM_CONFIG_DICT": "api_key: synthetic-test-secret",
                "PYTHONPATH": "/root/resources_servers/my_server:/root",
                "UV_CACHE_DIR": "default uv cache dir",
            },
            stdout="stdout",
            stderr="stderr",
            start_new_session=False,
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_project_root_precedes_existing_extra_roots(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)
        monkeypatch.setattr(
            nemo_gym.server_runtime.supervisor.server_venv,
            "environ",
            {"NEMO_GYM_EXTRA_ROOTS": "/plugin-a:/plugin-b"},
        )

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
            project_root=Path("/root"),
        )

        assert Popen_mock.call_args.kwargs["env"]["NEMO_GYM_EXTRA_ROOTS"] == "/root:/plugin-a:/plugin-b"

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
            start_new_session=False,
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
    def test_tee_logs_with_server_name(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {
            "uv_cache_dir": "default uv cache dir",
            "nemo_gym_log_dir": str(tmp_path / "gym_logs"),
        }

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
            server_name="my_resources/my_server",
        )

        expected_args = call(
            f"set -o pipefail; (my command) 2>&1 | tee -a {tmp_path}/gym_logs/my_resources_my_server.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_tee_logs_falls_back_to_dir_name(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {
            "uv_cache_dir": "default uv cache dir",
            "nemo_gym_log_dir": str(tmp_path / "gym_logs"),
        }

        run_command(
            command="my command",
            working_dir_path=Path("/root/resources_servers/my_server"),
        )

        expected_args = call(
            f"set -o pipefail; (my command) 2>&1 | tee -a {tmp_path}/gym_logs/my_server.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args


class TestLegacySetupContract:
    def _installation_command(self, command: str) -> str:
        # Keep the dependency-command assertions independent of the shell quoting
        # used to pass that command as one argument to the setup runner.
        args = shlex.split(command)
        return f"cd {args[1]} && {args[args.index('--command') + 1]}"

    def _setup_server_dir(self, tmp_path: Path) -> Path:
        server_dir = tmp_path / "first_level" / "second_level"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "nemo-gym"\n')

        return server_dir.absolute()

    def _debug_global_config_dict(self, tmp_path: Path) -> dict:
        return _TestGlobalConfig._default_global_config_dict_values.fget(None) | {UV_VENV_DIR_KEY_NAME: str(tmp_path)}

    def test_sanity(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path),
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -c <(printf '%s\\n' 'ray==test ray version') -r requirements.txt 'openai==test openai version' > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_requirements_uses_server_local_overrides(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "overrides.txt").write_text("dependency==2\n")

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path),
            prefix="my server name",
        )

        assert (
            "uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -c <(printf '%s\\n' 'ray==test ray version') --override overrides.txt -r requirements.txt"
            in self._installation_command(actual_command)
        )

    @pytest.mark.parametrize("has_marker", [False, True])
    def test_existing_venv_skips_setup_only_when_complete(self, tmp_path: Path, capsys, has_marker: bool) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        config = self._debug_global_config_dict(tmp_path) | {"skip_venv_if_present": True}
        before = setup_env_command(server_dir, config, "policy")

        # A cancelled install leaves bin/python and bin/activate behind without the marker.
        (server_dir / ".venv/bin").mkdir(parents=True)
        (server_dir / ".venv/bin/python").touch()
        (server_dir / ".venv/bin/activate").touch()
        if has_marker:
            (server_dir / ".venv" / SETUP_COMPLETE_MARKER).touch()

        after = setup_env_command(server_dir, config, "policy")
        if has_marker:
            assert after == f"cd {server_dir} && source {server_dir}/.venv/bin/activate"
        else:
            assert after == before
            err = capsys.readouterr().err
            assert f"{server_dir}/.venv has no {SETUP_COMPLETE_MARKER}" in err
            assert f"touch {server_dir}/.venv/{SETUP_COMPLETE_MARKER}" in err
        assert "--skip-if-ready" in shlex.split(before)
        assert "uv pip install" in self._installation_command(before)

    def test_head_server_deps(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"head_server_deps": ["dep 1", "dep 2"]},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -c <(printf '%s\\n' 'ray==test ray version') -r requirements.txt 'dep 1' 'dep 2' > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_python_version(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"python_version": "my python version"},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'my python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -c <(printf '%s\\n' 'ray==test ray version') -r requirements.txt 'openai==test openai version' > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_uv_pip_set_python(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"uv_pip_set_python": True},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') --python {server_dir}/.venv/bin/python -c <(printf '%s\\n' 'ray==test ray version') -r requirements.txt 'openai==test openai version' > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_pip_install_verbose(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"pip_install_verbose": True},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -v -c <(printf '%s\\n' 'ray==test ray version') -r requirements.txt 'openai==test openai version' > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_pyproject_requirements_raises_error(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "pyproject.toml").write_text("")

        with raises(RuntimeError, match="Found both pyproject.toml and requirements.txt"):
            setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )

    def test_missing_pyproject_requirements_raises_error(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "requirements.txt").unlink()

        with raises(RuntimeError, match="Missing pyproject.toml or requirements.txt"):
            setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )

    def test_pyproject(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "pyproject.toml").write_text("")
        (server_dir / "requirements.txt").unlink()

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path),
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -c <(printf '%s\\n' 'ray==test ray version') '-e .' 'openai==test openai version' > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_uv_venv_dir_with_install(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        uv_venv_dir = tmp_path / "uv_venv_dir"

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"uv_venv_dir": str(uv_venv_dir)},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {uv_venv_dir}/first_level/second_level/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {uv_venv_dir}/first_level/second_level/.venv/bin/activate && uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -c <(printf '%s\\n' 'ray==test ray version') -r requirements.txt 'openai==test openai version' > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_uv_venv_dir_path_is_shared_with_cleanup(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        uv_venv_dir = tmp_path / "uv_venv_dir"

        actual_path = get_venv_path(
            server_dir,
            self._debug_global_config_dict(tmp_path) | {"uv_venv_dir": str(uv_venv_dir)},
        )

        assert actual_path == uv_venv_dir / "first_level/second_level/.venv"

    @pytest.mark.parametrize("version", ["0.3.0", "0.3.0rc0", "1.0.0", "2.1.3rc1"])
    def test_installs_from_pypi_when_not_editable(
        self, tmp_path: Path, version: str, monkeypatch: MonkeyPatch
    ) -> None:
        server_dir = (tmp_path / "first_level" / "second_level").absolute()
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        with patch("importlib.metadata.version", return_value=version):
            actual_command = setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && (echo 'nemo-gym=={version}' && {{ grep -v -i -E {_OUTSIDE_FILTER} requirements.txt || [ $? -eq 1 ]; }}) | uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -c <(printf '%s\\n' 'ray==test ray version') -r /dev/stdin 'openai==test openai version' > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    @pytest.mark.parametrize("version", ["0.3.0", "0.3.0rc0", "1.0.0", "2.1.3rc1"])
    def test_installs_from_pypi_when_not_editable_pyproject(
        self, tmp_path: Path, version: str, monkeypatch: MonkeyPatch
    ) -> None:
        server_dir = (tmp_path / "first_level" / "second_level").absolute()
        server_dir.mkdir(parents=True)
        (server_dir / "pyproject.toml").write_text("")
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        with patch("importlib.metadata.version", return_value=version):
            actual_command = setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python 'test python version' {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -c <(printf '%s\\n' 'ray==test ray version') 'nemo-gym=={version}' && uv pip install --overrides <(printf '%s\\n' 'ray==test ray version') -c <(printf '%s\\n' 'ray==test ray version') --no-sources-package nemo-gym --overrides <(printf '%s\\n' nemo-gym=={version}) '-e .' 'openai==test openai version' > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == self._installation_command(actual_command)

    def test_installs_from_pypi_keeps_requested_extras(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        server_dir = (tmp_path / "first_level" / "second_level").absolute()
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("-e nemo-gym[dev,sandbox] @ ../../\npytest\n")
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)
        for name in ("NEMO_GYM_ALLOW_PRERELEASE", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_INDEX_STRATEGY"):
            monkeypatch.delenv(name, raising=False)

        with patch("importlib.metadata.version", return_value="0.3.0"):
            actual_command = setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )

        assert (
            f"(echo 'nemo-gym[dev,sandbox]==0.3.0' && {{ grep -v -i -E {_OUTSIDE_FILTER} requirements.txt || [ $? -eq 1 ]; }})"
            in (self._installation_command(actual_command))
        )


class TestLocalNemoGymExtras:
    @pytest.mark.parametrize(
        ("requirements", "expected"),
        [
            ("-e nemo-gym[dev] @ ../../\n", "[dev]"),
            ("-e nemo-gym[dev,sandbox] @ ../..\n", "[dev,sandbox]"),
            ("-e nemo-gym[ dev , sandbox ] @ ../../  # local checkout\n", "[dev,sandbox]"),
            ("nemo_gym[sandbox] @ ../../\n", "[sandbox]"),
            ("-e nemo-gym @ ../../\n", ""),
            ("-e ../../\n", ""),
            ("-e ../../benchmarks/automationbench\n", ""),
            ("pytest\n", ""),
            ("-e nemo-gym[dev] @ ../../\n-e nemo-gym[sandbox,dev] @ ../../\n", "[dev,sandbox]"),
        ],
    )
    def test_reads_extras_from_the_local_gym_requirement(
        self, tmp_path: Path, requirements: str, expected: str
    ) -> None:
        requirements_fpath = tmp_path / "requirements.txt"
        requirements_fpath.write_text(requirements)

        assert _local_nemo_gym_extras(requirements_fpath) == expected

    def test_recognizes_every_shipped_local_gym_requirement(self) -> None:
        unrecognized = [
            f"{path.relative_to(PARENT_DIR)}: {line.strip()}"
            for path in PARENT_DIR.glob("*/*/requirements.txt")
            for line in path.read_text().splitlines()
            if "nemo-gym" in line and "../.." in line and not _NEMO_GYM_REQUIREMENT_RE.match(line)
        ]

        assert not unrecognized, "PyPI installs would drop the extras on these lines:\n" + "\n".join(unrecognized)


class TestGetNemoGymInstallFlags:
    """Test _get_nemo_gym_install_flags helper function."""

    def test_no_env_vars_returns_empty(self, monkeypatch: MonkeyPatch) -> None:
        """When no env vars are set, should return empty string."""
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == ""

    def test_prerelease_flag(self, monkeypatch: MonkeyPatch) -> None:
        """When NEMO_GYM_ALLOW_PRERELEASE=true, should add --pre and --index-strategy."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == "--pre --index-strategy unsafe-best-match 'fastapi<1.0' "

    def test_prerelease_false(self, monkeypatch: MonkeyPatch) -> None:
        """When NEMO_GYM_ALLOW_PRERELEASE=false, should not add flags."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "false")
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == ""

    def test_index_url(self, monkeypatch: MonkeyPatch) -> None:
        """Should include UV_INDEX_URL if set."""
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.setenv("UV_INDEX_URL", "https://test.pypi.org/simple/")
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == "--index-url https://test.pypi.org/simple/ "

    def test_extra_index_url(self, monkeypatch: MonkeyPatch) -> None:
        """Should include UV_EXTRA_INDEX_URL if set."""
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://pypi.org/simple/")
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == "--extra-index-url https://pypi.org/simple/ "

    def test_explicit_index_strategy(self, monkeypatch: MonkeyPatch) -> None:
        """Explicit UV_INDEX_STRATEGY should override auto-set from prerelease."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
        monkeypatch.setenv("UV_INDEX_STRATEGY", "first-match")

        flags = _get_nemo_gym_install_flags()
        # Should have --pre but use explicit strategy, not auto-set unsafe-best-match
        assert flags == "--pre 'fastapi<1.0' --index-strategy first-match "

    def test_all_flags_combined(self, monkeypatch: MonkeyPatch) -> None:
        """Test all flags together."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
        monkeypatch.setenv("UV_INDEX_URL", "https://test.pypi.org/simple/")
        monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://pypi.org/simple/")
        monkeypatch.setenv("UV_INDEX_STRATEGY", "unsafe-best-match")

        flags = _get_nemo_gym_install_flags()
        assert (
            flags
            == "--pre 'fastapi<1.0' --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match "
        )


class TestGetNemoGymVersionSpec:
    """Test _get_nemo_gym_version_spec helper function."""

    def test_non_editable_detects_version(self) -> None:
        """For non-editable installs, should detect and pin to parent version."""
        with patch("importlib.metadata.version", return_value="0.2.1rc0"):
            version_spec = _get_nemo_gym_version_spec()
            assert version_spec == "==0.2.1rc0"

    def test_non_editable_stable_version(self) -> None:
        """Should work with stable versions too."""
        with patch("importlib.metadata.version", return_value="0.2.0"):
            version_spec = _get_nemo_gym_version_spec()
            assert version_spec == "==0.2.0"

    def test_package_not_found_returns_empty(self) -> None:
        """If nemo-gym is not installed, should return empty string gracefully."""
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            version_spec = _get_nemo_gym_version_spec()
            assert version_spec == ""


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
            "set -o pipefail; (my command) 2>&1 | tee -i -a /tmp/gym_logs/my_resources_my_server.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
            start_new_session=False,
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
            "set -o pipefail; (my command) 2>&1 | tee -i -a /tmp/gym_logs/my_server.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/root/resources_servers/my_server", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
            start_new_session=False,
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args


class TestIsGymCheckout:
    """NVIDIA-NeMo/Gym#4725: an external repository with its own pyproject.toml is not a Gym checkout."""

    @pytest.mark.parametrize(
        ("pyproject", "expected"),
        [
            ('[project]\nname = "nemo-gym"\n', True),
            ('[project]\nname = "NeMo_Gym"\n', True),
            ('[project]\nname = "auto-ontology-eval"\n', False),
            ("", False),
            ("not = [valid toml", False),
            ('project = "nemo-gym"\n', False),
            (b'[project]\nname = "\xff"\n', False),
            (None, False),
        ],
    )
    def test_only_the_nemo_gym_project_counts(self, tmp_path: Path, pyproject, expected) -> None:
        if isinstance(pyproject, bytes):
            (tmp_path / "pyproject.toml").write_bytes(pyproject)
        elif pyproject is not None:
            (tmp_path / "pyproject.toml").write_text(pyproject)
        assert _is_gym_checkout(tmp_path) is expected

    @staticmethod
    def _install_command(server_dir: Path, tmp_path: Path) -> str:
        args = shlex.split(
            setup_env_command(
                dir_path=server_dir.absolute(),
                global_config_dict=_TestGlobalConfig._default_global_config_dict_values.fget(None)
                | {UV_VENV_DIR_KEY_NAME: str(tmp_path)},
                prefix="my server name",
            )
        )
        return args[args.index("--command") + 1]

    def test_an_external_repository_installs_gym_as_a_package(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        server_dir = tmp_path / "resources_servers" / "my_server"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "auto-ontology-eval"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)

        def install_command() -> str:
            return self._install_command(server_dir, tmp_path)

        assert f"grep -v -i -E {_OUTSIDE_FILTER} requirements.txt" in install_command()

        (tmp_path / "pyproject.toml").write_text('[project]\nname = "nemo-gym"\n')
        assert "grep -v" not in install_command()

    def test_an_external_repository_next_to_an_editable_gym_installs_that_checkout(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """`gym env init` in an external repository writes `-e nemo-gym[dev] @ <path to the Gym checkout>`. The server
        venv gets the parent's editable checkout with those extras, and keeps the server's other local requirements."""
        server_dir = tmp_path / "resources_servers" / "my_server"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text(
            "-e nemo-gym[dev] @ ../../../Gym\n-e ../../shared_lib\nnemo-gym==0.7.0\npytest\n"
        )
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "auto-ontology-eval"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", True)

        command = self._install_command(server_dir, tmp_path)
        assert "nemo-gym==" not in command
        requirements_pipe = command[command.index("(echo") : command.index(") | uv pip install") + 1]
        piped = subprocess.run(
            ["bash", "-c", requirements_pipe], cwd=server_dir, capture_output=True, text=True, check=False
        ).stdout.splitlines()
        assert piped == [f"-e {PARENT_DIR}[dev]", "-e ../../shared_lib", "pytest"]

        (server_dir / "requirements.txt").unlink()
        (server_dir / "pyproject.toml").write_text('[project]\nname = "my-server"\n')
        command = self._install_command(server_dir, tmp_path)
        assert f"{shlex.quote(f'-e {PARENT_DIR}')} && " in command
        # The only Gym pin left is the override with the checkout's own version, which keeps that install.
        version = importlib.metadata.version("nemo-gym")
        assert f"--no-sources-package nemo-gym --overrides <(printf '%s\\n' nemo-gym=={version}) '-e .'" in command
        assert command.count("nemo-gym==") == 1

    @pytest.mark.parametrize("editable", [True, False])
    def test_an_external_server_with_a_lock_still_gets_the_parents_gym(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, editable: bool
    ) -> None:
        """Its uv.lock pins its own Gym source; the locked installer would install that instead of the parent's Gym."""
        server_dir = tmp_path / "resources_servers" / "my_server"
        server_dir.mkdir(parents=True)
        (server_dir / "pyproject.toml").write_text('[project]\nname = "my-server"\n')
        (server_dir / "uv.lock").write_text('version = 1\n[[package]]\nname = "nemo-gym"\nversion = "0.1.0"\n')
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "auto-ontology-eval"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", editable)

        with patch("importlib.metadata.version", return_value="0.9.0"):
            command = self._install_command(server_dir, tmp_path)
        assert "_locked_env.py" not in command
        assert (f"{shlex.quote(f'-e {PARENT_DIR}')} && " in command) is editable
        assert ("'nemo-gym==0.9.0' && " in command) is not editable
        assert "--no-sources-package nemo-gym --overrides <(printf '%s\\n' nemo-gym==0.9.0) '-e .'" in command

    @pytest.mark.parametrize("editable", [True, False])
    def test_an_external_server_that_is_not_a_package_installs_its_requirements_with_the_parents_gym(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, editable: bool
    ) -> None:
        server_dir = tmp_path / "resources_servers" / "my_server"
        server_dir.mkdir(parents=True)
        (server_dir / "pyproject.toml").write_text(
            '[project]\nname = "my-server"\n[tool.uv]\npackage = false\n[tool.uv.sources]\nhelper = { path = "../helper" }\n'
        )
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "auto-ontology-eval"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", editable)

        with patch("importlib.metadata.version", return_value="0.9.0"):
            command = self._install_command(server_dir, tmp_path)
        # Only Gym's own source is replaced, so the server's helper source still applies.
        assert (
            "--no-sources-package nemo-gym --overrides <(printf '%s\\n' nemo-gym==0.9.0) --requirements pyproject.toml"
            in command
        )
        assert "--no-sources " not in command
        assert (f"{shlex.quote(f'-e {PARENT_DIR}')} && " in command) is editable
        assert ("'nemo-gym==0.9.0' && " in command) is not editable

    @staticmethod
    def _piped_requirements(command: str, server_dir: Path) -> list[str]:
        pipe = command[command.index("(echo") : command.index(") | uv pip install") + 1]
        return subprocess.run(["bash", "-c", pipe], cwd=server_dir, capture_output=True, text=True).stdout.splitlines()

    def test_an_external_repository_keeps_its_own_project_and_drops_its_gym_pin(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """The #4725 repository installs its own project with `-e ../..` and pins `nemo-gym==0.7.0` as a workaround.
        With Gym installed from the package index, the server venv gets the parent's version and the project."""
        server_dir = tmp_path / "resources_servers" / "auto_ontology_sql"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("-e ../..\nnemo-gym==0.7.0\npytest\n")
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "auto-ontology-eval"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)
        for name in ("NEMO_GYM_ALLOW_PRERELEASE", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_INDEX_STRATEGY"):
            monkeypatch.delenv(name, raising=False)

        with patch("importlib.metadata.version", return_value="0.8.0"):
            command = self._install_command(server_dir, tmp_path)
        assert self._piped_requirements(command, server_dir) == ["nemo-gym==0.8.0", "-e ../..", "pytest"]

    def test_a_built_in_server_in_a_wheel_install_still_drops_every_parent_path(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Under the install root, `../..` is site-packages and points at files a wheel does not ship."""
        site_packages = tmp_path / "site-packages"
        server_dir = site_packages / "resources_servers" / "example"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text(
            "-e nemo-gym @ ../../\n-e ../../benchmarks/automationbench\npytest\n"
        )
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)
        monkeypatch.setattr(nemo_gym.server_runtime.supervisor.server_venv, "PARENT_DIR", site_packages)

        with patch("importlib.metadata.version", return_value="0.8.0"):
            command = self._install_command(server_dir, tmp_path)
        assert "grep -v -F '../..' requirements.txt" in command
        assert self._piped_requirements(command, server_dir)[1:] == ["pytest"]

    def test_an_external_repository_nested_in_an_editable_gym_keeps_its_own_project(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Inside an editable Gym checkout, a nested external repository's `../..` is itself, not site-packages."""
        gym = tmp_path / "Gym"
        repo = gym / "examples" / "external_repo"
        server_dir = repo / "resources_servers" / "example"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("-e ../..\nnemo-gym==0.7.0\npytest\n")
        (repo / "pyproject.toml").write_text('[project]\nname = "external-repo"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", True)
        monkeypatch.setattr(nemo_gym.server_runtime.supervisor.server_venv, "PARENT_DIR", gym)

        command = self._install_command(server_dir, tmp_path)
        assert self._piped_requirements(command, server_dir) == [f"-e {gym}", "-e ../..", "pytest"]

    def test_equivalent_spellings_of_the_gym_package_are_replaced(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        server_dir = tmp_path / "resources_servers" / "my_server"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text(
            "NeMo-Gym==0.7.0\nnemo.gym[dev]==0.7.0\nnemo__gym\nnemo-gym-foo==1.0\npytest\n"
        )
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "auto-ontology-eval"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)
        for name in ("NEMO_GYM_ALLOW_PRERELEASE", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_INDEX_STRATEGY"):
            monkeypatch.delenv(name, raising=False)

        with patch("importlib.metadata.version", return_value="0.8.0"):
            command = self._install_command(server_dir, tmp_path)
        assert self._piped_requirements(command, server_dir) == ["nemo-gym[dev]==0.8.0", "nemo-gym-foo==1.0", "pytest"]

    def test_a_built_in_pyproject_server_in_a_wheel_install_still_ignores_every_source(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        site_packages = tmp_path / "site-packages"
        server_dir = site_packages / "resources_servers" / "example"
        server_dir.mkdir(parents=True)
        (server_dir / "pyproject.toml").write_text('[project]\nname = "example"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)
        monkeypatch.setattr(nemo_gym.server_runtime.supervisor.server_venv, "PARENT_DIR", site_packages)

        with patch("importlib.metadata.version", return_value="0.8.0"):
            command = self._install_command(server_dir, tmp_path)
        assert "--no-sources --overrides <(printf '%s\\n' nemo-gym==0.8.0) '-e .'" in command

    @pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
    def test_an_external_pyproject_server_keeps_its_other_local_sources(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Only Gym's source is dropped: a local helper declared in `[tool.uv.sources]` still resolves (offline)."""
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "pyproject.toml").write_text('[project]\nname = "my-helper"\nversion = "0.1.0"\n')
        server_dir = tmp_path / "resources_servers" / "srv"
        server_dir.mkdir(parents=True)
        (server_dir / "pyproject.toml").write_text(
            '[project]\nname = "srv"\nversion = "0.1.0"\ndependencies = ["my-helper"]\n'
            '[tool.uv.sources]\nmy-helper = { path = "../../lib" }\n'
        )
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "auto-ontology-eval"\nversion = "0.1.0"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)

        with patch("importlib.metadata.version", return_value="0.8.0"):
            command = self._install_command(server_dir, tmp_path)
        assert "--no-sources-package nemo-gym --overrides <(printf '%s\\n' nemo-gym==0.8.0) '-e .'" in command
        resolved = subprocess.run(
            ["uv", "pip", "compile", "--offline", "-q", "--no-sources-package", "nemo-gym", "pyproject.toml"],
            cwd=server_dir,
            capture_output=True,
            text=True,
        )
        assert resolved.returncode == 0, resolved.stderr
        assert "../../lib" in resolved.stdout

    @pytest.mark.parametrize(
        ("requirements", "site_packages"),
        [("nemo-gym==0.7.0\n", False), ("-e nemo-gym @ ../../\n", True)],
        ids=["external repository", "built-in server in a wheel install"],
    )
    def test_requirements_of_only_gym_lines_install_under_pipefail(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, requirements: str, site_packages: bool
    ) -> None:
        """grep selects no line and exits 1; with `pipefail` (a `nemo_gym_log_dir` run) that must not fail setup."""
        root = tmp_path / "site-packages" if site_packages else tmp_path
        server_dir = root / "resources_servers" / "my_server"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text(requirements)
        if site_packages:
            monkeypatch.setattr(nemo_gym.server_runtime.supervisor.server_venv, "PARENT_DIR", root)
        else:
            (tmp_path / "pyproject.toml").write_text('[project]\nname = "auto-ontology-eval"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", False)
        for name in ("NEMO_GYM_ALLOW_PRERELEASE", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_INDEX_STRATEGY"):
            monkeypatch.delenv(name, raising=False)

        with patch("importlib.metadata.version", return_value="0.8.0"):
            command = self._install_command(server_dir, tmp_path)
        pipe = command[command.index("(echo") : command.index(") | uv pip install") + 1]

        def run() -> subprocess.CompletedProcess:
            script = f"set -o pipefail; {pipe} | cat"
            return subprocess.run(["bash", "-c", script], cwd=server_dir, capture_output=True, text=True)

        piped = run()
        assert piped.returncode == 0, piped.stderr
        assert piped.stdout.splitlines() == ["nemo-gym==0.8.0"]
        # A real grep error, such as a missing file, still fails the install.
        (server_dir / "requirements.txt").unlink()
        assert run().returncode != 0

    @pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
    def test_a_pinned_gym_dependency_with_extras_keeps_the_parents_checkout(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """A server pinning `nemo-gym[sandbox]==0.7.0` keeps the parent's editable Gym and still gets the extra's
        dependencies. uv plans the server install offline against a venv holding a stand-in editable Gym, as the first
        install leaves it; without the override, uv would replace that Gym with 0.7.0 from the package index."""
        dependency = tmp_path / "sandbox_dep"
        dependency.mkdir()
        (dependency / "pyproject.toml").write_text('[project]\nname = "sandbox-dep"\nversion = "1.0.0"\n')
        gym = tmp_path / "Gym"
        gym.mkdir()
        repo = tmp_path / "repo"
        server_dir = repo / "resources_servers" / "srv"
        server_dir.mkdir(parents=True)
        (server_dir / "pyproject.toml").write_text(
            '[project]\nname = "srv"\nversion = "0.1.0"\ndependencies = ["nemo-gym[sandbox]==0.7.0"]\n'
            "[tool.uv]\npackage = false\n"
        )
        (repo / "pyproject.toml").write_text('[project]\nname = "auto-ontology-eval"\nversion = "0.1.0"\n')
        monkeypatch.setattr(nemo_gym, "_is_editable_install", True)
        monkeypatch.setattr(nemo_gym.server_runtime.supervisor.server_venv, "PARENT_DIR", gym)

        with patch("importlib.metadata.version", return_value="0.9.0"):
            command = self._install_command(server_dir, tmp_path)
        assert f"{shlex.quote(f'-e {gym}[sandbox]')} && " in command
        [override] = re.findall(r"--no-sources-package nemo-gym --overrides <\(printf '%s\\n' '([^']+)'\)", command)
        assert override == "nemo-gym[sandbox]==0.9.0"

        # The venv after the first install: the parent's Gym, editable, with the extra declared but not yet installed.
        venv = tmp_path / "venv"
        subprocess.run(["uv", "venv", "--offline", "-q", "--python", sys.executable, str(venv)], check=True)
        [site_packages] = venv.glob("lib/python*/site-packages")
        dist_info = site_packages / "nemo_gym-0.9.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: nemo-gym\nVersion: 0.9.0\nProvides-Extra: sandbox\n"
            f"Requires-Dist: sandbox-dep @ {dependency.as_uri()} ; extra == 'sandbox'\n"
        )
        (dist_info / "INSTALLER").write_text("uv\n")
        (dist_info / "RECORD").write_text("")
        (dist_info / "direct_url.json").write_text(json.dumps({"url": gym.as_uri(), "dir_info": {"editable": True}}))
        (server_dir / "overrides.in").write_text(override + "\n")
        # Stands in for the package index: uv takes a URL only from a direct requirement or a constraint.
        (server_dir / "index.in").write_text(f"sandbox-dep @ {dependency.as_uri()}\n")

        def plan(*overrides: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["uv", "pip", "install", "--dry-run", "--offline", "--python", str(venv / "bin" / "python")]
                + [
                    "--no-sources-package",
                    "nemo-gym",
                    *overrides,
                    "-c",
                    "index.in",
                    "--requirements",
                    "pyproject.toml",
                ],
                cwd=server_dir,
                capture_output=True,
                text=True,
            )

        planned = plan("--overrides", "overrides.in")
        assert planned.returncode == 0, planned.stderr
        assert "+ sandbox-dep @ " in planned.stderr
        assert "nemo-gym" not in planned.stderr  # the parent's Gym stays as it is
        # Without it the pin wins: uv replaces the parent's Gym with 0.7.0, or fails offline with no 0.7.0 cached.
        replaced = plan()
        assert "+ nemo-gym==0.7.0" in replaced.stderr if replaced.returncode == 0 else "0.7.0" in replaced.stderr
