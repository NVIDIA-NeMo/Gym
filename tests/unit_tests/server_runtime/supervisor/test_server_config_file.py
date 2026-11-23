# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A config too long for an environment variable reaches a server in a file, when the server's Gym reads files."""

import errno
import os
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

import nemo_gym.server_runtime.supervisor.server_supervisor as server_supervisor
from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.global_config import (
    MAX_CONFIG_DICT_ENV_BYTES,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME,
    SKIP_VENV_IF_PRESENT_KEY_NAME,
    UV_VENV_DIR_KEY_NAME,
)
from nemo_gym.server_runtime.supervisor.server_venv import get_venv_path
from nemo_gym.utils.venv_setup import SETUP_COMPLETE_MARKER
from tests.unit_tests.config.test_global_config import TestGlobalConfig as _TestGlobalConfig


# What a server does first: read the config its parent gave it.
_COUNT_TASKS = (
    "from nemo_gym.config.global_config import maybe_get_global_config_dict; "
    "print(len(maybe_get_global_config_dict().tasks))"
)

# The config module of a Gym from before servers could read the config from a file: it reads the variable only.
_GYM_THAT_READS_THE_VARIABLE_ONLY = """\
import os

from omegaconf import OmegaConf

NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME = "NEMO_GYM_CONFIG_DICT"


def maybe_get_global_config_dict():
    config = os.getenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME)
    return OmegaConf.create(config) if config else None
"""


def _supervisor() -> server_supervisor.ServerSupervisor:
    supervisor = server_supervisor.ServerSupervisor()
    supervisor._processes = {}
    supervisor._head_server = None
    return supervisor


def _config_with_tasks(min_bytes: int) -> tuple[str, int]:
    """A config, as YAML, of a taskset's task-to-digest mapping that is longer than `min_bytes`."""
    line = "  task-{index:06d}: " + "a" * 64 + "\n"
    count = min_bytes // len(line.format(index=0)) + 1
    return "tasks:\n" + "".join(line.format(index=index) for index in range(count)), count


def _tasks_a_server_reads(env: dict[str, str], python: Path | str = sys.executable, cwd: Path | None = None) -> int:
    child = subprocess.run(
        [str(python), "-c", _COUNT_TASKS], env=os.environ | env, cwd=cwd, capture_output=True, text=True, check=True
    )
    return int(child.stdout)


def _server(
    tmp_path: Path, reused_venv_gym: str | None = None, name: str = "sample"
) -> tuple[server_supervisor._ConfiguredServer, dict]:
    """A configured server and its global config.

    Without `reused_venv_gym` the server's venv is set up in the run. With it, `skip_venv_if_present` reuses a
    complete venv whose Gym is this checkout's ("reads the file") or an earlier one ("reads the variable only").
    """
    server_dir = tmp_path / "resources_servers" / name
    server_dir.mkdir(parents=True)
    server = server_supervisor._ConfiguredServer(name, "resources_servers", name, {}, Path("app.py"), server_dir)
    config = _TestGlobalConfig._default_global_config_dict_values.fget(None) | {
        UV_VENV_DIR_KEY_NAME: str(tmp_path / "venvs"),
        SKIP_VENV_IF_PRESENT_KEY_NAME: reused_venv_gym is not None,
    }
    if reused_venv_gym is None:
        return server, config

    venv = get_venv_path(server_dir, config)
    (venv / "bin").mkdir(parents=True)
    python = f'exec "{sys.executable}" "$@"'
    if reused_venv_gym == "reads the variable only":
        gym_config = venv / "earlier_gym" / "nemo_gym" / "config"
        gym_config.mkdir(parents=True)
        (gym_config.parent / "__init__.py").touch()
        (gym_config / "__init__.py").touch()
        (gym_config / "global_config.py").write_text(_GYM_THAT_READS_THE_VARIABLE_ONLY)
        # Without `site`, so that an editable install of this checkout cannot come before the earlier Gym.
        packages = sysconfig.get_paths()["purelib"]
        python = f'PYTHONPATH="{venv / "earlier_gym"}:{packages}" exec "{sys.executable}" -S "$@"'
    (venv / "bin" / "python").write_text(f"#!/bin/sh\n{python}\n")
    (venv / "bin" / "python").chmod(0o755)
    (venv / "bin" / "activate").touch()
    (venv / SETUP_COMPLETE_MARKER).touch()
    return server, config


def test_a_short_config_stays_in_the_variable(tmp_path: Path) -> None:
    supervisor = _supervisor()
    server, global_config = _server(tmp_path)
    config = "tasks:\n  task-000000: digest\n"

    env = supervisor._config_dict_env(config, [server], global_config)

    assert env == {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: config, NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME: ""}
    assert supervisor._config_dict_file is None
    assert _tasks_a_server_reads(env) == 1


def test_a_long_config_reaches_the_server_in_a_private_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    supervisor = _supervisor()
    server, global_config = _server(tmp_path)
    config, count = _config_with_tasks(MAX_CONFIG_DICT_ENV_BYTES)

    env = supervisor._config_dict_env(config, [server], global_config)

    config_file = Path(env[NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME])
    assert env[NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME] == ""
    assert config_file.parent == tmp_path and config_file.read_text(encoding="utf-8") == config
    # The config holds API keys.
    assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
    assert _tasks_a_server_reads(env) == count

    supervisor.shutdown()
    assert not config_file.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="the limit on one environment variable is Linux's")
def test_a_config_the_operating_system_refuses_as_a_variable_starts_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    supervisor = _supervisor()
    server, global_config = _server(tmp_path)
    # Linux refuses one environment variable over 32 memory pages.
    config, count = _config_with_tasks(32 * os.sysconf("SC_PAGE_SIZE"))
    with pytest.raises(OSError) as refused:
        _tasks_a_server_reads({NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: config})
    assert refused.value.errno == errno.E2BIG

    try:
        assert _tasks_a_server_reads(supervisor._config_dict_env(config, [server], global_config)) == count
    finally:
        supervisor.shutdown()


def test_a_long_config_reaches_a_reused_venv_whose_gym_reads_files_in_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    supervisor = _supervisor()
    server, global_config = _server(tmp_path, reused_venv_gym="reads the file")
    config, count = _config_with_tasks(MAX_CONFIG_DICT_ENV_BYTES)

    env = supervisor._config_dict_env(config, [server], global_config)

    try:
        assert env[NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME] == ""
        assert Path(env[NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME]).read_text(encoding="utf-8") == config
        assert _tasks_a_server_reads(env) == count
    finally:
        supervisor.shutdown()


def test_a_long_config_stays_in_the_variable_for_a_reused_venv_whose_gym_reads_no_file(tmp_path: Path) -> None:
    supervisor = _supervisor()
    server, global_config = _server(tmp_path, reused_venv_gym="reads the variable only")
    venv = get_venv_path(server.dir_path, global_config)
    venv_files = sorted(venv.rglob("*"))
    # Over 120 KiB, and short enough for one environment variable on every Linux host.
    config, count = _config_with_tasks(MAX_CONFIG_DICT_ENV_BYTES)

    env = supervisor._config_dict_env(config, [server], global_config)

    assert env == {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: config, NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME: ""}
    assert supervisor._config_dict_file is None
    # Asking the venv's Gym changed nothing in the venv, and that Gym receives the whole config. It starts in the
    # server's directory, like a server: from this checkout `python -c` would import the checkout's Gym.
    assert sorted(venv.rglob("*")) == venv_files
    assert _tasks_a_server_reads(env, python=venv / "bin" / "python", cwd=server.dir_path) == count


def test_one_reused_venv_whose_gym_reads_no_file_keeps_the_variable_for_every_server(tmp_path: Path) -> None:
    supervisor = _supervisor()
    set_up_in_the_run, _ = _server(tmp_path, name="set_up_in_the_run")
    reused, global_config = _server(tmp_path, reused_venv_gym="reads the variable only", name="reused")
    config, _ = _config_with_tasks(MAX_CONFIG_DICT_ENV_BYTES)

    env = supervisor._config_dict_env(config, [set_up_in_the_run, reused], global_config)

    assert env == {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: config, NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME: ""}
    assert supervisor._config_dict_file is None


@pytest.mark.skipif(sys.platform != "linux", reason="the limit on one environment variable is Linux's")
def test_a_config_no_transport_can_carry_to_a_reused_venv_stops_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "config_files"))
    (tmp_path / "config_files").mkdir()
    supervisor = _supervisor()
    server, global_config = _server(tmp_path, reused_venv_gym="reads the variable only")
    config, _ = _config_with_tasks(32 * os.sysconf("SC_PAGE_SIZE"))

    with pytest.raises(ConfigError) as stopped:
        supervisor._config_dict_env(config, [server], global_config)

    message = str(stopped.value)
    assert "sample" in message and "Upgrade nemo-gym" in message and "++skip_venv_if_present=false" in message
    assert not list((tmp_path / "config_files").iterdir())


def test_a_server_inside_this_checkout_is_not_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Its launch imports this checkout's Gym before its venv's, so a reused venv with an earlier Gym gets the file."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    supervisor = _supervisor()
    server, global_config = _server(tmp_path, reused_venv_gym="reads the variable only")
    # This checkout is the folder that holds `resources_servers/`, and Gym was imported through a symlink to it.
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "imported_from").symlink_to(tmp_path)
    monkeypatch.setattr(server_supervisor, "PARENT_DIR", tmp_path / "imported_from")
    monkeypatch.setattr(server_supervisor, "run_command", MagicMock(side_effect=AssertionError("the venv was asked")))
    config, _ = _config_with_tasks(MAX_CONFIG_DICT_ENV_BYTES)

    env = supervisor._config_dict_env(config, [server], global_config)

    try:
        assert env[NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME] == ""
        assert Path(env[NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME]).read_text(encoding="utf-8") == config
    finally:
        supervisor.shutdown()


def test_a_shutdown_cut_short_while_it_waits_for_a_server_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    supervisor = _supervisor()
    server, global_config = _server(tmp_path)
    config, _ = _config_with_tasks(MAX_CONFIG_DICT_ENV_BYTES)
    config_file = Path(
        supervisor._config_dict_env(config, [server], global_config)[NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME]
    )
    # A second Ctrl-C arrives while the shutdown waits for a server to finish.
    supervisor._processes = {"sample": MagicMock(pid=123, **{"wait.side_effect": KeyboardInterrupt})}
    monkeypatch.setattr(server_supervisor, "_signal_server_group", MagicMock())

    with pytest.raises(KeyboardInterrupt):
        supervisor.shutdown()

    assert not config_file.exists()


def _start_with_a_long_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spinup: MagicMock
) -> tuple[server_supervisor.ServerSupervisor, list[dict[str, str]], DictConfig]:
    """Run `start()` as a dry run of two servers, with the launch, the head server and the readiness check replaced.

    Returns the supervisor, the environment each server was launched with, and the config.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    (tmp_path / "responses_api_agents" / "example").mkdir(parents=True)
    (tmp_path / "resources_servers" / "another").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    tasks = {f"task-{index:06d}": "a" * 64 for index in range(MAX_CONFIG_DICT_ENV_BYTES // 64)}
    config = OmegaConf.create(
        {
            "example": {"responses_api_agents": {"example": {"entrypoint": "app.py", "host": "127.0.0.1"}}},
            "another": {"resources_servers": {"another": {"entrypoint": "app.py", "host": "127.0.0.1"}}},
            "dry_run": True,
            UV_VENV_DIR_KEY_NAME: str(tmp_path),
            SKIP_VENV_IF_PRESENT_KEY_NAME: False,
            "tasks": tasks,
        }
    )
    run = MagicMock(return_value=MagicMock(pid=123))
    monkeypatch.setattr(server_supervisor, "run_command", run)
    # The launched servers are not real: the shutdown must not signal whatever process group 123 is on this host.
    monkeypatch.setattr(server_supervisor, "_signal_server_group", MagicMock())
    monkeypatch.setattr(server_supervisor, "_server_group_exists", MagicMock(return_value=False))
    monkeypatch.setattr(server_supervisor, "setup_env_command", lambda *_: "setup")
    monkeypatch.setattr(server_supervisor, "get_global_config_dict", MagicMock(return_value=config))
    head_server_client = MagicMock(**{"poll_for_status.return_value": "success"})
    monkeypatch.setattr(server_supervisor, "ServerClient", MagicMock(return_value=head_server_client))
    monkeypatch.setattr(server_supervisor.HeadServer, "run_webserver", MagicMock(return_value=(MagicMock(),) * 3))
    supervisor = server_supervisor.ServerSupervisor()
    supervisor.wait_for_dry_run_spinup = spinup
    try:
        supervisor.start(MagicMock())
    except RuntimeError:
        pass
    return supervisor, [launch.kwargs["extra_env"] for launch in run.call_args_list], config


def test_start_gives_every_server_a_long_config_in_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor, envs, config = _start_with_a_long_config(tmp_path, monkeypatch, MagicMock())
    config_file = Path(envs[0][NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME])

    assert len(envs) == 2
    for env in envs:
        assert env[NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME] == ""
        assert env[NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME] == str(config_file)
    assert OmegaConf.create(config_file.read_text(encoding="utf-8")) == config
    supervisor.shutdown()
    assert not config_file.exists()


def test_a_start_that_fails_after_writing_the_file_removes_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spinup = MagicMock(side_effect=RuntimeError("a server exited"))

    _, envs, _ = _start_with_a_long_config(tmp_path, monkeypatch, spinup)

    assert spinup.called
    assert not Path(envs[0][NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME]).exists()


def test_private_config_is_removed_on_interpreter_exit(tmp_path):
    script = """
import os, sys
from pathlib import Path
from nemo_gym.server_runtime.supervisor.server_supervisor import ServerSupervisor
supervisor = ServerSupervisor()
config = 'value: ' + 'x' * 140000
__import__('tempfile').tempdir = sys.argv[1]
env = supervisor._config_dict_env(config, [], {})
print(env['NEMO_GYM_CONFIG_DICT_FILE'], flush=True)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, check=True)
    path = Path(result.stdout.strip())
    assert path.parent == tmp_path
    assert not path.exists()


@pytest.mark.parametrize("during_start", [True, False])
def test_library_sigterm_removes_config_and_keeps_failure_exit(tmp_path, during_start):
    script = """
import os, signal, sys
from pathlib import Path
from nemo_gym.server_runtime.supervisor.server_supervisor import ServerSupervisor
class Supervisor(ServerSupervisor):
    def _start(self, config):
        self._config_dict_file = Path(sys.argv[1])
        self._config_dict_file.write_text('private')
        if sys.argv[2] == 'True':
            os.kill(os.getpid(), signal.SIGTERM)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
Supervisor().start(None)
os.kill(os.getpid(), signal.SIGTERM)
"""
    path = tmp_path / "config.yaml"
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(during_start)], capture_output=True, text=True
    )
    assert result.returncode == 143, result.stderr
    assert not path.exists()


def test_library_start_keeps_custom_sigterm_policy(monkeypatch):
    import signal

    supervisor = _supervisor()
    policy = lambda *_: None
    monkeypatch.setattr(signal, "getsignal", lambda _: policy)
    install = MagicMock()
    monkeypatch.setattr(signal, "signal", install)
    monkeypatch.setattr(supervisor, "_start", lambda _: None)
    supervisor.start(None)
    install.assert_not_called()


def test_library_start_on_worker_thread_does_not_install_signal_handler(monkeypatch):
    import signal
    from concurrent.futures import ThreadPoolExecutor

    supervisor = _supervisor()
    install = MagicMock()
    monkeypatch.setattr(signal, "signal", install)
    monkeypatch.setattr(supervisor, "_start", lambda _: None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(supervisor.start, None).result()
    install.assert_not_called()
