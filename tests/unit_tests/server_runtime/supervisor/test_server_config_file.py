# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A config too long for an environment variable reaches a server in a file."""

import errno
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import nemo_gym.server_runtime.supervisor.server_supervisor as server_supervisor
from nemo_gym.config.global_config import (
    MAX_CONFIG_DICT_ENV_BYTES,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME,
)


# What a server does first: read the config its parent gave it.
_COUNT_TASKS = (
    "from nemo_gym.config.global_config import maybe_get_global_config_dict; "
    "print(len(maybe_get_global_config_dict().tasks))"
)


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


def _tasks_a_server_reads(env: dict[str, str]) -> int:
    child = subprocess.run(
        [sys.executable, "-c", _COUNT_TASKS], env=os.environ | env, capture_output=True, text=True, check=True
    )
    return int(child.stdout)


def test_a_short_config_stays_in_the_variable() -> None:
    supervisor = _supervisor()
    config = "tasks:\n  task-000000: digest\n"

    env = supervisor._config_dict_env(config)

    assert env == {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: config, NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME: ""}
    assert supervisor._config_dict_file is None
    assert _tasks_a_server_reads(env) == 1


def test_a_long_config_reaches_the_server_in_a_private_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    supervisor = _supervisor()
    config, count = _config_with_tasks(MAX_CONFIG_DICT_ENV_BYTES)

    env = supervisor._config_dict_env(config)

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
    # Linux refuses one environment variable over 32 memory pages.
    config, count = _config_with_tasks(32 * os.sysconf("SC_PAGE_SIZE"))
    with pytest.raises(OSError) as refused:
        _tasks_a_server_reads({NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: config})
    assert refused.value.errno == errno.E2BIG

    try:
        assert _tasks_a_server_reads(supervisor._config_dict_env(config)) == count
    finally:
        supervisor.shutdown()
