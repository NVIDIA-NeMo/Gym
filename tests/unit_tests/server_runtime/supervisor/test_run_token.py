# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The launcher gives a run its token, and refuses a run that requires it when a server's Gym cannot send it."""

import os
from pathlib import Path

import pytest
from omegaconf import DictConfig

import nemo_gym.server_runtime.supervisor.server_supervisor as server_supervisor
from nemo_gym.config.config_types import ConfigError
from nemo_gym.core.run_token import (
    RUN_TOKEN_DIR_ENV,
    RUN_TOKEN_ENV,
    read_run_token_file,
    run_token_file,
    write_run_token_file,
)
from tests.unit_tests.server_runtime.supervisor.test_server_config_file import _server


@pytest.fixture(autouse=True)
def _token_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(RUN_TOKEN_DIR_ENV, str(tmp_path / "run-tokens"))
    monkeypatch.delenv(RUN_TOKEN_ENV, raising=False)


def test_the_launcher_gives_the_run_a_token_and_takes_it_back_at_shutdown() -> None:
    # The head server binds every address, and a client of this host reaches it on loopback.
    run = server_supervisor._RunToken.start(DictConfig({"host": "0.0.0.0", "port": 11000}))

    assert run.minted and os.environ[RUN_TOKEN_ENV] == run.token
    assert read_run_token_file("127.0.0.1", 11000) == run.token
    run.end()
    assert RUN_TOKEN_ENV not in os.environ
    assert not run_token_file("127.0.0.1", 11000).exists()


@pytest.mark.parametrize("head_server_config", [None, DictConfig({}), DictConfig({"host": "127.0.0.1"})])
def test_a_config_without_a_head_server_address_gets_a_token_and_no_file(
    head_server_config: DictConfig | None, tmp_path: Path
) -> None:
    run = server_supervisor._RunToken.start(head_server_config)

    assert run.minted and os.environ[RUN_TOKEN_ENV] == run.token and run.head_server_address is None
    assert not (tmp_path / "run-tokens").exists()
    run.end()
    assert RUN_TOKEN_ENV not in os.environ


def test_a_token_set_for_the_run_is_used_and_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_TOKEN_ENV, "shared")

    run = server_supervisor._RunToken.start(DictConfig({"host": "127.0.0.1", "port": 11000}))

    assert (run.minted, run.token) == (False, "shared")
    assert read_run_token_file("127.0.0.1", 11000) == "shared"
    run.end()
    # Whoever set it decides when it goes; the file goes with the run.
    assert os.environ[RUN_TOKEN_ENV] == "shared"
    assert read_run_token_file("127.0.0.1", 11000) is None


def test_a_run_that_ends_after_another_took_its_address_leaves_the_others_file() -> None:
    run = server_supervisor._RunToken.start(DictConfig({"host": "127.0.0.1", "port": 11000}))
    write_run_token_file("127.0.0.1", 11000, "the-next-run")

    run.end()

    assert read_run_token_file("127.0.0.1", 11000) == "the-next-run"


def test_a_token_file_that_cannot_be_written_does_not_stop_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unwritable(*args: object) -> None:
        raise PermissionError("read-only home")

    monkeypatch.setattr(server_supervisor, "write_run_token_file", unwritable)

    run = server_supervisor._RunToken.start(DictConfig({"host": "127.0.0.1", "port": 11000}))

    assert run.minted and os.environ[RUN_TOKEN_ENV] == run.token and run.head_server_address is None
    assert "Could not write the run's token file (read-only home)" in capsys.readouterr().out
    run.end()
    assert RUN_TOKEN_ENV not in os.environ


def test_a_reused_venv_whose_gym_sends_no_token_stops_a_run_that_requires_it(tmp_path: Path) -> None:
    server, global_config = _server(tmp_path, reused_venv_gym="reads the variable only")

    with pytest.raises(ConfigError) as stopped:
        server_supervisor._refuse_servers_that_cannot_send_the_run_token([server], global_config)

    message = str(stopped.value)
    assert message.startswith("require_run_token is set, but the reused venv of these servers holds a Gym")
    assert "sample" in message and "Upgrade nemo-gym" in message and "++skip_venv_if_present=false" in message


def test_venvs_set_up_in_the_run_or_holding_this_gym_send_the_token(tmp_path: Path) -> None:
    set_up_in_the_run, _ = _server(tmp_path, name="set_up_in_the_run")
    reused, global_config = _server(tmp_path, reused_venv_gym="this checkout", name="reused")

    server_supervisor._refuse_servers_that_cannot_send_the_run_token([set_up_in_the_run, reused], global_config)
