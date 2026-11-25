# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import stat
from pathlib import Path

import pytest

from nemo_gym.core.run_token import (
    RUN_TOKEN_DIR_ENV,
    RUN_TOKEN_ENV,
    is_run_token,
    mint_run_token,
    read_run_token_file,
    remove_run_token_file,
    run_token,
    run_token_file,
    write_run_token_file,
)


@pytest.fixture(autouse=True)
def _no_run_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(RUN_TOKEN_ENV, raising=False)
    monkeypatch.delenv(RUN_TOKEN_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_a_process_without_a_token_mints_one_that_its_children_inherit() -> None:
    assert run_token() is None
    assert mint_run_token() is True
    token = run_token()
    assert token is not None and len(token) >= 40
    assert os.environ[RUN_TOKEN_ENV] == token
    # A second launcher in the same process keeps it.
    assert mint_run_token() is False
    assert run_token() == token


def test_a_token_set_beforehand_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_TOKEN_ENV, "shared-by-the-user")
    assert mint_run_token() is False
    assert run_token() == "shared-by-the-user"


def test_an_empty_variable_counts_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_TOKEN_ENV, "")
    assert run_token() is None
    assert mint_run_token() is True and run_token()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("the-token", True), ("the-token ", False), ("other", False), ("", False), (None, False), ("thé-token", False)],
)
def test_only_the_runs_own_token_matches(monkeypatch: pytest.MonkeyPatch, value, expected: bool) -> None:
    monkeypatch.setenv(RUN_TOKEN_ENV, "the-token")
    assert is_run_token(value) is expected


def test_nothing_matches_outside_a_run() -> None:
    assert not is_run_token("anything")
    assert not is_run_token(None)


def test_the_file_is_readable_by_its_user_only_and_named_after_the_head_servers_address(tmp_path: Path) -> None:
    path = write_run_token_file("127.0.0.1", 11000, "the-token")

    assert path == tmp_path / "cache" / "nemo-gym" / "run-tokens" / "head-127.0.0.1-11000"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert read_run_token_file("127.0.0.1", 11000) == "the-token"
    # Another address, or another port, is another run.
    assert read_run_token_file("10.0.0.5", 11000) is None
    assert read_run_token_file("127.0.0.1", 11001) is None
    # A host that no file name can hold is spelled with underscores.
    assert run_token_file("[::1]", 11000).name == "head-___1_-11000"


def test_a_new_run_on_the_same_address_replaces_the_file() -> None:
    write_run_token_file("127.0.0.1", 11000, "first")
    write_run_token_file("127.0.0.1", 11000, "second")

    assert read_run_token_file("127.0.0.1", 11000) == "second"
    # The run that wrote "first" ends after "second" started: it leaves the newer run's file.
    remove_run_token_file("127.0.0.1", 11000, "first")
    assert read_run_token_file("127.0.0.1", 11000) == "second"
    remove_run_token_file("127.0.0.1", 11000, "second")
    assert not run_token_file("127.0.0.1", 11000).exists()
    # Removing a file that is gone is fine.
    remove_run_token_file("127.0.0.1", 11000, "second")


def test_an_empty_or_unreadable_file_holds_no_token() -> None:
    path = run_token_file("127.0.0.1", 11000)
    path.parent.mkdir(parents=True)
    path.write_text("  \n")
    assert read_run_token_file("127.0.0.1", 11000) is None
    path.write_bytes(b"\xff\xfe")
    assert read_run_token_file("127.0.0.1", 11000) is None


def test_a_directory_of_ones_own_holds_the_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(RUN_TOKEN_DIR_ENV, str(tmp_path / "elsewhere"))

    path = write_run_token_file("127.0.0.1", 11000, "the-token")

    assert path == tmp_path / "elsewhere" / "head-127.0.0.1-11000"
    assert read_run_token_file("127.0.0.1", 11000) == "the-token"


def test_the_cache_directory_defaults_to_the_users_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert run_token_file("127.0.0.1", 11000) == (
        tmp_path / "home" / ".cache" / "nemo-gym" / "run-tokens" / "head-127.0.0.1-11000"
    )
