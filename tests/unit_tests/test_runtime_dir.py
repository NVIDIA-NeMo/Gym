# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from nemo_gym import runtime_dir


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """A directory under /tmp: AF_UNIX paths are limited to about 100 bytes, and pytest's can be longer."""
    path = tempfile.mkdtemp(prefix="ngrt-", dir="/tmp")
    yield Path(path)
    shutil.rmtree(path, ignore_errors=True)


def dead_pid() -> int:
    """The PID of a process that has exited."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_a_dead_process_directory_with_closed_sockets_is_removed(short_dir: Path) -> None:
    stale = short_dir / f"ng-{dead_pid()}-0123abcd"
    stale.mkdir()
    # A socket file whose server is gone: bound, then closed without unlinking.
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(stale / "policy.sock"))
    (stale / "notes").write_text("x")

    assert runtime_dir.remove_stale_runtime_dirs(str(short_dir)) == [str(stale)]
    assert not stale.exists()


def test_a_live_process_directory_is_kept(tmp_path: Path) -> None:
    live = tmp_path / f"ng-{os.getpid()}-0123abcd"
    live.mkdir()

    assert runtime_dir.remove_stale_runtime_dirs(str(tmp_path)) == []
    assert live.exists()


def test_a_directory_whose_socket_accepts_is_kept_even_if_its_process_looks_dead(short_dir: Path) -> None:
    # A server in another PID namespace sharing /tmp: its PID means nothing here, but its socket answers.
    other = short_dir / f"ng-{dead_pid()}-0123abcd"
    other.mkdir()
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(other / "policy.sock"))
        server.listen()
        removed = runtime_dir.remove_stale_runtime_dirs(str(short_dir))

    assert removed == []
    assert other.exists()


def test_only_runtime_directory_names_are_considered(tmp_path: Path) -> None:
    pid = dead_pid()
    for name in (f"ng-{pid}", f"ng-{pid}-notahex!", "ng-abcdefgh", f"other-{pid}-0123abcd"):
        (tmp_path / name).mkdir()
    (tmp_path / f"ng-{pid}-89abcdef").write_text("a file, not a directory")

    assert runtime_dir.remove_stale_runtime_dirs(str(tmp_path)) == []
    assert len(list(tmp_path.iterdir())) == 5


def test_the_runtime_directory_is_created_once_per_process_and_removed_at_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_dir, "RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(runtime_dir, "_runtime_dir", None)
    stale = tmp_path / f"ng-{dead_pid()}-0123abcd"
    stale.mkdir()
    exits = []
    monkeypatch.setattr(runtime_dir.atexit, "register", lambda *call: exits.append(call))

    path = runtime_dir.server_runtime_dir()

    assert runtime_dir.server_runtime_dir() == path
    assert Path(path).parent == tmp_path and Path(path).name.startswith(f"ng-{os.getpid()}-")
    assert (Path(path).stat().st_mode & 0o777) == 0o700
    # Creating it removed the directory a killed server left behind.
    assert not stale.exists()
    [(remove, *args)] = exits
    remove(*args)
    assert not Path(path).exists()
