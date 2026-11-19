# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.core import worker_resources
from nemo_gym.core.worker_resources import (
    PROCESS_GROUPS_DIR_NAME,
    ProcessGroupRecord,
    recorded_process_group,
    recorded_process_groups,
    set_worker_routing_id,
    worker_mkdtemp,
    worker_routing_id,
    worker_workspace_dir,
    workspace_root,
)
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.local.provider import LocalProvider


ROUTING_ID = "b" * 32


@pytest.fixture
def worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This process as worker ``ROUTING_ID`` with its workspace root in ``tmp_path``."""
    monkeypatch.setenv(worker_resources.WORKSPACE_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(worker_resources, "_routing_id", None)
    set_worker_routing_id(ROUTING_ID)
    return tmp_path


def test_routing_id_is_generated_once_per_process_and_renewed_after_a_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_resources, "_routing_id", None)
    first = worker_routing_id()
    assert first == worker_routing_id()

    parent_pid = os.getpid()
    monkeypatch.setattr(worker_resources.os, "getpid", lambda: parent_pid + 1)
    assert worker_routing_id() != first


def test_session_routing_sets_the_routing_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_resources, "_routing_id", None)
    set_worker_routing_id(ROUTING_ID)
    assert worker_routing_id() == ROUTING_ID


@pytest.mark.parametrize("routing_id", ["", "a/b", "..", "."])
def test_a_routing_id_must_be_one_path_component(routing_id: str) -> None:
    with pytest.raises(ValueError, match="path component"):
        set_worker_routing_id(routing_id)


def test_workspace_root_defaults_to_a_per_user_temporary_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(worker_resources.WORKSPACE_ROOT_ENV_VAR, raising=False)
    root = workspace_root()
    assert root.parent == Path(worker_resources.tempfile.gettempdir())
    assert root.name == f"nemo-gym-workspaces-{os.getuid()}"


def test_workspaces_are_created_under_the_root_and_routing_id(worker: Path) -> None:
    directory = worker_workspace_dir()
    workspace = Path(worker_mkdtemp(prefix="task-"))

    assert directory == worker / ROUTING_ID
    assert directory.stat().st_mode & 0o777 == 0o700
    assert workspace.parent == directory
    assert workspace.name.startswith("task-")


def test_the_default_root_is_created_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(worker_resources.WORKSPACE_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(worker_resources.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(worker_resources, "_routing_id", None)
    set_worker_routing_id(ROUTING_ID)

    directory = worker_workspace_dir()

    assert directory.parent.stat().st_mode & 0o777 == 0o700
    assert directory.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("planted", ["symlink", "writable by others"])
def test_a_default_root_someone_else_could_control_is_refused(
    planted: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(worker_resources.WORKSPACE_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(worker_resources.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    root = workspace_root()
    if planted == "symlink":
        (tmp_path / "elsewhere").mkdir(mode=0o700)
        root.symlink_to(tmp_path / "elsewhere")
    else:
        root.mkdir()
        root.chmod(0o777)

    with pytest.raises(PermissionError, match="owned by this user and writable by nobody else"):
        worker_workspace_dir()
    assert not (tmp_path / "elsewhere" / worker_routing_id()).exists()


def test_a_configured_root_may_be_shared_but_the_worker_directory_is_private(worker: Path) -> None:
    worker.chmod(0o1777)

    assert worker_workspace_dir().stat().st_mode & 0o777 == 0o700


def test_a_planted_worker_directory_is_refused(worker: Path) -> None:
    (worker / "elsewhere").mkdir(mode=0o700)
    (worker / ROUTING_ID).symlink_to(worker / "elsewhere")

    with pytest.raises(PermissionError):
        worker_workspace_dir()


def test_a_clean_exit_removes_the_worker_directory_only_once_it_is_empty(worker: Path) -> None:
    directory = worker_workspace_dir()
    (directory / PROCESS_GROUPS_DIR_NAME).mkdir()
    kept = Path(worker_mkdtemp())

    worker_resources._remove_if_empty(directory)
    assert kept.exists() and not (directory / PROCESS_GROUPS_DIR_NAME).exists()

    kept.rmdir()
    worker_resources._remove_if_empty(directory)
    assert not directory.exists()


def test_replacing_a_routing_id_already_used_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(worker_resources, "_routing_id", None)
    set_worker_routing_id("c" * 32)
    set_worker_routing_id(ROUTING_ID)
    assert "already used" not in caplog.text

    used = worker_routing_id()
    set_worker_routing_id("d" * 32)
    assert f"replaces {used}, which this worker already used" in caplog.text


def test_a_child_process_group_is_recorded_while_it_runs(worker: Path) -> None:
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        with recorded_process_group(child.pid) as record:
            assert record is not None and record.pgid == child.pid
            assert recorded_process_groups(ROUTING_ID) == [record]
            assert record.is_running()
            child.kill()
            child.wait()
            assert not record.is_running()
    finally:
        child.kill()
        child.wait()
    assert recorded_process_groups(ROUTING_ID) == []


def test_a_group_whose_leader_exited_still_runs_while_a_member_does(worker: Path) -> None:
    # The usual way a command leaves a background server: the shell exits, its child keeps the group.
    shell = subprocess.Popen(["sh", "-c", "sleep 30 & echo $!"], start_new_session=True, stdout=subprocess.PIPE)
    try:
        with recorded_process_group(shell.pid) as record:
            assert record is not None
            shell.stdout.readline()  # the member started before the shell exits
            shell.wait()
            assert record.is_running()
            os.killpg(shell.pid, signal.SIGKILL)
            for _ in range(500):
                if not record.is_running():
                    break
                time.sleep(0.01)
            assert not record.is_running()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(shell.pid, signal.SIGKILL)
        shell.wait()
        shell.stdout.close()


def test_a_record_whose_process_id_was_reused_is_not_running(worker: Path) -> None:
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        with recorded_process_group(child.pid) as record:
            assert record is not None
            # Another process with the same ID started at a different time: the record must not match it.
            assert not ProcessGroupRecord(pgid=child.pid, start_ticks=record.start_ticks - 1).is_running()
    finally:
        child.kill()
        child.wait()


def test_a_child_that_does_not_lead_its_group_is_not_running(worker: Path) -> None:
    child = subprocess.Popen(["sleep", "30"])
    try:
        with recorded_process_group(child.pid) as record:
            assert record is not None
            assert not record.is_running()
    finally:
        child.kill()
        child.wait()


def test_two_records_of_one_process_id_are_removed_independently(worker: Path) -> None:
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        with recorded_process_group(child.pid) as outer:
            with recorded_process_group(child.pid):
                assert len(recorded_process_groups(ROUTING_ID)) == 2
            assert recorded_process_groups(ROUTING_ID) == [outer]
    finally:
        child.kill()
        child.wait()


def test_a_child_already_reaped_runs_unrecorded(worker: Path) -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    child.wait()
    with recorded_process_group(child.pid) as record:
        assert record is None
        assert recorded_process_groups(ROUTING_ID) == []


def test_unreadable_records_are_skipped(worker: Path) -> None:
    records = worker / ROUTING_ID / PROCESS_GROUPS_DIR_NAME
    records.mkdir(parents=True)
    (records / "a").write_text("1.0")
    (records / "b").write_text("garbage")
    (records / "c").write_text("456 789")

    assert recorded_process_groups(ROUTING_ID) == [ProcessGroupRecord(pgid=456, start_ticks=789)]
    assert recorded_process_groups("never-started") == []


async def test_the_local_provider_works_and_runs_commands_under_this_worker(worker: Path) -> None:
    provider = LocalProvider()
    handle = await provider.create(SandboxSpec())
    records = shlex.quote(str(worker / ROUTING_ID / PROCESS_GROUPS_DIR_NAME))
    try:
        assert handle.raw["workspace"].parent == worker / ROUTING_ID
        # The command's shell leads its own process group, recorded while it runs.
        # The record is written right after the spawn returns, so the shell waits for it rather than racing the parent.
        command = (
            f"for _ in $(seq 500); do ls {records}/$$-* >/dev/null 2>&1 && break; sleep 0.01; done; "
            f"echo $$; cat {records}/$$-*"
        )
        result = await provider.exec(handle, command)
        shell_pid, recorded_pgid, _ticks = result.stdout.split()
        assert recorded_pgid == shell_pid
        assert recorded_process_groups(ROUTING_ID) == []
    finally:
        await provider.close(handle)
    assert recorded_process_groups(ROUTING_ID) == []
    assert not handle.raw["workspace"].exists()
