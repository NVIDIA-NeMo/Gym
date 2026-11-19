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
import threading
import time
from pathlib import Path

import pytest

from nemo_gym.core import worker_resources
from nemo_gym.core.worker_resources import (
    PROCESS_GROUPS_DIR_NAME,
    ProcessGroupRecord,
    record_process_group,
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
def test_a_child_process_group_is_recorded_until_forgotten(worker: Path) -> None:
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        record = record_process_group(child.pid)
        assert record is not None and record.pgid == child.pid
        assert recorded_process_groups(ROUTING_ID) == [record]
        assert record.is_running()
        child.kill()
        child.wait()
        assert not record.is_running()
        record.forget()
    finally:
        child.kill()
        child.wait()
    assert recorded_process_groups(ROUTING_ID) == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
def test_ending_a_record_kills_its_group_and_removes_the_record(worker: Path) -> None:
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        record = record_process_group(child.pid)
        assert record is not None
        # The reap reads the record back from its file and ends it the same way.
        (read_back,) = recorded_process_groups(ROUTING_ID)
        read_back.end()
        assert child.wait(timeout=5) == -signal.SIGKILL
        assert recorded_process_groups(ROUTING_ID) == []
    finally:
        child.kill()
        child.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
def test_a_group_whose_leader_exited_still_runs_while_a_member_does(worker: Path) -> None:
    # The usual way a command leaves a background server: the shell exits, its child keeps the group.
    shell = subprocess.Popen(["sh", "-c", "sleep 30 & echo $!"], start_new_session=True, stdout=subprocess.PIPE)
    try:
        record = record_process_group(shell.pid)
        assert record is not None
        member = int(shell.stdout.readline())  # the member started before the shell exits
        shell.wait()
        # The process ID counter has not wrapped since the record was written, so no other group can hold the ID.
        assert record.is_running()
        tracked = record.track_members()
        assert tracked is not None and [pid for pid, _ in tracked.members] == [member]
        assert recorded_process_groups(ROUTING_ID) == [tracked]
        assert tracked.is_running()
        record = tracked
        record.end()
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
def test_a_record_whose_process_id_was_reused_is_not_running(worker: Path) -> None:
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        record = record_process_group(child.pid)
        assert record is not None
        # Another process with the same ID started at a different time: the record must not match it.
        reused = ProcessGroupRecord(pgid=child.pid, start_ticks=record.start_ticks - 1)
        assert not reused.is_running()
        reused.end()
        assert child.poll() is None
        record.forget()
    finally:
        child.kill()
        child.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
def test_a_child_that_does_not_lead_its_group_is_not_running(worker: Path) -> None:
    child = subprocess.Popen(["sleep", "30"])
    try:
        record = record_process_group(child.pid)
        assert record is not None
        assert not record.is_running()
        record.forget()
    finally:
        child.kill()
        child.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
def test_two_records_of_one_process_id_are_removed_independently(worker: Path) -> None:
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        outer = record_process_group(child.pid)
        inner = record_process_group(child.pid)
        assert outer is not None and inner is not None
        assert len(recorded_process_groups(ROUTING_ID)) == 2
        inner.forget()
        assert recorded_process_groups(ROUTING_ID) == [outer]
        outer.forget()
    finally:
        child.kill()
        child.wait()


def test_a_child_already_reaped_runs_unrecorded(worker: Path) -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    child.wait()
    assert record_process_group(child.pid) is None
    assert recorded_process_groups(ROUTING_ID) == []


def test_unreadable_records_are_skipped(worker: Path) -> None:
    records = worker / ROUTING_ID / PROCESS_GROUPS_DIR_NAME
    records.mkdir(parents=True)
    (records / "a").write_text("1.0")
    (records / "b").write_text("garbage")
    (records / "c").write_text("456 789 - 457:790")
    (records / "d").write_text("456 789")

    assert recorded_process_groups(ROUTING_ID) == [
        ProcessGroupRecord(pgid=456, start_ticks=789, members=((457, 790),))
    ]
    assert recorded_process_groups("never-started") == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
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
        shell_pid, recorded_pgid, _ticks, _last_pid = result.stdout.split()
        assert recorded_pgid == shell_pid
        assert recorded_process_groups(ROUTING_ID) == []
    finally:
        await provider.close(handle)
    assert recorded_process_groups(ROUTING_ID) == []
    assert not handle.raw["workspace"].exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
async def test_the_local_provider_keeps_a_background_process_recorded_until_close(worker: Path) -> None:
    provider = LocalProvider()
    handle = await provider.create(SandboxSpec())
    try:
        result = await provider.exec(handle, "sleep 300 >/dev/null 2>&1 & echo $!")
        background = int(result.stdout)
        # The command returned, but its background process keeps the group, so a peer's reap can still find it.
        (record,) = recorded_process_groups(ROUTING_ID)
        assert record.is_running() and os.getpgid(background) == record.pgid
    finally:
        await provider.close(handle)
    for _ in range(500):
        if not record.is_running():
            break
        time.sleep(0.01)
    assert not record.is_running()
    assert recorded_process_groups(ROUTING_ID) == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
def test_a_stale_record_never_claims_a_group_that_reused_its_id(worker: Path) -> None:
    # Another leaderless group now holds the number: its leader exited, and its member keeps the group.
    other = subprocess.Popen(["sh", "-c", "sleep 30 & echo $!"], start_new_session=True, stdout=subprocess.PIPE)
    try:
        other_member = int(other.stdout.readline())
        other.wait()
        start_ticks = worker_resources._start_ticks(other_member)
        assert start_ticks is not None
        # A stale record of an earlier group with the same ID, whose own leader and members are gone.
        stale = ProcessGroupRecord(pgid=other.pid, start_ticks=1, members=((other_member, start_ticks + 1),))
        untracked = ProcessGroupRecord(pgid=other.pid, start_ticks=1)

        assert not stale.is_running() and not untracked.is_running()
        stale.end()
        untracked.end()
        assert os.getpgid(other_member) == other.pid
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(other.pid, signal.SIGKILL)
        other.stdout.close()


def test_concurrent_first_use_hands_out_one_routing_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_resources, "_routing_id", None)
    real_uuid4 = worker_resources.uuid.uuid4
    entered = threading.Barrier(8, timeout=5)

    def slow_uuid4() -> object:
        # Widen the window between the check and the assignment, where an unsynchronized getter would race.
        time.sleep(0.01)
        return real_uuid4()

    monkeypatch.setattr(worker_resources.uuid, "uuid4", slow_uuid4)
    seen: list[str] = []

    def first_use() -> None:
        entered.wait()
        seen.append(worker_routing_id())

    threads = [threading.Thread(target=first_use) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(seen)) == 1 and seen[0] == worker_routing_id()


@pytest.mark.parametrize(
    ("pid", "since", "now", "passed"),
    [
        (500, 400, 450, False),  # not reached yet
        (500, 400, 500, True),  # allocated again
        (500, 400, 900, True),
        (500, 500, 900, False),  # the counter was at the group ID when confirmed
        (500, 900, 100, False),  # wrapped, but not yet back up to the group ID
        (500, 900, 600, True),  # wrapped and passed it
        (950, 900, 100, True),  # passed it before wrapping
    ],
)
def test_the_counter_passes_a_process_id_only_by_reaching_it_again(
    pid: int, since: int, now: int, passed: bool
) -> None:
    assert worker_resources._counter_passed(pid, since=since, now=now) is passed


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
def test_a_leaderless_group_is_trusted_only_until_the_counter_wraps_past_its_id(worker: Path) -> None:
    group = subprocess.Popen(["sh", "-c", "sleep 30 & echo $!"], start_new_session=True, stdout=subprocess.PIPE)
    try:
        member = int(group.stdout.readline())
        group.wait()
        current = worker_resources._last_pid()
        assert current is not None

        trusted = ProcessGroupRecord(pgid=group.pid, start_ticks=1, last_pid=current)
        # A counter that went past the group ID since the record was confirmed could have handed it to another group.
        wrapped = ProcessGroupRecord(pgid=group.pid, start_ticks=1, last_pid=group.pid - 1)

        # A tracked member still in the group proves the group is the original one, whatever the counter did.
        tracked = ProcessGroupRecord(
            pgid=group.pid,
            start_ticks=1,
            members=((member, worker_resources._start_ticks(member)),),
            last_pid=group.pid - 1,
        )

        assert trusted.is_running()
        assert not wrapped.is_running()
        assert tracked.is_running()
        wrapped.end()
        assert os.getpgid(member) == group.pid
        trusted.end()
        for _ in range(500):
            if not trusted.is_running():
                break
            time.sleep(0.01)
        assert not trusted.is_running()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group.pid, signal.SIGKILL)
        group.stdout.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
async def test_the_local_provider_kills_a_process_started_after_its_command_returned(worker: Path) -> None:
    provider = LocalProvider()
    handle = await provider.create(SandboxSpec())
    try:
        # A wrapper that starts the server a second after the command returns, then exits.
        await provider.exec(handle, "(sleep 1; exec sleep 300 &) >/dev/null 2>&1 & echo started")
        (record,) = recorded_process_groups(ROUTING_ID)
        for _ in range(500):
            if not any(worker_resources._start_ticks(pid) for pid, _ in record.members):
                break
            time.sleep(0.01)
        late = [pid for pid in worker_resources._processes_in_group(record.pgid)]
        assert late and not any(pid in dict(record.members) for pid in late)
        # A later command confirms the group again, so the record keeps trusting it.
        await provider.exec(handle, "true")
    finally:
        await provider.close(handle)
    for _ in range(500):
        if not worker_resources._processes_in_group(record.pgid):
            break
        time.sleep(0.01)
    assert worker_resources._processes_in_group(record.pgid) == []
    assert recorded_process_groups(ROUTING_ID) == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc is required")
async def test_a_later_command_drops_the_record_of_a_background_group_that_ended(worker: Path) -> None:
    provider = LocalProvider()
    handle = await provider.create(SandboxSpec())
    try:
        await provider.exec(handle, "sleep 0.5 >/dev/null 2>&1 & echo started")
        assert len(recorded_process_groups(ROUTING_ID)) == 1
        time.sleep(1)
        await provider.exec(handle, "true")
        assert recorded_process_groups(ROUTING_ID) == []
        assert handle.raw["process_groups"] == []
    finally:
        await provider.close(handle)
