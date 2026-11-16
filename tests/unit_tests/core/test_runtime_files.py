# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anyio
import pytest
from omegaconf import DictConfig

import nemo_gym.config.global_config as global_config
import nemo_gym.core.runtime_files as runtime_files
from nemo_gym.core.episode_context import rollout_context
from nemo_gym.core.runtime_files import (
    async_temp_folder,
    open_temp_folder,
    remove_stale_temp_folders,
    remove_temp_folder,
    run_temp_folder,
    save_folder,
    temp_folder,
)


def _request_folders(parent: Path) -> list[Path]:
    """The request folders under ``parent``: everything inside its per-process folders."""
    return [folder for process in parent.iterdir() for folder in process.iterdir()]


@pytest.fixture
def set_global_config(monkeypatch: pytest.MonkeyPatch):
    def set_(values: dict | None) -> None:
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None if values is None else DictConfig(values))
        monkeypatch.delenv(global_config.NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

    return set_


@pytest.fixture(autouse=True)
def system_temp(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the system temporary directory at a test folder, so no test writes into the host's own /tmp."""
    folder = tmp_path_factory.mktemp("system-temp")
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
    monkeypatch.setattr(runtime_files, "_default_temp_dirs", {})
    monkeypatch.setattr(runtime_files, "_run_folders", {})
    return folder


class TestDefaultTempDir:
    def test_is_a_private_folder_per_user_under_a_folder_shared_like_tmp(self, set_global_config, system_temp) -> None:
        set_global_config(None)

        with temp_folder("workspace") as path:
            user_folder = system_temp / "nemo_gym" / str(os.getuid())
            assert path.parent.parent == user_folder

        assert stat.S_IMODE(os.lstat(system_temp / "nemo_gym").st_mode) == 0o1777
        assert stat.S_IMODE(os.lstat(user_folder).st_mode) == 0o700
        assert not path.exists()

    @pytest.mark.parametrize("problem", ["symlink", "not-sticky"])
    def test_an_unsafe_shared_folder_falls_back_to_one_per_user(
        self, set_global_config, system_temp, tmp_path: Path, caplog, problem: str
    ) -> None:
        set_global_config(None)
        shared = system_temp / "nemo_gym"
        if problem == "symlink":
            (tmp_path / "elsewhere").mkdir()
            shared.symlink_to(tmp_path / "elsewhere")
        else:
            shared.mkdir()
            os.chmod(shared, 0o777)

        with caplog.at_level("WARNING"), temp_folder("workspace") as path:
            assert path.parent.parent == system_temp / f"nemo_gym-{os.getuid()}"

        assert "Not using" in caplog.text
        assert not any((tmp_path / "elsewhere").iterdir()) if problem == "symlink" else True

    def test_a_user_folder_that_is_a_symlink_is_refused_with_the_setting_named(
        self, set_global_config, system_temp, tmp_path: Path
    ) -> None:
        set_global_config(None)
        (tmp_path / "elsewhere").mkdir()
        shared = system_temp / "nemo_gym"
        shared.mkdir()
        os.chmod(shared, 0o1777)
        (shared / str(os.getuid())).symlink_to(tmp_path / "elsewhere")

        with pytest.raises(PermissionError, match="default temp_dir"):
            with temp_folder("workspace"):
                pass

    def test_a_temp_dir_that_cannot_be_created_names_the_folder_and_the_setting(
        self, set_global_config, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "a-file"
        blocker.write_text("")
        set_global_config({"temp_dir": str(blocker / "temp")})

        with pytest.raises(OSError, match="chosen by the temp_dir setting"):
            with temp_folder("workspace"):
                pass


class TestSessionAndRunFolders:
    def test_a_session_folder_lasts_until_it_is_removed_and_removing_twice_is_fine(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})

        folder = open_temp_folder("session")
        (folder / "state.json").write_text("{}")
        assert folder.exists()

        assert remove_temp_folder(folder) is True
        assert not folder.exists()
        assert remove_temp_folder(folder) is True

    def test_a_failed_removal_is_reported_and_a_later_removal_retries(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        folder = open_temp_folder("session")
        real_rmtree = shutil.rmtree

        def failing(path, *args, **kwargs):
            raise PermissionError(13, "busy")

        monkeypatch.setattr(shutil, "rmtree", failing)
        assert remove_temp_folder(folder) is False
        assert folder.exists()

        monkeypatch.setattr(shutil, "rmtree", real_rmtree)
        assert remove_temp_folder(folder) is True
        assert not folder.exists()

    def test_a_run_folder_is_the_same_for_every_call_with_its_name(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})

        first = run_temp_folder("image-cache")
        assert run_temp_folder("image-cache") == first
        assert run_temp_folder("venvs") != first
        assert first.is_dir() and first.name.startswith("image-cache-")

    def test_a_forked_child_gets_run_folders_of_its_own(self, set_global_config, tmp_path: Path, monkeypatch) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        monkeypatch.setattr(runtime_files, "_process_folders", {})
        parent_folder = run_temp_folder("image-cache")
        real_pid = os.getpid()

        # A forked child has a new process id and inherits the parent's tables.
        monkeypatch.setattr(os, "getpid", lambda: real_pid + 1)
        child_folder = run_temp_folder("image-cache")

        assert child_folder != parent_folder
        assert child_folder.parent != parent_folder.parent

    def test_the_exit_handler_removes_run_folders(self, set_global_config, tmp_path: Path, monkeypatch) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        monkeypatch.setattr(runtime_files, "_process_folders", {})
        folder = run_temp_folder("image-cache")
        (folder / "a.txt").write_text("x")

        runtime_files._remove_own_process_folders(os.getpid())

        assert not folder.parent.exists()


class TestSaveFolder:
    @staticmethod
    def _logs(tmp_path: Path) -> Path:
        logs = tmp_path / "logs"
        (logs / "sub").mkdir(parents=True)
        (logs / "report.json").write_text("{}")
        (logs / "sub" / "test_output.txt").write_text("ok")
        return logs

    async def test_copies_into_the_server_results_by_name_and_rollout(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)

        with rollout_context("4-2-a1"):
            kept = await save_folder(logs, server_name="swebench", name="verification", label="astropy-1-attempt0")

        assert kept is not None
        assert kept.parent == tmp_path / "results" / "swebench" / "verification" / "4-2-a1"
        assert "-astropy-1-attempt0-" in kept.name
        assert (kept / "report.json").read_text() == "{}"
        assert (kept / "sub" / "test_output.txt").read_text() == "ok"
        # The source is left as it was; the temporary folder's own block removes it.
        assert (logs / "report.json").exists()

    async def test_copies_contents_only_not_permissions_or_times(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        os.utime(logs / "report.json", (1, 1))
        os.chmod(logs / "report.json", 0o600)

        kept = await save_folder(logs, server_name="swebench", name="verification")

        umask = os.umask(0)
        os.umask(umask)
        assert (kept / "report.json").stat().st_mtime != 1
        assert stat.S_IMODE((kept / "report.json").stat().st_mode) == 0o666 & ~umask

    async def test_outside_a_rollout_and_on_repeat_each_copy_gets_its_own_folder(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)

        first = await save_folder(logs, server_name="swebench", name="verification", label="same")
        second = await save_folder(logs, server_name="swebench", name="verification", label="same")

        assert first != second
        assert first.parent == second.parent == tmp_path / "results" / "swebench" / "verification" / "no-rollout"

    async def test_the_server_output_dir_replaces_the_default(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})

        kept = await save_folder(
            self._logs(tmp_path), server_name="swebench", name="verification", output_dir=str(tmp_path / "mine")
        )

        assert kept is not None and kept.parent == tmp_path / "mine" / "verification" / "no-rollout"

    async def test_a_relative_output_dir_keeps_nothing_with_one_warning(
        self, set_global_config, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(runtime_files, "_warned_relative_output_dirs", set())
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)

        with caplog.at_level("WARNING"):
            assert await save_folder(logs, server_name="swebench", name="verification", output_dir="out") is None
            assert await save_folder(logs, server_name="swebench", name="verification", output_dir="out") is None

        assert caplog.text.count("is not an absolute path") == 1

    async def test_a_process_without_gym_configuration_keeps_nothing(self, set_global_config, tmp_path: Path) -> None:
        set_global_config(None)

        assert await save_folder(self._logs(tmp_path), server_name="swebench", name="verification") is None

    async def test_a_failed_copy_returns_none_and_leaves_no_partial_folder(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})

        def failing_copy(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(shutil, "copyfile", failing_copy)
        kept = await save_folder(self._logs(tmp_path), server_name="swebench", name="verification")

        assert kept is None
        assert not any((tmp_path / "results" / "swebench" / "verification" / "no-rollout").iterdir())

    async def test_a_failed_request_keeps_its_files_from_a_finally(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results"), "temp_dir": str(tmp_path / "temp")})
        kept = None

        with pytest.raises(RuntimeError):
            async with async_temp_folder("verification") as logs:
                try:
                    (logs / "test_output.txt").write_text("FAILED")
                    raise RuntimeError("harness failed")
                finally:
                    kept = await save_folder(logs, server_name="swebench", name="verification")

        assert kept is not None and (kept / "test_output.txt").read_text() == "FAILED"
        assert not logs.exists()

    async def test_nothing_is_copied_while_the_task_is_being_cancelled(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        started = asyncio.Event()
        kept: list[Path | None] = []

        async def request() -> None:
            try:
                started.set()
                await asyncio.sleep(3600)
            finally:
                kept.append(await save_folder(logs, server_name="swebench", name="verification"))

        task = asyncio.create_task(request())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert kept == [None]
        assert not (tmp_path / "results").exists()

    async def test_names_and_labels_stay_one_folder_under_the_root(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)

        with rollout_context("../escape"):
            kept = await save_folder(logs, server_name="../server", name="verification", label="../../org/repo")

        assert kept is not None
        assert kept.relative_to(tmp_path / "results").parts[0] == "..%2Fserver"
        assert ".." not in kept.parts

    async def test_a_name_must_be_one_safe_part_and_a_server_name_must_name_a_server(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)

        with pytest.raises(ValueError, match="name"):
            await save_folder(logs, server_name="swebench", name="../escape")
        for server_name in ("", ".", ".."):
            with pytest.raises(ValueError, match="server_name"):
                await save_folder(logs, server_name=server_name, name="verification")


class TestTempFolder:
    def test_folder_is_under_the_temporary_directory_and_removed_on_exit(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        # The system temporary directory, pointed at a test folder so the host's own /tmp is left alone.
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
        set_global_config(None)

        with temp_folder(name="workspace") as path:
            (path / "file.txt").write_text("x")
            assert path.parent.parent == Path(tempfile.gettempdir()) / "nemo_gym" / str(os.getuid())

        assert not path.exists()

    def test_folder_is_under_temp_dir_when_set(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})

        with temp_folder(name="workspace") as path:
            assert path.parent.parent == tmp_path / "temp"

        assert _request_folders(tmp_path / "temp") == []

    def test_folder_is_removed_when_the_block_raises(self, set_global_config) -> None:
        set_global_config(None)

        with pytest.raises(RuntimeError):
            with temp_folder(name="workspace") as path:
                (path / "file.txt").write_text("x")
                raise RuntimeError("boom")

        assert not path.exists()


def test_temp_dir_is_checked_once_on_the_writing_node(set_global_config, tmp_path: Path, monkeypatch) -> None:
    checked: list[tuple[str, Path]] = []
    monkeypatch.setattr(runtime_files, "warn_if_unsuitable", lambda setting, path: checked.append((setting, path)))
    monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
    set_global_config({"temp_dir": str(tmp_path / "temp")})

    with temp_folder(name="workspace"):
        pass
    with temp_folder(name="workspace"):
        pass

    assert checked == [("temp_dir", tmp_path / "temp")]


@pytest.mark.skipif(
    not Path("/proc/self/ns/pid").exists() or not Path("/proc/sys/kernel/random/boot_id").exists(),
    reason="needs Linux PID namespaces and a boot id",
)
class TestStaleTempFolders:
    def _dead_pid(self) -> int:
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait()
        return process.pid

    def _name(self, pid: int, *, host: str | None = None, namespace: int | None = None) -> str:
        host = host or Path("/proc/sys/kernel/random/boot_id").read_text().strip().replace("-", "")[:12]
        namespace = namespace or os.stat("/proc/self/ns/pid").st_ino
        return f"nemo-gym-{host}-{namespace}-{pid}"

    def test_only_folders_of_dead_processes_on_this_host_and_namespace_are_removed(self, tmp_path: Path) -> None:
        namespace = os.stat("/proc/self/ns/pid").st_ino
        dead = tmp_path / self._name(self._dead_pid())
        live = tmp_path / self._name(os.getpid())
        # The dead process's request folders go with its process folder.
        (dead / "workspace-abc" / "nested").mkdir(parents=True)
        (dead / "workspace-abc" / "nested" / "file.txt").write_text("x")
        other_namespace = tmp_path / self._name(self._dead_pid(), namespace=namespace + 1)
        # Another host's folder on a shared directory: same namespace number, a dead pid here, a live process there.
        other_host = tmp_path / self._name(self._dead_pid(), host="0" * 12)
        unrelated = tmp_path / "something-else"
        for folder in (live, other_namespace, other_host, unrelated):
            folder.mkdir()

        assert remove_stale_temp_folders(tmp_path) == [dead]
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            p.name for p in (live, other_namespace, other_host, unrelated)
        )

        # On storage private to this host, a folder from another boot id is from an earlier boot: no process of it runs.
        assert remove_stale_temp_folders(tmp_path, private_to_host=True) == [other_host]
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in (live, other_namespace, unrelated))

    def test_first_folder_in_a_temp_dir_clears_leftovers(self, set_global_config, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
        leftover = tmp_path / self._name(self._dead_pid())
        leftover.mkdir()
        set_global_config({"temp_dir": str(tmp_path)})

        with temp_folder(name="workspace") as path:
            assert path.parent.name == self._name(os.getpid())

        assert not leftover.exists()

    def test_leftovers_are_removed_on_a_shared_filesystem_too(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        # The boot id keeps other hosts' folders safe, so a shared temp_dir still gets an end of life.
        monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
        monkeypatch.setattr(runtime_files, "warn_if_unsuitable", lambda setting, path: "lustre")
        leftover = tmp_path / self._name(self._dead_pid())
        leftover.mkdir()
        set_global_config({"temp_dir": str(tmp_path)})

        with temp_folder(name="workspace"):
            pass

        assert not leftover.exists()

    def test_a_folder_name_with_an_impossible_pid_is_kept_and_never_fails_the_request(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
        crafted = tmp_path / self._name(10**30)
        crafted.mkdir()
        set_global_config({"temp_dir": str(tmp_path)})

        with temp_folder(name="workspace") as path:
            assert path.exists()

        assert crafted.exists()

    def test_a_failing_cleanup_never_fails_the_request(self, set_global_config, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())

        def broken(parent: Path) -> list[Path]:
            raise OSError("listing failed")

        monkeypatch.setattr(runtime_files, "remove_stale_temp_folders", broken)
        set_global_config({"temp_dir": str(tmp_path)})

        with temp_folder(name="workspace") as path:
            assert path.exists()

    def test_another_users_folders_stay(self, tmp_path: Path, monkeypatch) -> None:
        folder = tmp_path / self._name(self._dead_pid())
        folder.mkdir()
        real_uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

        assert remove_stale_temp_folders(tmp_path) == []
        assert folder.exists()

    def test_files_and_symlinks_with_a_matching_name_are_left_alone(self, tmp_path: Path) -> None:
        (tmp_path / self._name(self._dead_pid())).write_text("")
        target = tmp_path / "target"
        target.mkdir()
        (target / "keep").write_text("")
        (tmp_path / self._name(self._dead_pid())).symlink_to(target)

        assert remove_stale_temp_folders(tmp_path) == []
        assert (target / "keep").exists()

    def test_a_process_this_user_cannot_signal_counts_as_running(self, tmp_path: Path) -> None:
        if os.getuid() == 0:
            pytest.skip("root may signal every process")
        # PID 1 is alive in every namespace, and a process of another user answers with PermissionError.
        folder = tmp_path / self._name(1)
        folder.mkdir()

        assert remove_stale_temp_folders(tmp_path) == []
        assert folder.exists()


def test_request_folders_share_one_folder_per_process(set_global_config, tmp_path: Path) -> None:
    set_global_config({"temp_dir": str(tmp_path)})

    with temp_folder(name="workspace") as first, temp_folder(name="harness") as second:
        assert first.parent == second.parent
        assert first.parent.parent == tmp_path
        assert first.name.startswith("workspace-") and second.name.startswith("harness-")

    assert [p.name for p in tmp_path.iterdir()] == [first.parent.name]


def test_exit_removes_this_process_folder_and_a_forked_child_does_not(
    set_global_config, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runtime_files, "_process_folders", {})
    set_global_config({"temp_dir": str(tmp_path)})
    with temp_folder(name="workspace") as path:
        (path / "file.txt").write_text("x")
        process_folder = path.parent

        # A forked child inherits the exit handler and the record of this folder, but must leave it alone.
        real_pid = os.getpid()
        monkeypatch.setattr(os, "getpid", lambda: real_pid + 1)
        runtime_files._remove_own_process_folders(real_pid)
        monkeypatch.setattr(os, "getpid", lambda: real_pid)
        assert process_folder.exists()

        runtime_files._remove_own_process_folders(os.getpid())

    assert not process_folder.exists()


def test_a_stale_folder_with_this_process_name_is_cleared_before_use(
    set_global_config, tmp_path: Path, monkeypatch
) -> None:
    # Only an earlier process with the same id, now dead, can have left a folder with this name.
    monkeypatch.setattr(runtime_files, "_process_folders", {})
    stale = tmp_path / runtime_files._process_folder_name()
    (stale / "workspace-old").mkdir(parents=True)
    set_global_config({"temp_dir": str(tmp_path)})

    with temp_folder(name="workspace") as path:
        assert path.parent == stale

    assert not (stale / "workspace-old").exists()


def test_a_temp_dir_argument_replaces_the_setting_for_code_without_gym_configuration(
    set_global_config, tmp_path: Path
) -> None:
    set_global_config(None)

    with temp_folder(name="harbor-job", temp_dir=tmp_path / "from-server") as path:
        assert path.parent.parent == tmp_path / "from-server"

    assert not path.exists()


class TestAsyncTempFolder:
    async def test_folder_is_removed_when_the_task_is_cancelled(self, set_global_config) -> None:
        set_global_config(None)
        created: list[Path] = []
        entered = asyncio.Event()

        async def work() -> None:
            async with async_temp_folder(name="workspace") as path:
                created.append(path)
                (path / "file.txt").write_text("x")
                entered.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(work())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert created and not created[0].exists()


async def test_folder_is_removed_when_cancelled_while_it_is_created(
    set_global_config, tmp_path: Path, monkeypatch
) -> None:
    set_global_config({"temp_dir": str(tmp_path)})
    started, release, created = threading.Event(), threading.Event(), threading.Event()
    create = runtime_files._create_temp_folder
    made: list[Path] = []

    def slow_create(kind: str, parent: Path | None) -> Path:
        started.set()
        release.wait(5)
        made.append(create(kind, parent))
        created.set()
        return made[0]

    monkeypatch.setattr(runtime_files, "_create_temp_folder", slow_create)

    async def work() -> None:
        async with async_temp_folder(name="workspace"):
            pass

    task = asyncio.create_task(work())
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    # The folder must really be created after the cancel, or the test would prove nothing.
    assert await asyncio.to_thread(created.wait, 5)
    for _ in range(200):
        if not made[0].exists():
            break
        await asyncio.sleep(0.01)

    assert not made[0].exists()
    assert _request_folders(tmp_path) == []


async def test_folder_is_removed_when_an_anyio_cancel_scope_fires(set_global_config, tmp_path: Path) -> None:
    import anyio

    set_global_config({"temp_dir": str(tmp_path)})

    with anyio.move_on_after(0.05):
        async with async_temp_folder(name="workspace"):
            await anyio.sleep(3600)

    for _ in range(100):
        if not _request_folders(tmp_path):
            break
        await asyncio.sleep(0.01)
    assert _request_folders(tmp_path) == []


def _cleanup_left_after_a_repeated_cancellation(cancel_kind: str) -> list[Path]:
    """Cancel again while cleanup waits behind a busy worker thread, as under load, and return what is left."""

    async def main() -> list[Path]:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        created: list[Path] = []
        entered = asyncio.Event()
        scopes: list[anyio.CancelScope] = []

        async def work() -> None:
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                async with async_temp_folder(name="workspace") as path:
                    created.append(path)
                    entered.set()
                    await asyncio.sleep(3600)

        task = asyncio.create_task(work())
        await entered.wait()
        gate = threading.Event()
        busy = loop.run_in_executor(None, gate.wait)
        if cancel_kind == "anyio":
            # anyio delivers the cancellation again at every await in the scope, as the disconnect middleware does.
            scopes[0].cancel()
        else:
            task.cancel()
            await asyncio.sleep(0.05)
            task.cancel()
        try:
            await asyncio.wait_for(task, 5)
        except asyncio.CancelledError:
            pass
        gate.set()
        await busy
        await loop.run_in_executor(None, lambda: None)
        executor.shutdown(wait=True)
        return [path for path in created if path.exists()]

    return asyncio.run(main())


@pytest.mark.parametrize("cancel_kind", ["anyio", "task"])
def test_cleanup_survives_a_repeated_cancellation_while_queued(
    set_global_config, tmp_path: Path, cancel_kind: str
) -> None:
    set_global_config({"temp_dir": str(tmp_path / "temp")})

    assert _cleanup_left_after_a_repeated_cancellation(cancel_kind) == []
    assert _request_folders(tmp_path / "temp") == []


def test_concurrent_first_use_of_a_parent_loses_no_request(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime_files, "_process_folders", {})
    monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
    count = 16
    start, written = threading.Barrier(count), threading.Barrier(count)
    errors: list[BaseException] = []
    missing: list[Path] = []

    def request(index: int) -> None:
        try:
            start.wait(5)
            with temp_folder(name="workspace", temp_dir=tmp_path) as path:
                (path / f"{index}.txt").write_text("x")
                # Every request holds its folder open until all have written, so a lost file shows here.
                written.wait(5)
                if not (path / f"{index}.txt").exists():
                    missing.append(path)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=request, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    assert missing == []


def test_a_process_folder_removed_from_outside_is_made_again(set_global_config, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime_files, "_process_folders", {})
    set_global_config({"temp_dir": str(tmp_path)})
    with temp_folder(name="workspace") as first:
        pass
    shutil.rmtree(first.parent)

    with temp_folder(name="workspace") as second:
        assert second.parent == first.parent


def test_a_forked_child_gets_a_free_first_use_lock(monkeypatch) -> None:
    held = threading.Lock()
    held.acquire()
    monkeypatch.setattr(runtime_files, "_first_use_lock", held)

    # What os.register_at_fork runs in the child after a fork.
    runtime_files._reset_locks_after_fork()

    assert runtime_files._first_use_lock is not held
    assert runtime_files._first_use_lock.acquire(blocking=False)
