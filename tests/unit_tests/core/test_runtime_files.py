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
import atexit
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
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
    async_remove_temp_dir,
    async_request_temp_dir,
    async_session_temp_dir,
    process_temp_dir,
    remove_temp_dir,
    request_temp_dir,
    save_dir,
    session_temp_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LINUX_IDS = pytest.mark.skipif(
    not Path("/proc/self/ns/pid").exists() or not Path("/proc/sys/kernel/random/boot_id").exists(),
    reason="needs Linux PID namespaces and a boot id",
)
NOT_ROOT = pytest.mark.skipif(os.getuid() == 0, reason="root writes through folder permissions")


def _request_folders(parent: Path) -> list[Path]:
    """The request folders under ``parent``: everything inside its process folders."""
    return [folder for process in parent.iterdir() for folder in process.iterdir()]


def _umask() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return umask


@pytest.fixture
def set_global_config(monkeypatch: pytest.MonkeyPatch):
    def set_(values: dict | None) -> None:
        monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None if values is None else DictConfig(values))
        monkeypatch.delenv(global_config.NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)

    return set_


@pytest.fixture(autouse=True)
def system_temp(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the system temporary directory at a test folder and start each test with no per-process state."""
    folder = tmp_path_factory.mktemp("system-temp")
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    monkeypatch.setattr(runtime_files, "_default_temp_dir", None)
    monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
    monkeypatch.setattr(runtime_files, "_process_dirs", {})
    monkeypatch.setattr(runtime_files, "_warned", set())
    return folder


class TestDefaultTempDir:
    def test_process_folders_are_private_under_a_configured_temp_dir(self, set_global_config, tmp_path: Path) -> None:
        # A configured temp_dir, unlike the default, may be a folder other users can open.
        set_global_config({"temp_dir": str(tmp_path)})
        old_umask = os.umask(0o022)
        try:
            process = process_temp_dir("cache")
        finally:
            os.umask(old_umask)

        assert stat.S_IMODE(os.lstat(process).st_mode) == 0o700
        assert stat.S_IMODE(os.lstat(process.parent).st_mode) == 0o700

    def test_is_a_private_folder_per_user_in_the_system_temp(self, set_global_config, system_temp) -> None:
        set_global_config(None)

        with request_temp_dir("workspace") as path:
            user_folder = system_temp / f"nemo_gym-{os.getuid()}"
            assert path.parent.parent == user_folder

        assert stat.S_IMODE(os.lstat(user_folder).st_mode) == 0o700
        assert not path.exists()

    @pytest.mark.parametrize("problem", ["symlink", "file", "other-user"])
    def test_a_default_that_is_not_this_users_own_folder_is_refused(
        self, set_global_config, system_temp, tmp_path: Path, monkeypatch, problem: str
    ) -> None:
        set_global_config(None)
        real_uid = os.getuid()
        if problem == "other-user":
            # The folder this test creates is owned by the real user, so it is another user's to a fake one.
            monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
        folder = system_temp / f"nemo_gym-{os.getuid()}"
        if problem == "symlink":
            (tmp_path / "elsewhere").mkdir()
            folder.symlink_to(tmp_path / "elsewhere")
        elif problem == "file":
            folder.write_text("")
        else:
            folder.mkdir(mode=0o700)

        with pytest.raises(PermissionError, match="Set temp_dir"):
            with request_temp_dir("workspace"):
                pass
        if problem == "symlink":
            assert not any((tmp_path / "elsewhere").iterdir())

    def test_an_existing_default_others_can_write_is_made_private(
        self, set_global_config, system_temp, caplog
    ) -> None:
        set_global_config(None)
        # Such as one a launcher made with the default umask, or for every user.
        folder = system_temp / f"nemo_gym-{os.getuid()}"
        folder.mkdir()
        os.chmod(folder, 0o777)

        with caplog.at_level("WARNING"):
            with request_temp_dir("workspace") as path:
                assert path.parent.parent == folder

        assert stat.S_IMODE(os.lstat(folder).st_mode) == 0o700
        assert "private to this user: it was mode 777" in caplog.text

    @NOT_ROOT
    def test_an_unwritable_system_temp_names_the_setting(self, set_global_config, system_temp: Path) -> None:
        set_global_config(None)
        os.chmod(system_temp, 0o555)
        try:
            with pytest.raises(PermissionError, match="Could not create the default temp_dir .*Set temp_dir"):
                with request_temp_dir("workspace"):
                    pass
        finally:
            os.chmod(system_temp, 0o755)

    def test_a_default_removed_from_outside_is_made_again_and_private(self, set_global_config, system_temp) -> None:
        set_global_config(None)
        with request_temp_dir("workspace") as first:
            pass
        shutil.rmtree(first.parent.parent)

        with request_temp_dir("workspace") as second:
            assert second.parent.parent == first.parent.parent

        assert stat.S_IMODE(os.lstat(first.parent.parent).st_mode) == 0o700


class TestTempDirChoice:
    def test_the_setting_replaces_the_default(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})

        with request_temp_dir("workspace") as path:
            assert path.parent.parent == tmp_path / "temp"

        assert _request_folders(tmp_path / "temp") == []

    def test_the_argument_replaces_the_setting_for_code_without_gym_configuration(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "from-config")})

        with request_temp_dir("harbor-job", temp_dir=tmp_path / "from-server") as path:
            assert path.parent.parent == tmp_path / "from-server"

        assert not path.exists()

    def test_the_argument_expands_home_and_refuses_a_relative_path(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config(None)
        monkeypatch.setenv("HOME", str(tmp_path))

        with request_temp_dir("workspace", temp_dir="~/gym-temp") as path:
            assert path.parent.parent == tmp_path / "gym-temp"
        # A relative path would resolve against the server's own source folder.
        with pytest.raises(ValueError, match="absolute"):
            with request_temp_dir("workspace", temp_dir="gym-temp"):
                pass

    def test_a_temp_dir_that_cannot_be_created_names_the_folder_and_the_setting(
        self, set_global_config, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "a-file"
        blocker.write_text("")
        set_global_config({"temp_dir": str(blocker / "temp")})

        with pytest.raises(OSError, match="chosen by the temp_dir setting"):
            with request_temp_dir("workspace"):
                pass

    def test_temp_dir_is_checked_once_on_the_writing_node(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        checked: list[tuple[str, Path]] = []
        monkeypatch.setattr(runtime_files, "warn_if_unsuitable", lambda setting, path: checked.append((setting, path)))
        set_global_config({"temp_dir": str(tmp_path / "temp")})

        with request_temp_dir("workspace"):
            pass
        with request_temp_dir("workspace"):
            pass

        assert checked == [("temp_dir", tmp_path / "temp")]

    def test_a_temp_dir_removed_from_outside_is_checked_and_made_again(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        checked: list[Path] = []
        monkeypatch.setattr(runtime_files, "warn_if_unsuitable", lambda setting, path: checked.append(path))
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        with request_temp_dir("workspace") as first:
            pass
        # A cleaner of old temporary files can remove the whole folder, not only this process's.
        shutil.rmtree(tmp_path / "temp")

        with request_temp_dir("workspace") as second:
            assert second.parent == first.parent
        assert process_temp_dir("cache").is_dir()

        assert checked == [tmp_path / "temp", tmp_path / "temp"]

    def test_a_name_must_be_one_safe_part(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path)})

        for name in ("../escape", "", "Upper", "a.b"):
            with pytest.raises(ValueError, match="name"):
                session_temp_dir(name)
            with pytest.raises(ValueError, match="name"):
                process_temp_dir(name)


class TestRequestTempDir:
    def test_folders_of_one_process_share_one_process_folder(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path)})

        with request_temp_dir("workspace") as first, request_temp_dir("harness") as second:
            assert first.parent == second.parent
            assert first.parent.parent == tmp_path
            assert first.name.startswith("workspace-") and second.name.startswith("harness-")

        assert [p.name for p in tmp_path.iterdir()] == [first.parent.name]

    def test_folder_is_removed_when_the_block_raises(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path)})

        with pytest.raises(RuntimeError):
            with request_temp_dir("workspace") as path:
                (path / "file.txt").write_text("x")
                raise RuntimeError("boom")

        assert not path.exists()

    @NOT_ROOT
    def test_read_only_folders_inside_are_removed_too(self, set_global_config, tmp_path: Path) -> None:
        # Tools such as the Go module cache leave folders this user cannot write into.
        set_global_config({"temp_dir": str(tmp_path)})

        with request_temp_dir("workspace") as path:
            (path / "cache" / "module").mkdir(parents=True)
            (path / "cache" / "module" / "file.go").write_text("x")
            os.chmod(path / "cache" / "module", 0o500)
            os.chmod(path / "cache", 0o500)
            # A folder with no permissions at all cannot even be listed until it is made readable.
            (path / "locked").mkdir()
            (path / "locked" / "file").write_text("x")
            os.chmod(path / "locked", 0)

        assert not path.exists()

    def test_an_entry_that_cannot_be_removed_leaves_the_rest_removed(
        self, set_global_config, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        # Such as a file a container wrote as root into a bind mount: everything else must still go.
        set_global_config({"temp_dir": str(tmp_path)})
        unlink = os.unlink

        def refuse_stuck(path, *args, **kwargs):
            if os.path.basename(path) == "stuck":
                raise PermissionError(1, "Operation not permitted", path)
            return unlink(path, *args, **kwargs)

        with caplog.at_level("WARNING"):
            with request_temp_dir("workspace") as path:
                for name in ("a", "stuck", "z"):
                    (path / name).mkdir()
                    (path / name / "file").write_text("x")
                (path / "stuck" / "file").rename(path / "stuck" / "stuck")
                monkeypatch.setattr(os, "unlink", refuse_stuck)
            monkeypatch.setattr(os, "unlink", unlink)

        assert sorted(p.name for p in path.iterdir()) == ["stuck"]
        assert list((path / "stuck").iterdir()) == [path / "stuck" / "stuck"]
        assert "Could not remove" in caplog.text

    def test_an_unexpected_error_while_removing_never_replaces_the_requests_own(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        # A very deep tree makes shutil.rmtree raise RecursionError.
        set_global_config({"temp_dir": str(tmp_path)})

        def deep(path, *args, **kwargs):
            raise RecursionError("maximum recursion depth exceeded")

        with pytest.raises(RuntimeError, match="the request's own"):
            with request_temp_dir("workspace"):
                monkeypatch.setattr(shutil, "rmtree", deep)
                raise RuntimeError("the request's own error")

    def test_removing_leftovers_holds_up_no_other_parent_or_session_removal(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "first")})
        session = session_temp_dir("session")
        started, release = threading.Event(), threading.Event()
        remove_stale = runtime_files._remove_stale_process_dirs

        def slow_cleanup(parent: Path) -> list[Path]:
            # Leftovers can be large trees of a dead process.
            if parent.name == "second":
                started.set()
                release.wait(10)
            return remove_stale(parent)

        monkeypatch.setattr(runtime_files, "_remove_stale_process_dirs", slow_cleanup)
        slow = threading.Thread(target=lambda: session_temp_dir("other", temp_dir=tmp_path / "second"))
        slow.start()
        try:
            assert started.wait(5)
            done = threading.Event()
            other = threading.Thread(
                target=lambda: (
                    remove_temp_dir(session),
                    session_temp_dir("third", temp_dir=tmp_path / "third"),
                    done.set(),
                )
            )
            other.start()
            assert done.wait(2)
        finally:
            release.set()
            slow.join(5)
            other.join(5)
        assert not session.exists()

    def test_concurrent_first_use_of_a_parent_loses_no_request(self, tmp_path: Path) -> None:
        count = 16
        start, written = threading.Barrier(count), threading.Barrier(count)
        errors: list[BaseException] = []
        missing: list[Path] = []

        def request(index: int) -> None:
            try:
                start.wait(5)
                with request_temp_dir("workspace", temp_dir=tmp_path) as path:
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

        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        assert missing == []


class TestAsyncRequestTempDir:
    async def test_folder_is_removed_when_the_task_is_cancelled(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path)})
        created: list[Path] = []
        entered = asyncio.Event()

        async def work() -> None:
            async with async_request_temp_dir("workspace") as path:
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
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path)})
        started, release, created = threading.Event(), threading.Event(), threading.Event()
        create = runtime_files._create_temp_dir
        made: list[Path] = []

        def slow_create(name: str, temp_dir: Path | None, **kwargs: str) -> Path:
            started.set()
            release.wait(5)
            made.append(create(name, temp_dir, **kwargs))
            created.set()
            return made[0]

        monkeypatch.setattr(runtime_files, "_create_temp_dir", slow_create)

        async def work() -> None:
            async with async_request_temp_dir("workspace"):
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

    async def test_folder_is_removed_when_an_anyio_cancel_scope_fires(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path)})

        with anyio.move_on_after(0.05):
            async with async_request_temp_dir("workspace"):
                await anyio.sleep(3600)

        for _ in range(100):
            if not _request_folders(tmp_path):
                break
            await asyncio.sleep(0.01)
        assert _request_folders(tmp_path) == []

    async def test_folders_are_made_and_removed_while_the_default_executor_is_busy(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path)})
        loop = asyncio.get_running_loop()
        default = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(default)
        gate = threading.Event()
        # Holds the default executor's only thread, as slow sandbox calls and DNS lookups can.
        busy = loop.run_in_executor(None, gate.wait, 10)
        try:

            async def work() -> Path:
                async with async_request_temp_dir("workspace") as path:
                    (path / "out.txt").write_text("x")
                return path

            path = await asyncio.wait_for(work(), 2)
        finally:
            gate.set()
            await busy
            default.shutdown(wait=True)

        assert not path.exists()


def _cleanup_left_after_a_repeated_cancellation(cancel_kind: str, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Cancel again while cleanup waits behind a busy worker thread, as under load, and return what is left."""

    async def main() -> list[Path]:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        monkeypatch.setattr(runtime_files, "_temp_dir_executor", executor)
        created: list[Path] = []
        entered = asyncio.Event()
        scopes: list[anyio.CancelScope] = []

        async def work() -> None:
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                async with async_request_temp_dir("workspace") as path:
                    created.append(path)
                    entered.set()
                    await asyncio.sleep(3600)

        task = asyncio.create_task(work())
        await entered.wait()
        gate = threading.Event()
        busy = loop.run_in_executor(executor, gate.wait)
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
        await loop.run_in_executor(executor, lambda: None)
        executor.shutdown(wait=True)
        return [path for path in created if path.exists()]

    return asyncio.run(main())


@pytest.mark.parametrize("cancel_kind", ["anyio", "task"])
def test_cleanup_survives_a_repeated_cancellation_while_queued(
    set_global_config, tmp_path: Path, cancel_kind: str, monkeypatch
) -> None:
    set_global_config({"temp_dir": str(tmp_path / "temp")})

    assert _cleanup_left_after_a_repeated_cancellation(cancel_kind, monkeypatch) == []
    assert _request_folders(tmp_path / "temp") == []


class TestSessionTempDir:
    def test_a_session_folder_lasts_until_it_is_removed_and_removing_twice_is_fine(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})

        folder = session_temp_dir("session")
        (folder / "state.json").write_text("{}")
        assert folder.exists()

        assert remove_temp_dir(folder) is True
        assert not folder.exists()
        assert remove_temp_dir(folder) is True

    async def test_a_coroutine_creates_a_session_folder_off_the_event_loop(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        threads: list[str] = []
        create = runtime_files._create_temp_dir

        def recording_create(name: str, temp_dir: Path | None, **kwargs: str) -> Path:
            threads.append(threading.current_thread().name)
            return create(name, temp_dir, **kwargs)

        monkeypatch.setattr(runtime_files, "_create_temp_dir", recording_create)

        folder = await async_session_temp_dir("session")

        assert threads and threads[0].startswith("nemo-gym-temp-dir")
        assert folder.is_dir()
        assert await async_remove_temp_dir(folder) is True

    async def test_a_session_folder_created_after_its_caller_was_cancelled_is_removed(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        started, release, created = threading.Event(), threading.Event(), threading.Event()
        create = runtime_files._create_temp_dir
        made: list[Path] = []

        def slow_create(name: str, temp_dir: Path | None, **kwargs: str) -> Path:
            started.set()
            release.wait(5)
            made.append(create(name, temp_dir, **kwargs))
            created.set()
            return made[0]

        monkeypatch.setattr(runtime_files, "_create_temp_dir", slow_create)
        task = asyncio.create_task(async_session_temp_dir("session"))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await asyncio.to_thread(created.wait, 5)
        for _ in range(200):
            if not made[0].exists():
                break
            await asyncio.sleep(0.01)

        assert not made[0].exists()

    async def test_a_coroutine_removes_a_session_folder_off_the_event_loop(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        folder = session_temp_dir("session")
        (folder / "state.json").write_text("{}")
        threads: list[str] = []
        remove = runtime_files.remove_tree

        def recording_remove(path: str | Path) -> bool:
            threads.append(threading.current_thread().name)
            return remove(path)

        monkeypatch.setattr(runtime_files, "remove_tree", recording_remove)

        assert await async_remove_temp_dir(folder) is True
        assert not folder.exists()
        assert threads and threads[0].startswith("nemo-gym-temp-dir")
        assert await async_remove_temp_dir(tmp_path) is False
        assert tmp_path.exists()

    async def test_a_cancelled_caller_still_gets_its_session_folder_removed(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        folder = session_temp_dir("session")
        executor = ThreadPoolExecutor(max_workers=1)
        monkeypatch.setattr(runtime_files, "_temp_dir_executor", executor)
        gate = threading.Event()
        # The pool's only thread is busy, so the removal waits in its queue, as under load.
        busy = asyncio.get_running_loop().run_in_executor(executor, gate.wait, 10)
        try:
            removal = asyncio.create_task(async_remove_temp_dir(folder))
            await asyncio.sleep(0.05)
            removal.cancel()
            with pytest.raises(asyncio.CancelledError):
                await removal
        finally:
            gate.set()
            await busy
            executor.shutdown(wait=True)

        assert not folder.exists()

    def test_a_failed_removal_is_reported_and_a_later_removal_retries(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        folder = session_temp_dir("session")
        real_rmtree = shutil.rmtree

        def failing(path, *args, **kwargs):
            raise OSError(16, "busy")

        monkeypatch.setattr(shutil, "rmtree", failing)
        assert remove_temp_dir(folder) is False
        assert folder.exists()

        monkeypatch.setattr(shutil, "rmtree", real_rmtree)
        assert remove_temp_dir(folder) is True
        assert not folder.exists()

    def test_only_a_session_folder_of_this_process_is_removed(self, set_global_config, tmp_path: Path, caplog) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        session = session_temp_dir("session")
        process = process_temp_dir("cache")
        elsewhere = tmp_path / "results"
        elsewhere.mkdir()
        # Folders with the right shape under another process's folder are that process's to remove.
        other = tmp_path / "temp" / "nemo-gym-000000000000-1-1-00000000" / session.name
        other.mkdir(parents=True)
        # A request folder, and one a tool made beside the session folders, are not session folders.
        with request_temp_dir("session") as request_folder:
            pass
        request_folder.mkdir()
        tool_made = session.parent / "tool-output"
        tool_made.mkdir()

        with caplog.at_level("WARNING"):
            for path in (
                elsewhere,
                Path(""),
                session.parent,
                process,
                other,
                request_folder,
                tool_made,
                Path(session.name),
                # pathlib keeps "..", so this names temp_dir itself while its parent looks like a process folder.
                session.parent / "..",
                session.parent / "." / ".." / session.parent.name,
            ):
                assert remove_temp_dir(path) is False

        for path in (elsewhere, session, process, other, request_folder, tool_made):
            assert path.exists()
        assert "is not a session folder of this process" in caplog.text
        # Warned once in all, so a stream of wrong paths does not grow what the process remembers.
        assert caplog.text.count("is not a session folder") == 1
        assert len(runtime_files._warned) == 1


class TestProcessTempDir:
    def test_is_the_same_for_every_call_with_its_name(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})

        first = process_temp_dir("image-cache")
        assert process_temp_dir("image-cache") == first
        assert process_temp_dir("venvs") != first
        assert first.is_dir() and first.name == "process.image-cache"

    def test_concurrent_first_calls_get_the_same_folder(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path)})
        start = threading.Barrier(8)
        results: list[Path] = []

        def call() -> None:
            start.wait(5)
            results.append(process_temp_dir("image-cache"))

        threads = [threading.Thread(target=call) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert len(results) == 8 and len(set(results)) == 1
        assert [p.name for p in results[0].parent.iterdir()] == ["process.image-cache"]

    def test_a_folder_removed_from_outside_is_made_again(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path)})
        first = process_temp_dir("image-cache")
        shutil.rmtree(first.parent)

        assert process_temp_dir("image-cache") == first
        assert first.is_dir()

    @pytest.mark.parametrize("removed", ["process folder", "temp_dir"])
    def test_a_late_recovery_keeps_what_another_thread_made_again(
        self, set_global_config, tmp_path: Path, removed: str
    ) -> None:
        # Two threads hit the same outside removal; one makes the folders again before the other recovers.
        set_global_config({"temp_dir": str(tmp_path / "temp")})
        session = session_temp_dir("session")
        shutil.rmtree(session.parent if removed == "process folder" else tmp_path / "temp")
        again = session_temp_dir("session")
        (again / "state.json").write_text("{}")

        runtime_files._forget(tmp_path / "temp")
        late = session_temp_dir("session")

        assert (again / "state.json").exists()
        assert late.parent == again.parent


class TestProcessLifetime:
    def test_the_exit_handler_is_registered_once_for_this_process(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        registered: list[tuple] = []
        monkeypatch.setattr(runtime_files, "_exit_handler_registered", False)
        monkeypatch.setattr(atexit, "register", lambda *args: registered.append(args))
        set_global_config({"temp_dir": str(tmp_path)})

        process_temp_dir("cache")
        with request_temp_dir("workspace", temp_dir=tmp_path / "other"):
            pass

        assert registered == [(runtime_files._remove_own_process_dirs, os.getpid())]

    def test_exit_removes_this_process_folder_and_a_forked_child_does_not(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"temp_dir": str(tmp_path)})
        with request_temp_dir("workspace") as path:
            (path / "file.txt").write_text("x")
            process_folder = path.parent

            # A forked child inherits the exit handler and the record of this folder, but must leave it alone.
            real_pid = os.getpid()
            monkeypatch.setattr(os, "getpid", lambda: real_pid + 1)
            runtime_files._remove_own_process_dirs(real_pid)
            monkeypatch.setattr(os, "getpid", lambda: real_pid)
            assert process_folder.exists()

            runtime_files._remove_own_process_dirs(os.getpid())

        assert not process_folder.exists()


_CHILD = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    import nemo_gym.core.runtime_files as rf

    temp_dir = Path(sys.argv[1])
    process = rf.process_temp_dir("image-cache", temp_dir=temp_dir)
    (process / "a").write_text("x")
    session = rf.session_temp_dir("session", temp_dir=temp_dir)
    if sys.argv[2] == "exit":
        sys.exit(0)
    # Hold the lock, as another thread of the parent might at a fork, then fork and exit the child normally.
    rf._first_use_lock.acquire()
    parent_executors = (rf._save_executor, rf._temp_dir_executor)
    parent_token = rf._process_token
    pid = os.fork()
    if pid == 0:
        ok = rf._first_use_lock.acquire(blocking=False)
        ok = ok and rf._save_executor is not parent_executors[0] and rf._temp_dir_executor is not parent_executors[1]
        ok = ok and rf._process_token != parent_token
        ok = ok and not rf._checked_temp_parents and not rf._process_dirs and rf._default_temp_dir is None
        rf._first_use_lock.release()
        child = rf.session_temp_dir("child", temp_dir=temp_dir)
        ok = ok and child.parent != session.parent and not rf.remove_temp_dir(session)
        sys.exit(0 if ok else 3)
    _, status = os.waitpid(pid, 0)
    print(os.waitstatus_to_exitcode(status), process.exists(), session.exists(), flush=True)
    """
)


def _run_child(tmp_path: Path, mode: str) -> subprocess.CompletedProcess:
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    return subprocess.run(
        [sys.executable, str(script), str(tmp_path / "temp"), mode],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )


def test_a_process_removes_its_folders_when_it_exits(tmp_path: Path) -> None:
    result = _run_child(tmp_path, "exit")

    assert result.returncode == 0, result.stderr
    assert list((tmp_path / "temp").iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_gets_state_of_its_own_and_leaves_its_parents_folders(tmp_path: Path) -> None:
    result = _run_child(tmp_path, "fork")

    assert result.returncode == 0, result.stderr
    # The child's exit code, then whether the parent's process and session folders survived the child's exit.
    assert result.stdout.split() == ["0", "True", "True"]
    # The parent then exits and removes its own folder; the child removed its own.
    assert list((tmp_path / "temp").iterdir()) == []


class TestSaveDir:
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
            kept = await save_dir(logs, server_name="swebench", name="verification", label="astropy-1-attempt0")

        assert kept is not None
        assert kept.parent == tmp_path / "results" / "swebench" / "verification" / "4-2-a1"
        assert "-astropy-1-attempt0-" in kept.name
        assert (kept / "report.json").read_text() == "{}"
        assert (kept / "sub" / "test_output.txt").read_text() == "ok"
        # The source is left as it was; the temporary folder's own block removes it.
        assert (logs / "report.json").exists()

    async def test_copies_contents_with_the_users_usual_permissions(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        os.utime(logs / "report.json", (1, 1))
        os.chmod(logs / "report.json", 0o600)
        # Temporary folders are private; kept ones should be readable as the user's other files are.
        os.chmod(logs / "sub", 0o700)

        kept = await save_dir(logs, server_name="swebench", name="verification")

        umask = _umask()
        assert (kept / "report.json").stat().st_mtime != 1
        assert stat.S_IMODE((kept / "report.json").stat().st_mode) == 0o666 & ~umask
        assert stat.S_IMODE((kept / "sub").stat().st_mode) == 0o777 & ~umask

    async def test_symlinks_stay_symlinks_and_special_files_are_skipped(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        (logs / "link").symlink_to("/etc/hostname")
        os.mkfifo(logs / "pipe")

        kept = await save_dir(logs, server_name="swebench", name="verification")

        assert kept is not None
        assert os.readlink(kept / "link") == "/etc/hostname"
        assert not (kept / "pipe").exists()

    async def test_outside_a_rollout_and_on_repeat_each_copy_gets_its_own_folder(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)

        first = await save_dir(logs, server_name="swebench", name="verification", label="same")
        second = await save_dir(logs, server_name="swebench", name="verification", label="same")

        assert first != second
        assert first.parent == second.parent == tmp_path / "results" / "swebench" / "verification" / "no-rollout"

    async def test_the_server_output_dir_replaces_the_default(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})

        kept = await save_dir(
            self._logs(tmp_path), server_name="swebench", name="verification", output_dir=str(tmp_path / "mine")
        )

        assert kept is not None and kept.parent == tmp_path / "mine" / "verification" / "no-rollout"

    async def test_a_relative_output_dir_keeps_nothing_with_one_warning(
        self, set_global_config, tmp_path: Path, caplog
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)

        with caplog.at_level("WARNING"):
            assert await save_dir(logs, server_name="swebench", name="verification", output_dir="out") is None
            assert await save_dir(logs, server_name="swebench", name="verification", output_dir="out") is None

        assert caplog.text.count("is not an absolute path") == 1

    async def test_a_process_without_results_dir_keeps_nothing_with_one_warning(
        self, set_global_config, tmp_path: Path, caplog
    ) -> None:
        set_global_config(None)

        with caplog.at_level("WARNING"):
            assert await save_dir(self._logs(tmp_path), server_name="swebench", name="verification") is None
            assert await save_dir(tmp_path / "logs", server_name="swebench", name="verification") is None

        assert caplog.text.count("No results_dir") == 1

    async def test_a_failed_copy_returns_none_and_leaves_no_partial_folder(
        self, set_global_config, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        copied = []
        copyfileobj = shutil.copyfileobj

        def fail_second_copy(*args, **kwargs):
            if copied:
                raise OSError(28, "No space left on device")
            copied.append(copyfileobj(*args, **kwargs))

        monkeypatch.setattr(shutil, "copyfileobj", fail_second_copy)
        with caplog.at_level("WARNING"):
            kept = await save_dir(self._logs(tmp_path), server_name="swebench", name="verification")

        assert kept is None and copied
        assert not any((tmp_path / "results" / "swebench" / "verification" / "no-rollout").iterdir())
        assert caplog.text.count("Could not copy") == 1

    async def test_a_copy_appears_under_its_name_only_when_complete(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        seen: list[list[str]] = []
        copy = shutil.copyfileobj

        def watching_copy(reader, writer, *args, **kwargs):
            # What a reader of the results sees mid-copy, as when the process is killed here.
            kept_root = tmp_path / "results" / "swebench" / "verification" / "no-rollout"
            seen.append(sorted(path.name.rsplit("-", 1)[-1] for path in kept_root.iterdir()))
            return copy(reader, writer, *args, **kwargs)

        monkeypatch.setattr(shutil, "copyfileobj", watching_copy)
        kept = await save_dir(logs, server_name="swebench", name="verification")

        assert kept is not None and (kept / "report.json").exists()
        assert seen and all(names == [kept.name.rsplit("-", 1)[-1] + ".partial"] for names in seen)
        assert sorted(path.name for path in kept.parent.iterdir()) == [kept.name]

    async def test_a_failed_copy_never_removes_a_folder_it_did_not_create(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        existing = tmp_path / "results" / "swebench" / "verification" / "no-rollout" / "taken"
        existing.mkdir(parents=True)
        (existing / "earlier.log").write_text("x")
        monkeypatch.setattr(runtime_files, "_kept_dir", lambda *args: existing)

        assert await save_dir(self._logs(tmp_path), server_name="swebench", name="verification") is None
        assert (existing / "earlier.log").exists()

    async def test_a_copy_never_replaces_an_empty_folder_already_at_its_destination(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        # A rename onto an empty folder would replace it, so the copy must refuse before it starts.
        set_global_config({"results_dir": str(tmp_path / "results")})
        existing = tmp_path / "results" / "swebench" / "verification" / "no-rollout" / "taken"
        existing.mkdir(parents=True)
        inode = existing.stat().st_ino
        monkeypatch.setattr(runtime_files, "_kept_dir", lambda *args: existing)

        assert await save_dir(self._logs(tmp_path), server_name="swebench", name="verification") is None
        assert existing.stat().st_ino == inode and not any(existing.iterdir())
        assert sorted(path.name for path in existing.parent.iterdir()) == ["taken"]

    async def test_a_copy_that_fails_after_its_request_is_cancelled_is_logged(
        self, set_global_config, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        # The cancelled request no longer receives the copy's error, so the copy itself must report it.
        set_global_config({"results_dir": str(tmp_path / "results")})
        started, release = threading.Event(), threading.Event()

        def slow_failing_copy(*args, **kwargs):
            started.set()
            release.wait(5)
            raise OSError(2, "No such file or directory")

        monkeypatch.setattr(shutil, "copyfileobj", slow_failing_copy)
        with caplog.at_level("WARNING"):
            save = asyncio.create_task(save_dir(self._logs(tmp_path), server_name="swebench", name="verification"))
            await asyncio.to_thread(started.wait, 5)
            save.cancel()
            with pytest.raises(asyncio.CancelledError):
                await save
            release.set()
            for _ in range(200):
                if "Could not copy" in caplog.text:
                    break
                await asyncio.sleep(0.01)

        assert "Could not copy" in caplog.text

    async def test_a_copy_running_when_its_request_is_cancelled_stops_and_keeps_nothing(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        started, release = threading.Event(), threading.Event()
        copy = shutil.copyfileobj

        def slow_copy(*args, **kwargs):
            started.set()
            release.wait(5)
            return copy(*args, **kwargs)

        monkeypatch.setattr(shutil, "copyfileobj", slow_copy)
        save = asyncio.create_task(save_dir(logs, server_name="swebench", name="verification"))
        await asyncio.to_thread(started.wait, 5)
        save.cancel()
        with pytest.raises(asyncio.CancelledError):
            await save
        release.set()
        kept_root = tmp_path / "results" / "swebench" / "verification"
        for _ in range(200):
            if not any(kept_root.rglob("*-*")):
                break
            await asyncio.sleep(0.01)

        # The request may remove its source next, so a partial copy would look kept but miss files.
        assert list(kept_root.glob("*/*")) == []

    @NOT_ROOT
    async def test_what_cannot_be_read_is_skipped_and_the_rest_kept(
        self, set_global_config, tmp_path: Path, caplog
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        (logs / "unreadable.log").write_text("x")
        os.chmod(logs / "unreadable.log", 0)
        (logs / "locked").mkdir()
        os.chmod(logs / "locked", 0)

        try:
            with caplog.at_level("WARNING"):
                kept = await save_dir(logs, server_name="swebench", name="verification")
        finally:
            os.chmod(logs / "locked", 0o700)

        assert kept is not None
        assert (kept / "report.json").read_text() == "{}"
        assert (kept / "sub" / "test_output.txt").read_text() == "ok"
        assert not (kept / "unreadable.log").exists()
        assert "without 2 entries it could not read" in caplog.text

    async def test_a_failed_request_keeps_its_files_from_a_finally(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results"), "temp_dir": str(tmp_path / "temp")})
        kept = None

        with pytest.raises(RuntimeError):
            async with async_request_temp_dir("verification") as logs:
                try:
                    (logs / "test_output.txt").write_text("FAILED")
                    raise RuntimeError("harness failed")
                finally:
                    kept = await save_dir(logs, server_name="swebench", name="verification")

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
                kept.append(await save_dir(logs, server_name="swebench", name="verification"))

        task = asyncio.create_task(request())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert kept == [None]
        assert not (tmp_path / "results").exists()

    async def test_nothing_is_copied_when_an_anyio_cancel_scope_fires(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        kept: list[Path | None] = []

        with anyio.move_on_after(0.05):
            try:
                await anyio.sleep(3600)
            finally:
                kept.append(await save_dir(logs, server_name="swebench", name="verification"))

        assert kept == [None]
        assert not (tmp_path / "results").exists()

    async def test_a_save_after_an_expired_timeout_block_keeps_the_files(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        inside: list[Path | None] = []

        try:
            async with asyncio.timeout(0.01):
                try:
                    await asyncio.sleep(3600)
                finally:
                    # Inside the block the task counts as being cancelled, so nothing is kept.
                    inside.append(await save_dir(logs, server_name="swebench", name="verification"))
        except TimeoutError:
            after = await save_dir(logs, server_name="swebench", name="verification")

        assert inside == [None]
        assert after is not None and (after / "report.json").exists()

    async def test_copies_run_on_threads_of_their_own(self, set_global_config, tmp_path: Path, monkeypatch) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)
        loop = asyncio.get_running_loop()
        # A small default executor, so copies queued on it would starve other work.
        default = ThreadPoolExecutor(max_workers=4)
        loop.set_default_executor(default)
        lock, release = threading.Lock(), threading.Event()
        running = [0, 0]  # now, most at once
        copy = runtime_files.copy_tree

        def blocked_copy(src: Path, dest: Path, **kwargs: object) -> Path:
            with lock:
                running[0] += 1
                running[1] = max(running)
            release.wait(10)
            with lock:
                running[0] -= 1
            return copy(src, dest, **kwargs)

        monkeypatch.setattr(runtime_files, "copy_tree", blocked_copy)
        saves = [asyncio.create_task(save_dir(logs, server_name="swebench", name="verification")) for _ in range(8)]
        try:
            for _ in range(200):
                if running[0] >= runtime_files._SAVE_WORKERS:
                    break
                await asyncio.sleep(0.01)
            # Other file work, and DNS lookups, still get a thread while every copy waits.
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 2) == "free"
            assert running[1] == runtime_files._SAVE_WORKERS
        finally:
            release.set()
            kept = await asyncio.gather(*saves)
            default.shutdown(wait=True)

        assert all(path is not None for path in kept)

    async def test_a_long_rollout_id_still_keeps_the_files(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        # Within the rollout id length limit, but three times as long once percent-encoded.
        rollout = "/" * 200

        with rollout_context(rollout):
            kept = await save_dir(self._logs(tmp_path), server_name="swebench", name="verification")

        assert kept is not None and (kept / "report.json").exists()
        assert len(kept.parent.name.encode()) <= 255

    async def test_names_and_labels_stay_one_folder_under_the_root(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)

        with rollout_context("../escape"):
            kept = await save_dir(logs, server_name="../server", name="verification", label="../../org/repo")

        assert kept is not None
        assert kept.relative_to(tmp_path / "results").parts[0] == "..%2Fserver"
        assert ".." not in kept.parts

    @pytest.mark.parametrize("rollout", [".", ".."])
    async def test_a_dot_rollout_id_keeps_its_own_folder(
        self, set_global_config, tmp_path: Path, rollout: str
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})

        with rollout_context(rollout):
            kept = await save_dir(self._logs(tmp_path), server_name="swebench", name="verification")

        assert kept is not None
        assert kept.parent == tmp_path / "results" / "swebench" / "verification" / rollout.replace(".", "%2E")

    async def test_a_long_label_still_keeps_the_files(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})

        kept = await save_dir(self._logs(tmp_path), server_name="swebench", name="verification", label="x" * 300)

        assert kept is not None and len(kept.name.encode()) <= 255

    async def test_a_name_must_be_one_safe_part_and_a_server_name_must_name_a_server(
        self, set_global_config, tmp_path: Path
    ) -> None:
        set_global_config({"results_dir": str(tmp_path / "results")})
        logs = self._logs(tmp_path)

        with pytest.raises(ValueError, match="name"):
            await save_dir(logs, server_name="swebench", name="../escape")
        for server_name in ("", ".", ".."):
            with pytest.raises(ValueError, match="server_name"):
                await save_dir(logs, server_name=server_name, name="verification")


@LINUX_IDS
class TestStaleProcessDirs:
    def _dead_pid(self) -> int:
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait()
        return process.pid

    def _name(self, pid: int, *, host: str | None = None, namespace: int | None = None) -> str:
        host = host or Path("/proc/sys/kernel/random/boot_id").read_text().strip().replace("-", "")[:12]
        namespace = namespace or os.stat("/proc/self/ns/pid").st_ino
        return f"nemo-gym-{host}-{namespace}-{pid}-00000000"

    def test_only_folders_of_dead_processes_on_this_host_boot_and_namespace_are_removed(self, tmp_path: Path) -> None:
        namespace = os.stat("/proc/self/ns/pid").st_ino
        dead = tmp_path / self._name(self._dead_pid())
        live = tmp_path / self._name(os.getpid())
        # The dead process's request folders go with its process folder.
        (dead / "workspace-abc" / "nested").mkdir(parents=True)
        (dead / "workspace-abc" / "nested" / "file.txt").write_text("x")
        other_namespace = tmp_path / self._name(self._dead_pid(), namespace=namespace + 1)
        # Another host's or boot's folder on a shared directory: the same namespace number and a pid dead here.
        other_host = tmp_path / self._name(self._dead_pid(), host="0" * 12)
        unrelated = tmp_path / "something-else"
        for folder in (live, other_namespace, other_host, unrelated):
            folder.mkdir()

        assert runtime_files._remove_stale_process_dirs(tmp_path) == [dead]
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            p.name for p in (live, other_namespace, other_host, unrelated)
        )

    def test_first_folder_in_a_temp_dir_clears_leftovers(self, set_global_config, tmp_path: Path) -> None:
        leftover = tmp_path / self._name(self._dead_pid())
        leftover.mkdir()
        set_global_config({"temp_dir": str(tmp_path)})

        with request_temp_dir("workspace") as path:
            assert runtime_files._PROCESS_DIR.fullmatch(path.parent.name)["pid"] == str(os.getpid())

        assert not leftover.exists()

    def test_a_new_process_that_reuses_a_dead_pid_keeps_its_folder(self, tmp_path: Path, monkeypatch) -> None:
        pid = self._dead_pid()
        dead = tmp_path / self._name(pid)
        dead.mkdir()
        made: list[Path] = []

        def reused_after_the_check(checked: int) -> bool:
            # Between the liveness check and the removal, a new process gets the same pid and makes its folder.
            with monkeypatch.context() as patch:
                patch.setattr(os, "getpid", lambda: checked)
                folder = tmp_path / runtime_files._process_dir_name()
            folder.mkdir(exist_ok=True)
            (folder / "in-use").write_text("")
            made.append(folder)
            return False

        monkeypatch.setattr(runtime_files, "process_may_be_running", reused_after_the_check)

        assert runtime_files._remove_stale_process_dirs(tmp_path) == [dead]
        assert (made[0] / "in-use").exists()

    def test_an_unknown_host_removes_nothing(self, tmp_path: Path, monkeypatch) -> None:
        # Without a boot id no folder can be told apart from another host's.
        folder = tmp_path / self._name(self._dead_pid())
        folder.mkdir()
        monkeypatch.setattr(runtime_files, "_host_id", lambda: runtime_files._UNKNOWN_HOST)

        assert runtime_files._remove_stale_process_dirs(tmp_path) == []
        assert folder.exists()

    def test_a_missing_parent_removes_nothing(self, tmp_path: Path) -> None:
        assert runtime_files._remove_stale_process_dirs(tmp_path / "missing") == []

    def test_a_folder_name_with_an_impossible_pid_is_kept_and_never_fails_the_request(
        self, set_global_config, tmp_path: Path
    ) -> None:
        crafted = tmp_path / self._name(10**30)
        crafted.mkdir()
        set_global_config({"temp_dir": str(tmp_path)})

        with request_temp_dir("workspace") as path:
            assert path.exists()

        assert crafted.exists()

    def test_a_failing_cleanup_never_fails_the_request(self, set_global_config, tmp_path: Path, monkeypatch) -> None:
        def broken(parent: Path) -> list[Path]:
            raise OSError("listing failed")

        monkeypatch.setattr(runtime_files, "_remove_stale_process_dirs", broken)
        set_global_config({"temp_dir": str(tmp_path)})

        with request_temp_dir("workspace") as path:
            assert path.exists()

    def test_another_users_folders_stay(self, tmp_path: Path, monkeypatch) -> None:
        folder = tmp_path / self._name(self._dead_pid())
        folder.mkdir()
        real_uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

        assert runtime_files._remove_stale_process_dirs(tmp_path) == []
        assert folder.exists()

    def test_files_and_symlinks_with_a_matching_name_are_left_alone(self, tmp_path: Path) -> None:
        (tmp_path / self._name(self._dead_pid())).write_text("")
        target = tmp_path / "target"
        target.mkdir()
        (target / "keep").write_text("")
        (tmp_path / self._name(self._dead_pid())).symlink_to(target)

        assert runtime_files._remove_stale_process_dirs(tmp_path) == []
        assert (target / "keep").exists()

    @NOT_ROOT
    def test_a_process_this_user_cannot_signal_counts_as_running(self, tmp_path: Path) -> None:
        # PID 1 is alive in every namespace, and a process of another user answers with PermissionError.
        folder = tmp_path / self._name(1)
        folder.mkdir()

        assert runtime_files._remove_stale_process_dirs(tmp_path) == []
        assert folder.exists()
