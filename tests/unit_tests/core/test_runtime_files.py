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
from nemo_gym.config.config_types import BaseRunServerInstanceConfig
from nemo_gym.core.episode_context import rollout_context
from nemo_gym.core.runtime_files import (
    async_temp_folder,
    debug_folder,
    remove_stale_temp_folders,
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


class TestDebugFolder:
    def test_off_unless_the_global_or_server_switch_is_on(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_files": False, "results_dir": str(tmp_path)})
        assert debug_folder(server_name="swebench", kind="verification") is None
        assert debug_folder(server_name="swebench", kind="verification", debug_files=False) is None

        set_global_config(None)
        assert debug_folder(server_name="swebench", kind="verification") is None

    def test_lands_under_the_server_results_by_kind_and_rollout(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_files": True, "results_dir": str(tmp_path)})

        with rollout_context("4-2-a1"):
            path = debug_folder(server_name="swebench", kind="verification", label="astropy__astropy-1")

        assert path is not None
        assert path.parent == tmp_path / "swebench" / "debug" / "verification" / "4-2-a1"
        assert "-astropy__astropy-1-" in path.name
        # Nothing is created until the server writes into the folder.
        assert not path.exists()

    def test_outside_a_rollout_uses_a_fixed_segment(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_files": True, "results_dir": str(tmp_path)})

        path = debug_folder(server_name="vision2code", kind="render")

        assert path is not None
        assert path.parent == tmp_path / "vision2code" / "debug" / "render" / "no-rollout"

    def test_repeated_calls_in_one_rollout_get_separate_folders(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_files": True, "results_dir": str(tmp_path)})

        with rollout_context("0-0"):
            first = debug_folder(server_name="swebench", kind="verification", label="same")
            second = debug_folder(server_name="swebench", kind="verification", label="same")

        assert first != second
        assert first.parent == second.parent

    def test_label_with_slashes_stays_one_folder(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_files": True, "results_dir": str(tmp_path)})

        path = debug_folder(server_name="swebench", kind="verification", label="../../org/repo")

        assert path is not None
        assert path.parent == tmp_path / "swebench" / "debug" / "verification" / "no-rollout"
        assert "/" not in path.name

    def test_server_switch_overrides_the_global_one(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_files": False, "results_dir": str(tmp_path)})
        path = debug_folder(server_name="swebench", kind="verification", debug_files=True)
        assert path is not None
        assert path.is_relative_to(tmp_path / "swebench" / "debug")

        set_global_config({"debug_files": True, "results_dir": str(tmp_path)})
        assert debug_folder(server_name="swebench", kind="verification", debug_files=False) is None

    def test_every_server_config_carries_its_own_switch(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_files": False, "results_dir": str(tmp_path)})
        server = {"host": "127.0.0.1", "port": 8000, "entrypoint": "app.py", "name": "swebench"}

        follows_global = BaseRunServerInstanceConfig(**server)
        assert follows_global.debug_files is None
        assert (
            debug_folder(server_name="swebench", kind="verification", debug_files=follows_global.debug_files) is None
        )

        turned_on = BaseRunServerInstanceConfig(**server, debug_files=True)
        path = debug_folder(server_name="swebench", kind="verification", debug_files=turned_on.debug_files)
        assert path is not None and path.is_relative_to(tmp_path / "swebench" / "debug")

    def test_server_output_dir_replaces_the_default_output(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_files": True, "results_dir": str(tmp_path / "results")})

        path = debug_folder(server_name="swebench", kind="verification", output_dir=tmp_path / "mine")

        assert path is not None
        assert path.parent == tmp_path / "mine" / "debug" / "verification" / "no-rollout"

    def test_relative_output_dir_turns_debug_files_off_with_one_warning(
        self, set_global_config, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        # A relative path would land in the server's source folder; debug files never fail a request instead.
        monkeypatch.setattr(runtime_files, "_warned_relative_output_dirs", set())
        set_global_config({"debug_files": True, "results_dir": str(tmp_path)})

        with caplog.at_level("WARNING"):
            assert debug_folder(server_name="swebench", kind="verification", output_dir=Path("out")) is None
            assert debug_folder(server_name="swebench", kind="verification", output_dir="out") is None

        assert caplog.text.count("is not an absolute path") == 1

    def test_a_process_without_gym_configuration_writes_nothing(self, set_global_config) -> None:
        # A Ray task or helper process sees no results_dir; it gets its folder from the server instead.
        set_global_config(None)

        assert debug_folder(server_name="swebench", kind="verification", debug_files=True) is None

    def test_kind_must_be_one_safe_segment(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"debug_files": True, "results_dir": str(tmp_path)})

        with pytest.raises(ValueError, match="kind"):
            debug_folder(server_name="swebench", kind="../escape")


class TestTempFolder:
    def test_folder_is_under_the_temporary_directory_and_removed_on_exit(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        # The system temporary directory, pointed at a test folder so the host's own /tmp is left alone.
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
        set_global_config(None)

        with temp_folder(kind="workspace") as path:
            (path / "file.txt").write_text("x")
            assert path.parent.parent == Path(tempfile.gettempdir())

        assert not path.exists()

    def test_folder_is_under_temp_dir_when_set(self, set_global_config, tmp_path: Path) -> None:
        set_global_config({"temp_dir": str(tmp_path / "temp")})

        with temp_folder(kind="workspace") as path:
            assert path.parent.parent == tmp_path / "temp"

        assert _request_folders(tmp_path / "temp") == []

    def test_folder_is_removed_when_the_block_raises(self, set_global_config) -> None:
        set_global_config(None)

        with pytest.raises(RuntimeError):
            with temp_folder(kind="workspace") as path:
                (path / "file.txt").write_text("x")
                raise RuntimeError("boom")

        assert not path.exists()

    def test_keep_as_moves_the_folder_to_debug_files(self, set_global_config, tmp_path: Path) -> None:
        set_global_config(None)
        keep_as = tmp_path / "debug" / "verification" / "leaf"

        with temp_folder(kind="verification", keep_as=keep_as) as path:
            (path / "report.json").write_text("{}")

        assert not path.exists()
        assert (keep_as / "report.json").read_text() == "{}"

    def test_failing_to_keep_does_not_fail_the_block(self, set_global_config, tmp_path: Path) -> None:
        set_global_config(None)
        blocker = tmp_path / "not-a-folder"
        blocker.write_text("")

        with temp_folder(kind="verification", keep_as=blocker / "leaf") as path:
            (path / "report.json").write_text("{}")

        assert not path.exists()


def test_temp_dir_is_checked_once_on_the_writing_node(set_global_config, tmp_path: Path, monkeypatch) -> None:
    checked: list[tuple[str, Path]] = []
    monkeypatch.setattr(runtime_files, "warn_if_unsuitable", lambda setting, path: checked.append((setting, path)))
    monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
    set_global_config({"temp_dir": str(tmp_path / "temp")})

    with temp_folder(kind="workspace"):
        pass
    with temp_folder(kind="workspace"):
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

    def test_first_folder_in_a_temp_dir_clears_leftovers(self, set_global_config, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
        leftover = tmp_path / self._name(self._dead_pid())
        leftover.mkdir()
        set_global_config({"temp_dir": str(tmp_path)})

        with temp_folder(kind="workspace") as path:
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

        with temp_folder(kind="workspace"):
            pass

        assert not leftover.exists()

    def test_a_folder_name_with_an_impossible_pid_is_kept_and_never_fails_the_request(
        self, set_global_config, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())
        crafted = tmp_path / self._name(10**30)
        crafted.mkdir()
        set_global_config({"temp_dir": str(tmp_path)})

        with temp_folder(kind="workspace") as path:
            assert path.exists()

        assert crafted.exists()

    def test_a_failing_cleanup_never_fails_the_request(self, set_global_config, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(runtime_files, "_checked_temp_parents", set())

        def broken(parent: Path) -> list[Path]:
            raise OSError("listing failed")

        monkeypatch.setattr(runtime_files, "remove_stale_temp_folders", broken)
        set_global_config({"temp_dir": str(tmp_path)})

        with temp_folder(kind="workspace") as path:
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

    with temp_folder(kind="workspace") as first, temp_folder(kind="harness") as second:
        assert first.parent == second.parent
        assert first.parent.parent == tmp_path
        assert first.name.startswith("workspace-") and second.name.startswith("harness-")

    assert [p.name for p in tmp_path.iterdir()] == [first.parent.name]


def test_exit_removes_this_process_folder_and_a_forked_child_does_not(
    set_global_config, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runtime_files, "_process_folders", {})
    set_global_config({"temp_dir": str(tmp_path)})
    with temp_folder(kind="workspace") as path:
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

    with temp_folder(kind="workspace") as path:
        assert path.parent == stale

    assert not (stale / "workspace-old").exists()


def test_parent_replaces_temp_dir_for_code_without_gym_configuration(set_global_config, tmp_path: Path) -> None:
    set_global_config(None)

    with temp_folder(kind="harbor-job", parent=tmp_path / "from-server") as path:
        assert path.parent.parent == tmp_path / "from-server"

    assert not path.exists()


class TestAsyncTempFolder:
    async def test_folder_is_removed_when_the_task_is_cancelled(self, set_global_config) -> None:
        set_global_config(None)
        created: list[Path] = []
        entered = asyncio.Event()

        async def work() -> None:
            async with async_temp_folder(kind="workspace") as path:
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
        async with async_temp_folder(kind="workspace"):
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


def test_debug_folder_segments_are_quoted(set_global_config, tmp_path: Path) -> None:
    set_global_config({"debug_files": True, "results_dir": str(tmp_path)})

    with rollout_context("../escape"):
        path = debug_folder(server_name="swebench", kind="verification")

    assert path is not None
    assert path.is_relative_to(tmp_path / "swebench" / "debug" / "verification")
    assert ".." not in path.parts


async def test_folder_is_removed_when_an_anyio_cancel_scope_fires(set_global_config, tmp_path: Path) -> None:
    import anyio

    set_global_config({"temp_dir": str(tmp_path)})

    with anyio.move_on_after(0.05):
        async with async_temp_folder(kind="workspace"):
            await anyio.sleep(3600)

    for _ in range(100):
        if not _request_folders(tmp_path):
            break
        await asyncio.sleep(0.01)
    assert _request_folders(tmp_path) == []


def test_debug_folder_rejects_a_server_name_that_leaves_the_root(set_global_config, tmp_path: Path) -> None:
    set_global_config({"debug_files": True, "results_dir": str(tmp_path)})

    for name in ("", ".", ".."):
        with pytest.raises(ValueError, match="server_name"):
            debug_folder(server_name=name, kind="verification")


def test_debug_folder_accepts_a_string_output_dir(set_global_config, tmp_path: Path) -> None:
    set_global_config({"debug_files": True})

    path = debug_folder(server_name="swebench", kind="verification", output_dir=str(tmp_path))

    assert path is not None and path.is_relative_to(tmp_path / "debug")


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
                async with async_temp_folder(kind="workspace") as path:
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


def test_a_server_name_with_a_slash_stays_one_segment(set_global_config, tmp_path: Path) -> None:
    set_global_config({"debug_files": True, "results_dir": str(tmp_path)})

    path = debug_folder(server_name="../escape", kind="verification")

    assert path is not None
    assert path.relative_to(tmp_path).parts[0] == "..%2Fescape"


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
            with temp_folder(kind="workspace", parent=tmp_path) as path:
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
    with temp_folder(kind="workspace") as first:
        pass
    shutil.rmtree(first.parent)

    with temp_folder(kind="workspace") as second:
        assert second.parent == first.parent


def test_the_server_name_is_checked_only_where_it_is_used(set_global_config, tmp_path: Path) -> None:
    set_global_config({"debug_files": False, "results_dir": str(tmp_path)})
    assert debug_folder(server_name="", kind="verification") is None

    path = debug_folder(server_name="", kind="verification", debug_files=True, output_dir=tmp_path)
    assert path is not None and path.is_relative_to(tmp_path)


def test_a_forked_child_gets_a_free_first_use_lock(monkeypatch) -> None:
    held = threading.Lock()
    held.acquire()
    monkeypatch.setattr(runtime_files, "_first_use_lock", held)

    # What os.register_at_fork runs in the child after a fork.
    runtime_files._reset_first_use_lock()

    assert runtime_files._first_use_lock is not held
    assert runtime_files._first_use_lock.acquire(blocking=False)
