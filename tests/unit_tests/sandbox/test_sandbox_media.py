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
import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

import anyio
import pytest

import nemo_gym.core
import nemo_gym.sandbox
import nemo_gym.sandbox.utils
from nemo_gym.core.media import MediaFile
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.transfer import SandboxTransferError, SandboxUnavailableError
from nemo_gym.sandbox.utils import MEDIA_DIRECTORY, media_path, stage_media


PNG = MediaFile(
    "image/png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
)
JPEG = MediaFile("image/jpeg", b"\xff\xd8\xff\xe0\x00")


class DirectorySandbox:
    """Maps sandbox paths into a local directory, as file transfer would, and runs commands there with bash.

    Uploads keep the source file's mode, as copying providers do, and need the parent directory to exist,
    so staging must create it before uploading.
    """

    def __init__(
        self, root: Path, *, fail_upload_after: int | None = None, fail_rename_to: str = "", bsd_mv: bool = False
    ) -> None:
        self.root = root
        self.fail_upload_after = fail_upload_after
        self.fail_rename_to = fail_rename_to
        self.bsd_mv = bsd_mv
        self.uploads = 0
        self.commands: list[tuple[str, float | None]] = []

    def local(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    async def upload(self, source: Path, destination: str) -> None:
        if self.fail_upload_after is not None and self.uploads >= self.fail_upload_after:
            raise ConnectionError("upload failed")
        self.uploads += 1
        target = self.local(destination)
        data = source.read_bytes()
        # Write in two halves with a yield between, so a reader of the destination could see a partial file.
        target.write_bytes(data[: len(data) // 2])
        await asyncio.sleep(0)
        with target.open("ab") as handle:
            handle.write(data[len(data) // 2 :])
        shutil.copymode(source, target)

    async def exec(self, command: str, *, cwd: str, timeout_s: float | None = None) -> SandboxExecResult:
        self.commands.append((command, timeout_s))
        # A real exec awaits the provider, so a cancellation can reach it here.
        await asyncio.sleep(0)
        local_command = command.replace(MEDIA_DIRECTORY, str(self.local(MEDIA_DIRECTORY)))
        if self.fail_rename_to:
            # Fail the rename onto one target, as a full disk or a permission error would.
            local_command = (
                f'mv() {{ case "$4" in *{self.fail_rename_to}) return 1;; esac; command mv "$@"; }}; {local_command}'
            )
        if self.bsd_mv:
            # BSD mv, as on macOS, has no -T and rejects it.
            local_command = (
                'mv() { case "$1" in -*T*) echo "mv: illegal option -- T" >&2; return 64;; esac; command mv "$@"; }; '
                + local_command
            )
        completed = subprocess.run(["bash", "-c", local_command], capture_output=True, text=True)
        return SandboxExecResult(stdout=completed.stdout, stderr=completed.stderr, return_code=completed.returncode)

    def media_files(self) -> list[str]:
        return sorted(path.name for path in self.local(MEDIA_DIRECTORY).iterdir())

    def partial_files(self) -> list[str]:
        return [name for name in self.media_files() if name.endswith(".partial")]


def is_rename(command: str) -> bool:
    return " mv -f -- " in command


def test_stage_names_each_file_by_its_content_and_returns_the_paths_in_order(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    paths = asyncio.run(stage_media(sandbox, [JPEG, PNG, PNG]))
    png_path = f"{MEDIA_DIRECTORY}/{hashlib.sha256(PNG.data).hexdigest()}.png"
    jpeg_path = f"{MEDIA_DIRECTORY}/{hashlib.sha256(JPEG.data).hexdigest()}.jpg"
    assert paths == [jpeg_path, png_path, png_path] == [media_path(JPEG), media_path(PNG), media_path(PNG)]
    assert [sandbox.local(path).read_bytes() for path in paths] == [JPEG.data, PNG.data, PNG.data]
    assert sandbox.uploads == 2
    assert sandbox.media_files() == sorted(Path(path).name for path in {png_path, jpeg_path})


def test_the_default_user_creates_the_media_directory_before_any_upload(tmp_path: Path) -> None:
    # Uploads may run as another user, such as root on Docker; the rename needs a directory the exec user can write.
    sandbox = DirectorySandbox(tmp_path)
    asyncio.run(stage_media(sandbox, [PNG]))
    assert sandbox.commands[0][0] == f"mkdir -p -- {MEDIA_DIRECTORY}"


def test_staged_files_are_readable_by_every_user_under_a_private_umask(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    previous = os.umask(0o077)
    try:
        [path] = asyncio.run(stage_media(sandbox, [PNG]))
    finally:
        os.umask(previous)
    assert sandbox.local(path).stat().st_mode & 0o777 == 0o644


def test_concurrent_stagings_of_one_image_never_expose_a_partial_file(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    target = sandbox.local(media_path(PNG))
    seen = []

    async def watch() -> None:
        # A session reading the shared path must see either no file or the whole image.
        while True:
            if target.exists():
                seen.append(target.read_bytes())
            await asyncio.sleep(0)

    async def run() -> list[list[str]]:
        watcher = asyncio.create_task(watch())
        try:
            return await asyncio.gather(stage_media(sandbox, [PNG]), stage_media(sandbox, [PNG]))
        finally:
            watcher.cancel()

    first, second = asyncio.run(run())
    assert first == second == [media_path(PNG)]
    assert target.read_bytes() == PNG.data
    assert seen and all(data == PNG.data for data in seen)
    assert sandbox.media_files() == [target.name]


@pytest.mark.parametrize("failing", [PNG, JPEG])
def test_a_failed_rename_anywhere_in_the_chain_removes_the_partial_files(tmp_path: Path, failing: MediaFile) -> None:
    sandbox = DirectorySandbox(tmp_path, fail_rename_to=Path(media_path(failing)).name)
    with pytest.raises(SandboxTransferError, match="Cannot rename the staged media files"):
        asyncio.run(stage_media(sandbox, [PNG, JPEG]))
    assert sandbox.partial_files() == []


def test_a_directory_at_the_target_path_fails_the_staging(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    sandbox.local(media_path(PNG)).mkdir(parents=True)
    with pytest.raises(SandboxTransferError, match="is a directory"):
        asyncio.run(stage_media(sandbox, [PNG]))
    assert sandbox.partial_files() == []
    assert list(sandbox.local(media_path(PNG)).iterdir()) == []


def test_staging_works_with_an_mv_that_has_no_target_directory_option(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path, bsd_mv=True)
    paths = asyncio.run(stage_media(sandbox, [PNG, JPEG]))
    assert [sandbox.local(path).read_bytes() for path in paths] == [PNG.data, JPEG.data]
    assert sandbox.partial_files() == []


@pytest.mark.parametrize("fail_upload", ["before writing", "partway"])
def test_a_failed_upload_removes_the_partial_files(tmp_path: Path, fail_upload: str) -> None:
    class Sandbox(DirectorySandbox):
        async def upload(self, source: Path, destination: str) -> None:
            if self.uploads == 1:
                if fail_upload == "partway":
                    self.local(destination).write_bytes(source.read_bytes()[:4])
                raise ConnectionError("upload failed")
            await super().upload(source, destination)

    sandbox = Sandbox(tmp_path)
    with pytest.raises(ConnectionError):
        asyncio.run(stage_media(sandbox, [PNG, JPEG]))
    assert sandbox.media_files() == []


class RenameSandbox(DirectorySandbox):
    """Runs every command except the rename, whose outcome the test chooses."""

    def __init__(self, root: Path, rename: SandboxExecResult | BaseException | asyncio.Event) -> None:
        super().__init__(root)
        self.rename = rename

    async def exec(self, command: str, *, cwd: str, timeout_s: float | None = None) -> SandboxExecResult:
        if not is_rename(command):
            return await super().exec(command, cwd=cwd, timeout_s=timeout_s)
        self.commands.append((command, timeout_s))
        if isinstance(self.rename, asyncio.Event):
            await self.rename.wait()
        if isinstance(self.rename, BaseException):
            raise self.rename
        return self.rename


@pytest.mark.parametrize(
    ("rename", "error"),
    [
        (
            SandboxExecResult(stdout=None, stderr="deadline", return_code=124, error_type="timeout"),
            SandboxUnavailableError,
        ),
        (
            SandboxExecResult(stdout=None, stderr="lost", return_code=0, error_type="sandbox_error"),
            SandboxUnavailableError,
        ),
        (SandboxExecResult(stdout=None, stderr="denied", return_code=1), SandboxTransferError),
        (ConnectionError("sandbox gone"), ConnectionError),
    ],
)
def test_a_failed_rename_raises_by_its_cause_and_removes_the_partial_files(
    tmp_path: Path, rename: SandboxExecResult | BaseException, error: type[BaseException]
) -> None:
    sandbox = RenameSandbox(tmp_path, rename)
    with pytest.raises(error):
        asyncio.run(stage_media(sandbox, [PNG]))
    assert sandbox.partial_files() == []
    # The cleanup is bounded, so an unresponsive sandbox cannot hold the caller.
    assert sandbox.commands[-1][0].startswith("rm -f -- ") and sandbox.commands[-1][1] == 30


def test_a_cancelled_staging_removes_the_partial_files(tmp_path: Path) -> None:
    async def run() -> RenameSandbox:
        sandbox = RenameSandbox(tmp_path, asyncio.Event())
        task = asyncio.create_task(stage_media(sandbox, [PNG]))
        while not task.done() and not any(is_rename(command) for command, _ in sandbox.commands):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return sandbox

    assert asyncio.run(run()).partial_files() == []


def test_a_staging_cancelled_by_an_anyio_scope_removes_the_partial_files(tmp_path: Path) -> None:
    # An anyio scope stays cancelled, as when a request is cancelled on client disconnect, so the cleanup must be
    # shielded to run at all.
    async def run() -> RenameSandbox:
        sandbox = RenameSandbox(tmp_path, asyncio.Event())
        async with anyio.create_task_group() as group:
            group.start_soon(stage_media, sandbox, [PNG])
            with anyio.fail_after(5):
                while not any(is_rename(command) for command, _ in sandbox.commands):
                    await asyncio.sleep(0)
            group.cancel_scope.cancel()
        return sandbox

    assert asyncio.run(run()).partial_files() == []


def test_a_cleanup_that_does_not_finish_is_abandoned_after_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(nemo_gym.sandbox.utils, "_CLEANUP_TIMEOUT_S", 0.05)

    class Sandbox(RenameSandbox):
        async def exec(self, command: str, *, cwd: str, timeout_s: float | None = None) -> SandboxExecResult:
            if command.startswith("rm "):
                await asyncio.Event().wait()
            return await super().exec(command, cwd=cwd, timeout_s=timeout_s)

    sandbox = Sandbox(tmp_path, SandboxExecResult(stdout=None, stderr="denied", return_code=1))
    with pytest.raises(SandboxTransferError):
        asyncio.run(asyncio.wait_for(stage_media(sandbox, [PNG]), timeout=5))
    assert "Gave up removing partial media uploads" in caplog.text


def test_a_failing_cleanup_keeps_the_original_error_and_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    class Sandbox(DirectorySandbox):
        async def exec(self, command: str, *, cwd: str, timeout_s: float | None = None) -> SandboxExecResult:
            if command.startswith("rm "):
                return SandboxExecResult(stdout=None, stderr="rm: permission denied", return_code=1)
            return await super().exec(command, cwd=cwd, timeout_s=timeout_s)

    with caplog.at_level(logging.WARNING), pytest.raises(ConnectionError, match="upload failed"):
        asyncio.run(stage_media(Sandbox(tmp_path, fail_upload_after=0), [PNG]))
    assert "rm: permission denied" in caplog.text


def test_a_cleanup_that_raises_keeps_the_original_error(tmp_path: Path) -> None:
    class Sandbox(DirectorySandbox):
        async def exec(self, command: str, *, cwd: str, timeout_s: float | None = None) -> SandboxExecResult:
            if command.startswith("rm "):
                raise ConnectionError("sandbox gone")
            return await super().exec(command, cwd=cwd, timeout_s=timeout_s)

    with pytest.raises(ConnectionError, match="upload failed"):
        asyncio.run(stage_media(Sandbox(tmp_path, fail_upload_after=0), [PNG]))


def test_a_media_directory_that_cannot_be_created_fails_before_any_upload(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    sandbox.local(MEDIA_DIRECTORY).parent.mkdir(parents=True)
    sandbox.local(MEDIA_DIRECTORY).write_text("a file, not a directory")
    with pytest.raises(SandboxTransferError, match="Cannot create the media directory"):
        asyncio.run(stage_media(sandbox, [PNG]))
    assert sandbox.uploads == 0


def test_staging_nothing_runs_no_command(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    assert asyncio.run(stage_media(sandbox, [])) == []
    assert sandbox.commands == []


def test_the_staging_helpers_are_public() -> None:
    assert nemo_gym.sandbox.stage_media is stage_media
    assert nemo_gym.sandbox.media_path is media_path
    assert nemo_gym.sandbox.MEDIA_DIRECTORY == MEDIA_DIRECTORY
    assert {"download_bytes", "upload_bytes"} <= set(dir(nemo_gym.sandbox))
    assert nemo_gym.core.MediaFile is MediaFile
    assert callable(nemo_gym.core.split_text_and_media)
