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
import subprocess
from pathlib import Path

import pytest

from nemo_gym.core.media import MediaFile
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.utils import MEDIA_DIRECTORY, media_path, stage_media


PNG = MediaFile(
    "image/png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
)
JPEG = MediaFile("image/jpeg", b"\xff\xd8\xff\xe0\x00")


class DirectorySandbox:
    """Maps sandbox paths into a local directory, as file transfer would, and runs commands there with bash."""

    def __init__(self, root: Path, *, fail_upload_after: int | None = None, fail_rename_to: str = "") -> None:
        self.root = root
        self.fail_upload_after = fail_upload_after
        self.fail_rename_to = fail_rename_to
        self.uploads = 0

    def local(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    async def upload(self, source: Path, destination: str) -> None:
        if self.fail_upload_after is not None and self.uploads >= self.fail_upload_after:
            raise ConnectionError("upload failed")
        self.uploads += 1
        target = self.local(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        # Write in two halves with a yield between, so a reader of the destination could see a partial file.
        target.write_bytes(data[: len(data) // 2])
        await asyncio.sleep(0)
        with target.open("ab") as handle:
            handle.write(data[len(data) // 2 :])

    async def exec(self, command: str, *, cwd: str) -> SandboxExecResult:
        local_command = command.replace(MEDIA_DIRECTORY, str(self.local(MEDIA_DIRECTORY)))
        if self.fail_rename_to:
            # Fail the rename onto one target, as a full disk or a permission error would.
            local_command = (
                f'mv() {{ case "$4" in *{self.fail_rename_to}) return 1;; esac; command mv "$@"; }}; {local_command}'
            )
        completed = subprocess.run(["bash", "-c", local_command], capture_output=True, text=True)
        return SandboxExecResult(stdout=completed.stdout, stderr=completed.stderr, return_code=completed.returncode)

    def media_files(self) -> list[str]:
        return sorted(path.name for path in self.local(MEDIA_DIRECTORY).iterdir())


def test_stage_names_each_file_by_its_content_and_returns_the_paths_in_order(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    paths = asyncio.run(stage_media(sandbox, [PNG, JPEG, PNG]))
    png_path = f"{MEDIA_DIRECTORY}/{hashlib.sha256(PNG.data).hexdigest()}.png"
    jpeg_path = f"{MEDIA_DIRECTORY}/{hashlib.sha256(JPEG.data).hexdigest()}.jpg"
    assert paths == [png_path, jpeg_path, png_path] == [media_path(PNG), media_path(JPEG), media_path(PNG)]
    assert [sandbox.local(path).read_bytes() for path in paths] == [PNG.data, JPEG.data, PNG.data]
    assert sandbox.uploads == 2
    assert sandbox.media_files() == sorted(Path(path).name for path in {png_path, jpeg_path})


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


def test_failed_upload_removes_the_partial_files(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path, fail_upload_after=1)
    with pytest.raises(ConnectionError):
        asyncio.run(stage_media(sandbox, [PNG, JPEG]))
    assert sandbox.media_files() == []


def test_failed_rename_removes_the_partial_files_and_raises(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path, fail_rename_to=Path(media_path(PNG)).name)
    with pytest.raises(RuntimeError, match="Cannot stage media files"):
        asyncio.run(stage_media(sandbox, [JPEG, PNG]))
    assert [name for name in sandbox.media_files() if name.endswith(".partial")] == []


def test_staging_nothing_runs_no_command(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    assert asyncio.run(stage_media(sandbox, [])) == []
    assert not sandbox.local(MEDIA_DIRECTORY).exists()
