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
import shutil
from pathlib import Path

from nemo_gym.core.media import MediaFile
from nemo_gym.sandbox.utils import upload_media


PNG = MediaFile("image/png", b"\x89PNG\r\n\x1a\n\x00\xff")
JPEG = MediaFile("image/jpeg", b"\xff\xd8\xff\xe0\x00")


class DirectorySandbox:
    """Maps sandbox paths into a local directory, as file transfer would."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def local(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    async def upload(self, source: Path, destination: str) -> None:
        self.local(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, self.local(destination))


def test_upload_writes_each_file_with_its_extension_in_order(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    paths = asyncio.run(upload_media(sandbox, files=[PNG, JPEG], directory="/session/media"))
    assert paths == ["/session/media/media-0.png", "/session/media/media-1.jpg"]
    assert [sandbox.local(path).read_bytes() for path in paths] == [PNG.data, JPEG.data]
