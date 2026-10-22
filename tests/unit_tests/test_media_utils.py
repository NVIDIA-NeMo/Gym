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
from types import SimpleNamespace

import pytest

from nemo_gym.media_utils import (
    MediaFile,
    UnsupportedContentError,
    download_media,
    split_text_and_media,
    upload_media,
)


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

    async def download(self, source: str, destination: Path) -> None:
        shutil.copyfile(self.local(source), destination)


def test_data_uri_round_trip() -> None:
    uri = PNG.to_data_uri()
    assert uri.startswith("data:image/png;base64,")
    assert MediaFile.from_data_uri(uri) == PNG


def test_split_keeps_text_and_decodes_media_in_order() -> None:
    content = [
        {"type": "input_text", "text": "first"},
        {"type": "input_image", "image_url": PNG.to_data_uri(), "detail": "auto"},
        SimpleNamespace(type="input_text", text="second"),
        {"type": "input_image", "image_url": {"url": JPEG.to_data_uri()}},
    ]
    assert split_text_and_media(content) == ("first\nsecond", [PNG, JPEG])
    assert split_text_and_media("just text") == ("just text", [])


@pytest.mark.parametrize(
    ("part", "message"),
    [
        ({"type": "input_image", "image_url": "https://example.com/a.png"}, "base64 data URI"),
        ({"type": "input_image", "file_id": "file-1"}, "base64 data URI"),
        ({"type": "input_image", "image_url": "data:image/tiff;base64,AAAA"}, "Unsupported media type"),
        ({"type": "input_image", "image_url": "data:image/png,AAAA"}, "base64 data URI"),
        ({"type": "input_image", "image_url": "data:image/png;base64,@@@"}, "invalid base64"),
        ({"type": "input_file", "file_data": "AAAA"}, "input_file"),
        ({"type": "input_audio"}, "input_audio"),
    ],
)
def test_split_rejects_content_it_cannot_pass_on(part: dict, message: str) -> None:
    with pytest.raises(UnsupportedContentError, match=message):
        split_text_and_media([{"type": "input_text", "text": "look"}, part])


def test_upload_and_download_round_trip(tmp_path: Path) -> None:
    sandbox = DirectorySandbox(tmp_path)
    paths = asyncio.run(upload_media(sandbox, [PNG, JPEG], "/session/media"))
    assert paths == ["/session/media/media-0.png", "/session/media/media-1.jpg"]
    assert sandbox.local(paths[0]).read_bytes() == PNG.data
    assert [asyncio.run(download_media(sandbox, path)) for path in paths] == [PNG, JPEG]


def test_download_rejects_an_unknown_extension(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not media")
    with pytest.raises(UnsupportedContentError, match="notes.txt"):
        asyncio.run(download_media(DirectorySandbox(tmp_path), "/notes.txt"))
