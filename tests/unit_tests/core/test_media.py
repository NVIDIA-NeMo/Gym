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
from types import SimpleNamespace

import pytest

from nemo_gym.core.media import MediaFile, UnsupportedContentError, media_type_for_path, split_text_and_media


PNG = MediaFile("image/png", b"\x89PNG\r\n\x1a\n\x00\xff")
JPEG = MediaFile("image/jpeg", b"\xff\xd8\xff\xe0\x00")


def test_data_uri_round_trip() -> None:
    uri = PNG.to_data_uri()
    assert uri.startswith("data:image/png;base64,")
    assert MediaFile.from_data_uri(uri) == PNG


@pytest.mark.parametrize(
    ("header", "media_type"),
    [
        ("image/PNG;base64", "image/png"),
        ("image/jpg;base64", "image/jpeg"),
        ("image/png;charset=utf-8;base64", "image/png"),
    ],
)
def test_data_uri_accepts_other_spellings_of_a_supported_type(header: str, media_type: str) -> None:
    payload = PNG.to_data_uri().partition(",")[2]
    assert MediaFile.from_data_uri(f"data:{header},{payload}") == MediaFile(media_type, PNG.data)


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


@pytest.mark.parametrize(
    ("path", "media_type"),
    [("/a/b.png", "image/png"), ("x.JPG", "image/jpeg"), ("x.jpeg", "image/jpeg"), ("x.webp", "image/webp")],
)
def test_media_type_comes_from_the_extension(path: str, media_type: str) -> None:
    assert media_type_for_path(path) == media_type


def test_an_unknown_extension_names_no_media_type() -> None:
    with pytest.raises(UnsupportedContentError, match="notes.txt"):
        media_type_for_path("/notes.txt")
