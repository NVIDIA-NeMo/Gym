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
import base64
from types import SimpleNamespace

import pytest

from nemo_gym.core.media import MediaFile, UnsupportedContentError, split_text_and_media


PNG = MediaFile(
    "image/png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
)
JPEG = MediaFile("image/jpeg", b"\xff\xd8\xff\xe0\x00")
PAYLOAD = PNG.to_data_uri().partition(",")[2]


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
    payload = JPEG.to_data_uri().partition(",")[2] if media_type == "image/jpeg" else PAYLOAD
    assert MediaFile.from_data_uri(f"data:{header},{payload}").media_type == media_type


@pytest.mark.parametrize(
    "uri",
    [f"image/png;base64,{PAYLOAD}", f"data:image/png;base64;charset=utf-8,{PAYLOAD}"],
)
def test_data_uri_must_start_with_data_and_end_its_header_with_base64(uri: str) -> None:
    with pytest.raises(UnsupportedContentError, match="base64 data URI"):
        MediaFile.from_data_uri(uri)


@pytest.mark.parametrize(
    ("data", "media_type"),
    [
        (PNG.data, "image/png"),
        (JPEG.data, "image/jpeg"),
        (b"GIF89a\x01\x00", "image/gif"),
        (b"RIFF\x10\x00\x00\x00WEBPVP8 ", "image/webp"),
    ],
)
def test_the_content_decides_the_media_type_of_a_mislabeled_file(data: bytes, media_type: str) -> None:
    # Gym's own data labels raw images as image/png whatever their format, and harnesses such as Pi
    # read the type from the bytes, so the file must be named for what it holds.
    uri = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
    assert MediaFile.from_data_uri(uri) == MediaFile(media_type, data)


@pytest.mark.parametrize("data", [b"", b"plain text, not an image", b"\x89PNG\r\n\x1a\n\x00\xff"])
def test_data_uri_must_hold_data_of_a_supported_media_type(data: bytes) -> None:
    # Pi skips an empty file and reads one it does not recognize as text, such as a PNG without its IHDR chunk,
    # so neither reaches the model as media.
    with pytest.raises(UnsupportedContentError, match="no data of a supported media type"):
        MediaFile.from_data_uri(f"data:image/png;base64,{base64.b64encode(data).decode()}")


def test_media_file_holds_data_of_its_own_type() -> None:
    with pytest.raises(UnsupportedContentError, match="not image/jpeg"):
        MediaFile("image/jpeg", PNG.data)


def test_repr_leaves_out_the_data() -> None:
    assert repr(PNG) == "MediaFile(media_type='image/png')"


def test_split_keeps_text_and_decodes_media_in_order() -> None:
    content = [
        {"type": "input_text", "text": "first"},
        {"type": "input_image", "image_url": PNG.to_data_uri(), "detail": "auto"},
        SimpleNamespace(type="input_text", text="second"),
        {"type": "input_image", "image_url": {"url": JPEG.to_data_uri()}},
    ]
    assert split_text_and_media(content) == ("first\nsecond", [PNG, JPEG])
    assert split_text_and_media("just text") == ("just text", [])


def test_split_accepts_text_parts_and_reads_a_missing_text_as_empty() -> None:
    content = [{"type": "text", "text": "first"}, {"type": "input_text"}, {"type": "input_text", "text": "last"}]
    assert split_text_and_media(content) == ("first\n\nlast", [])


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
    "data",
    [
        b"GIF87a\x01\x00",
        # A RIFF size of 10 has the byte \n, which a signature without re.DOTALL would not match.
        b"RIFF\x0a\x00\x00\x00WEBPVP8 ",
    ],
)
def test_every_variant_of_a_signature_is_accepted(data: bytes) -> None:
    assert MediaFile.from_data_uri(f"data:image/png;base64,{base64.b64encode(data).decode()}").data == data


@pytest.mark.parametrize(
    "data",
    [
        b"RIFF\x10\x00\x00\x00WAVEfmt ",
        b"\xff\xd8\x00\x00",
        b"GIF8\x00a",
        b"padding\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
    ],
)
def test_signatures_must_match_whole_and_at_the_start(data: bytes) -> None:
    with pytest.raises(UnsupportedContentError, match="no data of a supported media type"):
        MediaFile.from_data_uri(f"data:image/png;base64,{base64.b64encode(data).decode()}")
