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
"""Media carried in Responses messages, such as an ``input_image`` part holding a base64 data URI.

``split_text_and_media`` decodes a message into its text and its media files,
and ``MediaFile`` converts between bytes and data URIs.
``nemo_gym.sandbox.stage_media`` writes the decoded files into a sandbox for a harness CLI to read.

Images are the supported media today.
Supporting another kind means adding its media types to ``_MEDIA_TYPE_EXTENSIONS`` and ``_MEDIA_TYPE_SIGNATURES``
and its Responses part type to ``_PART_URL_FIELDS``, or a decoder for a part that does not hold a data URI.
"""

import base64
import binascii
import re
from collections.abc import Iterable
from dataclasses import dataclass, field


# Supported media types and the file extension each is written with.
_MEDIA_TYPE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}
# The leading bytes of each supported media type.
# Harnesses such as Pi tell an image's type from these bytes, not from its label or file extension.
_MEDIA_TYPE_SIGNATURES = {
    # The PNG standard puts the IHDR chunk first, and Pi reads a PNG without it as text.
    "image/png": re.compile(rb"\x89PNG\r\n\x1a\n.{4}IHDR", re.DOTALL),
    "image/jpeg": re.compile(rb"\xff\xd8\xff"),
    "image/gif": re.compile(rb"GIF8[79]a"),
    "image/webp": re.compile(rb"RIFF.{4}WEBP", re.DOTALL),
}
# Other spellings of a supported media type, such as the common misspelling image/jpg.
_MEDIA_TYPE_ALIASES = {"image/jpg": "image/jpeg"}
# Responses content part types that carry media, and the field that holds the data URI.
_PART_URL_FIELDS = {"input_image": "image_url"}
_TEXT_PART_TYPES = {"input_text", "text"}


class UnsupportedContentError(ValueError):
    """A message holds content that cannot be passed on as text or as a media file.

    The caller should reject the request rather than drop the content.
    """


def _media_type_of(data: bytes) -> str | None:
    return next(
        (media_type for media_type, signature in _MEDIA_TYPE_SIGNATURES.items() if signature.match(data)), None
    )


@dataclass(frozen=True)
class MediaFile:
    """Decoded media content and its media type, such as ``image/png``.

    The content always starts with the bytes that identify its media type.
    """

    media_type: str
    # Left out of the repr so logs and assertion messages do not print the whole file.
    data: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.media_type not in _MEDIA_TYPE_EXTENSIONS:
            raise UnsupportedContentError(
                f"Unsupported media type {self.media_type!r}; supported: {sorted(_MEDIA_TYPE_EXTENSIONS)}"
            )
        if _media_type_of(self.data) != self.media_type:
            raise UnsupportedContentError(f"Media data is not {self.media_type}")

    @property
    def extension(self) -> str:
        """The file extension, without the dot, that files of this media type are written with."""
        return _MEDIA_TYPE_EXTENSIONS[self.media_type]

    @classmethod
    def from_data_uri(cls, uri: str) -> "MediaFile":
        """Decode a ``data:<media type>[;<parameter>]...;base64,<payload>`` URI.

        The URI must name a supported media type, and the payload must hold data of a supported media type.
        When the two differ, as for a PNG labeled ``image/jpeg``, the payload's own type wins,
        so the file gets the extension its content needs.
        """
        header, separator, payload = uri.partition(",")
        media_type, *parameters = header.removeprefix("data:").split(";")
        if not uri.startswith("data:") or not separator or parameters[-1:] != ["base64"]:
            raise UnsupportedContentError("Media must be a base64 data URI; URLs and file IDs are not fetched")
        # Media types are case-insensitive.
        media_type = media_type.strip().lower()
        media_type = _MEDIA_TYPE_ALIASES.get(media_type, media_type)
        if media_type not in _MEDIA_TYPE_EXTENSIONS:
            raise UnsupportedContentError(
                f"Unsupported media type {media_type!r}; supported: {sorted(_MEDIA_TYPE_EXTENSIONS)}"
            )
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as error:
            raise UnsupportedContentError("Media data URI holds invalid base64") from error
        content_type = _media_type_of(data)
        if content_type is None:
            raise UnsupportedContentError(
                f"Media data URI labeled {media_type!r} holds no data of a supported media type; "
                f"supported: {sorted(_MEDIA_TYPE_EXTENSIONS)}"
            )
        return cls(media_type=content_type, data=data)

    def to_data_uri(self) -> str:
        """Encode as a ``data:<media type>;base64,<payload>`` URI."""
        return f"data:{self.media_type};base64,{base64.b64encode(self.data).decode()}"


def _part_field(part: object, name: str) -> object:
    return part.get(name) if isinstance(part, dict) else getattr(part, name, None)


def split_text_and_media(content: str | Iterable[object]) -> tuple[str, list[MediaFile]]:
    """Return a message's text, joined across text parts, and its decoded media in order.

    Each part may be a mapping or a Responses content part model.
    Raises ``UnsupportedContentError`` for a part that is neither text nor supported media.
    """
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    media: list[MediaFile] = []
    for part in content:
        part_type = _part_field(part, "type")
        if part_type in _TEXT_PART_TYPES:
            texts.append(_part_field(part, "text") or "")
        elif part_type in _PART_URL_FIELDS:
            url = _part_field(part, _PART_URL_FIELDS[part_type])
            if isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str):
                raise UnsupportedContentError(f"{part_type} must hold a base64 data URI")
            media.append(MediaFile.from_data_uri(url))
        else:
            raise UnsupportedContentError(f"Unsupported content part type {part_type!r}")
    return "\n".join(texts), media
