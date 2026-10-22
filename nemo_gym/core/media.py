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
``nemo_gym.agents`` writes the decoded files into a sandbox for a harness CLI to read.

Images are the supported media today.
Supporting another kind means adding its media types to ``MEDIA_TYPE_EXTENSIONS``
and its Responses part type to ``_PART_URL_FIELDS``, or a decoder for a part that does not hold a data URI.
"""

import base64
import binascii
import posixpath
from dataclasses import dataclass
from typing import Any, Iterable


# Supported media types and the file extension each is written with.
MEDIA_TYPE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}
_EXTENSION_MEDIA_TYPES = {extension: media_type for media_type, extension in MEDIA_TYPE_EXTENSIONS.items()}
_EXTENSION_MEDIA_TYPES["jpeg"] = "image/jpeg"
# Other spellings of a supported media type, such as the common misspelling image/jpg.
_MEDIA_TYPE_ALIASES = {"image/jpg": "image/jpeg"}
# Responses content part types that carry media, and the field that holds the data URI.
_PART_URL_FIELDS = {"input_image": "image_url"}
_TEXT_PART_TYPES = {"input_text", "text"}


class UnsupportedContentError(ValueError):
    """A message holds content that cannot be passed on as text or as a media file.

    The caller should reject the request rather than drop the content.
    """


@dataclass(frozen=True)
class MediaFile:
    """Decoded media content and its media type, such as ``image/png``."""

    media_type: str
    data: bytes

    def __post_init__(self) -> None:
        if self.media_type not in MEDIA_TYPE_EXTENSIONS:
            raise UnsupportedContentError(
                f"Unsupported media type {self.media_type!r}; supported: {sorted(MEDIA_TYPE_EXTENSIONS)}"
            )

    @property
    def extension(self) -> str:
        return MEDIA_TYPE_EXTENSIONS[self.media_type]

    @classmethod
    def from_data_uri(cls, uri: str) -> "MediaFile":
        """Decode a ``data:<media type>[;<parameter>]...;base64,<payload>`` URI."""
        header, separator, payload = uri.partition(",")
        media_type, *parameters = header.removeprefix("data:").split(";")
        if not uri.startswith("data:") or not separator or parameters[-1:] != ["base64"]:
            raise UnsupportedContentError("Media must be a base64 data URI; URLs and file IDs are not fetched")
        try:
            data = base64.b64decode(payload, validate=True)
        except binascii.Error as error:
            raise UnsupportedContentError("Media data URI holds invalid base64") from error
        # Media types are case-insensitive.
        media_type = media_type.strip().lower()
        return cls(media_type=_MEDIA_TYPE_ALIASES.get(media_type, media_type), data=data)

    def to_data_uri(self) -> str:
        return f"data:{self.media_type};base64,{base64.b64encode(self.data).decode()}"


def media_type_for_path(path: str) -> str:
    """Return the supported media type a file's extension names, such as ``image/png`` for ``a.PNG``."""
    extension = posixpath.splitext(path)[1].removeprefix(".").lower()
    if extension not in _EXTENSION_MEDIA_TYPES:
        raise UnsupportedContentError(f"Cannot tell the media type of {path!r} from its extension")
    return _EXTENSION_MEDIA_TYPES[extension]


def _part_field(part: Any, name: str) -> Any:
    return part.get(name) if isinstance(part, dict) else getattr(part, name, None)


def split_text_and_media(content: str | Iterable[Any]) -> tuple[str, list[MediaFile]]:
    """Return a message's text, joined across text parts, and its decoded media in order.

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
