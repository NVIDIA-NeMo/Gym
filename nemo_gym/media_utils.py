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
"""Media in Responses messages and in sandboxes.

Harness agents receive media as Responses content parts, such as an ``input_image`` part holding a base64 data URI.
Harness CLIs read media from files, so an agent decodes the parts with ``split_text_and_media``
and writes them into the sandbox with ``upload_media``.
``download_media`` reads a sandbox file back, and ``MediaFile.to_data_uri`` turns it into a data URI for a part.

Images are the supported media today.
Supporting another kind means adding its media types to ``MEDIA_TYPE_EXTENSIONS``
and its Responses part type to ``_PART_URL_FIELDS``, or a decoder for a part that does not hold a data URI.
"""

import base64
import binascii
import posixpath
from dataclasses import dataclass
from typing import Any, Iterable

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.utils import download_bytes, upload_bytes


# Supported media types and the file extension each is written with.
MEDIA_TYPE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}
_EXTENSION_MEDIA_TYPES = {extension: media_type for media_type, extension in MEDIA_TYPE_EXTENSIONS.items()}
_EXTENSION_MEDIA_TYPES["jpeg"] = "image/jpeg"
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
        """Decode a ``data:<media type>;base64,<payload>`` URI."""
        header, separator, payload = uri.partition(",")
        media_type, _, encoding = header.removeprefix("data:").partition(";")
        if not uri.startswith("data:") or not separator or encoding != "base64":
            raise UnsupportedContentError("Media must be a base64 data URI; URLs and file IDs are not fetched")
        try:
            data = base64.b64decode(payload, validate=True)
        except binascii.Error as error:
            raise UnsupportedContentError("Media data URI holds invalid base64") from error
        return cls(media_type=media_type, data=data)

    def to_data_uri(self) -> str:
        return f"data:{self.media_type};base64,{base64.b64encode(self.data).decode()}"


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


async def upload_media(sandbox: AsyncSandbox, files: Iterable[MediaFile], directory: str) -> list[str]:
    """Write each file into ``directory`` in the sandbox and return the sandbox paths, in order."""
    paths = []
    for index, file in enumerate(files):
        path = posixpath.join(directory, f"media-{index}.{file.extension}")
        await upload_bytes(sandbox, path=path, data=file.data)
        paths.append(path)
    return paths


async def download_media(sandbox: AsyncSandbox, path: str) -> MediaFile:
    """Read a media file from the sandbox, taking its media type from the file extension."""
    extension = posixpath.splitext(path)[1].removeprefix(".").lower()
    if extension not in _EXTENSION_MEDIA_TYPES:
        raise UnsupportedContentError(f"Cannot tell the media type of {path!r} from its extension")
    return MediaFile(media_type=_EXTENSION_MEDIA_TYPES[extension], data=await download_bytes(sandbox, path=path))
