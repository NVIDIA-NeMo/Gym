# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multimodal staging extras: the compact token delta and a media summary.

A vision-language engine runs on an *expanded* prompt (one media token per projected
embedding) but tokenizes the chat render in *compact* form (one media token per image
or video). The canonical ``token_ids_delta`` is expanded, which is what a trainer
needs. Splicing a previous turn into the next compact render needs the compact form,
so a call that carried media also stages ``compact_token_ids_delta``. ``media`` names
the media the engine saw (modality and per-frame sizes) so a framework that stores the
pixel tensors beside the row can verify it has the same media before training on it.

Both values ride ``StagedCallRecord.extras`` and are therefore bound by
``extras_digest``; text-only calls stage no extras at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator
from typing_extensions import Self


COMPACT_TOKEN_IDS_DELTA_FIELD = "compact_token_ids_delta"
MEDIA_FIELD = "media"

MediaModality = Literal["image", "video"]


class MediaCaptureExtras(BaseModel):
    """The geometry of one call's media, as digest-safe JSON values.

    The pixel tensors themselves are too large for the extras envelope and travel
    beside the row in framework storage; this digest-covered geometry is what a
    framework checks them against before training on the row.
    """

    model_config = ConfigDict(extra="forbid")

    modality: MediaModality
    # Dynamic resolution: per image (stills) or per frame (video), [height, width].
    imgs_sizes: list[list[StrictInt]] | None = None
    # Frames per video item; None for still images.
    num_frames: list[StrictInt] | None = None
    # Static tiling: tiles per image; None for dynamic resolution.
    num_tiles: list[StrictInt] | None = None

    @model_validator(mode="after")
    def _validate_geometry(self) -> Self:
        if self.imgs_sizes is None and self.num_tiles is None:
            raise ValueError("media must carry imgs_sizes (dynamic resolution) or num_tiles (static tiling)")
        for size in self.imgs_sizes or ():
            if len(size) != 2 or any(value <= 0 for value in size):
                raise ValueError("media imgs_sizes entries must be [height, width] with positive values")
        if any(value <= 0 for value in self.num_tiles or ()):
            raise ValueError("media num_tiles entries must be positive")
        if self.num_frames is not None:
            if any(value <= 0 for value in self.num_frames):
                raise ValueError("media num_frames entries must be positive")
            if self.imgs_sizes is not None and sum(self.num_frames) != len(self.imgs_sizes):
                raise ValueError("media num_frames must partition imgs_sizes exactly")
        return self


def build_compact_token_ids_delta(
    compact_prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    *,
    compact_prev_len: int,
) -> list[int]:
    """Return the compact-space delta: the new compact prompt suffix plus the generation.

    Mirrors ``build_staging_delta`` in the compact token space. ``compact_prev_len`` is
    the compact length of the parent chain the worker spliced in; the engine prompt must
    extend it.
    """
    if compact_prev_len < 0 or compact_prev_len > len(compact_prompt_token_ids):
        raise ValueError(
            f"compact_prev_len {compact_prev_len} is outside the compact prompt length {len(compact_prompt_token_ids)}"
        )
    return [int(token_id) for token_id in compact_prompt_token_ids[compact_prev_len:]] + [
        int(token_id) for token_id in generated_token_ids
    ]


def build_multimodal_extras(
    *,
    compact_token_ids_delta: Sequence[int] | None,
    media: Mapping[str, Any] | MediaCaptureExtras | None,
) -> dict[str, Any] | None:
    """Assemble the multimodal extras envelope, or ``None`` when the call carried no media.

    ``compact_token_ids_delta`` may be omitted for a call whose engine prompt was
    already expanded by the client (``media_tokens_preexpanded``): the compact and
    expanded spaces coincide and consumers fall back to ``token_ids_delta``.
    """
    if media is None:
        if compact_token_ids_delta is not None:
            raise ValueError("compact_token_ids_delta requires a media summary")
        return None
    summary = media if isinstance(media, MediaCaptureExtras) else MediaCaptureExtras.model_validate(dict(media))
    extras: dict[str, Any] = {MEDIA_FIELD: summary.model_dump(mode="json")}
    if compact_token_ids_delta is not None:
        extras[COMPACT_TOKEN_IDS_DELTA_FIELD] = [int(token_id) for token_id in compact_token_ids_delta]
    return extras


def parse_multimodal_extras(
    extras: Mapping[str, Any] | None,
) -> tuple[list[int] | None, MediaCaptureExtras | None]:
    """Read the multimodal envelope back out of a staged ``extras`` mapping."""
    if extras is None:
        return None, None
    compact = extras.get(COMPACT_TOKEN_IDS_DELTA_FIELD)
    media = extras.get(MEDIA_FIELD)
    if media is None:
        if compact is not None:
            raise ValueError("compact_token_ids_delta requires a media summary")
        return None, None
    if compact is not None and (
        not isinstance(compact, list) or any(type(token_id) is not int for token_id in compact)
    ):
        raise ValueError("compact_token_ids_delta must be a list of ints")
    return (None if compact is None else list(compact)), MediaCaptureExtras.model_validate(media)


__all__ = [
    "COMPACT_TOKEN_IDS_DELTA_FIELD",
    "MEDIA_FIELD",
    "MediaCaptureExtras",
    "build_compact_token_ids_delta",
    "build_multimodal_extras",
    "parse_multimodal_extras",
]
