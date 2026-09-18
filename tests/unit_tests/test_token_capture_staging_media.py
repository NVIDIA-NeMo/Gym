# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal staging extras: the compact delta and media geometry a VLM call stages."""

from typing import Any

import pytest

from nemo_gym.token_id_capture.staging import (
    COMPACT_TOKEN_IDS_DELTA_FIELD,
    MEDIA_FIELD,
    MediaCaptureExtras,
    build_compact_token_ids_delta,
    build_multimodal_extras,
    compute_extras_digest,
    parse_multimodal_extras,
)


def _media(**overrides: Any) -> dict[str, Any]:
    media: dict[str, Any] = {"modality": "image", "imgs_sizes": [[4, 4]], "num_frames": None, "num_tiles": None}
    media.update(overrides)
    return media


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({}, None),
        ({"modality": "video", "imgs_sizes": [[4, 4], [4, 4]], "num_frames": [2]}, None),
        ({"imgs_sizes": [[4]]}, r"\[height, width\]"),
        ({"imgs_sizes": [[0, 4]]}, "positive"),
        ({"imgs_sizes": None}, "imgs_sizes .* or num_tiles"),
        ({"num_frames": [3], "imgs_sizes": [[4, 4], [4, 4]]}, "partition imgs_sizes"),
        ({"modality": "audio"}, "modality"),
        ({"pixels": [1]}, "extra"),
    ],
)
def test_media_extras_validate_geometry(overrides: dict[str, Any], error: str | None) -> None:
    if error is None:
        assert MediaCaptureExtras.model_validate(_media(**overrides)).model_dump(mode="json") == _media(**overrides)
    else:
        with pytest.raises(ValueError, match=error):
            MediaCaptureExtras.model_validate(_media(**overrides))


@pytest.mark.parametrize(
    ("compact_prev_len", "expected"),
    [(0, [1, 9, 2, 3]), (1, [9, 2, 3]), (4, "outside the compact prompt length")],
)
def test_compact_delta_mirrors_the_expanded_delta_rule(compact_prev_len: int, expected: Any) -> None:
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            build_compact_token_ids_delta([1, 9, 2], [3], compact_prev_len=compact_prev_len)
    else:
        assert build_compact_token_ids_delta([1, 9, 2], [3], compact_prev_len=compact_prev_len) == expected


@pytest.mark.parametrize(
    ("compact", "media", "expected"),
    [
        (None, None, None),
        ([9, 2, 3], _media(), {COMPACT_TOKEN_IDS_DELTA_FIELD: [9, 2, 3], MEDIA_FIELD: _media()}),
        # Pre-expanded engine prompts stage the geometry alone.
        (None, _media(), {MEDIA_FIELD: _media()}),
    ],
    ids=["text", "multimodal", "media-only"],
)
def test_multimodal_extras_round_trip(compact, media, expected) -> None:
    extras = build_multimodal_extras(compact_token_ids_delta=compact, media=media)
    assert extras == expected
    parsed_compact, parsed_media = parse_multimodal_extras(extras)
    assert parsed_compact == compact
    assert parsed_media == (None if media is None else MediaCaptureExtras.model_validate(media))


def test_multimodal_extras_reject_orphan_compact_delta_and_pin_the_digest() -> None:
    with pytest.raises(ValueError, match="requires a media summary"):
        build_multimodal_extras(compact_token_ids_delta=[1], media=None)
    with pytest.raises(ValueError, match="list of ints"):
        parse_multimodal_extras({COMPACT_TOKEN_IDS_DELTA_FIELD: ["1"], MEDIA_FIELD: _media()})
    assert parse_multimodal_extras({"routed_experts": "x"}) == (None, None)
    # The envelope is plain JSON scalars/lists/dicts, so it is digest-safe and stable.
    extras = build_multimodal_extras(compact_token_ids_delta=[9, 2, 3], media=_media())
    assert compute_extras_digest(extras) == "a2527b4e92079fe6fd3c099d561b5b3bf596d003d9650d5b7094e1d92b0b3eea"
