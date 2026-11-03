# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route envelope codec and span-classification decision-table tests."""

import base64
import struct

import pytest

from nemo_gym.token_id_capture.staging.digest import compute_extras_digest
from nemo_gym.token_id_capture.staging.routes import (
    RouteSpanMode,
    classify_route_span,
    decode_routed_experts,
    encode_routed_experts,
    routed_experts_token_count,
)


def _envelope(values: list[int], *, shape: str, dtype: str = "int16") -> str:
    encoded = base64.b64encode(struct.pack(f"<{len(values)}h", *values)).decode("ascii")
    return f"nrlre1:{dtype}:{shape}:{encoded}"


def test_envelope_round_trips_through_encode_and_decode() -> None:
    values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[-1, -1], [9, 10]]]
    for dtype in ("int8", "int16", "int32"):
        payload = encode_routed_experts(values, dtype=dtype)
        fragment = decode_routed_experts(payload)
        assert fragment.values == values
        assert fragment.dtype == dtype
        assert (fragment.num_layers, fragment.topk) == (2, 2)
        assert routed_experts_token_count(payload) == 3


def test_encode_binds_the_digest_to_the_encoded_payload() -> None:
    values = [[[1, 2]], [[3, 4]]]
    payload = encode_routed_experts(values, dtype="int16")
    assert compute_extras_digest({"routed_experts": payload}) == compute_extras_digest(
        {"routed_experts": _envelope([1, 2, 3, 4], shape="2x1x2")}
    )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [[[1, 2]], [[3]]],
        [[[1, 2]], []],
        [[[1, 2]], [[3, 4], [5, 6]]],
        [[[1.5, 2]]],
    ],
)
def test_encode_rejects_ragged_or_non_integer_routes(values: list) -> None:
    with pytest.raises(ValueError):
        encode_routed_experts(values)


def test_encode_rejects_unsupported_dtype_and_overflow() -> None:
    with pytest.raises(ValueError):
        encode_routed_experts([[[1]]], dtype="int64")
    with pytest.raises(ValueError):
        encode_routed_experts([[[2**20]]], dtype="int8")


@pytest.mark.parametrize(
    "payload",
    [
        "nrlre2:int16:1x1x1:AA==",
        "nrlre1:float16:1x1x1:AA==",
        "nrlre1:int16:1x1:AA==",
        "nrlre1:int16:0x1x1:",
        "nrlre1:int16:1x1x1:!!",
        _envelope([1, 2, 3], shape="1x1x2"),
    ],
)
def test_malformed_envelopes_are_rejected(payload: str) -> None:
    with pytest.raises(ValueError):
        decode_routed_experts(payload)


@pytest.mark.parametrize(
    ("carry_len", "generation_len", "staged_route_len", "expected"),
    [
        # Fragment covers the whole span.
        (2, 3, 5, "full"),
        (0, 3, 3, "full"),
        (2, 0, 2, "full"),
        # Fragment covers at least the generated suffix.
        (2, 3, 3, "tail"),
        (2, 3, 4, "tail"),
        (0, 1, 2, "tail"),
        # No usable fragment.
        (2, 3, 0, "sentinel"),
        (2, 0, 1, "sentinel"),
        (2, 3, 2, "sentinel"),
        (0, 0, 0, "sentinel"),
    ],
)
def test_span_classification_decision_table(
    carry_len: int, generation_len: int, staged_route_len: int, expected: RouteSpanMode
) -> None:
    assert (
        classify_route_span(
            carry_len=carry_len,
            generation_len=generation_len,
            staged_route_len=staged_route_len,
        )
        == expected
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"carry_len": -1, "generation_len": 0, "staged_route_len": 0},
        {"carry_len": 0, "generation_len": -1, "staged_route_len": 0},
        {"carry_len": 0, "generation_len": 0, "staged_route_len": -1},
        {"carry_len": 0.5, "generation_len": 0, "staged_route_len": 0},
        {"carry_len": True, "generation_len": 0, "staged_route_len": 0},
    ],
)
def test_span_classification_rejects_invalid_metadata(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        classify_route_span(**kwargs)
