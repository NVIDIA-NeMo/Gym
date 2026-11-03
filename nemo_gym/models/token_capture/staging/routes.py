# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable validation, encoding, and decoding for staged routed-expert envelopes.

The envelope wire format and its codec live here because ``extras_digest`` is
computed over the *encoded* payload: whoever defines what staged route bytes
mean also defines what makes them authentic. The span-classification decision
table (:func:`classify_route_span`) is the pure-metadata rule for how one
staged fragment contributes to a linearized row; frameworks apply it instead
of mirroring it.
"""

from __future__ import annotations

import base64
import binascii
import struct
from dataclasses import dataclass
from typing import Any, Literal, Sequence


ROUTED_EXPERTS_ENVELOPE_VERSION = 1
ROUTED_EXPERTS_MAGIC = "nrlre1"
MISSING_ROUTE_SENTINEL = -1

RouteSpanMode = Literal["full", "tail", "sentinel"]

_DTYPE_FORMATS = {
    "int8": ("b", 1),
    "int16": ("h", 2),
    "int32": ("i", 4),
}


@dataclass(frozen=True)
class RoutedExpertsFragment:
    """One decoded ``[tokens][layers][topk]`` routed-expert fragment."""

    values: list[list[list[int]]]
    dtype: str
    num_layers: int
    topk: int


def _validate_nested_routes(payload: Any) -> RoutedExpertsFragment:
    if type(payload) is not list:
        raise ValueError("legacy routed_experts must be a nested list")
    if not payload:
        raise ValueError("routed_experts must contain at least one token row")
    num_layers: int | None = None
    topk: int | None = None
    values: list[list[list[int]]] = []
    for token_row in payload:
        if type(token_row) is not list or not token_row:
            raise ValueError("each routed_experts token row must contain layers")
        if num_layers is None:
            num_layers = len(token_row)
        elif len(token_row) != num_layers:
            raise ValueError("routed_experts layer count must be constant")
        normalized_row: list[list[int]] = []
        for layer_row in token_row:
            if type(layer_row) is not list or not layer_row:
                raise ValueError("each routed_experts layer row must contain experts")
            if topk is None:
                topk = len(layer_row)
            elif len(layer_row) != topk:
                raise ValueError("routed_experts top-k width must be constant")
            if any(type(expert) is not int for expert in layer_row):
                raise ValueError("routed_experts values must be exact integers")
            normalized_row.append(list(layer_row))
        values.append(normalized_row)
    assert num_layers is not None and topk is not None
    return RoutedExpertsFragment(
        values=values,
        dtype="legacy-int",
        num_layers=num_layers,
        topk=topk,
    )


def decode_routed_experts(payload: Any) -> RoutedExpertsFragment:
    """Decode the supported v1 base64 envelope or validate legacy nested lists."""
    if not isinstance(payload, str):
        return _validate_nested_routes(payload)
    parts = payload.split(":", 3)
    if len(parts) != 4 or parts[0] != ROUTED_EXPERTS_MAGIC:
        raise ValueError(f"unsupported routed_experts envelope; expected {ROUTED_EXPERTS_MAGIC}")
    _, dtype, shape_text, encoded = parts
    dtype_info = _DTYPE_FORMATS.get(dtype)
    if dtype_info is None:
        raise ValueError(f"unsupported routed_experts dtype {dtype!r}")
    dimensions = shape_text.split("x")
    if len(dimensions) != 3:
        raise ValueError("routed_experts envelope shape must have three dimensions")
    try:
        token_count, num_layers, topk = (int(dimension) for dimension in dimensions)
    except ValueError as error:
        raise ValueError("routed_experts envelope shape must contain integers") from error
    if token_count <= 0 or num_layers <= 0 or topk <= 0:
        raise ValueError("routed_experts envelope dimensions must be positive")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("routed_experts envelope contains invalid base64") from error
    format_code, item_size = dtype_info
    element_count = token_count * num_layers * topk
    if len(raw) != element_count * item_size:
        raise ValueError(
            f"routed_experts byte length does not match its declared shape ({len(raw)} != {element_count * item_size})"
        )
    flat = struct.unpack(f"<{element_count}{format_code}", raw)
    values: list[list[list[int]]] = []
    cursor = 0
    for _ in range(token_count):
        token_row: list[list[int]] = []
        for _ in range(num_layers):
            token_row.append(list(flat[cursor : cursor + topk]))
            cursor += topk
        values.append(token_row)
    return RoutedExpertsFragment(
        values=values,
        dtype=dtype,
        num_layers=num_layers,
        topk=topk,
    )


def encode_routed_experts(values: Sequence[Sequence[Sequence[int]]], *, dtype: str = "int16") -> str:
    """Encode a ``[tokens][layers][topk]`` nested list as the v1 base64 envelope.

    This is the reference inverse of :func:`decode_routed_experts` for the
    ``nrlre1`` wire format; array-backed producers may emit the same envelope
    from contiguous little-endian buffers without round-tripping through
    nested lists.
    """
    dtype_info = _DTYPE_FORMATS.get(dtype)
    if dtype_info is None:
        raise ValueError(f"unsupported routed_experts dtype {dtype!r}")
    fragment = _validate_nested_routes(list(values))
    format_code, _ = dtype_info
    flat = [expert for token_row in fragment.values for layer_row in token_row for expert in layer_row]
    try:
        raw = struct.pack(f"<{len(flat)}{format_code}", *flat)
    except struct.error as error:
        raise ValueError(f"routed_experts values do not fit dtype {dtype!r}") from error
    encoded = base64.b64encode(raw).decode("ascii")
    shape_text = f"{len(fragment.values)}x{fragment.num_layers}x{fragment.topk}"
    return f"{ROUTED_EXPERTS_MAGIC}:{dtype}:{shape_text}:{encoded}"


def classify_route_span(*, carry_len: int, generation_len: int, staged_route_len: int) -> RouteSpanMode:
    """Classify how one staged fragment contributes to a linearized row.

    ``full``: the fragment covers the span's whole carry+generation token
    range. ``tail``: the fragment covers at least the generated suffix, whose
    last ``generation_len`` rows are used. ``sentinel``: no usable fragment —
    the consumer fills the span with :data:`MISSING_ROUTE_SENTINEL`.
    """
    for name, value in (
        ("carry_len", carry_len),
        ("generation_len", generation_len),
        ("staged_route_len", staged_route_len),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    expected = carry_len + generation_len
    if staged_route_len > 0 and staged_route_len == expected:
        return "full"
    if staged_route_len > 0 and 0 < generation_len <= staged_route_len:
        return "tail"
    return "sentinel"


def routed_experts_token_count(payload: Any) -> int:
    """Inspect the token dimension without materializing the route tensor."""
    if not isinstance(payload, str):
        if type(payload) is not list or not payload:
            raise ValueError("legacy routed_experts must be a non-empty nested list")
        return len(payload)
    parts = payload.split(":", 3)
    if len(parts) != 4 or parts[0] != ROUTED_EXPERTS_MAGIC:
        raise ValueError(f"unsupported routed_experts envelope; expected {ROUTED_EXPERTS_MAGIC}")
    _, dtype, shape_text, _ = parts
    if dtype not in _DTYPE_FORMATS:
        raise ValueError(f"unsupported routed_experts dtype {dtype!r}")
    dimensions = shape_text.split("x")
    if len(dimensions) != 3:
        raise ValueError("routed_experts envelope shape must have three dimensions")
    try:
        token_count, num_layers, topk = (int(dimension) for dimension in dimensions)
    except ValueError as error:
        raise ValueError("routed_experts envelope shape must contain integers") from error
    if token_count <= 0 or num_layers <= 0 or topk <= 0:
        raise ValueError("routed_experts envelope dimensions must be positive")
    return token_count
