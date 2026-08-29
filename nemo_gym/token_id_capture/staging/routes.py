# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable validation and decoding for staged routed-expert envelopes."""

from __future__ import annotations

import base64
import binascii
import struct
from dataclasses import dataclass
from typing import Any


ROUTED_EXPERTS_ENVELOPE_VERSION = 1
ROUTED_EXPERTS_MAGIC = "nrlre1"
MISSING_ROUTE_SENTINEL = -1

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
