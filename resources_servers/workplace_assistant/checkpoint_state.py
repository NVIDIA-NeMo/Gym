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
"""Checkpoint the tables used by Workplace tools without pandas type inference."""

import json
import math
from io import StringIO
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, JsonValue


class WorkplaceCheckpointState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    # Version zero is the unversioned split-JSON format from the initial adapter.
    schema_version: Literal[0, 1] = 0
    containers: dict[str, dict[str, str]]


class _FrameState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    columns: list[str]
    index: list[int]
    data: list[list[JsonValue]]
    dtypes: list[Literal["object", "bool", "int64", "float64"]]
    range_index: list[int] | None
    index_name: str | None
    columns_name: str | None
    nan_cells: list[list[int]]


def encode_frame(frame: pd.DataFrame) -> str:
    """Preserve the scalar types, missing values, and integer indices used by tools."""
    if frame.index.dtype != "int64" or frame.columns.has_duplicates:
        raise ValueError("Workplace checkpoints require an integer index and unique columns")
    payload = frame.to_dict(orient="split")
    nan_cells = []
    for row_index, row in enumerate(payload["data"]):
        for column_index, value in enumerate(row):
            if isinstance(value, float) and math.isnan(value):
                row[column_index] = None
                nan_cells.append([row_index, column_index])
    state = _FrameState(
        **payload,
        dtypes=[str(dtype) for dtype in frame.dtypes],
        range_index=(
            [frame.index.start, frame.index.stop, frame.index.step] if isinstance(frame.index, pd.RangeIndex) else None
        ),
        index_name=frame.index.name,
        columns_name=frame.columns.name,
        nan_cells=nan_cells,
    )
    # Python's JSON encoder preserves float precision; pandas.to_json rounds it.
    return json.dumps(state.model_dump(), allow_nan=False)


def decode_frame(payload: str, columns: list[str], *, schema_version: int) -> pd.DataFrame:
    if schema_version == 0:
        return _decode_legacy_frame(payload, columns)

    state = _FrameState.model_validate_json(payload)
    width = len(state.columns)
    if state.columns != columns or len(state.dtypes) != width:
        raise ValueError("Workplace checkpoint columns do not match the tool table")
    if len(state.index) != len(state.data) or any(len(row) != width for row in state.data):
        raise ValueError("Workplace checkpoint table dimensions are inconsistent")

    nan_cells: set[tuple[int, int]] = set()
    for position in state.nan_cells:
        if len(position) != 2:
            raise ValueError("Invalid Workplace checkpoint NaN position")
        row, column = position
        if not (0 <= row < len(state.data) and 0 <= column < width):
            raise ValueError("Workplace checkpoint NaN position is outside the table")
        if (row, column) in nan_cells or state.data[row][column] is not None:
            raise ValueError("Invalid Workplace checkpoint NaN placeholder")
        nan_cells.add((row, column))

    if state.range_index is not None:
        if len(state.range_index) != 3 or state.range_index[2] == 0:
            raise ValueError("Invalid Workplace checkpoint RangeIndex")
        indices = range(*state.range_index)
        if len(indices) != len(state.index) or list(indices) != state.index:
            raise ValueError("Workplace checkpoint RangeIndex does not match its labels")
        index = pd.RangeIndex(*state.range_index, name=state.index_name)
    else:
        index = pd.Index(state.index, dtype="int64", name=state.index_name)

    # Start with object columns so None and mixed string/numeric cells survive.
    frame = pd.DataFrame(state.data, columns=state.columns, dtype=object)
    for row, column in nan_cells:
        frame.iat[row, column] = float("nan")
    for column_index, (column, dtype) in enumerate(zip(state.columns, state.dtypes, strict=True)):
        if dtype != "object":
            expected_types = {"bool": (bool,), "int64": (int,), "float64": (int, float)}[dtype]
            for row_index, row in enumerate(state.data):
                if (row_index, column_index) in nan_cells:
                    if dtype != "float64":
                        raise ValueError("NaN cannot be restored into a Workplace integer or boolean column")
                elif type(row[column_index]) not in expected_types:
                    raise ValueError(f"Workplace checkpoint value does not match dtype {dtype}")
        frame[column] = frame[column].astype(dtype)
    frame.index = index
    frame.columns.name = state.columns_name
    return frame


def _decode_legacy_frame(payload: str, columns: list[str]) -> pd.DataFrame:
    """Retain the initial adapter's semantics for snapshots without dtype metadata."""
    encoded = json.loads(payload)
    if not isinstance(encoded, dict) or set(encoded) != {"columns", "index", "data"}:
        raise ValueError("Invalid legacy Workplace checkpoint table")
    if encoded["columns"] != columns:
        raise ValueError("Workplace checkpoint columns do not match the tool table")
    frame = pd.read_json(StringIO(payload), orient="split", dtype=False, convert_dates=False)
    for column_index, column in enumerate(columns):
        if any(row[column_index] is None for row in encoded["data"]):
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    index_values = encoded["index"]
    if index_values == list(range(len(index_values))):
        frame.index = pd.RangeIndex(len(index_values))
    return frame


def restore_tool_tables(tool_env: dict[str, Any], state: WorkplaceCheckpointState) -> None:
    """Validate every expected table before the caller publishes the new environment."""
    containers = tool_env["containers"]
    if set(state.containers) != set(containers):
        raise ValueError("Workplace checkpoint must contain every tool container")
    for name, container in containers.items():
        frames = state.containers[name]
        expected = {key: value for key, value in vars(container).items() if isinstance(value, pd.DataFrame)}
        if set(frames) != set(expected):
            raise ValueError(f"Workplace checkpoint must contain every table for {name}")
        for attribute, original in expected.items():
            frame = decode_frame(frames[attribute], list(original.columns), schema_version=state.schema_version)
            setattr(container, attribute, frame)
