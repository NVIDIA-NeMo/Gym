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
"""Task-data schema for the arc_agi_3_pivot server.

A row is one pivot of a pivot pool (``create_dataset.py``). The server reads every task field from
``verifier_metadata``, so each top-level field carries ``legacy_location``. ``pivot`` is the pool row the server
restores; the server parses it with ``PivotRow`` (``app.pool_row``), so preparation validates what the runtime reads.
"""

from pydantic import BaseModel, ConfigDict, Field


class PivotRow(BaseModel):
    """The pool row columns a window is restored from. Paths are relative to the server's ``pool_dir``."""

    # A misspelled column would otherwise be dropped and the restore fail later.
    model_config = ConfigDict(extra="forbid")

    pivot_json: str = Field(
        min_length=1,
        description="The pivot, JSON-encoded: its recording (snapshot.snap_file, snap_sha256), game, call and po_d0.",
    )
    harness: str = Field(min_length=1, description="The harness revision the pivot was recorded with.")
    harness_env: str | None = Field(default=None, description="The recorded harness env file.")
    harness_env_sha256: str | None = Field(default=None, description="The sha256 the harness env file must have.")
    thinking_effort: str | None = Field(
        default=None,
        description="The effort the pivot's run was sampled with, when its collector set it past the harness's request.",
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    pivot: PivotRow = Field(
        json_schema_extra={"consumed_by": ["prompt", "verify"], "legacy_location": "verifier_metadata"},
    )
    pivot_id: str | None = Field(
        default=None,
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    game_id: str | None = Field(
        default=None,
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    level: int | None = Field(
        default=None,
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    po_d0: int | None = Field(
        default=None,
        description="The oracle distance at the pivot when it was recorded.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    reasoning_effort: str | None = Field(
        default=None,
        description="The thinking effort of the window's live requests; wins over the server's reasoning_effort.",
        json_schema_extra={"consumed_by": ["prompt"], "legacy_location": "verifier_metadata"},
    )
