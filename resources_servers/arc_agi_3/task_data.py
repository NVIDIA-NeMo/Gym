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
"""An ARC-AGI-3 row (``create_dataset.py``): one pinned game version. Both ARC-AGI-3 agents use this schema.

The verifier and both harnesses read every field from ``verifier_metadata`` only.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


_LEGACY = {"legacy_location": "verifier_metadata"}
Sha256 = Annotated[str, Field(pattern="^[0-9a-f]{64}$")]


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    game_id: str = Field(
        min_length=1,
        description="Short game id (ls20); the game files' directory under environments_dir.",
        json_schema_extra={"consumed_by": ["verify"], **_LEGACY},
    )
    full_game_id: str = Field(
        min_length=1,
        description="Pinned game version id (ls20-9607627b); the game played and the summary key.",
        json_schema_extra={"consumed_by": ["verify", "metrics"], **_LEGACY},
    )
    version: str = Field(
        min_length=1,
        description="The game version's directory name under <environments_dir>/<game_id>/.",
        json_schema_extra={"consumed_by": ["verify"], **_LEGACY},
    )
    files_sha256: dict[str, Sha256] = Field(
        min_length=1,
        description="sha256 of each game file except metadata.json; the loaded game must match exactly.",
        json_schema_extra={"consumed_by": ["verify", "provenance"], **_LEGACY},
    )
    number_of_levels: int | None = Field(
        default=None,
        ge=1,
        description="Levels of the game.",
        json_schema_extra={"consumed_by": ["provenance"], **_LEGACY},
    )
    baseline_actions: list[int] | None = Field(
        default=None,
        description="Human baseline actions per level (the verifier uses the loaded game's copy).",
        json_schema_extra={"consumed_by": ["provenance"], **_LEGACY},
    )
    tags: list[str] | None = Field(
        default=None,
        description="The game's metadata tags.",
        json_schema_extra={"consumed_by": ["provenance"], **_LEGACY},
    )
    dispatch_index: int | None = Field(
        default=None,
        ge=0,
        description="The row's harness dispatch index (the kit's game order), read by the agents; 0 when absent.",
        json_schema_extra=_LEGACY,
    )
