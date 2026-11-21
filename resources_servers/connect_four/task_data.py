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
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    game_id: Optional[int] = Field(
        default=None,
        description="Row identifier; never read by the server.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed for the opponent's tie-breaking and random moves; unseeded when absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    opponent_depth: Optional[int] = Field(
        default=None,
        ge=0,
        le=6,
        description="Opponent search depth in plies: 0 plays uniformly random legal moves (default 2).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    model_first: Optional[bool] = Field(
        default=None,
        description="Whether the model makes the opening move (default true).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
