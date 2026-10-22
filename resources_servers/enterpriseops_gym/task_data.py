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
"""Task fields consumed by the external EnterpriseOps MCP integration.

Legacy dataset rows nest these fields in ``verifier_metadata``. Gym normalizes that
container before validation. Seeding requires a nonempty gym list; optional scoring
and provenance fields retain the runtime's defaults.
"""

from pydantic import BaseModel, ConfigDict, Field, JsonValue


_VM = {"legacy_location": "verifier_metadata"}


class GymServerConfig(BaseModel):
    """A task's MCP gym and SQL seed; trusted runtime configuration owns its endpoint."""

    model_config = ConfigDict(extra="allow")

    mcp_server_name: str
    mcp_server_url: str
    seed_database_file: str
    context: dict[str, JsonValue] | None = None
    user_info: str | dict[str, JsonValue] | None = None


class Verifier(BaseModel):
    """An upstream database, response, or tool-execution verification rule."""

    model_config = ConfigDict(extra="allow")

    verifier_type: str | None = None
    name: str | None = None
    description: str | None = None
    gym_name: str | None = None
    validation_config: dict[str, JsonValue] | None = None


class TaskData(BaseModel):
    """EnterpriseOps task metadata after Gym unwraps ``verifier_metadata``."""

    model_config = ConfigDict(extra="allow")

    task_id: str | None = Field(default=None, json_schema_extra={**_VM, "consumed_by": ["provenance"]})
    domain: str | None = Field(default=None, json_schema_extra={**_VM, "consumed_by": ["metrics"]})
    mode: str | None = Field(default=None, json_schema_extra={**_VM, "consumed_by": ["provenance"]})
    gym_servers_config: list[GymServerConfig] = Field(
        min_length=1, json_schema_extra={**_VM, "consumed_by": ["verify"]}
    )
    verifiers: list[Verifier] | None = Field(
        default=None, json_schema_extra={**_VM, "consumed_by": ["verify", "metrics"]}
    )
    selected_tools: list[str] | None = Field(default=None, json_schema_extra={**_VM, "consumed_by": ["prompt"]})
    restricted_tools: list[str] | None = Field(default=None, json_schema_extra={**_VM, "consumed_by": ["prompt"]})
    tool_to_gym: dict[str, str] | None = Field(default=None, json_schema_extra={**_VM, "consumed_by": ["verify"]})
    selected_tool_aliases: dict[str, str] | None = Field(
        default=None, json_schema_extra={**_VM, "consumed_by": ["provenance"]}
    )
    unavailable_selected_tools: list[str] | None = Field(
        default=None, json_schema_extra={**_VM, "consumed_by": ["provenance"]}
    )
