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

"""Task-data schema for the enterpriseops_gym server (ServiceNow EnterpriseOps-Gym port).

Rows nest everything inside an untyped ``verifier_metadata`` bucket — both
``EnterpriseOpsSeedSessionRequest`` (app.py:124) and ``EnterpriseOpsVerifyRequest``
(app.py:133) type it ``Dict[str, Any]`` with ``extra="allow"`` — so the schema is written
flat with ``legacy_location`` annotations.

Every field is Optional because the server reads the bucket exclusively through
``.get(...) or <default>`` (app.py:227, :325, :339, :465) and never 422s on its contents:
a row missing ``gym_servers_config`` seeds zero databases and scores an empty verifier set
rather than failing. Required-ness here mirrors that wire contract, not what a well-formed
row happens to carry.

Shapes were derived from every committed row plus a full 649-task oracle conversion: all
eight fields appear in 655/655 rows, across 744 ``gym_servers_config`` entries and 3,532
``verifiers``. ``user_info`` is the only genuinely variable field — absent from 44 entries
and carrying either a str or a dict where present.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class GymServerConfig(BaseModel):
    """One MCP gym server the task talks to; drives seeding, tool routing, and verifier fan-out.

    Mirrors the dicts read at app.py:227 (seed) and app.py:325 (run). Tool merge order and
    duplicate-name resolution follow this list's order, which is EOG parity behaviour.
    """

    model_config = ConfigDict(extra="allow")

    mcp_server_name: Optional[str] = Field(
        default=None,
        description="Gym identifier; joined against each verifier's `gym_name` and the `tool_to_gym` map. "
        "A value that matches no running gym causes every verifier referencing it to be skipped.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    mcp_server_url: Optional[str] = Field(
        default=None,
        description="Base URL of the MCP gym container. Overridden at runtime by `gym_url_pools` replicas.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    seed_database_file: Optional[str] = Field(
        default=None,
        description="Path of the seed SQL file, relative to the configured `seed_sql_root`.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-gym request headers (e.g. `x-user-email`) sent with every tool call and verifier query.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    user_info: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Upstream persona blob, carried for provenance only. Either a str or a dict depending on "
        "the source task; unread by this server.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )


class Verifier(BaseModel):
    """One scoring check, ported bug-for-bug from upstream `benchmark/verifier.py`.

    Results are keyed by `name`, so duplicate-named verifiers collapse and only the last is
    scored — an intentionally preserved upstream quirk (PARITY.md section 1).
    """

    model_config = ConfigDict(extra="allow")

    verifier_type: Optional[str] = Field(
        default=None,
        description="One of `database_state`, `response_check`, or `tool_execution` "
        "(verifier_engine.py:209-215). The public oracle split is entirely `database_state`.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    name: Optional[str] = Field(
        default=None,
        description="Verifier identifier and the collapse key for parity scoring.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable statement of what the check asserts.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    gym_name: Optional[str] = Field(
        default=None,
        description="Gym this verifier runs against, matched to `gym_servers_config[].mcp_server_name`. "
        "Verifiers naming an unknown gym are skipped entirely rather than failed (EOG parity).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    validation_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Verifier-type-dependent payload. For `database_state`: `query`, `expected_value` "
        "(str, int, or bool), and `comparison_type`.",
        json_schema_extra={"consumed_by": ["verify"]},
    )


class TaskData(BaseModel):
    """Task-owned fields of an enterpriseops_gym dataset row, flattened out of `verifier_metadata`."""

    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description="Upstream EOG task identifier; used for provenance and for the mass-skip warning.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    domain: Optional[str] = Field(
        default=None,
        description="Source domain (calendar, csm, drive, email, hr, itsm, teams, hybrid). Aggregate metrics "
        "emit per-domain success and pass rates, and `macro_success_rate` is their mean.",
        json_schema_extra={"consumed_by": ["metrics"], **_VM},
    )
    mode: Optional[str] = Field(
        default=None,
        description="Upstream tool-set mode the row was converted under (`oracle` for the public split).",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    gym_servers_config: Optional[List[GymServerConfig]] = Field(
        default=None,
        description="MCP gym servers this task uses, in tool-merge order.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    verifiers: Optional[List[Verifier]] = Field(
        default=None,
        description="Checks run against final database state after the rollout.",
        json_schema_extra={"consumed_by": ["verify", "metrics"], **_VM},
    )
    selected_tools: Optional[List[str]] = Field(
        default=None,
        description="Tool names the task exposes; empty means all tools, as upstream. Names with no matching "
        "schema in the tool snapshots are dropped and logged (convert_tasks.py:115).",
        json_schema_extra={"consumed_by": ["prompt"], **_VM},
    )
    restricted_tools: Optional[List[str]] = Field(
        default=None,
        description="Tool names removed after `selected_tools` filtering.",
        json_schema_extra={"consumed_by": ["prompt"], **_VM},
    )
    tool_to_gym: Optional[Dict[str, str]] = Field(
        default=None,
        description="Tool name to owning `mcp_server_name`, resolved at conversion time so the proxy can route "
        "each call without per-task tools/list discovery.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
