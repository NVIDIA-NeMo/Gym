# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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
"""Task-data schema for the blobfish_enterprise_ops server (NeMo Gym ``task_data.py`` convention).

Rows are prompt-only: ``responses_create_params`` carries the system and user messages plus the
suite's function tools, and the task fields ride at the row top level (no ``verifier_metadata``).
The only field the server reads is ``task_id``: ``/seed_session`` builds a fresh world for that task
package and ``/verify`` scores the episode with the package's own verifier, so no answer, oracle plan
or expected state is ever on the wire. Everything else is provenance the emitter records per suite
(``suite``, ``benchmark``, ``world_id``, ``license``, ``source``) or per task (``category``,
``difficulty``, ``metric`` where the package's ``task.toml`` carries them). ``task_source`` is added
by NeMo Gym during collation. ``extra="allow"`` per the repository rule; nothing is silently dropped.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description=(
            "Task package id inside the served suite (e.g. 'erpbench-007'); /seed_session builds that "
            "package's world and /verify runs its verifier. The system prompt also states it because "
            "some suites' tools take it as an argument."
        ),
        json_schema_extra={"consumed_by": ["seed_session", "verify"]},
    )
    suite: str = Field(
        description="Suite name the row belongs to (e.g. 'ERPBench-100'); provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    benchmark: str = Field(
        description="Benchmark name from the task package metadata; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    world_id: str = Field(
        description="Identifier of the world/tenant catalog the task runs in; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    license: str = Field(
        description="SPDX identifier of the task data licence (CC-BY-4.0 for the published suites); provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source: str = Field(
        description="Where the task packages are published (Harbor Hub dataset URL); provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    category: Optional[str] = Field(
        default=None,
        description="Workflow family or category from the task package, when the suite records one; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="Difficulty label from the task package, when the suite records one; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metric: Optional[str] = Field(
        default=None,
        description="Name of the suite's score (e.g. 'ERPScore'); provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task_source: Optional[str] = Field(
        default=None,
        description="Agent instance that collated the row; written by NeMo Gym, not by the emitter.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
