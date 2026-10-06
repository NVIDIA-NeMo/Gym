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
"""Row schema for the timewarp resources server (see nemo_gym/task_data.py for the protocol)."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: int = Field(
        description="TimeWarp task id (1-103 test, 104-231 train).", json_schema_extra={"consumed_by": ["provenance"]}
    )
    ui_version: int = Field(
        ge=1,
        le=6,
        description="UI era the task runs in; selects the site URLs the browser may visit.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    start_site: Literal["wiki", "news", "webshop"] = Field(
        description="Site whose home page the browser opens on.", json_schema_extra={"consumed_by": ["prompt"]}
    )
    sites: List[Literal["wiki", "news", "webshop"]] = Field(
        description="Sites the task needs; more than one makes it a multi-site task.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    intent: str = Field(
        description="The task question, shown to the policy and to the LLM judge.",
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
    eval_types: List[Literal["string_match", "number_match", "list_match", "exact_match", "llm_judge"]] = Field(
        min_length=1,
        description="TimeWarp verifiers to run; they are AND-ed.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    reference_answers: Dict[str, Any] = Field(
        description="Each verifier's spec (must_include, number_match, list_match, fuzzy_match, ...).",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    revision: Optional[int] = Field(
        default=None,
        description="Revision of the task's upstream eval spec.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
