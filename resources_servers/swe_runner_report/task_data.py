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
"""Task-data schema for the swe_runner_report resources server.

Mirrors ``app.SweRunnerReportInstanceRequest``. Test ids are the runner's own ``<spec path>::<full test name>`` form,
with the spec path relative to ``workdir``, exactly as recorded when the task was validated.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One SWE task graded by its own test-runner report."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    language: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    workdir: str = Field(default="/app/repo", json_schema_extra={"consumed_by": ["verify"]})
    image_ref: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    test_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    test_argv: list[str] = Field(min_length=1, json_schema_extra={"consumed_by": ["verify"]})
    results_format: Literal["vitest", "mocha"] = Field(json_schema_extra={"consumed_by": ["verify"]})
    spec_files: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    source_prefixes: list[str] = Field(min_length=1, json_schema_extra={"consumed_by": ["verify"]})
    source_suffixes: list[str] = Field(min_length=1, json_schema_extra={"consumed_by": ["verify"]})
    setup_commands: list[list[str]] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    setup_timeout_s: int = Field(default=900, gt=0, json_schema_extra={"consumed_by": ["verify"]})
    test_timeout_s: int = Field(default=1200, gt=0, json_schema_extra={"consumed_by": ["verify"]})
    FAIL_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    PASS_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
