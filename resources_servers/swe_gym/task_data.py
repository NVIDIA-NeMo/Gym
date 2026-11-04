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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One SWE-Gym/SWE-Gym task, as written by prepare_swe_gym.py."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    repo: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    # Selects the repo's SWE-bench eval spec (test command, install step); see swebench_specs.py.
    version: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    base_commit: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    test_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    language: str = Field(default="python", json_schema_extra={"consumed_by": ["verify", "provenance"]})
    # docker.io/xingyaoww/sweb.eval.x86_64.<instance_id with "__" -> "_s_">:latest, derived in prepare.
    image_name: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    dataset_name: str = Field(default="SWE-Gym/SWE-Gym", json_schema_extra={"consumed_by": ["provenance"]})
    # Upstream spells these in caps; kept as-received so a row round-trips unchanged.
    FAIL_TO_PASS: list[str] | Any = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    PASS_TO_PASS: list[str] | Any = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
