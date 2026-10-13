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
    """One R2E-Gym/R2E-Gym-Subset task, as written by prepare_r2e_gym.py."""

    model_config = ConfigDict(extra="allow")

    # "<repo_name>__<commit_hash>"; the Hub row has no id of its own.
    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    repo_name: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    commit_hash: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    # The fixing commit's Python-file diff, rebuilt from parsed_commit_content (r2e_patch.golden_patch).
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    # The Hub's full statement, [ISSUE] tags included; the prompt is the text inside the tags.
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    language: str = Field(default="python", json_schema_extra={"consumed_by": ["verify", "provenance"]})
    docker_image: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    # docker.io/<docker_image>, derived in prepare.
    image_name: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    # JSON string: hidden test id -> status the fixed code produces; the exact-match target.
    expected_output_json: str | Any = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    dataset_name: str = Field(default="R2E-Gym/R2E-Gym-Subset", json_schema_extra={"consumed_by": ["provenance"]})
    modified_files: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    num_non_test_files: int = Field(default=0, json_schema_extra={"consumed_by": ["provenance"]})
    num_non_test_func_methods: int = Field(default=0, json_schema_extra={"consumed_by": ["provenance"]})
    num_non_test_lines: int = Field(default=0, json_schema_extra={"consumed_by": ["provenance"]})
