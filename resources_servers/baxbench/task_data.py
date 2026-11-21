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
"""Task-data schema for BaxBench rows; mirrors ``BaxBenchTaskMetadata`` in app.py, which verify() validates."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: Optional[str] = Field(
        default=None,
        description="Upstream's task prompt (Scenario.build_prompt); filled into the prompt config. Absent from rows "
        "that already carry responses_create_params.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    task_id: str = Field(
        description="'<framework id>/<scenario id>'; labels sandboxes and groups insecure_among_correct by framework.",
        json_schema_extra={"consumed_by": ["verify", "metrics"], **_VM},
    )
    env_id: str = Field(
        description="Upstream framework id, e.g. 'Python-Flask'.", json_schema_extra={"consumed_by": ["verify"], **_VM}
    )
    image_slug: str = Field(
        description="Framework image directory under sandbox/images, filled into image_template.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    scenario_id: str = Field(
        description="Upstream scenario id, e.g. 'Login'.", json_schema_extra={"consumed_by": ["verify"], **_VM}
    )
    language: str = Field(
        description="Framework language; echoed for per-language metrics.",
        json_schema_extra={"consumed_by": ["verify", "metrics"], **_VM},
    )
    framework: str = Field(
        description="Framework name; echoed in the verify response.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    is_multi_file: bool = Field(
        description="Whether the answer lists several files (FILEPATH blocks) or one code block.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    code_filename: Optional[str] = Field(
        default=None,
        description="File a single-file answer is written to; null for multi-file frameworks.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    functional_tests: List[str] = Field(
        description="Upstream functional test function names, one sandbox each.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    security_tests: List[str] = Field(
        description="Upstream exploit function names, one sandbox each.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    upstream_sha: str = Field(
        description="Upstream commit the prompts were built from; the sandbox image must carry the same one.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
