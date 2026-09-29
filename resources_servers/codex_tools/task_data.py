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


_FROM_VERIFIER_METADATA = "verifier_metadata"


class TaskData(BaseModel):
    """One codex_tools coding task. Unset fields fall back to the resources server config."""

    model_config = ConfigDict(extra="allow")

    repo_path: Optional[str] = Field(
        default=None,
        description="Git repository to work on.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": _FROM_VERIFIER_METADATA},
    )
    base_ref: Optional[str] = Field(
        default=None,
        description="Commit-ish the workspace starts from and the diff is taken against.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": _FROM_VERIFIER_METADATA},
    )
    check_command: Optional[str] = Field(
        default=None,
        description="Shell command run in the workspace at verify time; exit status 0 earns reward 1.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": _FROM_VERIFIER_METADATA},
    )
    task_id: Optional[str] = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
