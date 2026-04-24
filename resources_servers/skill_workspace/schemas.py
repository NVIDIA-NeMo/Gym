# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
)


class SkillWorkspaceResourcesServerConfig(BaseResourcesServerConfig):
    workspace_root: Optional[str] = None
    bash_timeout_default_seconds: int = 30
    bash_timeout_hard_cap_seconds: int = 120
    output_cap_bytes: int = 50_000
    max_concurrent_bash: int = 16


class SkillWorkspaceSeedSessionRequest(BaseSeedSessionRequest):
    skill_path: str
    scenario_id: int
    files: list[str] = Field(default_factory=list)
    # Gate skill-adjacent payload on these flags so the "without skill" arm is a
    # true control. SKILL.md is never seeded (it belongs in the system prompt
    # when with_skill=True, nowhere otherwise). Defaults preserve pre-existing
    # behavior for non-eval callers.
    with_references: bool = True
    with_scripts: bool = True


class SkillWorkspaceSeedSessionResponse(BaseSeedSessionResponse):
    env_id: str


class RunBashRequest(BaseModel):
    env_id: str
    cmd: str
    timeout_seconds: Optional[int] = None


class RunBashResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    truncated: bool
    timed_out: bool


class ReadFileRequest(BaseModel):
    env_id: str
    path: str


class ReadFileResponse(BaseModel):
    content: str
    truncated: bool


class CloseRequest(BaseModel):
    env_id: str


class CloseResponse(BaseModel):
    message: str
    success: bool
