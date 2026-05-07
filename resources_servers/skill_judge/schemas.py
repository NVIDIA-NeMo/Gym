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
"""Schemas for the skill_judge resources server.

`AssertionGradeResponse` is additive: it subclasses `BaseVerifyResponse` to add
per-assertion grades without changing the base contract (reward is still a
single float — the fraction of assertions satisfied).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


EVIDENCE_MAX_CHARS = 200


class SkillJudgeResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "skill_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_system_message: Optional[str] = (
        "You grade whether an AI assistant's response satisfies a list of "
        "behavioral assertions. You judge only what is literally present in the "
        "response or tool-call log. Output a JSON array only — no prose, no "
        "markdown fences."
    )
    judge_prompt_template_fpath: str = "prompt_templates/skill_judge.txt"


class AssertionGrade(BaseModel):
    id: int
    satisfied: bool
    evidence: str = ""


class JudgeUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ToolCallLogEntry(BaseModel):
    """Compact record of one tool call from the agent's trajectory."""

    name: str
    arguments: str = ""
    exit_code: Optional[int] = None
    stdout_snippet: str = ""
    stderr_snippet: str = ""
    truncated: bool = False


class JudgeRequest(BaseModel):
    prompt: str
    expected_output: Optional[str] = None
    response: str
    tool_calls: list[ToolCallLogEntry] = Field(default_factory=list)
    assertions: list[str]


class JudgeResponse(BaseModel):
    grades: list[AssertionGrade]
    reward: float
    judge_usage: JudgeUsage = Field(default_factory=JudgeUsage)
    parse_error: Optional[str] = None


class SkillJudgeVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict] = None


class AssertionGradeResponse(BaseVerifyResponse):
    """Extends BaseVerifyResponse with per-assertion grades.

    `reward` (inherited) is the fraction of assertions satisfied ∈ [0, 1].
    """

    model_config = ConfigDict(extra="allow")

    grades: list[AssertionGrade]
    judge_usage: JudgeUsage = Field(default_factory=JudgeUsage)
    parse_error: Optional[str] = None
