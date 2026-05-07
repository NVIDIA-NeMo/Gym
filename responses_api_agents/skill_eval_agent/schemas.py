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
"""Schemas for the skill_eval_agent.

This agent wires together three servers:
  - skill_workspace: provides run_bash / read_file tools in a sandbox.
  - skill_judge: grades the rollout against behavioral assertions.
  - model_server: the policy being evaluated.

Its /run endpoint implements the with-skill-vs-without-skill delta methodology
by seeding the workspace, optionally prepending SKILL.md, looping model↔tools,
and forwarding the result + captured tool-call log to the judge.
"""

from typing import Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef


class SkillEvalAgentConfig(BaseResponsesAPIAgentConfig):
    workspace_server: ResourcesServerRef
    judge_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 8
    # If True, inject run_bash/read_file tool schemas when the incoming request
    # has no tools. Lets the JSONL stay minimal.
    inject_tools: bool = True


class SkillEvalAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[dict] = None


class SkillEvalAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
