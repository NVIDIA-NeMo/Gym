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
"""Host adapter for the parallel_thinking workflow executed inside its task sandbox."""

from typing import ClassVar

from pydantic import Field

from responses_api_agents.langgraph_agent.app import LangGraphAgentAdapter, LangGraphAgentConfig


DEFAULT_PERSPECTIVE_PROMPTS = [
    "Approach this problem step by step using logical deduction.",
    "Consider all possible cases and use process of elimination.",
    "Work backwards from the constraints to determine the answer.",
    "Identify the key relationships and build a truth table or systematic analysis.",
]


class ParallelThinkingAgentConfig(LangGraphAgentConfig):
    num_parallel_paths: int = Field(default=4, ge=1)
    perspective_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_PERSPECTIVE_PROMPTS))


class ParallelThinkingAgent(LangGraphAgentAdapter):
    config: ParallelThinkingAgentConfig
    workflow: ClassVar[str] = "parallel_thinking"
    calls_model_from_sandbox: ClassVar[bool] = True


if __name__ == "__main__":
    ParallelThinkingAgent.run_webserver()
