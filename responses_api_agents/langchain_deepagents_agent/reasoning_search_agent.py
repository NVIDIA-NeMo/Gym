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
"""Compatibility names for the Tavily search profile; bundled configs use app.py."""

from typing import Literal

from responses_api_agents.langchain_deepagents_agent.app import DeepAgentsAgent, DeepAgentsAgentConfig


class ReasoningSearchDeepAgentConfig(DeepAgentsAgentConfig):
    """Preserve the existing search config for callers importing its public class."""

    tool_profile: Literal["tavily_search"] = "tavily_search"
    tavily_api_key: str


class ReasoningSearchDeepAgent(DeepAgentsAgent):
    """Compatibility subclass using the shared in-process implementation."""

    config: ReasoningSearchDeepAgentConfig


if __name__ == "__main__":
    ReasoningSearchDeepAgent.run_webserver()
