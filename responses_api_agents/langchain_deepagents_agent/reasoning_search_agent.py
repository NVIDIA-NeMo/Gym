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
"""Concrete langchain_deepagents_agent instance: a TavilySearch-based deepagents graph wired in-tree, so
model calls go through Gym's own model_server. Reused across both the reasoning_gym and tavily_search
combo configs (see claude_code_agent/simple_agent for the repo-wide precedent of one agent class serving
many resources servers via config, not langgraph_agent's subclasses, which just haven't been pointed at a
second resources server yet).
"""

from typing import Optional

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tavily import TavilySearch

from responses_api_agents.langchain_deepagents_agent.app import DeepAgentsAgent, DeepAgentsAgentConfig


class ReasoningSearchDeepAgentConfig(DeepAgentsAgentConfig):
    tavily_api_key: str
    max_search_results: int = 5
    system_prompt: Optional[str] = None
    # Required, no default: without it, deepagents' SummarizationMiddleware falls back to a hardcoded,
    # model-agnostic threshold (170k tokens, keep last 6 messages) unrelated to the real model.
    max_input_tokens: int


class ReasoningSearchDeepAgent(DeepAgentsAgent):
    config: ReasoningSearchDeepAgentConfig

    def build_agent(self, model: BaseChatModel):
        # BaseChatModel.profile is a plain, already-inherited field (langchain_core) — deepagents reads
        # model.profile["max_input_tokens"] when computing its summarization trigger/keep thresholds.
        model.profile = {"max_input_tokens": self.config.max_input_tokens}
        return create_deep_agent(
            model=model,
            tools=[
                TavilySearch(max_results=self.config.max_search_results, tavily_api_key=self.config.tavily_api_key)
            ],
            system_prompt=self.config.system_prompt,
        )


if __name__ == "__main__":
    ReasoningSearchDeepAgent.run_webserver()
