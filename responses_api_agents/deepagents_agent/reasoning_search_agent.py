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
"""Concrete deepagents_agent instance: the exact same TavilySearch-based deepagents graph as
examples/langchain_deepagent, wired in-tree instead of via remote_agent. Reused across both the
reasoning_gym and tavily_search combo configs, matching how the remote_agent version already reuses one
agent object across both its configs today (see claude_code_agent/simple_agent for the repo-wide
precedent of one agent class serving many resources servers via config, not langgraph_agent's subclasses,
which just haven't been pointed at a second resources server yet).
"""

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tavily import TavilySearch

from responses_api_agents.deepagents_agent.app import DeepAgentsAgent, DeepAgentsAgentConfig


SYSTEM_PROMPT = "Answer the user, then restate your final answer wrapped in <answer></answer> tags."


class ReasoningSearchDeepAgentConfig(DeepAgentsAgentConfig):
    tavily_api_key: str
    max_search_results: int = 5


class ReasoningSearchDeepAgent(DeepAgentsAgent): #TODO: change name
    config: ReasoningSearchDeepAgentConfig

    def build_agent(self, model: BaseChatModel): #TODO: do we need BaseChatModel here? 
        return create_deep_agent(
            model=model,
            tools=[TavilySearch(max_results=self.config.max_search_results, tavily_api_key=self.config.tavily_api_key)],
            system_prompt=SYSTEM_PROMPT, #TODO: remove system prompt probably? 
        )


if __name__ == "__main__":
    ReasoningSearchDeepAgent.run_webserver()
