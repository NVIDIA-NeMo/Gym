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
from haystack import Pipeline
from haystack.components.agents import Agent

from responses_api_agents.haystack_agent.chat_generator import NeMoGymResponsesChatGenerator
from responses_api_agents.haystack_agent.example.example_tools import calculator, wiki_search_tool


MODEL_SERVER_NAME = "policy_model"
MAX_AGENT_STEPS = 20


def build() -> Pipeline:
    agent = Agent(
        chat_generator=NeMoGymResponsesChatGenerator(server_name=MODEL_SERVER_NAME),
        tools=[wiki_search_tool, calculator],
        system_prompt="You are a helpful assistant that can answer questions and help with tasks. You have access to two tools: calculator and wiki_search. You can use the calculator tool to perform arithmetic operations on two numbers. You can use the wiki_search tool to search Wikipedia for information. Use the tools to help you answer questions and check the answer correctness.",
        exit_conditions=["text"],
        max_agent_steps=MAX_AGENT_STEPS,
    )
    pipe = Pipeline()
    pipe.add_component("agent", agent)
    return pipe


def main() -> None:
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, "..", "configs")
    os.makedirs(configs_dir, exist_ok=True)
    pipeline_path = os.path.join(configs_dir, "example_pipeline_with_tools.yaml")
    with open(pipeline_path, "w") as f:
        f.write(build().dumps())
    print(f"Wrote {pipeline_path}")


if __name__ == "__main__":
    main()
