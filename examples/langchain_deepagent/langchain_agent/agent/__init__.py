# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The simplest possible deepagent: one model, one tool (Tavily web search).

Standalone and importable — see run_agent.py. The `agent` object below is also what
../service.py imports directly (`from agent import agent`) to run this as a Gym remote_agent;
see ../README.md.
"""

import os

from dotenv import load_dotenv


load_dotenv()  # picks up .env before ChatOpenAI()/TavilySearch() read env vars below

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


model = ChatOpenAI(
    # Free by default so running this example costs nothing beyond Tavily calls.
    model=os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    # OpenRouter supports the Responses API (POST /v1/responses) — use it over Chat Completions
    # for better reasoning-trace fidelity across tool-calling turns. See
    # https://reference.langchain.com/python/deepagents/graph/create_deep_agent
    use_responses_api=True,
)

agent = create_deep_agent(
    model=model,
    tools=[TavilySearch(max_results=5)],
    # reasoning_gym's verifier extracts the answer from <answer></answer> tags in the final
    # message; without this, correct reasoning still scores 0. See ../configs/config_reasoning_gym.yaml.
    system_prompt="Answer the user, then restate your final answer wrapped in <answer></answer> tags.",
)
