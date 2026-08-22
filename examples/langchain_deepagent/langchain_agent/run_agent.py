# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the agent with a prompt from the command line.
uv run run_agent.py "When is the next Valkyries game?"
"""

import sys

from agent import agent


result = agent.invoke({"messages": [{"role": "user", "content": sys.argv[1]}]})
content = result["messages"][-1].content
if isinstance(content, list):  # use_responses_api=True returns content blocks, not a plain string
    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
print(content)
