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
"""Trivial Haystack-side tools for the example pipeline.

Tools live entirely on the Haystack side (they are NOT proxied to the NeMo Gym resources
server). Add your own ``Tool``s / ``PipelineTool``s (e.g. the
retrieval sub-pipeline from an agentic-RAG setup) as needed.
"""

from typing import Annotated, Literal

from haystack.tools import tool


try:
    from haystack_integrations.tools.tavily import TavilyWebSearchTool
except ImportError as exc:
    raise RuntimeError("This feature requires tavily-haystack. Install it in the Haystack agent venv.") from exc


@tool
def calculator(
    operation: Annotated[
        Literal["add", "subtract", "multiply", "divide", "power"],
        "Operation to perform",
    ],
    a: Annotated[float, "First operand"],
    b: Annotated[float, "Second operand"],
) -> float:
    """Perform arithmetic on two numbers."""

    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    if operation == "power":
        return a**b


wiki_search_tool = TavilyWebSearchTool(
    name="wiki_search",
    top_k=1,
    search_params={
        "include_domains": ["wikipedia.org"],
        "name": "wiki_search",
        "description": "Search Wikipedia for information",
    },
)
