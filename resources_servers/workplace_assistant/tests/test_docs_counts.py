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
"""Keep the documented Workplace Assistant tool and database counts in sync with the code.

`get_tools` is the single source of truth: it always registers the company directory
lookup and then adds one toolkit per name in `TOOLKITS`. Pages and READMEs that quote a
tool count are listed here so adding or removing a tool fails this test instead of
silently making the docs wrong.
"""

from pathlib import Path

from resources_servers.workplace_assistant.utils import get_tools


REPO_ROOT = Path(__file__).resolve().parents[3]

# The toolkits `app.py` seeds every session with. The company directory is not listed
# because `get_tools` registers it unconditionally.
TOOLKITS = [
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
]

# Every place that quotes the tool count, and the exact text it must contain.
DOCUMENTED_TOOL_COUNTS = {
    "fern/versions/latest/pages/evaluation/environment-list.mdx": "Workplace tasks: {n} tools, 5 databases",
    "fern/versions/latest/pages/environment-tutorials/mcp-resources-server.mdx": "a real environment with {n} tools",
    "fern/versions/latest/pages/environment-tutorials/real-world-environment/generating-training-data.mdx": (
        "The Workplace Assistant uses {n} tools across five databases plus a company directory lookup"
    ),
    "fern/versions/latest/pages/training-tutorials/nemo-rl-grpo/about-workplace-assistant.mdx": (
        "The full task includes all {n} tools"
    ),
    "environments/workplace_assistant/README.md": "five databases, {n} tools",
    "resources_servers/workplace_assistant/README.md": "five databases, {n} tools",
    "resources_servers/workplace_assistant/app.py": "Register all {n} workplace tools",
}


class TestDocsCounts:
    def test_get_tools_registers_one_container_per_toolkit_plus_the_company_directory(self):
        tool_env = get_tools(TOOLKITS)

        assert sorted(tool_env["containers"]) == sorted(TOOLKITS + ["company_directory"])
        assert sorted(s["name"] for s in tool_env["schemas"]) == sorted(tool_env["functions"])

    def test_documented_tool_count_matches_get_tools(self):
        tool_count = len(get_tools(TOOLKITS)["functions"])

        for path, template in DOCUMENTED_TOOL_COUNTS.items():
            text = (REPO_ROOT / path).read_text()
            expected = template.format(n=tool_count)
            assert expected in text, f"{path} does not say {expected!r}; get_tools returns {tool_count} tools"

    def test_only_the_company_directory_is_read_only(self):
        """The five toolkits back a mutable database; the company directory only looks up addresses."""
        tool_env = get_tools(TOOLKITS)

        company_directory_tools = [name for name in tool_env["functions"] if name.startswith("company_directory_")]
        assert company_directory_tools == ["company_directory_find_email_address"]
