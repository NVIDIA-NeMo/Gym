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
"""The BrowseComp tools as the model sees them, one list per workspace mode.

``browsecomp_agent`` reads them from each row's ``responses_create_params.tools``. A harness that takes its tools
from the Resources Server's MCP grant gets the same names, descriptions and parameters, because ``mcp_tools`` in
``app.py`` advertises these definitions in place of the handlers' own schemas.
"""

from typing import Any


# workspace: none. Search and browse return page content inline; the example rows carry these.
INLINE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search",
        "description": (
            "Web Search API, works like Google Search. "
            "All queries will be searched in parallel. "
            "If you want to search with multiple keywords, "
            "put them in a single query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Search queries. All queries are executed in parallel.",
                }
            },
            "required": ["queries"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "browse",
        "description": (
            "Visit specific webpage(s) and return their full text content. "
            "Use this to read the complete content of web pages found during search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URL(s) of webpage(s) to visit.",
                },
                "goal": {
                    "type": "string",
                    "description": "What specific information you are looking for.",
                },
            },
            "required": ["urls"],
        },
        "strict": False,
    },
]

# workspace: per_session. Search and browse save pages to the session's workspace, and bash_command reads them;
# benchmarks/browsecomp/prepare.py writes these into the benchmark rows.
WORKSPACE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search",
        "description": (
            "Web Search API, works like Google Search. "
            "All queries will be searched in parallel. "
            "If you want to search with multiple keywords, "
            "put them in a single query. "
            "Each result's full raw content is saved to "
            "pages/<idx>_search_<slug>_rN.txt under the current workspace; "
            "the tool response returns the per-result title, URL, snippet, "
            "and [Saved to] path. Use bash_command to read the saved files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ("Search queries. All queries are executed in parallel."),
                }
            },
            "required": ["queries"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "browse",
        "description": (
            "Visit specific webpage(s) and save their full text content "
            "to local files. Use this to fetch the complete content of "
            "web pages found during search. Each page is written to "
            "pages/<idx>_browse_<slug>.txt under the current workspace; "
            "the tool response returns the per-URL title, [Saved to] "
            "path, byte count, and a short preview. Use bash_command "
            "to read the full content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URL(s) of webpage(s) to visit.",
                },
                "goal": {
                    "type": "string",
                    "description": ("What specific information you are looking for."),
                },
            },
            "required": ["urls"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "bash_command",
        "description": (
            "Run a shell command in the current sample's workspace and "
            "return stdout/stderr (truncated to ~12 KB head+tail).\n\n"
            "cwd layout (pinned per sample; do not navigate outside):\n"
            "  pages/<idx:04d>_<kind>_<slug>[_rN].txt    one file per search result or browsed page\n"
            "  manifest.tsv                              page_id<TAB>kind<TAB>source<TAB>title<TAB>bytes\n\n"
            "Useful recipes:\n"
            "  ls pages/ | wc -l                              # how many pages saved so far\n"
            "  ls pages/ | head                                # glance at filenames (slugged from query/URL)\n"
            '  grep -l "<phrase>" pages/*.txt | head           # which files mention X\n'
            '  grep -m 5 -B 2 -A 4 "<phrase>" pages/0042_*     # context around hits in one file\n'
            "  head -c 2000 pages/0042_*                       # peek at start of a file\n"
            "  sed -n '100,200p' pages/0042_*                  # specific line range\n"
            "  cut -f3 manifest.tsv | sort -u                  # all queries / URLs you've tried\n\n"
            "Each call is one-shot (cwd resets between calls). Chain with "
            "`;` or `&&` inside one keystrokes string.\n\n"
            "Read-only allow-list: only inspection commands are permitted "
            "(ls cat grep head tail sed cut sort uniq wc tr nl strings file find "
            "diff echo printf cd), composed with pipes / for-loops / conditionals. "
            "Anything else — destructive, network, interpreter, install, or "
            "redirection-to-file (rm, mv, curl, python, pip, `>`, …) — is blocked "
            "and returns `[blocked: …]`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keystrokes": {
                    "type": "string",
                    "description": "Shell command to run.",
                },
                "duration": {
                    "type": "number",
                    "description": "Max seconds to wait (1-60, default 10).",
                },
            },
            "required": ["keystrokes", "duration"],
        },
        "strict": False,
    },
]
