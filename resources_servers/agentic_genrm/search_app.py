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
"""Web search and page browsing for the agentic GenRM, served to OpenCode over Gym MCP.

The search RL harness (browsecomp_advanced_harness) unchanged underneath: provider keys, the domain
exclusion list, the per-result contamination filter and the shared cache all stay on this host, and
the sandbox reaches the tools only through Gym MCP. On top of it:

- Only `search` and `browse` are exposed, with the descriptions the browsecomp agent gives the model
  (the harness routes carry none, and its read-only bash and workspace routes do not apply here).
- Leak filters for preference data. HelpSteer3 and the arena preference sets are public with their
  labels, so a judge that finds its own row reads the answer. Extra excluded domains, result-URL
  substrings and result-text patterns come from config and extend the harness's own lists.
- Per-session metrics are bounded. A tool server never receives /verify, which is what clears them
  in the harness.
"""

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional

import mcp.types as types

import resources_servers.browsecomp_advanced_harness.app as harness
from nemo_gym.mcp_auto_exposure import MCPTool
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


TOOL_SPECS = {
    "search": types.Tool(
        name="search",
        description=(
            "Web Search API, works like Google Search. All queries will be searched in parallel. "
            "If you want to search with multiple keywords, put them in a single query."
        ),
        inputSchema={
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
    ),
    "browse": types.Tool(
        name="browse",
        description=(
            "Visit specific webpage(s) and return their full text content. "
            "Use this to read the complete content of web pages found during search."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URL(s) of webpage(s) to visit.",
                },
                "goal": {"type": "string", "description": "What specific information you are looking for."},
            },
            "required": ["urls"],
        },
    ),
}


class _BoundedDefaultDict(OrderedDict):
    """defaultdict that drops its oldest keys beyond max_size."""

    def __init__(self, factory: Callable[[], object], max_size: int) -> None:
        super().__init__()
        self._factory = factory
        self._max_size = max_size

    def __missing__(self, key: str) -> object:
        value = self[key] = self._factory()
        while len(self) > self._max_size:
            self.popitem(last=False)
        return value


class GenRMSearchResourcesServerConfig(harness.TavilySearchResourcesServerConfig):
    use_judge: bool = False
    extra_exclude_domains: List[str] = []
    # Case-insensitive substrings of a result's URL.
    extra_contaminated_url_substrings: List[str] = []
    # Case-insensitive substrings anywhere in a serialized result (URL, title, text).
    extra_contamination_patterns: List[str] = []
    max_tracked_sessions: int = 20000


class GenRMSearchResourcesServer(harness.TavilySearchResourcesServer):
    config: GenRMSearchResourcesServerConfig

    def model_post_init(self, __context) -> None:
        # Servers run from their own directory; read a Gym-root-relative exclusion path from the root.
        exclusions = Path(self.config.exclude_domains_file_path)
        if not exclusions.is_absolute() and not exclusions.exists():
            self.config.exclude_domains_file_path = str(Path(__file__).resolve().parents[2] / exclusions)
        super().model_post_init(__context)
        self._exclude_domains.extend(d for d in self.config.extra_exclude_domains if d not in self._exclude_domains)
        # The harness filters through module-level lists; each server runs in its own process.
        for patterns, extra in (
            (harness.CONTAMINATED_URL_SUBSTRINGS, self.config.extra_contaminated_url_substrings),
            (harness.CONTAMINATION_PATTERNS, self.config.extra_contamination_patterns),
        ):
            patterns.extend(p.lower() for p in extra if p.lower() not in patterns)
        self._session_id_to_metrics = _BoundedDefaultDict(
            harness.TavilySearchMetrics, self.config.max_tracked_sessions
        )
        print(f"Excluded domains (with GenRM extras): {self._exclude_domains}")
        print(f"Contaminated URL substrings: {harness.CONTAMINATED_URL_SUBSTRINGS}")
        print(f"Contamination patterns: {harness.CONTAMINATION_PATTERNS}")

    def mcp_tools(self, harvested: List[MCPTool], catchall: Optional[object]) -> List[MCPTool]:
        return [replace(tool, tool=TOOL_SPECS[tool.name]) for tool in harvested if tool.name in TOOL_SPECS]


if __name__ == "__main__":
    GenRMSearchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GenRMSearchResourcesServer.run_webserver()  # noqa: F401
