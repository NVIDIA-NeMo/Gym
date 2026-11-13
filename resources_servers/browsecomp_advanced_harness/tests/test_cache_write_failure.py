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
"""A cache write that fails after a paid provider call must not fail the tool call."""

import os
import stat
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    BrowseRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


_EXCLUDE = str(Path(__file__).with_name("dummy_exclude_domains_file.json"))


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
async def test_unwritable_cache_entry_dirs_do_not_fail_search_or_browse(tmp_path) -> None:
    config = TavilySearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        tavily_api_key="test_key",  # pragma: allowlist secret
        exclude_domains_file_path=_EXCLUDE,
        search_cache_dir=str(tmp_path / "c"),
    )
    server = TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    # Quota, ENOSPC or a permission change mid-run: the entry directories stop accepting writes.
    dirs = [tmp_path / "c" / "search", tmp_path / "c" / "browse"]
    for d in dirs:
        d.chmod(stat.S_IRUSR | stat.S_IXUSR)
    server._async_tavily_clients = [
        MagicMock(
            search=AsyncMock(
                return_value={
                    "results": [{"title": "T", "url": "https://a.example/1", "content": "s", "raw_content": "LIVE"}]
                }
            ),
            extract=AsyncMock(return_value={"results": [{"url": "https://a.example/1", "raw_content": "PAGE"}]}),
        )
    ]
    request = Request({"type": "http", "session": {SESSION_ID_KEY: "s"}, "headers": []})
    try:
        search = await server.search(request, TavilySearchRequest(queries=["q1", "q2"]))
        browse = await server.browse(request, BrowseRequest(urls=["https://a.example/1"]))
    finally:
        for d in dirs:
            d.chmod(stat.S_IRWXU)
    assert search.results_string.count("LIVE") == 2
    assert "PAGE" in browse.results_string
