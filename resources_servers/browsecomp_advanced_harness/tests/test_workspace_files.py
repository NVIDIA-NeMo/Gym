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
"""/workspace_files lets a sandboxed agent copy the session's pages into its sandbox."""

import os
from unittest.mock import MagicMock

from pytest import fixture

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    WorkspaceFilesRequest,
)


_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dummy_exclude_domains_file.json"
)


def _server(**overrides) -> TavilySearchResourcesServer:
    config = TavilySearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        search_provider="exa",
        exa_api_key="test_exa_key",  # pragma: allowlist secret
        exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
        **overrides,
    )
    return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@fixture
def req() -> MagicMock:
    m = MagicMock()
    m.session = {SESSION_ID_KEY: "sid"}
    return m


async def test_returns_pages_after_since_plus_manifest(req, tmp_path) -> None:
    server = _server(workspace="per_session", workspace_root=str(tmp_path))
    writer = server._get_page_writer("sid")
    first = writer.write_browse_page("https://a.example/x", "A", "alpha body")
    second = writer.write_search_result("q", 1, "B", "https://b.example", "beta body")

    everything = await server.workspace_files(req, WorkspaceFilesRequest(since=0))
    assert everything.enabled
    assert [f.path for f in everything.files] == [first, second, "manifest.tsv"]
    assert everything.files[0].content == (tmp_path / "sid" / first).read_text()
    assert everything.next == 2

    newer = await server.workspace_files(req, WorkspaceFilesRequest(since=1))
    assert [f.path for f in newer.files] == [second, "manifest.tsv"]
    assert newer.files[-1].content == (tmp_path / "sid" / "manifest.tsv").read_text()

    nothing_new = await server.workspace_files(req, WorkspaceFilesRequest(since=2))
    assert nothing_new.files == []
    assert nothing_new.next == 2


async def test_disabled_workspace_reports_disabled(req) -> None:
    result = await _server().workspace_files(req, WorkspaceFilesRequest(since=0))
    assert not result.enabled
    assert result.files == []
