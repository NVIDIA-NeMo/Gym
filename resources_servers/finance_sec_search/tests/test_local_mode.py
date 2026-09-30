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
"""What edgar_search_mode selects, and what the server does once it has selected it.

The engine underneath is exercised in
resources_servers/sec_local_index/tests.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.server_utils import ServerClient
from resources_servers.finance_sec_search.app import (
    EdgarSearchRequest,
    FinanceAgentResourcesServer,
    FinanceAgentResourcesServerConfig,
)
from resources_servers.sec_local_index import local_edgar_search
from resources_servers.sec_local_index.live_edgar_search import LiveEdgarSearch
from resources_servers.sec_local_index.local_edgar_search import default_sidecar_path
from resources_servers.sec_local_index.scripts.build_local_edgar_metadata import build
from resources_servers.sec_local_index.tests.index_fixtures import build_index


def _server_config(tmp_path: Path, **overrides: object) -> FinanceAgentResourcesServerConfig:
    prompt_dir = Path(__file__).resolve().parents[1] / "prompt_templates"
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "finance_sec_search_test",
        "cache_dir": str(tmp_path / "cache"),
        "judge_prompt_template_fpath": str(prompt_dir / "finance_sec_search_judge.yaml"),
        "retrieval_system_prompt_fpath": str(prompt_dir / "finance_sec_search_retrieval.yaml"),
    }
    values.update(overrides)
    return FinanceAgentResourcesServerConfig(**values)


def _request() -> MagicMock:
    request = MagicMock()
    request.session = {"session_id": "test-session"}
    return request


def test_local_mode_without_an_index_fails_at_startup(tmp_path: Path) -> None:
    """Refused at boot rather than per search, which would surface as a rollout
    of failed tool calls."""
    with pytest.raises(ValueError, match="local_edgar_index_path is not set"):
        FinanceAgentResourcesServer(
            config=_server_config(tmp_path, edgar_search_mode="local"),
            server_client=MagicMock(spec=ServerClient),
        )


def test_live_mode_without_a_key_fails_at_startup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sec_api_key is not set"):
        FinanceAgentResourcesServer(
            config=_server_config(tmp_path, edgar_search_mode="live"),
            server_client=MagicMock(spec=ServerClient),
        )


def test_live_mode_uses_sec_api(tmp_path: Path) -> None:
    server = FinanceAgentResourcesServer(
        config=_server_config(tmp_path, edgar_search_mode="live", sec_api_key="test-key"),
        server_client=MagicMock(spec=ServerClient),
    )

    assert isinstance(server._edgar_search_service._backend, LiveEdgarSearch)


def test_an_index_alone_does_not_turn_on_edgar_search(tmp_path: Path) -> None:
    server = FinanceAgentResourcesServer(
        config=_server_config(tmp_path, local_edgar_index_path=str(build_index(tmp_path / "index.sqlite"))),
        server_client=MagicMock(spec=ServerClient),
    )

    assert server._edgar_search_service is None


def test_live_mode_wins_over_a_configured_index(tmp_path: Path) -> None:
    """Eval against sec-api.io while the training index stays configured."""
    server = FinanceAgentResourcesServer(
        config=_server_config(
            tmp_path,
            edgar_search_mode="live",
            sec_api_key="test-key",
            local_edgar_index_path=str(build_index(tmp_path / "index.sqlite")),
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    assert isinstance(server._edgar_search_service._backend, LiveEdgarSearch)


@pytest.mark.asyncio
async def test_server_routes_edgar_search_to_local_index(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    config = _server_config(
        tmp_path,
        edgar_search_mode="local",
        local_edgar_index_path=str(build_index(tmp_path / "index.sqlite")),
        local_edgar_metrics_dir=str(metrics_dir),
        max_end_date="2025-04-07",
    )
    server = FinanceAgentResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )

    response = await server.edgar_search(
        _request(),
        EdgarSearchRequest(
            search_query="quantum pineapple",
            form_types=["10-K"],
            ciks=["320193"],
        ),
    )

    results = json.loads(response.results)
    assert results[0]["ticker"] == "AAPL"
    metric_files = list(metrics_dir.glob("search-*.jsonl"))
    assert len(metric_files) == 1
    metric = json.loads(metric_files[0].read_text(encoding="utf-8"))
    assert metric["result_count"] == 1
    assert "completed_at_unix_seconds" in metric


@pytest.mark.asyncio
async def test_server_uses_sidecar_when_configured(tmp_path: Path) -> None:
    index = build_index(tmp_path / "index.sqlite")
    sidecar = default_sidecar_path(index)
    build(index, sidecar)

    server = FinanceAgentResourcesServer(
        config=_server_config(
            tmp_path,
            edgar_search_mode="local",
            local_edgar_index_path=str(index),
            local_edgar_metadata_path=str(sidecar),
            max_end_date="2025-04-07",
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    response = await server.edgar_search(
        _request(),
        EdgarSearchRequest(search_query="quantum pineapple", form_types=["10-K"]),
    )

    assert json.loads(response.results)[0]["ticker"] == "AAPL"


def test_server_refuses_to_boot_when_the_sidecar_is_required(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(local_edgar_search, "SLOW_METADATA_LIMIT_BYTES", 1)
    config = _server_config(
        tmp_path,
        edgar_search_mode="local",
        local_edgar_index_path=str(build_index(tmp_path / "index.sqlite")),
    )

    # Must fail the whole server, not degrade edgar_search to unavailable.
    with pytest.raises(ValidationError, match="no metadata sidecar"):
        FinanceAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_server_refuses_to_boot_on_a_malformed_index(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    sqlite3.connect(path).close()
    config = _server_config(tmp_path, edgar_search_mode="local", local_edgar_index_path=str(path))

    # Must fail the whole server, not degrade edgar_search to unavailable.
    with pytest.raises(ValidationError, match="missing required tables"):
        FinanceAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.asyncio
async def test_edgar_search_is_unavailable_when_no_mode_is_set(tmp_path: Path) -> None:
    server = FinanceAgentResourcesServer(
        config=_server_config(tmp_path),
        server_client=MagicMock(spec=ServerClient),
    )

    response = await server.edgar_search(
        _request(),
        EdgarSearchRequest(search_query="revenue"),
    )

    assert "edgar_search_mode is not set" in response.results
