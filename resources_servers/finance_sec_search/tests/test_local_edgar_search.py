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
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.server_utils import ServerClient
from resources_servers.finance_sec_search import local_edgar_search
from resources_servers.finance_sec_search.app import (
    EdgarSearchRequest,
    FinanceAgentResourcesServer,
    FinanceAgentResourcesServerConfig,
)
from resources_servers.finance_sec_search.local_edgar_search import (
    LocalEdgarSearch,
    canonical_url_key,
    default_sidecar_path,
    normalize_request,
    translate_query,
)
from resources_servers.finance_sec_search.scripts.build_local_edgar_metadata import build
from resources_servers.finance_sec_search.scripts.convert_questions import (
    EDGAR_SEARCH_TOOL,
    PROMPT,
    convert_entry,
)


# Indexed paths are container-absolute; only the part below data/ is portable.
DUMP_PREFIX = "/workspace/outputs/finance/demo/workflow-2-download-sec/step-0-download/data"


def _index(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            accession_number TEXT NOT NULL,
            cik TEXT NOT NULL,
            company_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            description TEXT,
            form_type TEXT NOT NULL,
            document_type TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            url TEXT NOT NULL,
            canonical_url_key TEXT NOT NULL,
            source_path TEXT NOT NULL,
            body TEXT NOT NULL
        );
        CREATE UNIQUE INDEX documents_url_key ON documents(canonical_url_key);
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            body,
            content='documents',
            content_rowid='id'
        );
        """
    )
    rows = [
        (
            1,
            "0000320193-24-000001",
            "320193",
            "Apple Inc.",
            "AAPL",
            "10-K",
            "10-K",
            "10-K",
            "2024-11-01",
            "https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/aapl.htm",
            "320193:000032019324000001:aapl.htm",
            f"{DUMP_PREFIX}/AAPL/10-K/2024/0000320193-24-000001/primary-document.html",
            "quantum pineapple net income",
        ),
        (
            2,
            "0000789019-25-000001",
            "789019",
            "Microsoft Corporation",
            "MSFT",
            "EX-99.1",
            "8-K",
            "EX-99.1",
            "2025-04-08",
            "https://www.sec.gov/Archives/edgar/data/789019/000078901925000001/msft-ex991.htm",
            "789019:000078901925000001:msft-ex991.htm",
            f"{DUMP_PREFIX}/MSFT/8-K/2025/0000789019-25-000001/exhibits/EX-99.1.html",
            "quantum pineapple guidance",
        ),
    ]
    connection.executemany(
        """
        INSERT INTO documents (
            id, accession_number, cik, company_name, ticker, description,
            form_type, document_type, filing_date, url, canonical_url_key,
            source_path, body
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.executemany(
        "INSERT INTO documents_fts(rowid, body) VALUES (?, ?)",
        [(row[0], row[-1]) for row in rows],
    )
    connection.commit()
    connection.close()
    return path


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


def test_converter_preserves_prompt_and_exposes_edgar_search() -> None:
    converted = convert_entry(
        {"question": "What was revenue?", "expected_answer": "Example"},
        search_tool="edgar_search",
    )

    params = converted["responses_create_params"]
    assert params["input"][0]["content"] == PROMPT + "What was revenue?"
    assert [tool["name"] for tool in params["tools"]] == [
        "retrieve_information",
        "parse_html_page",
        "edgar_search",
        "submit_final_result",
    ]
    assert params["tools"][2] == EDGAR_SEARCH_TOOL


def test_query_language_translation() -> None:
    assert translate_query("apple revenue") == "apple AND revenue"
    assert translate_query('"net income"') == '"net income"'
    assert translate_query("apple OR microsoft") == "(apple OR microsoft)"
    assert translate_query("software NOT hardware") == "software NOT hardware"
    assert translate_query("cyber*") == "cyber*"


def test_search_contract_filters_and_cutoff(tmp_path: Path) -> None:
    search = LocalEdgarSearch(_index(tmp_path / "index.sqlite"))

    results = search.search(
        "quantum pineapple",
        form_types=["10-K"],
        ciks=["0000320193"],
        end_date="2030-01-01",
    )

    assert results == [
        {
            "accessionNo": "0000320193-24-000001",
            "cik": "320193",
            "companyNameLong": "Apple Inc.",
            "ticker": "AAPL",
            "description": "10-K",
            "formType": "10-K",
            "type": "10-K",
            "filingUrl": ("https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/aapl.htm"),
            "filedAt": "2024-11-01",
        }
    ]


def test_match_all_and_filtered_browse_fallback(tmp_path: Path) -> None:
    search = LocalEdgarSearch(_index(tmp_path / "index.sqlite"))

    match_all = search.search("*", form_types=["10-K"], ciks=["0000320193"])
    fallback = search.search(
        "Apple annual report 2024",
        form_types=["10-K"],
        ciks=["320193"],
    )

    assert match_all[0]["accessionNo"] == "0000320193-24-000001"
    assert fallback[0]["accessionNo"] == "0000320193-24-000001"


def test_index_schema_is_validated_at_startup(tmp_path: Path) -> None:
    path = tmp_path / "invalid.sqlite"
    sqlite3.connect(path).close()

    with pytest.raises(ValueError, match="documents"):
        LocalEdgarSearch(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"search_query": ""}, "search_query"),
        ({"search_query": "x", "start_date": "not-a-date"}, "start_date"),
        ({"search_query": "x", "form_types": "10-K"}, "form_types"),
        ({"search_query": "x", "ciks": ["AAPL"]}, "numeric strings"),
        ({"search_query": "x", "page": 0}, "page"),
        ({"search_query": "x", "top_n_results": 101}, "top_n_results"),
    ],
)
def test_request_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_request(**kwargs)


@pytest.mark.asyncio
async def test_server_routes_edgar_search_to_local_index(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    config = _server_config(
        tmp_path,
        local_edgar_index_path=str(_index(tmp_path / "index.sqlite")),
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


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://www.sec.gov/Archives/edgar/data/0000066740/000006674026000246/MMM-20260630.htm",
            "66740:000006674026000246:mmm-20260630.htm",
        ),
        (
            "https://www.sec.gov/Archives/edgar/data/66740/000006674026000246/ex%2010-1.htm",
            "66740:000006674026000246:ex 10-1.htm",
        ),
        ("https://example.test/not-edgar.htm", None),
        ("https://www.sec.gov/Archives/edgar/data/66740/000006674026000246/", None),
    ],
)
def test_canonical_url_key_normalizes_cik_and_filename(url: str, expected: str | None) -> None:
    assert canonical_url_key(url) == expected


def test_dump_paths_cover_primary_documents_and_exhibits(tmp_path: Path) -> None:
    search = LocalEdgarSearch(_index(tmp_path / "index.sqlite"))

    resolved = search.dump_paths_for_urls(
        [
            "https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/aapl.htm",
            "https://www.sec.gov/Archives/edgar/data/789019/000078901925000001/msft-ex991.htm",
            "https://www.sec.gov/Archives/edgar/data/1/000000000000000001/absent.htm",
        ]
    )

    assert resolved == {
        "320193:000032019324000001:aapl.htm": "AAPL/10-K/2024/0000320193-24-000001/primary-document.html",
        "789019:000078901925000001:msft-ex991.htm": "MSFT/8-K/2025/0000789019-25-000001/exhibits/EX-99.1.html",
    }


def test_dump_paths_are_unavailable_without_the_columns(tmp_path: Path) -> None:
    """An index built before source_path existed degrades instead of erroring."""
    search = LocalEdgarSearch(_varied_index(tmp_path / "legacy.sqlite", documents=4))

    assert search.supports_dump_paths is False
    assert search.dump_paths_for_urls(["https://www.sec.gov/Archives/edgar/data/1/2/a.htm"]) == {}


def _varied_index(path: Path, documents: int = 400) -> Path:
    """An index broad enough that filters, paging and ranking all have work to do."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            accession_number TEXT NOT NULL,
            cik TEXT NOT NULL,
            company_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            description TEXT,
            form_type TEXT NOT NULL,
            document_type TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            url TEXT NOT NULL,
            body TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            body,
            content='documents',
            content_rowid='id'
        );
        """
    )
    rows = []
    for number in range(1, documents + 1):
        company = number % 7
        rows.append(
            (
                number,
                f"000-{number:06d}",
                str(300000 + company),
                f"Company {company}",
                f"TCK{company}",
                "10-K" if number % 2 else "EX-1",
                "10-K" if number % 2 else "8-K",
                "10-K" if number % 2 else "EX-1",
                f"202{number % 5}-0{1 + number % 9}-1{number % 9}",
                f"https://example.test/{number}.htm",
                ("revenue growth " * (number % 9 + 1)) + ("pineapple" if number % 13 == 0 else ""),
            )
        )
    connection.executemany(
        """
        INSERT INTO documents (
            id, accession_number, cik, company_name, ticker, description,
            form_type, document_type, filing_date, url, body
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.executemany(
        "INSERT INTO documents_fts(rowid, body) VALUES (?, ?)",
        [(row[0], row[-1]) for row in rows],
    )
    connection.commit()
    connection.close()
    return path


SIDECAR_PARITY_QUERIES = [
    {"search_query": "revenue"},
    {"search_query": "revenue", "form_types": ["10-K"]},
    {"search_query": "revenue", "ciks": ["300001"]},
    {"search_query": "revenue", "form_types": ["10-K"], "ciks": ["300003"]},
    {"search_query": "pineapple"},
    {"search_query": "revenue growth"},
    {"search_query": "revenue OR pineapple"},
    {"search_query": "reven*"},
    {"search_query": "revenue NOT pineapple"},
    {"search_query": "*", "form_types": ["10-K"]},
    {"search_query": "revenue", "top_n_results": 7, "page": 2},
    {"search_query": "revenue", "start_date": "2022-01-01", "end_date": "2024-12-31"},
    {"search_query": "Company annual report", "ciks": ["300002"]},
]


@pytest.mark.parametrize("query", SIDECAR_PARITY_QUERIES, ids=lambda q: q["search_query"])
def test_metadata_sidecar_returns_identical_results(tmp_path: Path, query: dict) -> None:
    index = _varied_index(tmp_path / "index.sqlite")
    without_sidecar = LocalEdgarSearch(index, max_end_date="2030-01-01")
    assert not without_sidecar.uses_metadata_sidecar

    build(index, default_sidecar_path(index))
    with_sidecar = LocalEdgarSearch(index, max_end_date="2030-01-01")
    assert with_sidecar.uses_metadata_sidecar

    assert with_sidecar.search(**query) == without_sidecar.search(**query)


def test_sidecar_beside_the_index_is_discovered(tmp_path: Path) -> None:
    index = _varied_index(tmp_path / "index.sqlite")
    build(index, default_sidecar_path(index))

    assert LocalEdgarSearch(index).metadata_path == default_sidecar_path(index)


def test_configured_sidecar_must_exist(tmp_path: Path) -> None:
    index = _index(tmp_path / "index.sqlite")

    with pytest.raises(FileNotFoundError, match="sidecar"):
        LocalEdgarSearch(index, metadata_path=tmp_path / "absent.metadata")


def test_sidecar_built_from_another_index_is_rejected(tmp_path: Path) -> None:
    index = _varied_index(tmp_path / "index.sqlite")
    other = _varied_index(tmp_path / "other.sqlite", documents=200)
    build(other, default_sidecar_path(other))

    with pytest.raises(ValueError, match="covers 200 documents"):
        LocalEdgarSearch(index, metadata_path=default_sidecar_path(other))


def test_sidecar_is_rejected_when_the_index_changed_underneath_it(tmp_path: Path) -> None:
    index = _varied_index(tmp_path / "index.sqlite")
    build(index, default_sidecar_path(index))

    # id 1 is always sampled by the fingerprint, so this edit is caught deterministically
    # rather than with the sampling probability a random row would carry.
    connection = sqlite3.connect(index)
    connection.execute("UPDATE documents SET accession_number = '000-999999' WHERE id = 1")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="built from a different index"):
        LocalEdgarSearch(index, max_end_date="2030-01-01")


def test_large_index_without_a_sidecar_is_rejected(tmp_path: Path, monkeypatch) -> None:
    index = _varied_index(tmp_path / "index.sqlite")
    monkeypatch.setattr(local_edgar_search, "SLOW_METADATA_LIMIT_BYTES", 1)

    with pytest.raises(ValueError, match="no\nmetadata sidecar|no metadata sidecar"):
        LocalEdgarSearch(index, max_end_date="2030-01-01")

    build(index, default_sidecar_path(index))
    assert LocalEdgarSearch(index, max_end_date="2030-01-01").uses_metadata_sidecar


def test_server_refuses_to_boot_when_the_sidecar_is_required(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(local_edgar_search, "SLOW_METADATA_LIMIT_BYTES", 1)
    config = _server_config(
        tmp_path,
        local_edgar_index_path=str(_index(tmp_path / "index.sqlite")),
    )

    # Must fail the whole server, not degrade edgar_search to unavailable.
    with pytest.raises(ValidationError, match="no metadata sidecar"):
        FinanceAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_index_missing_a_metadata_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            cik TEXT NOT NULL,
            body TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            body,
            content='documents',
            content_rowid='id'
        );
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="missing required columns"):
        LocalEdgarSearch(path)


def test_server_refuses_to_boot_on_a_malformed_index(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    sqlite3.connect(path).close()
    config = _server_config(tmp_path, local_edgar_index_path=str(path))

    # Must fail the whole server, not degrade edgar_search to unavailable.
    with pytest.raises(ValidationError, match="missing required tables"):
        FinanceAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.asyncio
async def test_server_uses_sidecar_when_configured(tmp_path: Path) -> None:
    index = _index(tmp_path / "index.sqlite")
    sidecar = default_sidecar_path(index)
    build(index, sidecar)

    server = FinanceAgentResourcesServer(
        config=_server_config(
            tmp_path,
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


@pytest.mark.asyncio
async def test_edgar_search_requires_local_index_configuration(tmp_path: Path) -> None:
    server = FinanceAgentResourcesServer(
        config=_server_config(tmp_path),
        server_client=MagicMock(spec=ServerClient),
    )

    response = await server.edgar_search(
        _request(),
        EdgarSearchRequest(search_query="revenue"),
    )

    assert "local_edgar_index_path is not configured" in response.results
