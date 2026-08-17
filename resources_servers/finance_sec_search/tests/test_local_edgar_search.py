from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.finance_sec_search.app import (
    EdgarSearchRequest,
    FinanceAgentResourcesServer,
    FinanceAgentResourcesServerConfig,
)
from resources_servers.finance_sec_search.local_edgar_search import (
    LocalEdgarSearch,
    normalize_request,
    translate_query,
)
from resources_servers.finance_sec_search.scripts.convert_questions import (
    EDGAR_SEARCH_TOOL,
    PROMPT,
    convert_entry,
)


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
            body TEXT NOT NULL
        );
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
            "quantum pineapple guidance",
        ),
    ]
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
