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
"""Everything the SQLite-backed engine returns, independent of either server."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from resources_servers.sec_local_index import local_edgar_search
from resources_servers.sec_local_index.local_edgar_search import (
    LocalEdgarSearch,
    OutOfCoverageError,
    canonical_url_key,
    default_sidecar_path,
    normalize_request,
    translate_query,
)
from resources_servers.sec_local_index.scripts.build_local_edgar_metadata import build
from resources_servers.sec_local_index.tests.index_fixtures import build_index, build_varied_index


# The two cutoffs the library is asked to serve: finance_sec_search runs against
# the Vals v1 benchmark, finance_agent_v2 against finance_agent.tools.MAX_END_DATE.
V1_CUTOFF = "2025-04-07"
V2_CUTOFF = "2026-03-01"


def test_query_language_translation() -> None:
    assert translate_query("apple revenue") == "apple AND revenue"
    assert translate_query('"net income"') == '"net income"'
    assert translate_query("apple OR microsoft") == "(apple OR microsoft)"
    assert translate_query("software NOT hardware") == "software NOT hardware"
    assert translate_query("cyber*") == "cyber*"


def test_search_contract_filters_and_cutoff(tmp_path: Path) -> None:
    search = LocalEdgarSearch(build_index(tmp_path / "index.sqlite"), max_end_date=V1_CUTOFF)

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
    search = LocalEdgarSearch(build_index(tmp_path / "index.sqlite"), max_end_date=V1_CUTOFF)

    match_all = search.search("*", form_types=["10-K"], ciks=["0000320193"])
    fallback = search.search(
        "Apple annual report 2024",
        form_types=["10-K"],
        ciks=["320193"],
    )

    assert match_all[0]["accessionNo"] == "0000320193-24-000001"
    assert fallback[0]["accessionNo"] == "0000320193-24-000001"


def test_cutoff_comes_from_the_caller(tmp_path: Path) -> None:
    """One engine, two lineages: the fixture holds an 8-K filed 2025-04-08, one
    day past V1's cutoff and well inside V2's."""
    index = build_index(tmp_path / "index.sqlite")

    v1 = LocalEdgarSearch(index, max_end_date=V1_CUTOFF)
    v2 = LocalEdgarSearch(index, max_end_date=V2_CUTOFF)

    assert [row["ticker"] for row in v1.search("quantum pineapple")] == ["AAPL"]
    assert sorted(row["ticker"] for row in v2.search("quantum pineapple")) == ["AAPL", "MSFT"]


def test_max_end_date_has_no_default(tmp_path: Path) -> None:
    """Omitting it must fail loudly rather than inherit the other lineage's cutoff."""
    with pytest.raises(TypeError, match="max_end_date"):
        LocalEdgarSearch(build_index(tmp_path / "index.sqlite"))

    with pytest.raises(TypeError, match="max_end_date"):
        normalize_request("quantum pineapple")


def test_an_unknown_sidecar_version_is_refused(tmp_path: Path) -> None:
    index = build_varied_index(tmp_path / "index.sqlite")
    sidecar = default_sidecar_path(index)
    build(index, sidecar)
    connection = sqlite3.connect(sidecar)
    connection.execute("UPDATE sidecar_metadata SET value = '99' WHERE key = 'schema_version'")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="expected 1"):
        LocalEdgarSearch(index, max_end_date="2030-01-01")


def test_coverage_reports_the_indexed_span(tmp_path: Path) -> None:
    search = LocalEdgarSearch(build_index(tmp_path / "index.sqlite"), max_end_date=V2_CUTOFF)

    assert search.coverage == ("2024-11-01", "2025-04-08")


def test_window_outside_the_corpus_is_an_error_not_an_empty_list(tmp_path: Path) -> None:
    """An empty list reads as 'nothing matched', which sends the model looking for
    a better query when the corpus simply does not reach that far."""
    search = LocalEdgarSearch(build_index(tmp_path / "index.sqlite"), max_end_date=V2_CUTOFF)

    with pytest.raises(OutOfCoverageError, match="2024-11-01 through 2025-04-08"):
        search.search("quantum pineapple", start_date="2019-01-01", end_date="2019-12-31")


def test_window_inside_the_corpus_still_returns_an_empty_list(tmp_path: Path) -> None:
    """Only coverage is special-cased; a genuine miss stays a miss."""
    search = LocalEdgarSearch(build_index(tmp_path / "index.sqlite"), max_end_date=V2_CUTOFF)

    assert search.search("nonexistent terminology") == []


def test_partial_overlap_with_the_corpus_is_served(tmp_path: Path) -> None:
    search = LocalEdgarSearch(build_index(tmp_path / "index.sqlite"), max_end_date=V2_CUTOFF)

    results = search.search("quantum pineapple", start_date="1900-01-01", end_date="2024-12-31")

    assert [row["ticker"] for row in results] == ["AAPL"]


def test_index_schema_is_validated_at_startup(tmp_path: Path) -> None:
    path = tmp_path / "invalid.sqlite"
    sqlite3.connect(path).close()

    with pytest.raises(ValueError, match="documents"):
        LocalEdgarSearch(path, max_end_date=V1_CUTOFF)


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
        normalize_request(**kwargs, max_end_date=V1_CUTOFF)


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
    search = LocalEdgarSearch(build_index(tmp_path / "index.sqlite"), max_end_date=V1_CUTOFF)

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
    search = LocalEdgarSearch(build_varied_index(tmp_path / "legacy.sqlite", documents=4), max_end_date=V1_CUTOFF)

    assert search.supports_dump_paths is False
    assert search.dump_paths_for_urls(["https://www.sec.gov/Archives/edgar/data/1/2/a.htm"]) == {}


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
    index = build_varied_index(tmp_path / "index.sqlite")
    without_sidecar = LocalEdgarSearch(index, max_end_date="2030-01-01")
    assert not without_sidecar.uses_metadata_sidecar

    build(index, default_sidecar_path(index))
    with_sidecar = LocalEdgarSearch(index, max_end_date="2030-01-01")
    assert with_sidecar.uses_metadata_sidecar

    assert with_sidecar.search(**query) == without_sidecar.search(**query)


def test_sidecar_beside_the_index_is_discovered(tmp_path: Path) -> None:
    index = build_varied_index(tmp_path / "index.sqlite")
    build(index, default_sidecar_path(index))

    assert LocalEdgarSearch(index, max_end_date=V1_CUTOFF).metadata_path == default_sidecar_path(index)


def test_configured_sidecar_must_exist(tmp_path: Path) -> None:
    index = build_index(tmp_path / "index.sqlite")

    with pytest.raises(FileNotFoundError, match="sidecar"):
        LocalEdgarSearch(index, max_end_date=V1_CUTOFF, metadata_path=tmp_path / "absent.metadata")


def test_sidecar_built_from_another_index_is_rejected(tmp_path: Path) -> None:
    index = build_varied_index(tmp_path / "index.sqlite")
    other = build_varied_index(tmp_path / "other.sqlite", documents=200)
    build(other, default_sidecar_path(other))

    with pytest.raises(ValueError, match="covers 200 documents"):
        LocalEdgarSearch(index, max_end_date=V1_CUTOFF, metadata_path=default_sidecar_path(other))


def test_sidecar_is_rejected_when_the_index_changed_underneath_it(tmp_path: Path) -> None:
    index = build_varied_index(tmp_path / "index.sqlite")
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
    index = build_varied_index(tmp_path / "index.sqlite")
    monkeypatch.setattr(local_edgar_search, "SLOW_METADATA_LIMIT_BYTES", 1)

    with pytest.raises(ValueError, match="no\nmetadata sidecar|no metadata sidecar"):
        LocalEdgarSearch(index, max_end_date="2030-01-01")

    build(index, default_sidecar_path(index))
    assert LocalEdgarSearch(index, max_end_date="2030-01-01").uses_metadata_sidecar


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
        LocalEdgarSearch(path, max_end_date=V1_CUTOFF)
