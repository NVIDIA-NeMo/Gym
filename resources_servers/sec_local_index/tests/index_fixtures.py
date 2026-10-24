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
"""Indexes in the shape the builder produces, small enough to assert against.

Shared with the servers' own suites so a schema change is felt in one place.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


# Indexed paths are container-absolute; only the part below data/ is portable.
DUMP_PREFIX = "/workspace/outputs/finance/demo/workflow-2-download-sec/step-0-download/data"

FULL_SCHEMA = """
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

LEGACY_SCHEMA = """
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


def build_index(path: Path) -> Path:
    """Two filings, one either side of the V1 cutoff (2025-04-07)."""
    connection = sqlite3.connect(path)
    connection.executescript(FULL_SCHEMA)
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


def build_varied_index(path: Path, documents: int = 400) -> Path:
    """An index broad enough that filters, paging and ranking all have work to do.

    Built on the pre-source_path schema, so it doubles as the older-index case.
    """
    connection = sqlite3.connect(path)
    connection.executescript(LEGACY_SCHEMA)
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
