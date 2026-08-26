#!/usr/bin/env python3
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
"""Build the metadata sidecar used by the local EDGAR search backend.

The full-text index stores each filing's body in the same table as its metadata,
so reading a filing's form type or date pulls a multi-kilobyte record. Searches
need those columns for every candidate match, which turns a common-term query
into tens of thousands of large random reads.

This script copies the non-body columns into a separate, much smaller database.
Search reads metadata from the sidecar and never touches the body column, which
leaves query results identical while removing the dominant I/O cost.

Usage:
    python build_local_edgar_metadata.py --index INDEX.sqlite [--output SIDECAR]

The default output path is the one the search backend looks for automatically:
the index path with a ``.metadata`` suffix.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from resources_servers.finance_sec_search.local_edgar_search import (  # noqa: E402
    METADATA_COLUMNS,
    SIDECAR_SCHEMA_VERSION,
    default_sidecar_path,
    fingerprint_source_index,
)


BATCH_SIZE = 20_000


def build(index_path: Path, output_path: Path, *, batch_size: int = BATCH_SIZE) -> None:
    if not index_path.is_file():
        raise SystemExit(f"Index not found: {index_path}")

    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    for stale in (temporary_path, temporary_path.with_name(temporary_path.name + "-journal")):
        if stale.exists():
            stale.unlink()

    source = sqlite3.connect(f"file:{index_path}?mode=ro&immutable=1", uri=True)
    try:
        total = source.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        print(f"source documents: {total}", flush=True)

        destination = sqlite3.connect(str(temporary_path))
        try:
            # A larger page size cuts the number of reads needed to warm the
            # sidecar from shared storage.
            destination.execute("PRAGMA page_size = 8192")
            destination.execute("PRAGMA journal_mode = OFF")
            destination.execute("PRAGMA synchronous = OFF")
            _create_schema(destination)

            projection = ", ".join(METADATA_COLUMNS)
            cursor = source.execute(f"SELECT {projection} FROM documents ORDER BY id")
            placeholders = ", ".join("?" for _ in METADATA_COLUMNS)
            insert = f"INSERT INTO documents_meta ({projection}) VALUES ({placeholders})"

            copied = 0
            started = time.monotonic()
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                destination.executemany(insert, rows)
                destination.commit()
                copied += len(rows)
                elapsed = time.monotonic() - started
                rate = copied / elapsed if elapsed else 0.0
                print(
                    f"  copied {copied}/{total} ({100 * copied / total:.1f}%) {rate:.0f} rows/s",
                    flush=True,
                )

            if copied != total:
                raise SystemExit(f"Copied {copied} rows but the index holds {total}")

            print("creating indexes", flush=True)
            _create_indexes(destination)
            _record_provenance(destination, source, document_count=total)
            destination.commit()
            destination.execute("VACUUM")
            destination.commit()
        finally:
            destination.close()
    finally:
        source.close()

    os.replace(temporary_path, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"wrote {output_path} ({size_mb:.0f} MB)", flush=True)


def _create_schema(destination: sqlite3.Connection) -> None:
    destination.execute(
        """
        CREATE TABLE documents_meta (
            id INTEGER PRIMARY KEY,
            accession_number TEXT NOT NULL,
            cik TEXT NOT NULL,
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            form_type TEXT NOT NULL,
            document_type TEXT NOT NULL,
            description TEXT,
            filing_date TEXT NOT NULL,
            url TEXT NOT NULL
        )
        """
    )
    destination.execute(
        """
        CREATE TABLE sidecar_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _create_indexes(destination: sqlite3.Connection) -> None:
    destination.execute("CREATE INDEX documents_meta_cik ON documents_meta(cik, form_type, filing_date)")
    destination.execute("CREATE INDEX documents_meta_form ON documents_meta(form_type, filing_date)")
    destination.execute("CREATE INDEX documents_meta_date ON documents_meta(filing_date)")


def _record_provenance(
    destination: sqlite3.Connection,
    source: sqlite3.Connection,
    *,
    document_count: int,
) -> None:
    entries = {
        "schema_version": str(SIDECAR_SCHEMA_VERSION),
        "document_count": str(document_count),
        "source_fingerprint": fingerprint_source_index(source),
        "built_at_unix_seconds": str(int(time.time())),
    }
    destination.executemany(
        "INSERT OR REPLACE INTO sidecar_metadata (key, value) VALUES (?, ?)",
        sorted(entries.items()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path, help="Full-text index to read")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Sidecar destination (defaults to the index path plus '.metadata')",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    arguments = parser.parse_args()

    output = arguments.output or default_sidecar_path(arguments.index)
    build(arguments.index, output, batch_size=arguments.batch_size)


if __name__ == "__main__":
    main()
