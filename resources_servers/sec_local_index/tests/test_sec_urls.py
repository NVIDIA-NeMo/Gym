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
"""Each derivation the SEC URL parser exposes to its callers."""

from __future__ import annotations

import pytest

from resources_servers.sec_local_index.sec_urls import parse_sec_archives_url


PRIMARY = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"


def test_primary_document_derivations() -> None:
    parsed = parse_sec_archives_url(PRIMARY)

    assert parsed is not None
    assert parsed.cik == "320193"
    assert parsed.padded_cik == "0000320193"
    assert parsed.unpadded_cik == "320193"
    assert parsed.accession == "000032019324000123"
    assert parsed.dashed_accession == "0000320193-24-000123"
    assert parsed.document_path == "aapl-20240928.htm"
    assert parsed.document_basename == "aapl-20240928.htm"
    assert parsed.flat_document == "aapl-20240928.htm"


@pytest.mark.parametrize(
    "url",
    [
        PRIMARY + "?query=1",
        PRIMARY + "#section",
        PRIMARY + "?query=1#section",
    ],
)
def test_query_and_fragment_are_not_part_of_the_document(url: str) -> None:
    parsed = parse_sec_archives_url(url)

    assert parsed is not None
    assert parsed.document_path == "aapl-20240928.htm"


def test_exhibit_below_the_accession_keeps_its_subpath() -> None:
    parsed = parse_sec_archives_url(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/exhibits/ex-21.htm"
    )

    assert parsed is not None
    assert parsed.document_path == "exhibits/ex-21.htm"
    assert parsed.document_basename == "ex-21.htm"
    assert parsed.flat_document == "exhibits_ex-21.htm"


def test_basename_is_percent_decoded_and_lowercased() -> None:
    parsed = parse_sec_archives_url(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/AAPL%20Report.HTM"
    )

    assert parsed is not None
    assert parsed.document_basename == "aapl report.htm"
    assert parsed.document_path == "AAPL%20Report.HTM"


def test_accession_of_an_unexpected_length_is_left_alone() -> None:
    parsed = parse_sec_archives_url("https://www.sec.gov/Archives/edgar/data/320193/12345/doc.htm")

    assert parsed is not None
    assert parsed.dashed_accession == "12345"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/report.html",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany",
        "https://www.sec.gov/Archives/edgar/data/320193/",
    ],
)
def test_non_document_urls_do_not_parse(url: str) -> None:
    assert parse_sec_archives_url(url) is None
