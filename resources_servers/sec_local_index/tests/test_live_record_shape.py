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
"""The fields a search result carries, against real sec-api.io responses.

A model trained against local mode and evaluated against live mode reads the
same JSON either way, so the local rows have to carry what sec-api.io carries.
The fixture holds responses recorded from the live API.
"""

from __future__ import annotations

import json
from pathlib import Path

from resources_servers.sec_local_index.local_edgar_search import LocalEdgarSearch, normalize_request
from resources_servers.sec_local_index.tests.index_fixtures import build_index


RECORDS = json.loads((Path(__file__).parent / "data" / "sec_api_search_records.json").read_text())

# description is the one field sec-api.io omits rather than nulls.
ALWAYS_PRESENT_LIVE_FIELDS = set.intersection(*(set(record) for record in RECORDS))
ANY_LIVE_FIELD = set().union(*(set(record) for record in RECORDS))


def _local_record(tmp_path: Path) -> dict[str, object]:
    search = LocalEdgarSearch(build_index(tmp_path / "index.sqlite"), max_end_date="2026-03-01")
    try:
        results = search.execute(normalize_request(search_query="quantum pineapple", max_end_date="2026-03-01"))
    finally:
        search.close()
    assert results
    return results[0]


def test_a_local_result_carries_every_field_sec_api_returns(tmp_path: Path) -> None:
    assert ANY_LIVE_FIELD <= set(_local_record(tmp_path))


def test_a_local_result_invents_no_field_of_its_own(tmp_path: Path) -> None:
    assert set(_local_record(tmp_path)) <= ANY_LIVE_FIELD


def test_the_fields_sec_api_always_sends_are_never_null_locally(tmp_path: Path) -> None:
    record = _local_record(tmp_path)

    assert all(record[field] is not None for field in ALWAYS_PRESENT_LIVE_FIELDS)
