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
"""scripts/build_local_prices.py builds a store that local price_history serves."""

import datetime
import importlib.util
import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from resources_servers.finance_agent_v2.local_tools import LocalPriceHistory


_SCRIPT_FPATH = Path(__file__).resolve().parents[1] / "scripts" / "build_local_prices.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_local_prices", _SCRIPT_FPATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _price_args(ticker: str, start: str, end: str) -> dict:
    return {"ticker": ticker, "start_date": start, "end_date": end, "asset_class": "equity"}


@pytest.mark.asyncio
async def test_csv_input_is_served_by_local_price_history(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    source.write_text(
        "date,ticker,open,high,low,close,volume,vendor_id\n"
        "2024-01-03,aaa,11,12,10,11.5,100,x\n"
        "2024-01-02,AAA,10,11,9,10.5,100,x\n"
        "2024-01-02,BRK.B,400,401,399,400.5,5,x\n"
    )

    summary = builder.build([source], tmp_path / "store", drop_columns=["volume"], overwrite=False)
    output = await LocalPriceHistory(tmp_path / "store").execute(
        _price_args("AAA", "2024-01-01", "2024-01-31"), {}, logging.getLogger(__name__)
    )

    assert summary == {"tickers": 2, "skipped_tickers": 0, "rows": 3}
    assert output.output == "date,open,high,low,close\n2024-01-02,10.0,11.0,9.0,10.5\n2024-01-03,11.0,12.0,10.0,11.5"
    brk = await LocalPriceHistory(tmp_path / "store").execute(
        _price_args("BRK-B", "2024-01-02", "2024-01-02"), {}, logging.getLogger(__name__)
    )
    assert brk.output.splitlines()[1] == "2024-01-02,400.0,401.0,399.0,400.5"


def test_parquet_dates_and_missing_values(tmp_path: Path) -> None:
    table = pa.table(
        {
            "date": pa.array([datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)], pa.date32()),
            "ticker": ["AAA", "AAA"],
            "close": [10.0, 11.0],
            "adjClose": [5.0, None],
        }
    )
    pq.write_table(table, tmp_path / "prices.parquet")

    builder.build([tmp_path], tmp_path / "store", drop_columns=[], overwrite=False)
    records = [json.loads(line) for line in (tmp_path / "store" / "equity" / "AAA.jsonl").read_text().splitlines()]

    assert records == [
        {"date": "2024-01-02", "close": 10.0, "adjClose": 5.0},
        {"date": "2024-01-03", "close": 11.0, "adjClose": None},
    ]


@pytest.mark.asyncio
async def test_adjusted_only_input_is_served_as_adjusted(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    source.write_text("date,ticker,adjClose\n2024-01-02,AAA,5.25\n")

    builder.build([source], tmp_path / "store", drop_columns=[], overwrite=False)
    output = await LocalPriceHistory(tmp_path / "store").execute(
        _price_args("AAA", "2024-01-02", "2024-01-02"), {}, logging.getLogger(__name__)
    )

    assert output.output == "date,adjClose\n2024-01-02,5.25"


def test_input_without_prices_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    source.write_text("date,ticker,volume\n2024-01-02,AAA,100\n")

    with pytest.raises(SystemExit, match="no price column"):
        builder.build([source], tmp_path / "store", drop_columns=[], overwrite=False)


def test_duplicate_dates_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    source.write_text("date,ticker,close\n2024-01-02,AAA,10\n2024-01-02,AAA,11\n")

    with pytest.raises(SystemExit, match="more than one row"):
        builder.build([source], tmp_path / "store", drop_columns=[], overwrite=False)


def test_unusable_tickers_are_skipped(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    source.write_text("date,ticker,close\n2024-01-02,../AAA,10\n2024-01-02,BBB,11\n")

    summary = builder.build([source], tmp_path / "store", drop_columns=[], overwrite=False)

    assert summary["skipped_tickers"] == 1
    assert [path.name for path in (tmp_path / "store" / "equity").iterdir()] == ["BBB.jsonl"]


def test_an_existing_store_needs_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    source.write_text("date,ticker,close\n2024-01-02,AAA,10\n")
    builder.build([source], tmp_path / "store", drop_columns=[], overwrite=False)

    with pytest.raises(SystemExit, match="already exists"):
        builder.build([source], tmp_path / "store", drop_columns=[], overwrite=False)
    builder.build([source], tmp_path / "store", drop_columns=[], overwrite=True)
