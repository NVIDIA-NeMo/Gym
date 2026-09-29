#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the local price store read by ``price_history_mode: local``.

Input is a table of daily prices, as Parquet or CSV files (or directories of
them), with one row per ticker and trading day:

    date,ticker,open,high,low,close
    2024-01-02,AAPL,187.15,188.44,183.89,185.64

``date``, ``ticker`` and at least one price column are required. Every column
that upstream ``price_history`` prints (``open``, ``high``, ``low``, ``close``,
``adjClose``, ``volume``, ...) is carried through; other columns are ignored.
The output is ``<output>/equity/<TICKER>.jsonl`` with records in ascending
date order.

Usage (from the resource server venv):
    python scripts/build_local_prices.py daily_prices.parquet --output /data/local_prices
    python scripts/build_local_prices.py csv_dir/ --output /data/local_prices --drop-columns volume
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq
from finance_agent.tools import PriceHistory


REQUIRED = ("date", "ticker")
PRICE_COLUMNS = tuple(PriceHistory._COLUMNS[1:])
COUNT_COLUMNS = {"volume", "adjVolume", "tradesDone"}
VALUE_COLUMNS = ("open", "high", "low", "close", "adjOpen", "adjHigh", "adjLow", "adjClose")
# Must match LocalPriceHistory._SYMBOL; files for tickers it rejects could never be served.
SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


def input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix in {".parquet", ".csv"})
        else:
            files.append(path)
    if not files:
        raise SystemExit("no .parquet or .csv input files found")
    return files


def read_table(path: Path) -> pa.Table:
    if path.suffix == ".parquet":
        table = pq.read_table(path)
    elif path.suffix == ".csv":
        table = pv.read_csv(path, convert_options=pv.ConvertOptions(strings_can_be_null=True))
    else:
        raise SystemExit(f"unsupported input {path}; use .parquet or .csv")
    missing = [column for column in REQUIRED if column not in table.column_names]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(missing)}")
    if not any(column in table.column_names for column in VALUE_COLUMNS):
        raise SystemExit(f"{path} has no price column; expected one of {', '.join(VALUE_COLUMNS)}")
    keep = [column for column in ("date", "ticker", *PRICE_COLUMNS) if column in table.column_names]
    table = table.select(keep)
    for column in keep:
        if column in PRICE_COLUMNS and column not in COUNT_COLUMNS:
            table = table.set_column(keep.index(column), column, table[column].cast(pa.float64()))
    dates = table["date"]
    if pa.types.is_string(dates.type) or pa.types.is_large_string(dates.type):
        dates = pc.utf8_slice_codeunits(dates, 0, 10)
    else:
        dates = pc.strftime(dates.cast(pa.timestamp("s")), format="%Y-%m-%d")
    table = table.set_column(table.column_names.index("date"), "date", dates)
    tickers = pc.utf8_upper(pc.utf8_trim_whitespace(table["ticker"].cast(pa.string())))
    return table.set_column(table.column_names.index("ticker"), "ticker", tickers)


def clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build(paths: list[Path], output: Path, drop_columns: list[str], overwrite: bool) -> dict[str, int]:
    equity = output / "equity"
    if equity.exists():
        if not overwrite:
            raise SystemExit(f"{equity} already exists; pass --overwrite to replace it")
        shutil.rmtree(equity)
    tables = [read_table(path) for path in input_files(paths)]
    table = pa.concat_tables(tables, promote_options="default")
    columns = [column for column in table.column_names if column not in drop_columns]
    if not any(column in columns for column in VALUE_COLUMNS):
        raise SystemExit("--drop-columns would leave no price column")
    table = table.select(columns).sort_by([("ticker", "ascending"), ("date", "ascending")])

    equity.mkdir(parents=True)
    written = skipped = 0
    offset = 0
    for item in pc.value_counts(table["ticker"]).to_pylist():
        ticker, count = item["values"], item["counts"]
        rows = table.slice(offset, count).to_pylist()
        offset += count
        if not ticker or not SYMBOL.match(ticker):
            skipped += 1
            continue
        dates = [row["date"] for row in rows]
        if len(set(dates)) != len(dates):
            raise SystemExit(f"{ticker} has more than one row for the same date")
        with (equity / f"{ticker}.jsonl").open("w") as handle:
            for row in rows:
                record = {key: clean(value) for key, value in row.items() if key != "ticker"}
                handle.write(json.dumps(record) + "\n")
        written += 1
    return {"tickers": written, "skipped_tickers": skipped, "rows": table.num_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local price store for price_history_mode: local.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Parquet or CSV files, or directories of them.")
    parser.add_argument("--output", type=Path, required=True, help="Store root; use it as local_pricing_dir.")
    parser.add_argument("--drop-columns", nargs="*", default=[], help="Price columns to leave out, e.g. volume.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing <output>/equity.")
    args = parser.parse_args()
    unknown = sorted(set(args.drop_columns) - set(PRICE_COLUMNS))
    if unknown:
        print(f"ERROR: unknown columns for --drop-columns: {', '.join(unknown)}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(build(args.inputs, args.output, args.drop_columns, args.overwrite)))


if __name__ == "__main__":
    main()
