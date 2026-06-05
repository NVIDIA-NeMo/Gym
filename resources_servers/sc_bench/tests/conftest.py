# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for sc_bench resources server tests."""

from pathlib import Path

import pytest

from resources_servers.sc_bench.supchain_tools import configure_data_dir


CSV_DIR = Path(__file__).resolve().parents[1] / "data" / "csv"
LOCAL_SC_BENCH = Path(__file__).resolve().parents[3] / "SC-bench" / "data"


def _csv_available() -> bool:
    return (CSV_DIR / "TradeOrders.csv").exists() or (LOCAL_SC_BENCH / "TradeOrders.csv").exists()


pytestmark = pytest.mark.skipif(
    not _csv_available(),
    reason="SC-bench CSV tables not present; run ng_prepare_benchmark first",
)


@pytest.fixture(scope="module", autouse=True)
def configure_csv_dir():
    if (CSV_DIR / "TradeOrders.csv").exists():
        configure_data_dir(CSV_DIR)
    else:
        configure_data_dir(LOCAL_SC_BENCH)
