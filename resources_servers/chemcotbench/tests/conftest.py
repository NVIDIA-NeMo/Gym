# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare ChemCoTBench's external dependencies before test collection."""

import os
from urllib.error import URLError

import pytest

from resources_servers.chemcotbench.setup_molopt import ensure_molopt_runtime
from resources_servers.chemcotbench.setup_upstream import ensure_data, ensure_repository


# Set when the MolOpt oracles cannot be downloaded; tests that score molecule optimization skip with it.
_MOLOPT_UNAVAILABLE: str | None = None


def pytest_configure(config: pytest.Config) -> None:
    """Reuse or install the pinned evaluator, reference data, and MolOpt runtime."""
    global _MOLOPT_UNAVAILABLE
    ensure_repository(os.environ.get("CHEMCOTBENCH_TEST_REPO"))
    ensure_data(os.environ.get("CHEMCOTBENCH_TEST_DATA"))
    try:
        ensure_molopt_runtime(
            os.environ.get("CHEMCOTBENCH_TEST_MOLOPT_PYTHON"),
            os.environ.get("CHEMCOTBENCH_TEST_ORACLE_DIR"),
        )
    except (URLError, TimeoutError) as error:
        # Some CI networks cannot reach Harvard Dataverse; the rest of the suite needs no oracle.
        _MOLOPT_UNAVAILABLE = f"MolOpt oracles could not be downloaded: {error}"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _MOLOPT_UNAVAILABLE is None:
        return
    skip = pytest.mark.skip(reason=_MOLOPT_UNAVAILABLE)
    for item in items:
        if "optimization_server" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)
