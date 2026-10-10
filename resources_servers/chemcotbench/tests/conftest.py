# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare ChemCoTBench's external dependencies before test collection."""

import os

import pytest

from resources_servers.chemcotbench.setup_molopt import ensure_molopt_runtime
from resources_servers.chemcotbench.setup_upstream import ensure_data, ensure_repository


def pytest_configure(config: pytest.Config) -> None:
    """Reuse or install the pinned evaluator, reference data, and MolOpt runtime."""
    ensure_repository(os.environ.get("CHEMCOTBENCH_TEST_REPO"))
    ensure_data(os.environ.get("CHEMCOTBENCH_TEST_DATA"))
    ensure_molopt_runtime(
        os.environ.get("CHEMCOTBENCH_TEST_MOLOPT_PYTHON"),
        os.environ.get("CHEMCOTBENCH_TEST_ORACLE_DIR"),
    )
