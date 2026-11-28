# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest

from resources_servers.eqbench4 import upstream_rating
from resources_servers.eqbench4.tests.stand_in_solver import STAND_IN_FILES


@pytest.fixture(autouse=True)
def stand_in_solver_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests rate with the stand-in solver (tests/stand_in_solver.py) in place of the downloaded one: its files
    are pinned instead of the upstream revision's, and the servers and preparations under test write them."""
    pins = {path: hashlib.sha256(data).hexdigest() for path, data in STAND_IN_FILES.items()}
    monkeypatch.setattr(upstream_rating, "SOLVER_SHA256", pins)
