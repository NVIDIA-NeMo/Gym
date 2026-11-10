# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.agentic_if_realworld import rollout_driver


@pytest.fixture(autouse=True)
def stable_runtime_identity(monkeypatch):
    """Pin the driver's source fingerprint so a resume within one test sees the same identity.

    CI runs other servers' tests in the same checkout at the same time, and they can add or delete source
    files between a run and its resume. The test_runtime_fingerprint_* tests cover runtime_identity() itself.
    """
    monkeypatch.setattr(rollout_driver, "runtime_identity", lambda: {"sources": {"pinned": "for tests"}})
