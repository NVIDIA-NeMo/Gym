# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the pinned upstream checkout exists before collection.

The verifier reads its rubrics and category rules from that checkout, so without it the
tests would fail on a missing file rather than with a useful message. Fetching is
idempotent and a no-op once the checkout is already at the pinned revision.
"""

from benchmarks.kidbench.prepare import ensure_upstream


def pytest_configure(config) -> None:
    ensure_upstream()
