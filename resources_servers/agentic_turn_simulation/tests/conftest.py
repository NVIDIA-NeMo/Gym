# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for every test module."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def sleep():
    """Retry backoffs return at once; a test that takes `sleep` can check the delays."""
    with patch("asyncio.sleep", new=AsyncMock()) as mock:
        yield mock
