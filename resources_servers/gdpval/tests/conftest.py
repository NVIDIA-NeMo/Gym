# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def tavily_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The benchmark and standalone configs require the key and no longer default it."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
