# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

import pytest

from responses_api_agents.codex_agent.setup_codex import ensure_node


def pytest_configure(config: pytest.Config) -> None:
    # The sandbox launcher runs on Node; install it before collection so skip guards see it. A failed install
    # fails the run, so the launcher and runner tests never skip silently on Linux.
    if sys.platform == "linux":
        ensure_node()
