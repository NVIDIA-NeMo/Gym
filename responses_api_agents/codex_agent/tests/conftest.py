# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import sys

import pytest

from responses_api_agents.codex_agent.setup_codex import ensure_node


def pytest_configure(config: pytest.Config) -> None:
    # The sandbox launcher runs on Node; install it before collection so skip guards see it.
    if sys.platform != "linux":
        return
    try:
        ensure_node()
    except Exception:
        logging.getLogger(__name__).warning("Node.js is unavailable; Codex launcher tests will skip", exc_info=True)
