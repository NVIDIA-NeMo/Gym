# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier construction for the workload contract fixtures.

The fixture exercises the rule-based answer match; the optional LLM judge is configured per benchmark.
"""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from nemo_gym.client import ServerClient


if TYPE_CHECKING:
    from resources_servers.agentic_mme.app import AgenticMMEServer


def create_server() -> "AgenticMMEServer":
    from resources_servers.agentic_mme.app import AgenticMMEConfig, AgenticMMEServer

    return AgenticMMEServer(
        config=AgenticMMEConfig(host="127.0.0.1", port=8080, entrypoint="app.py", name="agentic_mme"),
        server_client=MagicMock(spec=ServerClient),
    )
