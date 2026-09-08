# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="",
            sandbox_config=dict(),
            host="",
            port=0,
            entrypoint="",
            name="",
        )
        TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
