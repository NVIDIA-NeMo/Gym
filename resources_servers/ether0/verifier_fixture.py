# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier construction for the workload contract fixtures."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from nemo_gym.client import ServerClient


if TYPE_CHECKING:
    from resources_servers.ether0.app import Ether0ResourcesServer


def create_server() -> "Ether0ResourcesServer":
    from resources_servers.ether0.app import Ether0ResourcesServer, Ether0ResourcesServerConfig

    return Ether0ResourcesServer(
        config=Ether0ResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="ether0",
            # The fixture cases use graders that need no oracle files.
            local_oracles=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
