# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier construction for the workload contract fixtures."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient


if TYPE_CHECKING:
    from resources_servers.chemcotbench.app import (
        ChemCoTBenchResourcesServer,
        ChemCoTBenchVerifyRequest,
        ChemCoTBenchVerifyResponse,
    )


def create_server() -> "ChemCoTBenchResourcesServer":
    from resources_servers.chemcotbench.app import ChemCoTBenchResourcesServer, ChemCoTBenchResourcesServerConfig

    return ChemCoTBenchResourcesServer(
        config=ChemCoTBenchResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="chemcotbench",
            enable_molopt=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


async def invoke(
    server: "ChemCoTBenchResourcesServer", request: "ChemCoTBenchVerifyRequest"
) -> "ChemCoTBenchVerifyResponse":
    """The fixture runner creates a fresh server for each case; reap its workers."""
    try:
        return await server.verify(request)
    finally:
        await server.close()
