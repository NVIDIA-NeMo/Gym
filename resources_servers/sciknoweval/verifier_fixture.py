# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier construction for the workload contract fixtures."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient


if TYPE_CHECKING:
    from resources_servers.sciknoweval.app import SciKnowEvalResourcesServer


def create_server() -> "SciKnowEvalResourcesServer":
    from resources_servers.sciknoweval.app import SciKnowEvalResourcesServer, SciKnowEvalResourcesServerConfig

    return SciKnowEvalResourcesServer(
        config=SciKnowEvalResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="sciknoweval",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
