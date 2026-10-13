# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier construction for the workload contract fixtures."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient


if TYPE_CHECKING:
    from resources_servers.asr_with_pc.app import ASRWithPCResourcesServer


def create_server() -> "ASRWithPCResourcesServer":
    from resources_servers.asr_with_pc.app import ASRWithPCConfig, ASRWithPCResourcesServer

    return ASRWithPCResourcesServer(
        config=ASRWithPCConfig(host="127.0.0.1", port=8080, entrypoint="app.py", name="asr_with_pc"),
        server_client=MagicMock(spec=ServerClient),
    )
