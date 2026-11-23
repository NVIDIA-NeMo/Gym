# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier construction for the verifier fixture (kept out of the production ``app.py``)."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from nemo_gym.client import ServerClient


if TYPE_CHECKING:
    from resources_servers.asr_with_pc.app import ASRWithPCResourcesServer


def create_server() -> "ASRWithPCResourcesServer":
    from resources_servers.asr_with_pc.app import ASRWithPCConfig, ASRWithPCResourcesServer

    return ASRWithPCResourcesServer(
        config=ASRWithPCConfig(name="asr_with_pc", host="127.0.0.1", port=8080, entrypoint="app.py", task_type="ASR"),
        server_client=MagicMock(spec=ServerClient),
    )
