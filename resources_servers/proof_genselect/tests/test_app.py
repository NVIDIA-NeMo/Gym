# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.proof_genselect.app import (
    ProofGenSelectResourcesServer,
    ProofGenSelectResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = ProofGenSelectResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        ProofGenSelectResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
