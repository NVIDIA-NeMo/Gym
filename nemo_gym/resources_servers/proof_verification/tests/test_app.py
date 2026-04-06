# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from nemo_gym.resources_servers.proof_verification.app import (
    ProofVerificationResourcesServer,
    ProofVerificationResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = ProofVerificationResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        )
        ProofVerificationResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
