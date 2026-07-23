# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from resources_servers.proof_judge.app import (
    ProofWithJudgeResourcesServer,
    ProofWithJudgeResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = ProofWithJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        )
        ProofWithJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
