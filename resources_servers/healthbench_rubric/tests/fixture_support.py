# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline fixtures: replace only the external judge, exercising the real verifier."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.healthbench_rubric.app import (
    HealthBenchRubricResourcesServer,
    HealthBenchRubricResourcesServerConfig,
)


def make_server(**overrides: Any) -> HealthBenchRubricResourcesServer:
    config = HealthBenchRubricResourcesServerConfig(
        host="127.0.0.1",
        port=11000,
        entrypoint="app.py",
        judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        judge_chat_completion_params={"messages": [], "temperature": 0.5, "max_tokens": 2048},
        **overrides,
    )
    return HealthBenchRubricResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def fixture_server() -> HealthBenchRubricResourcesServer:
    server = make_server()

    async def grade(conversation, rubric):
        return {"criteria_met": conversation.endswith("assistant: fixture correct"), "explanation": "Scripted judge"}

    server._grade_rubric = AsyncMock(side_effect=grade)
    return server
