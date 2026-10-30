# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free verifier-fixture cases: fixed persisted episodes, re-graded on the session-less path.

Each case names its episode by a ``deliverables_dir`` relative to the fixture directory, which the fixture server uses
as its ``artifacts_dir``. The cases use deterministic checks only, so no judge or task backend is reached.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import BaseModel

from nemo_gym.client import ServerClient


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "verifier_fixture"


def create_fixture_server() -> object:
    """A fresh server whose ``artifacts_dir`` holds the fixture's persisted episodes."""
    from resources_servers.agentic_if.app import AgenticIFResourcesServer, AgenticIFResourcesServerConfig

    config = AgenticIFResourcesServerConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        name="agentic_if",
        judge_model_server={"type": "responses_api_models", "name": "agentic_if_judge_model"},
        task_resources={
            "swe": {"type": "resources_servers", "name": "scale_swe_resources_server"},
            "tb21": {"type": "resources_servers", "name": "terminal_bench_2_1_resources_server"},
        },
        artifacts_dir=FIXTURE_DIR,
    )
    return AgenticIFResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def invoke_fixture(server: object, request: BaseModel) -> object:
    """Verify without a live session, as re-verification does."""
    return await server.verify(SimpleNamespace(session={}), request)
