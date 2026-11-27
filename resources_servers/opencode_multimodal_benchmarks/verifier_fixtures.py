# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline verifier contracts: scripted judge transport, real extraction and scoring."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.client import ServerClient
from nemo_gym.verification import VerifierFixture
from resources_servers.opencode_multimodal_benchmarks.verifier import VisualVerifyRequest, VisualVerifyResponse


class FixtureRequest(VisualVerifyRequest):
    judge_reply: str | None = None
    judge_failure: bool = False


def _server(*, agentx: bool = False) -> object:
    # Import lazily because the entrypoints export these fixture descriptors.
    from resources_servers.opencode_multimodal_benchmarks.agentx import AgentXConfig, AgentXServer
    from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksConfig, VisualBenchmarksServer

    config_type = AgentXConfig if agentx else VisualBenchmarksConfig
    server_type = AgentXServer if agentx else VisualBenchmarksServer
    return server_type(
        config=config_type(
            name="fixture",
            host="localhost",
            port=1,
            entrypoint="app.py",
            judge_server="fixture_judge",
            judge_model="scripted-judge",
            judge_attempts=1,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def _agentx_server() -> object:
    return _server(agentx=True)


async def _invoke(server: object, body: FixtureRequest) -> VisualVerifyResponse:
    from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksServer

    assert isinstance(server, VisualBenchmarksServer)
    if body.judge_failure:
        server.server_client.post = AsyncMock(side_effect=RuntimeError("scripted judge unavailable"))
    elif body.judge_reply is None:
        server.server_client.post = AsyncMock(side_effect=AssertionError("Unexpected judge call"))
    else:
        http = MagicMock(status=200)
        http.read = AsyncMock(
            return_value=json.dumps(
                {
                    "id": "fixture-verdict",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "scripted-judge",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": body.judge_reply},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
        )
        server.server_client.post = AsyncMock(return_value=http)
    request = VisualVerifyRequest.model_validate(body.model_dump(exclude={"judge_reply", "judge_failure"}))
    return await server.verify(request)


def visual_fixture() -> VerifierFixture:
    """Exercise every scoring mode of the shared visual verifier."""
    return VerifierFixture(
        server_factory=_server,
        request_model=FixtureRequest,
        invoke=_invoke,
        cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
    )


def agentx_fixture() -> VerifierFixture:
    """Exercise objective, blank, malformed, and unavailable Agent-X judging."""
    return VerifierFixture(
        server_factory=_agentx_server,
        request_model=FixtureRequest,
        invoke=_invoke,
        cases_path=Path(__file__).parent / "tests/agentx_verifier_cases.jsonl",
    )
