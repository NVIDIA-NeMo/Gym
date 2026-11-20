# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-free scorer cases: synthetic snapshots through the real verify path, with fixed judge replies.

Only the judge model call is replaced. Snapshot decoding and extraction, the artifact diff, prompt building,
reply parsing and rubric scoring all run as in a rollout.

``verify`` reports a grading failure as reward 0 with ``verifier_error`` set instead of raising, so that one bad
rollout does not stop a run. The fixture surfaces that field as an error, which keeps a failed grading distinct
from a judged zero.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef


if TYPE_CHECKING:
    from resources_servers.apex_agents.app import ApexFixtureRequest, ApexResourcesServer, ApexVerifyResponse


def create_fixture_server() -> ApexResourcesServer:
    """Construct an in-process verifier with no judge endpoint, world cache or artifact store."""
    from resources_servers.apex_agents.app import ApexResourcesServer, ApexResourcesServerConfig

    config = ApexResourcesServerConfig(
        host="127.0.0.1",
        port=9000,
        name="apex_agents_fixture",
        entrypoint="app.py",
        judge_model_server=ModelServerRef(type="responses_api_models", name="fixture_judge"),
        artifact_output_dir=None,
    )
    return ApexResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _snapshot(files: dict[str, str]) -> str:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _prompt_text(request_json: dict[str, Any]) -> str:
    content = request_json["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))


async def invoke_fixture(server: ApexResourcesServer, body: ApexFixtureRequest) -> ApexVerifyResponse:
    """Build the snapshots, answer each criterion's judge request with its fixed reply, and verify."""
    body.initial_artifact_snapshot_b64 = _snapshot({})
    if body.fixture_files is not None:
        body.artifact_snapshot_b64 = _snapshot(body.fixture_files)

    async def judge_post(*, json: dict[str, Any], **_: Any) -> MagicMock:
        prompt = _prompt_text(json)
        matches = [reply for criteria, reply in body.fixture_judge_replies.items() if criteria in prompt]
        if len(matches) != 1:
            raise AssertionError(f"expected one fixed judge reply for the request, found {len(matches)}")
        reply = MagicMock(ok=True)
        reply.read = AsyncMock(return_value=_completion(matches[0]))
        return reply

    server.server_client.post = AsyncMock(side_effect=judge_post)
    result = await server.verify(body)
    if result.verifier_error:
        raise ValueError(result.verifier_error)
    return result


def reseed_fixture(server: ApexResourcesServer, body: ApexFixtureRequest) -> None:
    """The scorer has no random state of its own; with fixed judge replies there is nothing to seed."""


def _completion(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode("utf-8")
