# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Lean server is stateless with a replayable verification for partial-rollout checkpoints."""

import os
import socket
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from omegaconf import DictConfig

from nemo_gym.server_utils import ServerClient
from resources_servers.math_formal_lean.app import (
    MathFormalLeanResourcesServer,
    MathFormalLeanResourcesServerConfig,
    MathFormalLeanVerifyRequest,
)


AUTH = {"authorization": "Bearer t"}
SANDBOX_HOST = os.environ.get("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1")
SANDBOX_PORT = int(os.environ.get("NEMO_SKILLS_SANDBOX_PORT", "6000"))


def _sandbox_reachable() -> bool:
    try:
        with socket.create_connection((SANDBOX_HOST, SANDBOX_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def _server(*, checkpoint: bool = False) -> MathFormalLeanResourcesServer:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "t"}} if checkpoint else {}
    )
    config = MathFormalLeanResourcesServerConfig(
        host="", port=0, entrypoint="", name="math_formal_lean", sandbox_host=SANDBOX_HOST, sandbox_port=SANDBOX_PORT
    )
    return MathFormalLeanResourcesServer(config=config, server_client=server_client)


def _response(text: str) -> dict:
    return {
        "id": "response",
        "created_at": 1.0,
        "model": "model",
        "object": "response",
        "output": [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def _verify_body(generation: str, *, turn_index: int = 2, statement: str = "example : 2 = 2 := by\n") -> dict:
    return {
        "responses_create_params": {"input": "Prove two equals two"},
        "response": _response(generation),
        "header": "import Mathlib\n",
        "formal_statement": statement,
        "turn_index": turn_index,
    }


def _control(checkpoint_id: str = "c1") -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5}


@pytest.mark.parametrize(
    ("generation", "compiler_output", "expected_reward"),
    [
        ("rfl", {"process_status": "completed", "stdout": "", "stderr": ""}, 1.0),
        ("wrong_tactic", {"process_status": "error", "stdout": "", "stderr": "unknown tactic 'wrong_tactic'"}, 0.0),
    ],
)
async def test_verification_replays_on_a_fresh_server(
    generation: str, compiler_output: dict, expected_reward: float
) -> None:
    request = MathFormalLeanVerifyRequest.model_validate(_verify_body(generation))
    server = _server()
    # An unrelated verification first: a replayed one must not depend on anything an earlier one left behind.
    server._sandbox_client.execute_lean4 = AsyncMock(
        return_value={"process_status": "error", "stdout": "", "stderr": "unrelated proof error"}
    )
    await server.verify(
        MathFormalLeanVerifyRequest.model_validate(
            _verify_body("wrong_tactic", turn_index=0, statement="example : False := by\n")
        )
    )

    results = []
    for verifier in (server, _server()):
        verifier._sandbox_client.execute_lean4 = AsyncMock(return_value=compiler_output)
        result = await verifier.verify(request)
        verifier._sandbox_client.execute_lean4.assert_awaited_once_with(
            code=f"import Mathlib\nexample : 2 = 2 := by\n{generation}", timeout=30.0
        )
        assert result.reward == expected_reward
        assert result.turn_index == 2
        assert result.needs_correction is (expected_reward == 0.0)
        if result.needs_correction:
            assert compiler_output["stderr"] in result.error_feedback
            assert generation in result.correction_prompt
            assert "unrelated proof error" not in result.correction_prompt
        results.append(result.model_dump())

    assert results[0] == results[1]


async def test_checkpoint_declaration_lets_verification_run_while_prepared(tmp_path: Path) -> None:
    server = _server(checkpoint=True)
    server._sandbox_client.execute_lean4 = AsyncMock(
        return_value={"process_status": "completed", "stdout": "", "stderr": ""}
    )
    app = server.setup_webserver()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://lean") as client:
        seed = await client.post("/ng-rollout/r/seed_session", json={"responses_create_params": {"input": "x"}})
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=_control(), headers=AUTH)
        # A replayed verification arrives from an episode the checkpoint did not wait for.
        during = await client.post("/ng-rollout/r/verify", json=_verify_body("rfl"))
        exported = await client.post(
            "/ng-control/v1/checkpoint/commit", json=_control() | {"checkpoint_dir": str(tmp_path)}, headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=_control(), headers=AUTH)

    assert seed.status_code == 200
    assert seed.headers["x-ng-checkpoint-verify"] == "replay"
    assert (status["mode"], status["verify"]) == ("stateless", "replay")
    assert prepared.json()["phase"] == "prepared"
    assert during.status_code == 200 and during.json()["reward"] == 1.0
    assert exported.status_code == 200 and exported.json()["manifest"]["record_count"] == 0


@pytest.mark.skipif(not _sandbox_reachable(), reason="needs a Lean sandbox at NEMO_SKILLS_SANDBOX_HOST:PORT")
@pytest.mark.parametrize(("generation", "expected_reward"), [("rfl", 1.0), ("exact absurd", 0.0)])
async def test_real_compilation_replays_to_the_same_result(generation: str, expected_reward: float) -> None:
    request = MathFormalLeanVerifyRequest.model_validate(
        _verify_body(generation) | {"header": "", "formal_statement": "example : 2 = 2 := by\n"}
    )
    first = await _server().verify(request)
    replayed = await _server().verify(request)

    assert first.reward == expected_reward
    assert first.model_dump(exclude={"compiler_output"}) == replayed.model_dump(exclude={"compiler_output"})
