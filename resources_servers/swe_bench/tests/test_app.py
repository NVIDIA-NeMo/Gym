# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import resources_servers.swe_bench.tests.test_swe_env  # noqa: F401  — registers fake-swe provider
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.swe_bench.app import (
    SweBenchResourcesServer,
    SweBenchResourcesServerConfig,
    SweBenchSeedSessionRequest,
    SweBenchVerifyRequest,
)


@pytest.fixture
def server() -> SweBenchResourcesServer:
    return SweBenchResourcesServer(
        config=SweBenchResourcesServerConfig(
            host="127.0.0.1",
            port=12346,
            entrypoint="app.py",
            name="swe_bench",
            sandbox_provider={"fake-swe": {}},
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def _sample_row() -> dict:
    inst = {
        "instance_id": "astropy__astropy-12907",
        "base_commit": "abc123",
        "test_patch": "",
        "FAIL_TO_PASS": '["tests/test_x.py::a"]',
        "PASS_TO_PASS": '["tests/test_x.py::b"]',
    }
    meta = {
        "instance_id": "astropy__astropy-12907",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "problem_statement": "Fix the bug.",
        "instance_dict": json.dumps(inst),
    }
    return {
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Fix the bug."}],
            metadata=meta,
        ),
        "verifier_metadata": meta,
    }


@pytest.mark.asyncio
async def test_seed_session_agent_in_env(server: SweBenchResourcesServer) -> None:
    body = SweBenchSeedSessionRequest(**_sample_row())
    resp = await server.seed_session(body)
    assert resp.environment == "swe_bench"
    assert resp.placement.topology == "agent_in_env"
    assert resp.sandbox.spec["image"].startswith("swebench/")
    assert resp.task.task_id == "astropy__astropy-12907"
    assert resp.task.harness_family == "swe-bench"
    assert resp.task.dataset_name == "princeton-nlp/SWE-bench_Verified"
    assert resp.verifier_metadata["instance_id"] == "astropy__astropy-12907"


@pytest.mark.asyncio
async def test_verify_empty_patch(server: SweBenchResourcesServer) -> None:
    row = _sample_row()
    row["verifier_metadata"] = {**row["verifier_metadata"], "model_patch": ""}
    body = SweBenchVerifyRequest(
        **row,
        response=NeMoGymResponse(
            id="r1",
            created_at=0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        ),
    )
    resp = await server.verify(body)
    assert resp.task_id == "astropy__astropy-12907"
    assert resp.environment == "swe_bench"
    assert resp.reward == 0.0
    assert resp.patch_exists is False
    assert resp.resolved is False
