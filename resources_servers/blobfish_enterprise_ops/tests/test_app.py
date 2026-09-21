# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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
import asyncio
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.dataset import build_row
from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.server import (
    BlobfishSuiteResourcesServer,
    BlobfishSuiteResourcesServerConfig,
)
from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.suite import Suite, public_tool_name


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_suite"
SUITES = [
    ("blobfish_erpbench_100", "BLOBFISH_ERPBENCH_100_DIR"),
    ("blobfish_factorybench_100", "BLOBFISH_FACTORYBENCH_100_DIR"),
    ("blobfish_salesbench_100", "BLOBFISH_SALESBENCH_100_DIR"),
    ("blobfish_devopsbench_100", "BLOBFISH_DEVOPSBENCH_100_DIR"),
    ("blobfish_dealbench_100", "BLOBFISH_DEALBENCH_100_DIR"),
    ("blobfish_ledgerbench_100", "BLOBFISH_LEDGERBENCH_100_DIR"),
    ("blobfish_counselbench_100", "BLOBFISH_COUNSELBENCH_100_DIR"),
]


def _server(suite_dir):
    config = BlobfishSuiteResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="blobfish_test", suite_dir=str(suite_dir)
    )
    return BlobfishSuiteResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response():
    return {
        "id": "resp_test",
        "created_at": time.time(),
        "model": "test",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class TestApp:
    def test_seed_call_verify_on_the_fixture(self):
        suite = Suite.load(FIXTURE)
        task = suite.get("mini-001")
        server = _server(FIXTURE)
        row = build_row(task, suite.name)
        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json=row).status_code == 200
            for step in task.oracle_steps:
                result = client.post(f"/{public_tool_name(step['tool'])}", json=step.get("arguments") or {})
                assert result.status_code == 200 and "error" not in result.json(), result.text
            body = client.post("/verify", json=row | {"response": _response()}).json()
            assert body["reward"] == pytest.approx(1.0) and body["strict_pass"] == 1.0
        with TestClient(server.setup_webserver()) as client:  # no work: dense reward stays low, strict fails
            assert client.post("/seed_session", json=row).status_code == 200
            body = client.post("/verify", json=row | {"response": _response()}).json()
            assert body["strict_pass"] == 0.0 and body["reward"] < 1.0

    def test_onboarding_verifier_fixture(self):
        exercise = pytest.importorskip("nemo_gym.verifier_fixture").exercise_verifier_fixture
        from resources_servers.blobfish_enterprise_ops.app import VERIFIER_FIXTURE

        results = asyncio.run(
            exercise(VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="seeded")
        )
        assert [result.kind for result in results] == ["full_reward", "zero_reward", "malformed", "determinism"]

    @pytest.mark.parametrize("env_name,env_var", SUITES)
    def test_downloaded_suite_loads(self, env_name, env_var):
        suite_dir = os.environ.get(env_var)
        if not suite_dir or not Path(suite_dir).exists():
            pytest.skip(f"set {env_var} to a downloaded suite for {env_name}")
        server = _server(suite_dir)
        assert len(server.suite) > 0
        assert server.setup_webserver() is not None
