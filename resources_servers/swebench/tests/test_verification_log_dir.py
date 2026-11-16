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
"""The SWE-bench harness logs stay out of the server's source folder and outlive a verify only when asked to."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.client import ServerClient
from nemo_gym.config.global_config import RESULTS_DIR_KEY_NAME
from nemo_gym.core import runtime_files
from nemo_gym.sandbox import SandboxHandle
from resources_servers.swebench import app as swebench_app
from resources_servers.swebench.app import SwebenchResourcesServer, SwebenchResourcesServerConfig


_REQUEST = {
    "repo": "astropy/astropy",
    "instance_id": "astropy__astropy-12907",
    "base_commit": "my base_commit",
    "patch": "my patch",
    "test_patch": "my test_patch",
    "problem_statement": "my problem_statement",
    "hints_text": "",
    "created_at": "my created_at",
    "version": "4.3",
    "FAIL_TO_PASS": "[]",
    "PASS_TO_PASS": "[]",
    "environment_setup_commit": "my environment_setup_commit",
    "difficulty": "my difficulty",
    "responses_create_params": {"input": []},
    "response": {
        "output": [],
        "id": "",
        "created_at": 0,
        "model": "",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
    "subset": "my subset",
    "split": "my split",
}


def _server(monkeypatch: MonkeyPatch, results_dir: Path, *, keep: bool) -> SwebenchResourcesServer:
    monkeypatch.setattr(
        runtime_files, "maybe_get_global_config_dict", lambda: {RESULTS_DIR_KEY_NAME: str(results_dir)}
    )
    config = SwebenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swebench",
        sandbox_provider="test",
        sandbox_config=dict(),
        is_verifying_golden_patch=True,
        keep_verification_logs=keep,
    )
    sandbox = MagicMock()
    sandbox._handle = SandboxHandle(sandbox_id="sandbox-123", provider_name="test-provider", raw=None)
    sandbox.stop = AsyncMock()
    monkeypatch.setattr(SwebenchResourcesServer, "_create_sandbox", AsyncMock(return_value=sandbox))
    return SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class _Harness:
    """Stands in for run_instance: writes a test log the way the harness does, then succeeds or fails."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.log_root: Path | None = None

    async def __call__(self, *, run_id: str, log_root: Path, **kwargs) -> dict:
        self.log_root = log_root
        instance_dir = log_root / run_id / "None" / _REQUEST["instance_id"]
        instance_dir.mkdir(parents=True)
        (instance_dir / "test_output.txt").write_text("test log")
        if self.error is not None:
            raise self.error
        return {"completed": True, "resolved": True}


def test_logs_go_to_a_temporary_folder_that_is_removed(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    harness = _Harness()
    monkeypatch.setattr(swebench_app, "run_instance", harness)

    response = TestClient(_server(monkeypatch, tmp_path, keep=False).setup_webserver()).post("/verify", json=_REQUEST)

    assert response.status_code == 200
    assert response.json()["test_output"] == "test log"
    assert response.json()["log_dir"] == ""
    assert not harness.log_root.is_relative_to(Path(swebench_app.__file__).parent)
    assert not harness.log_root.exists()
    assert not any(tmp_path.iterdir())


def test_kept_logs_are_copied_under_the_server_results(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    harness = _Harness()
    monkeypatch.setattr(swebench_app, "run_instance", harness)

    response = TestClient(_server(monkeypatch, tmp_path, keep=True).setup_webserver()).post("/verify", json=_REQUEST)

    assert response.status_code == 200
    log_dir = Path(response.json()["log_dir"])
    assert log_dir.relative_to(tmp_path).parts[:2] == ("swebench", "verification")
    assert "-astropy__astropy-12907-attempt1-" in log_dir.name
    [kept] = log_dir.rglob("test_output.txt")
    assert kept.read_text() == "test log"
    assert not harness.log_root.exists()


def test_a_verify_that_fails_still_removes_its_logs(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    harness = _Harness(error=RuntimeError("harness crashed"))
    monkeypatch.setattr(swebench_app, "run_instance", harness)

    TestClient(_server(monkeypatch, tmp_path, keep=False).setup_webserver(), raise_server_exceptions=False).post(
        "/verify", json=_REQUEST
    )

    assert harness.log_root is not None and not harness.log_root.exists()


def test_the_old_setting_still_loads_and_says_how_to_keep_logs(caplog: pytest.LogCaptureFixture) -> None:
    config = SwebenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        sandbox_provider="test",
        sandbox_config=dict(),
        clear_swebench_debug_logs=False,
    )

    assert config.keep_verification_logs is False
    assert "Set keep_verification_logs" in caplog.text
