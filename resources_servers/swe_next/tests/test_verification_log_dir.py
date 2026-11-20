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
"""Each verify writes its logs to a temporary folder that is removed afterwards, and keeps a copy only when asked."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config.global_config import RESULTS_DIR_KEY_NAME
from nemo_gym.core import runtime_files
from resources_servers.swe_next import app as swe_next_app
from resources_servers.swe_next.app import (
    SWENextResourcesServer,
    SWENextResourcesServerConfig,
    SWENextVerifyRequest,
)
from resources_servers.swe_next.verification import VerificationResult


@pytest.mark.asyncio
async def test_logs_go_to_a_temporary_folder_and_are_not_kept_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_dir = await _verify(monkeypatch, tmp_path, keep=False)

    assert not log_dir.is_relative_to(Path(swe_next_app.__file__).parent)
    assert not log_dir.exists()
    assert not any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_kept_logs_are_copied_under_the_server_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_dir = await _verify(monkeypatch, tmp_path, keep=True)

    # <results_dir>/<server>/verification/<rollout>/<time>-<instance id>-attempt<n>-<random>/
    [kept] = (tmp_path / "swe_next" / "verification").rglob("test_output.log")
    assert kept.read_text() == "test log"
    assert "-owner%2Frepo-1-attempt1-" in kept.parent.name
    assert not log_dir.exists()


_INSTANCE_ID = "owner/repo-1"


async def _verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, keep: bool) -> Path:
    """Run one golden-patch verify with results under ``tmp_path``; return the log folder the verifier wrote to."""
    monkeypatch.setattr(runtime_files, "maybe_get_global_config_dict", lambda: {RESULTS_DIR_KEY_NAME: str(tmp_path)})
    monkeypatch.setattr(swe_next_app, "get_global_config_dict", lambda: {})
    config = SWENextResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swe_next",
        sandbox_provider="docker",
        sandbox_config={},
        is_verifying_golden_patch=True,
        keep_verification_logs=keep,
    )
    server = SWENextResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "_stop_verifier_sandbox", AsyncMock())
    log_dirs: list[Path] = []

    async def verify(*, log_dir: Path, **kwargs: object) -> VerificationResult:
        log_dirs.append(log_dir)
        (log_dir / "test_output.log").write_text("test log")
        return VerificationResult(True, True, True, {}, "ok")

    monkeypatch.setattr(swe_next_app, "run_verification", verify)
    body = SWENextVerifyRequest.model_validate(
        {
            "instance_id": _INSTANCE_ID,
            "workdir": "/testbed",
            "image_ref": "example/image:tag",
            "patch": "diff --git a/a.py b/a.py\n",
            "expected_output_json": "{}",
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
        }
    )
    await server.verify(SimpleNamespace(session={SESSION_ID_KEY: "session"}), body)
    [log_dir] = log_dirs
    return log_dir
