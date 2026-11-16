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
"""Verification logs stay out of the server's source folder unless debug_dir names a folder."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config.global_config import DEBUG_DIR_KEY_NAME
from nemo_gym.core import runtime_files
from resources_servers.swe_next import app as swe_next_app
from resources_servers.swe_next.app import (
    SWENextResourcesServer,
    SWENextResourcesServerConfig,
    SWENextVerifyRequest,
)
from resources_servers.swe_next.verification import VerificationResult


_INSTANCE_ID = "owner/repo-1"


async def _verify_log_dir(monkeypatch: pytest.MonkeyPatch, log_root: Path | None) -> Path | None:
    """Run one golden-patch verify and return the log folder the verifier was handed."""
    config = SWENextResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swe_next",
        sandbox_provider="docker",
        sandbox_config={},
        is_verifying_golden_patch=True,
        debug_dir=log_root,
    )
    server = SWENextResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(server, "_stop_sandbox", AsyncMock())
    verify = AsyncMock(return_value=VerificationResult(True, True, True, {}, "ok"))
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
    return verify.await_args.kwargs["log_dir"]


@pytest.mark.asyncio
async def test_no_log_folder_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _verify_log_dir(monkeypatch, None) is None


@pytest.mark.asyncio
async def test_logs_go_under_the_configured_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_dir = await _verify_log_dir(monkeypatch, tmp_path)

    # <debug_dir>/verification/<rollout>/<agent instance>/<time>-<instance id>-<random>/attempt-1
    assert log_dir.relative_to(tmp_path).parts[0] == "verification"
    assert "-owner%2Frepo-1-" in log_dir.parent.name
    assert log_dir.name == "attempt-1"


@pytest.mark.asyncio
async def test_the_global_debug_dir_applies_under_the_server_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime_files, "maybe_get_global_config_dict", lambda: {DEBUG_DIR_KEY_NAME: str(tmp_path)})

    log_dir = await _verify_log_dir(monkeypatch, None)

    assert log_dir.relative_to(tmp_path).parts[:2] == ("swe_next", "verification")
