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
"""The OpenHands capture overlay patches ``ServerClient.request`` on both nemo-gym layouts.

The overlay runs inside the OpenHands runtime, whose nemo-gym is locked by nv-OpenHands and can
predate the ``nemo_gym.client`` package.
"""

import asyncio
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.client import ServerClient, http_client


OVERLAY = Path(__file__).resolve().parents[1] / "openhands_capture_overlay" / "sitecustomize.py"


def test_overlay_routes_the_model_server_to_its_override_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ServerClient, "request", ServerClient.request)  # undone after the test
    sent = AsyncMock(return_value="response")
    monkeypatch.setattr(http_client, "request", sent)
    monkeypatch.setenv("NEMO_GYM_MODEL_SERVER_NAME", "policy")
    monkeypatch.setenv("NEMO_GYM_MODEL_SERVER_BASE_URL", "http://model:8000/")

    runpy.run_path(str(OVERLAY))
    # The routed path does not use the client's own state, so no configured ServerClient is needed.
    routed = ServerClient.request(None, server_name="policy", url_path="/v1/responses", method="POST", json={})
    result = asyncio.run(routed)

    assert result == "response"
    sent.assert_awaited_once_with(method="POST", url="http://model:8000/v1/responses", _internal=True, json={})


def test_overlay_patches_a_nemo_gym_from_before_the_client_package(tmp_path: Path) -> None:
    # A stand-in for the locked nemo-gym: only nemo_gym.server_runtime.base_server, as before 0.8.0.
    (tmp_path / "nemo_gym").mkdir()
    (tmp_path / "nemo_gym" / "__init__.py").write_text("")
    (tmp_path / "nemo_gym" / "server_utils.py").write_text(
        textwrap.dedent(
            """
            class ServerClient:
                async def request(self, server_name, url_path, method, **kwargs):
                    return "original"

            async def request(**kwargs):
                return kwargs
            """
        )
    )
    (tmp_path / "pydantic.py").write_text("class BaseModel:\n    pass\n")
    check = textwrap.dedent(
        f"""
        import asyncio, os, runpy
        from nemo_gym.client import ServerClient
        os.environ["NEMO_GYM_MODEL_SERVER_NAME"] = "policy"
        os.environ["NEMO_GYM_MODEL_SERVER_BASE_URL"] = "http://model:8000"
        runpy.run_path({str(OVERLAY)!r})
        sent = asyncio.run(ServerClient().request(server_name="policy", url_path="/v1/responses", method="POST"))
        assert sent["url"] == "http://model:8000/v1/responses", sent
        """
    )
    # -S leaves out site-packages, so the installed nemo_gym cannot shadow the stand-in.
    result = subprocess.run(
        [sys.executable, "-S", "-c", check],
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
