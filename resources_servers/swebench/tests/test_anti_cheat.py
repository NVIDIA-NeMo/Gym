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
"""Tests for the shared anti-cheat git-scrub helper multiple resources servers call from their
own ``seed_session`` (see anti_cheat.py's module docstring)."""

from types import SimpleNamespace

import pytest

from resources_servers.swebench.anti_cheat import ANTI_CHEAT_SCRIPT_FPATH, apply_anti_cheat_setup


class _FakeSandbox:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.uploads: list[tuple[str, str]] = []
        self.exec_calls: list[dict] = []

    async def upload(self, local_path, remote_path: str) -> None:
        self.uploads.append((str(local_path), remote_path))

    async def exec(self, command: str, **kwargs):
        self.exec_calls.append({"command": command} | kwargs)
        return SimpleNamespace(return_code=self.return_code, stdout="", stderr="")


class TestApplyAntiCheatSetup:
    @pytest.mark.asyncio
    async def test_uploads_the_shared_script_into_the_workdir(self) -> None:
        sandbox = _FakeSandbox()
        await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server")
        assert sandbox.uploads == [(str(ANTI_CHEAT_SCRIPT_FPATH), "/testbed/anti_cheat_setup.sh")]

    @pytest.mark.asyncio
    async def test_runs_it_from_the_given_workdir(self) -> None:
        sandbox = _FakeSandbox()
        await apply_anti_cheat_setup(sandbox, "/workspace/repo", "inst-1", "some_server")
        assert len(sandbox.exec_calls) == 1
        call = sandbox.exec_calls[0]
        assert call["cwd"] == "/workspace/repo"
        assert "anti_cheat_setup.sh" in call["command"]
        assert "git reset --hard" in call["command"]

    @pytest.mark.asyncio
    async def test_a_failed_scrub_does_not_raise(self) -> None:
        """Best-effort: seed_session must still hand back a working (if unscrubbed) sandbox
        rather than failing the whole rollout over a cleanup step."""
        sandbox = _FakeSandbox(return_code=1)
        await apply_anti_cheat_setup(sandbox, "/testbed", "inst-1", "some_server")  # must not raise

    @pytest.mark.asyncio
    async def test_an_upload_exception_does_not_raise(self) -> None:
        class _BrokenSandbox(_FakeSandbox):
            async def upload(self, local_path, remote_path: str) -> None:
                raise RuntimeError("boom")

        await apply_anti_cheat_setup(_BrokenSandbox(), "/testbed", "inst-1", "some_server")  # must not raise
