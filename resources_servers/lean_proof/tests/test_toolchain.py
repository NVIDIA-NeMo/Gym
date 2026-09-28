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

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from resources_servers.lean_proof.toolchain import (
    PROBE_TIMEOUT,
    TOOLCHAIN_PROBE,
    ToolchainCheck,
    normalize_version,
    parse_lean_version,
)


class TestParsing:
    @pytest.mark.parametrize(
        "output,expected",
        [
            ({"stdout": '"4.19.0"', "stderr": ""}, "4.19.0"),
            ({"stdout": "", "stderr": "4.12.0"}, "4.12.0"),
            ({"stdout": "info: 4.19.0", "stderr": ""}, "4.19.0"),
            # A failed `import Mathlib` must not be read as a version.
            ({"stdout": "", "stderr": "error: unknown package 'Mathlib'"}, None),
            ({"stdout": "", "stderr": ""}, None),
        ],
    )
    def test_parse_lean_version(self, output, expected):
        assert parse_lean_version(output) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("leanprover/lean4:v4.19.0", "4.19.0"),
            ("v4.19.0", "4.19.0"),
            ("4.19.0", "4.19.0"),
            (None, ""),
        ],
    )
    def test_normalize_version(self, raw, expected):
        assert normalize_version(raw) == expected


class TestToolchainCheck:
    def _client(self, **output) -> MagicMock:
        client = MagicMock()
        client.execute_lean4 = AsyncMock(return_value={"stdout": "", "stderr": ""} | output)
        return client

    @pytest.mark.asyncio
    async def test_probe_imports_mathlib_with_its_own_budget(self):
        client = self._client(stdout='"4.19.0"')
        await ToolchainCheck("4.19.0").run(client)
        client.execute_lean4.assert_awaited_once_with(code=TOOLCHAIN_PROBE, timeout=PROBE_TIMEOUT)

    @pytest.mark.asyncio
    async def test_mismatch_is_logged_as_an_error(self, caplog):
        with caplog.at_level("ERROR"):
            await ToolchainCheck("4.19.0").run(self._client(stdout='"4.12.0"'))
        assert "MATHLIB MISMATCH" in caplog.text
        assert "4.12.0" in caplog.text and "4.19.0" in caplog.text

    @pytest.mark.asyncio
    async def test_matching_version_logs_no_error(self, caplog):
        with caplog.at_level("ERROR"):
            await ToolchainCheck("leanprover/lean4:v4.19.0").run(self._client(stdout='"4.19.0"'))
        assert caplog.text == ""

    @pytest.mark.asyncio
    async def test_row_pin_overrides_server_default(self, caplog):
        with caplog.at_level("ERROR"):
            await ToolchainCheck("4.12.0").run(self._client(stdout='"4.19.0"'), expected_override="v4.19.0")
        assert caplog.text == ""

    @pytest.mark.asyncio
    async def test_unusable_sandbox_is_logged(self, caplog):
        with caplog.at_level("ERROR"):
            await ToolchainCheck("4.19.0").run(self._client(stderr="error: boom"))
        assert "Could not determine" in caplog.text

    @pytest.mark.asyncio
    async def test_probe_runs_only_once(self):
        client = self._client(stdout='"4.19.0"')
        check = ToolchainCheck("4.19.0")
        await asyncio.gather(*(check.run(client) for _ in range(10)))
        assert client.execute_lean4.await_count == 1, "concurrent verifies must not each pay a Mathlib compile"
