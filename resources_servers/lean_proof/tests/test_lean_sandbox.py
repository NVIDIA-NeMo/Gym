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

"""Tests for the compile command LeanSandbox builds, and for running without a Gym config."""

from unittest.mock import AsyncMock

import pytest

from resources_servers.lean_proof.lean_sandbox import DEFAULT_LEAN_PROJECT_DIR, LeanSandbox


INLINE_PROVIDER = {"enroot": {"create": {"bypass_entrypoint": False}}}


def _sandbox(**overrides) -> LeanSandbox:
    return LeanSandbox(
        sandbox_provider=INLINE_PROVIDER,
        sandbox_config={"image": "gym-lean:test", **overrides},
    )


@pytest.mark.asyncio
async def test_compile_writes_the_file_and_removes_it():
    """The proof goes in through a heredoc, so quotes and unicode need no escaping."""
    lean = _sandbox()
    exec_mock = AsyncMock()
    lean.start = AsyncMock(return_value=type("S", (), {"exec": exec_mock})())

    await lean.compile('theorem t : "α ≫ β" := by\n  sorry', timeout_s=300)

    command = exec_mock.await_args.args[0]
    assert "cat > /tmp/attempt_" in command
    assert 'theorem t : "α ≫ β" := by' in command
    assert "lake env lean" in command
    assert "rm -f /tmp/attempt_" in command, "a long-lived sandbox must not accumulate one file per rollout"
    assert exec_mock.await_args.kwargs["cwd"] == DEFAULT_LEAN_PROJECT_DIR


@pytest.mark.asyncio
async def test_compile_gives_the_sandbox_headroom_over_the_lean_budget():
    """The sandbox, not the client, should be the one to report a timeout."""
    lean = _sandbox()
    exec_mock = AsyncMock()
    lean.start = AsyncMock(return_value=type("S", (), {"exec": exec_mock})())

    await lean.compile("import Mathlib", timeout_s=300)

    assert exec_mock.await_args.kwargs["timeout_s"] > 300


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides,expected", [({}, "lean"), ({"compile_user": None}, None)])
async def test_compile_runs_as_an_unprivileged_user(overrides, expected):
    """`#eval` runs during elaboration; as root a submission could rewrite the toolchain."""
    lean = _sandbox(**overrides)
    exec_mock = AsyncMock()
    lean.start = AsyncMock(return_value=type("S", (), {"exec": exec_mock})())

    await lean.compile("import Mathlib", timeout_s=10)

    assert exec_mock.await_args.kwargs["user"] == expected


@pytest.mark.asyncio
async def test_each_compile_uses_a_fresh_filename():
    """One sandbox serves many concurrent verifies."""
    lean = _sandbox()
    exec_mock = AsyncMock()
    lean.start = AsyncMock(return_value=type("S", (), {"exec": exec_mock})())

    await lean.compile("import Mathlib", timeout_s=10)
    first = exec_mock.await_args.args[0]
    await lean.compile("import Mathlib", timeout_s=10)
    second = exec_mock.await_args.args[0]

    assert first != second


@pytest.mark.asyncio
async def test_named_provider_without_a_global_config_is_a_clear_error():
    """A named provider needs the config; saying so beats a Hydra parse of sys.argv."""
    lean = LeanSandbox(sandbox_provider="sandbox", sandbox_config={"image": "gym-lean:test"})
    with pytest.raises(RuntimeError, match="no Gym global config"):
        await lean.start()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,warns", [({"enroot": {}}, True), ({"opensandbox": {}}, False)])
async def test_start_warns_when_the_provider_does_not_isolate(provider, warns, monkeypatch, caplog):
    """Enroot runs submitted code as the host user; the operator should be told."""
    monkeypatch.setattr(
        "resources_servers.lean_proof.lean_sandbox.AsyncSandbox",
        lambda _provider: type("S", (), {"start": AsyncMock()})(),
    )
    lean = LeanSandbox(sandbox_provider=provider, sandbox_config={"image": "gym-lean:test"})

    with caplog.at_level("WARNING"):
        await lean.start()

    assert ("no uid switch" in caplog.text) is warns
