# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for NVARC resource server.

All grid parsing tests go through Board.from_text() from problem.py.
Code extraction and subprocess execution tested independently.
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _app_dir)

import app as nvarc_app
from app import (
    NVARCResourcesServer,
    NVARCResourcesServerConfig,
    _execute_python,
    _extract_python_code,
    _parse_grid,
)

from nemo_gym.server_utils import ServerClient


# ============================================================================
# Load real examples
# ============================================================================

_data_path = os.path.join(_app_dir, "data", "example.jsonl")
_examples = []
if os.path.exists(_data_path):
    with open(_data_path) as f:
        for line in f:
            if line.strip():
                _examples.append(json.loads(line))

_transductive = [e for e in _examples if e.get("agent_mode") == "transductive"]
_inductive = [e for e in _examples if e.get("agent_mode") == "inductive"]


# ============================================================================
# Unit tests: Grid parsing (Board.from_text)
# ============================================================================


class TestParseGrid:
    def test_boxed_text_grid(self):
        assert _parse_grid(r"\boxed{1 2" + "\n" + "3 4}") == [[1, 2], [3, 4]]

    def test_text_grid_integers(self):
        assert _parse_grid("0 1 0\n1 1 1\n0 1 0") == [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    def test_thinking_stripped(self):
        result = _parse_grid("<think>reasoning</think>\\boxed{0 1\n1 0}")
        assert result is not None

    def test_invalid_returns_none(self):
        assert _parse_grid("no grid here at all") is None

    def test_empty_returns_none(self):
        assert _parse_grid("") is None

    def test_jagged_grid_rejected(self):
        # Board.from_text raises ValueError for jagged grids
        assert _parse_grid("0 1 2\n3 4") is None


class TestExtractPythonCode:
    def test_markdown_python_block(self):
        code = _extract_python_code("```python\ndef transform(g):\n    return g\n```")
        assert code is not None and "def transform" in code

    def test_bare_function(self):
        assert _extract_python_code("def transform(grid):\n    return [[0]]") is not None

    def test_no_code(self):
        assert _extract_python_code("just text") is None

    def test_thinking_stripped(self):
        code = _extract_python_code("<think>hmm</think>\n```python\ndef transform(g):\n    return g\n```")
        assert code is not None and "def transform" in code


# ============================================================================
# Unit tests: Subprocess execution
# ============================================================================


class TestExecutePython:
    def test_correct_transform(self):
        code = "def transform(grid):\n    return [[c + 1 for c in row] for row in grid]"
        result = asyncio.run(_execute_python(code, [[0, 1], [2, 3]], timeout_seconds=10))
        assert result == [[1, 2], [3, 4]]

    def test_identity(self):
        result = asyncio.run(_execute_python("def transform(g):\n    return g", [[5, 6]], timeout_seconds=10))
        assert result == [[5, 6]]

    def test_syntax_error_returns_none(self):
        result = asyncio.run(_execute_python("def transform(g):\n    return g +", [[0]], timeout_seconds=10))
        assert result is None

    def test_no_transform_returns_none(self):
        result = asyncio.run(_execute_python("x = 42", [[0]], timeout_seconds=10))
        assert result is None

    def test_runtime_error_returns_none(self):
        result = asyncio.run(_execute_python("def transform(g):\n    return g[999]", [[0]], timeout_seconds=10))
        assert result is None

    async def test_spawn_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            nvarc_app.asyncio,
            "create_subprocess_exec",
            AsyncMock(side_effect=OSError("process limit reached")),
        )

        assert await _execute_python("def transform(g):\n    return g", [[0]], timeout_seconds=1) is None

    async def test_timeout_terminates_and_reaps_child(self, monkeypatch):
        process = _FakeProcess(terminate_exits=True)
        monkeypatch.setattr(nvarc_app, "_PROCESS_TIMEOUT_GRACE_SECONDS", 0.01)
        monkeypatch.setattr(nvarc_app.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

        assert await _execute_python("def transform(g):\n    return g", [[0]], timeout_seconds=0) is None
        assert process.terminate_calls == 1
        assert process.kill_calls == 0
        assert process.returncode == -15

    async def test_timeout_kills_child_that_ignores_terminate(self, monkeypatch):
        process = _FakeProcess(terminate_exits=False)
        monkeypatch.setattr(nvarc_app, "_PROCESS_TIMEOUT_GRACE_SECONDS", 0.01)
        monkeypatch.setattr(nvarc_app, "_PROCESS_TERMINATION_GRACE_SECONDS", 0.01)
        monkeypatch.setattr(nvarc_app.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

        assert await _execute_python("def transform(g):\n    return g", [[0]], timeout_seconds=0) is None
        assert process.terminate_calls == 1
        assert process.kill_calls == 1
        assert process.returncode == -9

    async def test_cancellation_reaps_child_before_propagating(self, monkeypatch):
        process = _FakeProcess(terminate_exits=True)
        monkeypatch.setattr(nvarc_app, "_PROCESS_TIMEOUT_GRACE_SECONDS", 60.0)
        monkeypatch.setattr(nvarc_app.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
        task = asyncio.create_task(_execute_python("def transform(g):\n    return g", [[0]], timeout_seconds=30))
        await process.communicate_started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.terminate_calls == 1
        assert process.returncode == -15

    @pytest.mark.parametrize(
        ("returncode", "stdout", "expected"),
        [
            (0, b'{"success": true, "result": [[1]]}', [[1]]),
            (1, b"", None),
            (0, b"", None),
            (0, b"not json", None),
            (0, b'{"success": true, "result": [[1], [2, 3]]}', None),
        ],
        ids=["success", "nonzero-exit", "empty-output", "invalid-json", "invalid-board"],
    )
    async def test_completed_child_paths_are_reaped(self, monkeypatch, returncode, stdout, expected):
        process = _CompletedProcess(returncode=returncode, stdout=stdout)
        monkeypatch.setattr(nvarc_app.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

        assert await _execute_python("def transform(g):\n    return g", [[0]], timeout_seconds=1) == expected
        assert process.communicate_calls == 1


class _FakeProcess:
    def __init__(self, *, terminate_exits):
        self.returncode = None
        self.communicate_started = asyncio.Event()
        self._exited = asyncio.Event()
        self._terminate_exits = terminate_exits
        self.terminate_calls = 0
        self.kill_calls = 0

    async def communicate(self):
        self.communicate_started.set()
        await asyncio.Future()

    async def wait(self):
        await self._exited.wait()
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if self._terminate_exits:
            self.returncode = -15
            self._exited.set()

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9
        self._exited.set()


class _CompletedProcess:
    def __init__(self, *, returncode, stdout):
        self.returncode = returncode
        self.stdout = stdout
        self.communicate_calls = 0

    async def communicate(self):
        self.communicate_calls += 1
        return self.stdout, b""


async def test_inductive_execution_respects_per_worker_concurrency(monkeypatch):
    config = NVARCResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="nvarc",
        python_max_concurrency=2,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server = NVARCResourcesServer(config=config, server_client=server_client)
    active = 0
    max_active = 0

    async def execute(code, input_grid, timeout_seconds):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return input_grid

    monkeypatch.setattr(nvarc_app, "_execute_python", execute)

    results = await asyncio.gather(
        *(server._verify_inductive("def transform(g):\n    return g", [[index]]) for index in range(8))
    )

    assert max_active == 2
    assert results == [[[index]] for index in range(8)]


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux procfs")
async def test_repeated_failures_leave_no_children_or_file_descriptors():
    children_path = f"/proc/{os.getpid()}/task/{os.getpid()}/children"

    def child_pids():
        with open(children_path) as stream:
            return set(stream.read().split())

    baseline_children = child_pids()
    baseline_fd_count = len(os.listdir("/proc/self/fd"))

    results = await asyncio.gather(
        *(_execute_python("def transform(g):\n    return g +", [[index]], timeout_seconds=1) for index in range(32))
    )
    await asyncio.sleep(0.05)

    assert all(result is None for result in results)
    assert child_pids() == baseline_children
    assert len(os.listdir("/proc/self/fd")) <= baseline_fd_count


# ============================================================================
# Positive tests: correct answers from real examples
# ============================================================================


class TestTransductivePositive:
    @pytest.mark.parametrize("example", _transductive, ids=[e["task_id"] for e in _transductive])
    def test_correct_boxed(self, example):
        grid = example["expected_output"]
        # Simulate model response with correct grid in \boxed{}
        rows_text = "\n".join(" ".join(str(c) for c in row) for row in grid)
        response = f"<think>Analysis...</think>\n\\boxed{{{rows_text}}}"
        parsed = _parse_grid(response)
        assert parsed is not None, "Failed to parse correct grid"
        assert parsed == grid

    @pytest.mark.parametrize("example", _transductive[:3], ids=[e["task_id"] for e in _transductive[:3]])
    def test_correct_text_grid(self, example):
        grid = example["expected_output"]
        text = "\n".join(" ".join(str(c) for c in row) for row in grid)
        parsed = _parse_grid(text)
        assert parsed is not None
        assert parsed == grid


class TestInductivePositive:
    @pytest.mark.parametrize("example", _inductive, ids=[e["task_id"] for e in _inductive])
    def test_correct_hardcoded_transform(self, example):
        grid = example["expected_output"]
        code = f"def transform(input_grid):\n    return {json.dumps(grid)}\n"
        response = f"```python\n{code}```"
        extracted = _extract_python_code(response)
        assert extracted is not None
        result = asyncio.run(_execute_python(extracted, example["test_input"], timeout_seconds=10))
        assert result is not None, "Subprocess returned None"
        assert result == grid


# ============================================================================
# Negative tests: wrong/broken answers
# ============================================================================


class TestTransductiveNegative:
    @pytest.mark.parametrize("example", _transductive[:3], ids=[e["task_id"] for e in _transductive[:3]])
    def test_wrong_grid(self, example):
        wrong = [[0] * len(example["expected_output"][0])] * len(example["expected_output"])
        text = "\n".join(" ".join(str(c) for c in row) for row in wrong)
        response = f"\\boxed{{{text}}}"
        parsed = _parse_grid(response)
        assert parsed is not None, "Should still parse"
        assert parsed != example["expected_output"], "Should NOT match"

    def test_garbage_response(self):
        assert _parse_grid("I don't know the answer, sorry!") is None

    def test_wrong_shape(self):
        parsed = _parse_grid("1 2 3")
        assert parsed is not None  # Valid 1-row grid


class TestInductiveNegative:
    @pytest.mark.parametrize("example", _inductive[:3], ids=[e["task_id"] for e in _inductive[:3]])
    def test_wrong_transform(self, example):
        code = "def transform(grid):\n    return [[0 for c in row] for row in grid]"
        result = asyncio.run(_execute_python(code, example["test_input"], timeout_seconds=10))
        if result is not None:
            assert result != example["expected_output"]

    def test_infinite_loop(self):
        result = asyncio.run(_execute_python("def transform(g):\n    while True: pass", [[0]], timeout_seconds=3))
        assert result is None

    def test_import_os_blocked(self):
        result = asyncio.run(_execute_python("import os\ndef transform(g):\n    return g", [[0]], timeout_seconds=10))
        assert result is None

    def test_no_code_in_response(self):
        assert _extract_python_code("Here is my analysis but no code block") is None
