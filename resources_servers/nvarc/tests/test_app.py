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
import re
import sys

import pytest


_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _app_dir)

from problem import Board, ColorPalette


# ============================================================================
# Thin wrappers matching app.py logic (no nemo_gym import needed)
# ============================================================================


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_grid(text: str):
    """Same logic as app.py — Board.from_text() only."""
    text = _strip_thinking(text)
    try:
        board = Board.from_text(text, color_palette=ColorPalette.integers())
        if board.is_valid:
            return board.board
    except (ValueError, AttributeError, IndexError):
        pass
    return None


def _extract_python_code(text: str):
    """Same logic as app.py."""
    text = _strip_thinking(text)
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    if "def transform" in text:
        return text.strip()
    return None


# Read SUBPROCESS_TEMPLATE from app.py
import ast as _ast


with open(os.path.join(_app_dir, "app.py")) as _f:
    _tree = _ast.parse(_f.read())
SUBPROCESS_TEMPLATE = None
for _node in _ast.walk(_tree):
    if isinstance(_node, _ast.Assign):
        for _t in _node.targets:
            if isinstance(_t, _ast.Name) and _t.id == "SUBPROCESS_TEMPLATE":
                SUBPROCESS_TEMPLATE = _ast.literal_eval(_node.value)


async def _execute_python(code, input_grid, timeout_seconds=30):
    """Same logic as app.py — subprocess + Board validation."""
    script = SUBPROCESS_TEMPLATE.format(
        timeout_seconds=timeout_seconds,
        content_repr=repr(code),
        input_json=json.dumps(input_grid),
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds + 5)
    except asyncio.TimeoutError:
        return None
    if proc.returncode != 0:
        return None
    output = stdout.decode("utf-8", errors="replace").strip()
    if not output:
        return None
    try:
        result = json.loads(output)
        if result.get("success") and result.get("result") is not None:
            board = Board(board=result["result"])
            if board.is_valid:
                return board.board
    except (json.JSONDecodeError, Exception):
        pass
    return None


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
