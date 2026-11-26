# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task syntax is validated independently of candidate behavior."""

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest
from assertion_harness import check
from assertion_harness.check import InvalidAssertionTests, check_assertions


_CODE_GEN = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("failure", ["compilation", "memory limit"])
def test_worker_setup_failure_is_not_a_candidate_failure(monkeypatch, failure: str) -> None:
    # Fork keeps the injected setup failure in the worker, without imposing a real memory cap on pytest.
    monkeypatch.setattr(check, "multiprocessing", multiprocessing.get_context("fork"))
    monkeypatch.setattr(check, "address_space_bytes", lambda: 0)
    parent_pid = os.getpid()
    compile_tests = check._compile_tests

    def _compile(tests):
        if failure == "compilation" and os.getpid() != parent_pid:
            raise MemoryError("fixture compilation failed")
        return compile_tests(tests)

    def _limit(_):
        if failure == "memory limit":
            raise OSError("memory limit setup failed")
        return {}

    monkeypatch.setattr(check, "_compile_tests", _compile)
    monkeypatch.setattr(check, "limit_address_space", _limit)
    message = "fixture compilation failed" if failure == "compilation" else "memory limit setup failed"
    with pytest.raises(RuntimeError, match=f"Assertion worker setup failed:.*{message}"):
        check.check_assertions("answer = 42", ["assert answer == 42"], time_limit_s=1, memory_limit_mb=1, deadline_s=6)


@pytest.mark.parametrize("generation", ["answer = 42", "raise SystemExit(0)", "def broken(:"])
@pytest.mark.parametrize(
    "tests, index",
    [
        (["assert answer =="], 0),
        (["assert False", "assert answer =="], 1),
        (["pass\x00"], 0),
        # These parse successfully as ASTs but cannot compile as module statements.
        (["return 42"], 0),
        (["break"], 0),
        (["nonlocal answer"], 0),
        (["assert False", "return 42"], 1),
    ],
)
def test_invalid_tests_fail_before_worker_start(generation: str, tests: list[str], index: int) -> None:
    # On macOS, reaching memory measurement would raise FileNotFoundError instead.
    with pytest.raises(InvalidAssertionTests, match=f"index {index}") as info:
        check_assertions(generation, tests, time_limit_s=1, memory_limit_mb=256, deadline_s=6)
    assert isinstance(info.value.__cause__, (SyntaxError, ValueError))


@pytest.mark.parametrize("tests", [["assert answer =="], ["assert False", "assert answer =="], ["return 42"]])
def test_runner_rejects_invalid_tasks_without_executing_candidate(tmp_path: Path, tests: list[str]) -> None:
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    marker = tmp_path / "candidate-ran"
    request.write_text(
        json.dumps(
            {
                "format": "assertions",
                "generation": f"from pathlib import Path\nPath({str(marker)!r}).touch()",
                "tests": tests,
                "time_limit_s": 1,
                "memory_limit_mb": 256,
                "deadline_s": 6,
            }
        )
    )
    run = subprocess.run(
        [sys.executable, str(_CODE_GEN / "sandbox/sandbox_runner.py"), str(request), str(result)],
        env={**os.environ, "PYTHONPATH": str(_CODE_GEN)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert run.returncode != 0
    assert "InvalidAssertionTests: Invalid assertion test at index" in run.stderr
    assert not result.exists()
    assert not marker.exists()
