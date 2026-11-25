# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise timeout and worker-exit classification through the real sandbox runner."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from lcb_integration import testing_util


_CODE_GEN = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "generation, fn_name, timeout, deadline, expected, stage",
    [
        ("def square(n):\n    return n*n", "square", 3, 30, [True, True], None),
        ("n=int(input())\nprint(n*n)", None, 3, 30, [True, True], None),
        (
            "import time\ntime.sleep(10)\ndef square(n):\n    return n*n",
            "square",
            0.2,
            30,
            [-3],
            "compilation",
        ),
        # stdio imports execute during compilation, before the wrapped candidate function.
        ("import slow_import\nprint(int(input())**2)", None, 0.2, 30, [-3], "compilation"),
        (
            "import time\ndef square(n):\n    time.sleep(10)\n    return n*n",
            "square",
            0.2,
            30,
            [-3],
            "test",
        ),
        ("import time\ntime.sleep(10)\nprint(int(input())**2)", None, 0.2, 30, [-3], "test"),
        # The outer deadline must expire before the per-test timer.
        ("import time\ntime.sleep(60)", None, 30, 2, [-3, -3], "grading"),
        ("import os\nos._exit(7)", None, 3, 30, [-1, -1], "worker"),
        ("import os\nos._exit(0)", None, 3, 30, [-1, -1], "worker"),
        # An exception mentioning the timeout class is still a runtime error.
        ("raise ValueError('TimeoutException')", None, 3, 30, [-4], None),
        ("raise ValueError('bad initialization')\ndef square(n):\n    return n*n", "square", 3, 30, [-4], None),
    ],
)
def test_runner_failure_classification(
    tmp_path: Path,
    generation: str,
    fn_name: str | None,
    timeout: float,
    deadline: float,
    expected: list[bool | int],
    stage: str | None,
) -> None:
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    (tmp_path / "slow_import.py").write_text("import time\ntime.sleep(10)\n")
    request.write_text(
        json.dumps(
            {
                "sample": {
                    "input_output": json.dumps(
                        {
                            "inputs": ["3", "5"],
                            "outputs": ["9", "25"],
                            "fn_name": fn_name,
                        }
                    )
                },
                "generation": generation,
                "timeout": timeout,
                "debug": False,
                "deadline_s": deadline,
            }
        )
    )
    execution = subprocess.run(
        [sys.executable, str(_CODE_GEN / "sandbox/sandbox_runner.py"), str(request), str(result)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(_CODE_GEN), str(tmp_path)))},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert execution.returncode == 0, execution.stderr
    payload = json.loads(result.read_text())
    assert payload["result"] == expected, payload
    metadata = payload["metadata"]
    assert metadata.get("stage") == stage, metadata
    if stage in {"compilation", "test", "grading"}:
        assert metadata["error_code"] == -3
        assert metadata["error_message"] == "Time Limit Exceeded"
    elif stage == "worker":
        assert metadata["error_code"] == -1
        assert metadata["exit_code"] == (7 if "_exit(7)" in generation else 0)
    elif expected == [-4]:
        assert metadata["error_code"] == -4
    else:
        assert "execution time" in metadata


@pytest.mark.parametrize("fn_name, grader", [("square", "grade_call_based"), (None, "grade_stdio")])
def test_preparation_timeout(fn_name: str | None, grader: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the guard from changing the pytest process; inject the timer exception at the preparation boundary.
    monkeypatch.setattr(testing_util, "reliability_guard", lambda **kwargs: None)

    def timeout_during_preparation(**kwargs):
        raise testing_util.TimeoutException

    monkeypatch.setattr(testing_util, grader, timeout_during_preparation)
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        result, metadata = testing_util.run_test(
            {"inputs": ["3"], "outputs": ["9"], "fn_name": fn_name}, test="pass", timeout=10
        )
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
    assert result == [-3]
    assert metadata["error_code"] == -3
    assert metadata["stage"] == "preparation"
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
